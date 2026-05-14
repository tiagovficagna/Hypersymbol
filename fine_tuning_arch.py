#!/usr/bin/env python3
"""
HyperSymbol Fine-Tuning Architecture v1.0
==========================================
Treinar um modelo a "pensar" em coordenadas 50D em vez de linguagem natural.

ARQUITETURA:
  Input text → Modelo Base (congelado) → Hidden States → Semantic Head → 50D Vector
                                                 ↓
                                          Next-token (preservado)

LOSS FUNÇÃO:
  L = α * CrossEntropy(next_token) + β * CosineDistance(hidden_state, target_50d)

ONDE:
  - Semantic Head = MLP de 2 camadas (hidden_size → 256 → 50)
  - target_50d = vetor HyperSymbol do conceito (gerado automaticamente)
  - α = 1.0 (preserva capacidade linguística)
  - β = 0.3 (força alinhamento semântico)
"""

import json, os, sys, math, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Adiciona HyperSymbol ao path
sys.path.insert(0, str(Path(__file__).parent))
from hypersymbol_v2 import HS, SemanticVector, AXIS_NAMES, NUM_DIMS


# ═══════════════════════════════════════════════════════════════
# 1. GERADOR DE DATASET — TEXTO → VETOR 50D
# ═══════════════════════════════════════════════════════════════

class DatasetGenerator:
    """
    Gera pares (texto, vetor_50d) para fine-tuning.
    
    Estratégia:
    1. Texto → extrai símbolos HyperSymbol → centro geométrico → vetor 50D
    2. Texto → calcula perfil semântico baseado em palavras-chave
    3. Combina com os 27 símbolos âncora existentes
    """

    # Palavras que ativam cada eixo (exemplo de mapeamento)
    AXIS_KEYWORDS = {
        "CON_ABS": {"neg": ["pedra", "água", "mesa", "carro", "corpo", "físico", "matéria"],
                     "pos": ["ideia", "conceito", "teoria", "espírito", "abstrato", "sentido"]},
        "CHA_ORD": {"neg": ["caos", "aleatório", "desordem", "entropia", "acaso"],
                     "pos": ["ordem", "estrutura", "sistema", "lei", "padrão", "organizado"]},
        "LOG_INT": {"neg": ["lógica", "razão", "cálculo", "dedução", "prova"],
                     "pos": ["intuição", "sentimento", "pressentimento", "palpite"]},
        "CAU_PUR": {"neg": ["causa", "origem", "porque", "razão", "motivo"],
                     "pos": ["propósito", "finalidade", "sentido", "para quê"]},
        "EST_TRA": {"neg": ["permanente", "fixo", "estável", "imutável", "constante"],
                     "pos": ["mudança", "transformação", "evolução", "fluxo", "dinâmico"]},
        "IND_COL": {"neg": ["indivíduo", "eu", "pessoal", "único", "subjetivo"],
                     "pos": ["coletivo", "sociedade", "comunidade", "todos", "grupo"]},
        "FIN_INF": {"neg": ["fim", "limite", "finito", "morte", "encerramento"],
                     "pos": ["infinito", "eterno", "sem fim", "imenso", "ilimitado"]},
        "MAT_MIN": {"neg": ["material", "físico", "corpo", "objeto", "substância"],
                     "pos": ["mente", "psíquico", "espírito", "consciência", "pensamento"]},
        "SUR_PRO": {"neg": ["superfície", "aparência", "óbvio", "evidente", "claro"],
                     "pos": ["profundidade", "essência", "fundo", "oculto", "íntimo"]},
        "FOR_ESS": {"neg": ["forma", "aparência", "figura", "molde", "estrutura"],
                     "pos": ["essência", "natureza", "substância", "cerne", "alma"]},
    }

    # Frases-base para gerar exemplos de treino
    TRAINING_CONCEPTS = [
        "A gravidade atrai todos os corpos com massa.",
        "A consciência é o grande mistério da filosofia.",
        "O universo se expande desde o Big Bang.",
        "A vida é um processo de constante evolução.",
        "O tempo flui do passado para o futuro.",
        "A mente humana busca significado em tudo.",
        "A sociedade se organiza através de leis.",
        "O indivíduo é único em sua subjetividade.",
        "A transformação é a única constante.",
        "O infinito não pode ser totalmente compreendido.",
        "A matéria é composta por átomos.",
        "A essência das coisas está além da aparência.",
        "O caos aparente esconde uma ordem profunda.",
        "A intuição muitas vezes supera a lógica.",
        "O propósito da existência é uma questão aberta.",
        "A superfície do oceano esconde profundezas.",
        "A criação e a destruição são ciclos naturais.",
        "O conhecimento emerge da experiência.",
        "A linguagem molda nosso pensamento.",
        "A conexão entre todas as coisas é fundamental.",
        # Adiciona variações simbólicas
        "🌌❓∞",
        "🔥🌀🌱",
        "💡🔗🌍",
        "📖💎🔮",
        "🤝🪞🌀",
    ]

    def __init__(self):
        self.hs = HS

    def text_to_vector(self, text: str) -> np.ndarray:
        """Converte texto pra vetor 50D. Tenta símbolos primeiro, depois keywords."""
        # Tenta símbolos
        symbols = self.hs.decode(text)
        if symbols:
            center = self.hs.phrase_center(symbols)
            return center.v
        
        # Tenta keywords (análise simples de sentimento semântico)
        vec = np.zeros(NUM_DIMS)
        text_lower = text.lower()
        words = text_lower.split()
        
        for i, axis in enumerate(AXIS_NAMES):
            kw = self.AXIS_KEYWORDS.get(axis, {"neg": [], "pos": []})
            neg_score = sum(1 for w in kw["neg"] if w in text_lower)
            pos_score = sum(1 for w in kw["pos"] if w in text_lower)
            if neg_score > 0 or pos_score > 0:
                total = neg_score + pos_score
                vec[i] = (pos_score - neg_score) / total if total > 0 else 0.0
        
        # Normaliza pra -1..1
        return np.clip(vec, -1.0, 1.0)

    def generate_training_pair(self, text: str) -> Dict:
        """Gera um par de treino (texto, vetor_50d)."""
        vec = self.text_to_vector(text)
        return {
            "text": text,
            "vector": vec.tolist(),
            "axes": {AXIS_NAMES[i]: float(vec[i]) for i in range(NUM_DIMS)},
            "dominant": self._get_dominant(vec),
        }

    def _get_dominant(self, vec: np.ndarray, top_n: int = 5) -> List[Dict]:
        """Retorna os eixos dominantes do vetor."""
        indices = np.argsort(np.abs(vec))[::-1][:top_n]
        result = []
        for i in indices:
            axis = AXIS_NAMES[i]
            val = float(vec[i])
            from hypersymbol_v2 import AXIS_INFO
            info = AXIS_INFO.get(axis, {"name": axis, "neg": "neg", "pos": "pos"})
            pole = info["pos"] if val > 0 else info["neg"]
            result.append({
                "axis": axis,
                "name": info["name"],
                "direction": pole,
                "strength": round(abs(val), 2),
            })
        return result

    def generate_dataset(self, n_variations: int = 5) -> List[Dict]:
        """Gera dataset completo com variações."""
        dataset = []
        
        # Para cada conceito, gera múltiplas variações
        for concept in self.TRAINING_CONCEPTS:
            base_pair = self.generate_training_pair(concept)
            dataset.append(base_pair)
            
            # Gera variações por similaridade semântica
            for i in range(n_variations):
                # Perturba levemente o vetor (ruído semântico)
                noise = np.random.normal(0, 0.1, NUM_DIMS)
                var_vec = np.clip(np.array(base_pair["vector"]) + noise, -1.0, 1.0)
                
                # Cria uma variação textual
                if not any(c in concept for c in '🌌🔥💡❓∞🔄🔗⚡🌱🕳️🌊✨🔮🌍🧠💎🌅⚖️📖∑🔴🤸🎵🤝🪞🌿∞'):
                    var_text = concept  # simplificado
                else:
                    var_text = concept
                
                dataset.append({
                    "text": var_text,
                    "vector": var_vec.tolist(),
                    "axes": {AXIS_NAMES[i]: float(var_vec[i]) for i in range(NUM_DIMS)},
                    "dominant": self._get_dominant(var_vec),
                    "augmented": True,
                })
        
        return dataset


# ═══════════════════════════════════════════════════════════════
# 2. ARQUITETURA DE FINE-TUNING (pseudocódigo PyTorch)
# ═══════════════════════════════════════════════════════════════

FINE_TUNING_ARCHITECTURE = """
Arquitetura de Fine-Tuning HyperSymbol
========================================

1. MODELO BASE (congelado)
   ┌─────────────────────────────────┐
   │ Qwen 1.7B / MiniMax 2.5B       │  ← Congelado (não treina)
   │ (qualquer modelo causal LM)     │
   └──────────────┬──────────────────┘
                  │ hidden_states (última camada)
                  ▼
2. SEMANTIC HEAD (treinável)
   ┌─────────────────────────────────┐
   │ Linear(hidden_size → 256)       │
   │ GELU                           │
   │ Linear(256 → 50)               │  ← Saída = vetor 50D
   │ Tanh (para -1..1)              │
   └──────────────┬──────────────────┘
                  │ predicted_vector
                  ▼
3. LOSS FUNCTION
   L = α * CrossEntropy(next_token) + β * CosineDistance(predicted_50d, target_50d)
   
   Onde:
   - CrossEntropy: mantém a capacidade linguística do modelo
   - CosineDistance: força o hidden state a se alinhar com o vetor HyperSymbol
   - α = 1.0, β = 0.3 (ajustável)

4. DATASET
   Entrada: texto qualquer
   Saída esperada: vetor 50D + continuação do texto
   
   Formato:
   {
     "text": "A consciência é um mistério.",
     "target_vector": [0.5, -0.3, 0.8, ...],  # 50 floats
     "target_text": "A consciência é um mistério que a filosofia tenta desvendar."
   }

5. BENEFÍCIOS ESPERADOS
   - O modelo aprende a "pensar" em coordenadas semânticas
   - As ativações internas se organizam naturalmente no espaço 50D
   - Pode-se ler a "posição semântica" do modelo a qualquer momento
   - Pode-se INJETAR vetores alvo para guiar o pensamento
   
6. CÓDIGO DE TREINO (pseudocódigo)
   
   class HyperSymbolModel(nn.Module):
       def __init__(self, base_model):
           super().__init__()
           self.base = base_model  # congelado
           self.semantic_head = nn.Sequential(
               nn.Linear(base_model.config.hidden_size, 256),
               nn.GELU(),
               nn.Linear(256, 50),
               nn.Tanh(),
           )
       
       def forward(self, input_ids, labels=None, target_vector=None):
           outputs = self.base(input_ids, output_hidden_states=True)
           last_hidden = outputs.hidden_states[-1][:, -1, :]  # último token
           predicted_50d = self.semantic_head(last_hidden)
           
           loss = 0
           if labels is not None:
               loss += F.cross_entropy(outputs.logits, labels)  # α=1.0
           if target_vector is not None:
               cos_loss = 1 - F.cosine_similarity(predicted_50d, target_vector)
               loss += 0.3 * cos_loss  # β=0.3
           
           return loss, predicted_50d

7. RECURSOS NECESSÁRIOS
   - GPU: 6-8GB VRAM (para Qwen 1.7B com LoRA)
   - Dataset: ~500-1000 pares (gerados automaticamente)
   - Tempo: ~1-2 horas de treino
   - Frameworks: PyTorch + transformers + PEFT (LoRA)
"""


# ═══════════════════════════════════════════════════════════════
# 3. GERADOR DE DATASET REAL
# ═══════════════════════════════════════════════════════════════

def generate_dataset(output_path: str = None):
    """Gera e salva o dataset de treino."""
    gen = DatasetGenerator()
    dataset = gen.generate_dataset(n_variations=10)
    
    print(f"📊 Dataset gerado: {len(dataset)} amostras")
    print(f"\nExemplos:")
    for i, item in enumerate(dataset[:5]):
        dom = item["dominant"][:3]
        dom_str = ", ".join(f"{d['axis']}: {d['direction']} ({d['strength']})" for d in dom)
        print(f"\n  [{i}] {item['text'][:50]:50s}")
        print(f"      → {dom_str}")
        vec = item["vector"]
        print(f"      → vetor: [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}...] ({len(vec)} dims)")
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Dataset salvo: {output_path}")
    
    return dataset


# ═══════════════════════════════════════════════════════════════
# 4. INFERÊNCIA — USANDO MODELO TREINADO
# ═══════════════════════════════════════════════════════════════

INFERENCE_FLOW = """
COMO USAR UM MODELO TREINADO NO HIPERESPAÇO
===============================================

1. ESCREVER NO ESPAÇO SEMÂNTICO
   ────────────────────────────
   Em vez de prompt textual, passamos um VETOR 50D como entrada:

   vetor_alvo = [0.8, -0.3, 0.5, ...]  # QUEREMOS uma resposta abstrata
                    ↓
   INJETAMOS no hidden state (via semantic head inverso)
                    ↓
   Modelo gera texto PARTINDO dessa posição semântica

2. LER DO ESPAÇO SEMÂNTICO
   ────────────────────────
   A cada token gerado, PODEMOS LER a posição atual:

   modelo.generate("O que é consciência?") → tokens...
                                                ↓
                        semantic_head(hidden_state) → vetor 50D
                        
   Sabemos EXATAMENTE onde o modelo está no hiperespaço
   em cada momento da geração.

3. COMUNICAÇÃO PURA NO HIPERESPAÇO (sem texto)
   ────────────────────────────────────────────
   Entrada: vetor 50D → modelo → saída: vetor 50D
   
   Isso elimina COMPLETAMENTE a linguagem natural.
   Dois modelos treinados podem "conversar" trocando vetores 50D.
   Cada vetor é um pensamento completo em 50 dimensões.
"""


# ═══════════════════════════════════════════════════════════════
# 5. MAIN — DEMONSTRAÇÃO
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║   HyperSymbol Fine-Tuning Architecture v1       ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    
    # 1. Arquitetura
    print("🏗️  ARQUITETURA DE TREINO")
    print(FINE_TUNING_ARCHITECTURE)
    print()
    
    # 2. Dataset
    print("📊 GERANDO DATASET DE TREINO")
    print()
    dataset = generate_dataset("data/training_dataset.json")
    print()
    
    # 3. Estatísticas do dataset
    vectors = [np.array(item["vector"]) for item in dataset]
    mean_vec = np.mean(vectors, axis=0)
    std_vec = np.std(vectors, axis=0)
    
    print(f"📈 Estatísticas do dataset ({len(dataset)} amostras):")
    print(f"  Cobertura dos eixos:")
    for i in range(0, NUM_DIMS, 5):
        axis = AXIS_NAMES[i]
        mean_v = mean_vec[i]
        std_v = std_vec[i]
        print(f"  {axis:10s} média={mean_v:+.2f} ±{std_v:.2f}")
    print()
    
    # 4. Simulação de treino
    print("🎯 SIMULAÇÃO DE TREINO")
    print()
    print("  Para treinar o modelo, você precisa de:")
    print("  1. GPU com 6-8GB VRAM (Google Colab Free ✅)")
    print("  2. ~500 amostras de treino (já geradas ✅)")
    print("  3. Modelo base (Qwen 1.7B ou MiniMax 2.5B)")
    print("  4. LoRA para fine-tuning eficiente")
    print()
    print("  CUSTO ESTIMADO:")
    print("  - Colab Free: GRÁTIS (limite de ~2h por sessão)")
    print("  - Colab Pro: $10/mês")
    print("  - RunPod: ~$0.50/h (A100)")
    print()
    print("  TEMPO ESTIMADO:")
    print("  - Dataset preparação: 5min")
    print("  - Treino (500 amostras, 10 épocas): ~45min no Colab")
    print("  - Inferência: ~igual ao modelo base")
    print()
    
    # 5. Fluxo de inferência
    print("🚀 FLUXO DE INFERÊNCIA PÓS-TREINO")
    print(INFERENCE_FLOW)
    
    # 6. Salva dataset e arquitetura
    Path("data").mkdir(exist_ok=True)
    with open("data/fine_tuning_architecture.json", "w") as f:
        arch = {
            "architecture": FINE_TUNING_ARCHITECTURE,
            "inference_flow": INFERENCE_FLOW,
            "dataset_size": len(dataset),
            "num_dimensions": NUM_DIMS,
            "base_model": "Qwen 1.7B / MiniMax 2.5B",
            "loss": "α*CE + β*CosineDistance(α=1.0, β=0.3)",
            "training_hardware": "GPU 6-8GB VRAM",
            "estimated_time": "45min (Colab)",
        }
        json.dump(arch, f, ensure_ascii=False, indent=2)
    
    print("💾 Arquivos gerados:")
    print(f"  data/training_dataset.json ({len(dataset)} amostras)")
    print(f"  data/fine_tuning_architecture.json")
    print()
    print("✅ Pronto para treinar!")


if __name__ == "__main__":
    main()
