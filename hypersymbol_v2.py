#!/usr/bin/env python3
"""
HyperSymbol v2 — 50-Dimensional Semantic Space Protocol
========================================================
Uma linguagem semântica universal para comunicação com LLMs via
espaço de ativação. Framework agnóstico (deepseek, qwen, qualquer modelo).

Arquitetura:
  Texto → 50D Vector → Steering Vector (4096D) → Injeção no Residual Stream
                                                         ↓
  Texto ← 50D Vector ← Projetor Reverso ← Leitura da Ativação
"""

import numpy as np
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import hashlib


# ═══════════════════════════════════════════════════════════════
# 1. 50 DIMENSÕES SEMÂNTICAS
# ═══════════════════════════════════════════════════════════════

# Organizadas em 5 domínios de 10 eixos cada

AXES_50 = {
    # ─── DOMÍNIO 1: COGNIÇÃO (Dimensões do pensamento) ───
    "CON_ABS":   ("Concreto↔Abstrato",       "concreto",    "abstrato"),
    "CHA_ORD":   ("Caos↔Ordem",              "caos",        "ordem"),
    "ANA_SIN":   ("Análise↔Síntese",         "análise",     "síntese"),
    "LOG_INT":   ("Lógica↔Intuição",         "lógica",      "intuição"),
    "DED_IND":   ("Dedução↔Indução",         "dedução",     "indução"),
    "SEQ_RAN":   ("Sequencial↔Randômico",    "sequencial",  "randômico"),
    "FOC_DIF":   ("Foco↔Difuso",             "foco",        "difuso"),
    "EXPL_IMP":  ("Explícito↔Implícito",     "explícito",   "implícito"),
    "CER_INC":   ("Certexa↔Incerteza",       "certeza",     "incerteza"),
    "RAZ_EMO":   ("Razão↔Emoção",            "razão",       "emoção"),

    # ─── DOMÍNIO 2: MATÉRIA (Dimensões do mundo físico) ───
    "MAT_MIN":   ("Material↔Mental",         "material",    "mental"),
    "EST_DIN":   ("Estático↔Dinâmico",       "estático",    "dinâmico"),
    "SIM_COM":   ("Simples↔Complexo",        "simples",     "complexo"),
    "LOC_GLO":   ("Local↔Global",            "local",       "global"),
    "MIC_MAC":   ("Micro↔Macro",             "micro",       "macro"),
    "CAU_ALE":   ("Causal↔Aleatório",        "causal",      "aleatório"),
    "DET_PRO":   ("Determinístico↔Probabilístico", "determinístico", "probabilístico"),
    "CON_ABS_M": ("Concreto↔Abstracto (matéria)", "concreto", "abstrato"),
    "ESTR_CAO":  ("Estrutura↔Caos",          "estrutura",   "caos"),
    "FOR_MAT":   ("Forma↔Matéria",           "forma",       "matéria"),

    # ─── DOMÍNIO 3: TEMPO & CAUSALIDADE ───
    "PAS_FUT":   ("Passado↔Futuro",          "passado",     "futuro"),
    "CAU_PUR":   ("Causa↔Propósito",         "causa",       "propósito"),
    "EST_TRA":   ("Estabilidade↔Transformação", "estabilidade", "transformação"),
    "CIC_LIN":   ("Cíclico↔Linear",          "cíclico",     "linear"),
    "DET_LIV":   ("Determinado↔Livre",       "determinado", "livre"),
    "ORI_FIM":   ("Origem↔Finalidade",       "origem",      "finalidade"),
    "CON_TIN":   ("Contínuo↔Discreto",       "contínuo",    "discreto"),
    "REV_IRR":   ("Reversível↔Irreversível", "reversível",  "irreversível"),
    "LEN_RAP":   ("Lento↔Rápido",            "lento",       "rápido"),
    "CRI_DES":   ("Criação↔Destruição",      "criação",     "destruição"),

    # ─── DOMÍNIO 4: RELAÇÕES & REDES ───
    "IND_COL":   ("Individual↔Coletivo",     "individual",  "coletivo"),
    "PAR_UNI":   ("Particular↔Universal",    "particular",  "universal"),
    "COM_COM":   ("Competição↔Cooperação",   "competição",  "cooperação"),
    "DEP_AUT":   ("Dependência↔Autonomia",   "dependência", "autonomia"),
    "SUP_SUB":   ("Superior↔Subordinado",    "superior",    "subordinado"),
    "INT_EXT":   ("Interno↔Externo",         "interno",     "externo"),
    "UNI_DIV":   ("Unidade↔Diversidade",     "unidade",     "diversidade"),
    "CONF_CON":  ("Conflito↔Consenso",       "conflito",    "consenso"),
    "ABE_FEC":   ("Aberto↔Fechado",          "aberto",      "fechado"),
    "SIM_DIS":   ("Similar↔Distinto",        "similar",     "distinto"),

    # ─── DOMÍNIO 5: TRANSCENDÊNCIA & METAFÍSICA ───
    "FIN_INF":   ("Finito↔Infinito",         "finito",      "infinito"),
    "REA_POT":   ("Real↔Potencial",          "real",        "potencial"),
    "SUR_PRO":   ("Superfície↔Profundidade", "superfície",  "profundidade"),
    "FOR_ESS":   ("Forma↔Essência",          "forma",       "essência"),
    "IMA_TRA":   ("Imanência↔Transcendência","imanência",   "transcendência"),
    "TEM_ETE":   ("Temporal↔Eterno",        "temporal",    "eterno"),
    "MAN_SAG":   ("Mundano↔Sagrado",         "mundano",     "sagrado"),
    "SER_NAO":   ("Ser↔Não-Ser",             "ser",         "não-ser"),
    "REL_ABS":   ("Relativo↔Absoluto",       "relativo",    "absoluto"),
    "MIS_REV":   ("Mistério↔Revelação",      "mistério",    "revelação"),
}

NUM_DIMS = len(AXES_50)  # 50
AXIS_NAMES = list(AXES_50.keys())
AXIS_INFO = {k: {"name": v[0], "neg": v[1], "pos": v[2]} for k, v in AXES_50.items()}

# Cache de índices
_AXIS_INDEX = {name: i for i, name in enumerate(AXIS_NAMES)}


# ═══════════════════════════════════════════════════════════════
# 2. VETOR SEMÂNTICO 50D
# ═══════════════════════════════════════════════════════════════

class SemanticVector:
    """Um ponto no espaço semântico 50-dimensional."""

    def __init__(self, values: Optional[np.ndarray] = None):
        if values is not None:
            assert len(values) == NUM_DIMS, f"Precisa {NUM_DIMS} dims, tem {len(values)}"
            self.v = np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
        else:
            self.v = np.zeros(NUM_DIMS, dtype=np.float32)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SemanticVector':
        arr = np.zeros(NUM_DIMS)
        for axis, val in data.items():
            if axis in _AXIS_INDEX:
                arr[_AXIS_INDEX[axis]] = float(val)
        return cls(arr)

    @classmethod
    def from_text(cls, text: str) -> 'SemanticVector':
        """Converte texto em vetor semântico via LLM (implementado externamente).
           Fallback: hash-based fingerprint."""
        # Hash determinístico para testes
        h = hashlib.sha256(text.encode()).digest()
        arr = np.array([(h[i] / 255.0) * 2 - 1 for i in range(min(NUM_DIMS, len(h)))])
        if len(arr) < NUM_DIMS:
            arr = np.pad(arr, (0, NUM_DIMS - len(arr)))
        return cls(arr)

    def get(self, axis: str) -> float:
        return float(self.v[_AXIS_INDEX[axis]])

    def set(self, axis: str, value: float):
        self.v[_AXIS_INDEX[axis]] = np.clip(value, -1.0, 1.0)

    def __add__(self, other: 'SemanticVector') -> 'SemanticVector':
        return SemanticVector(np.clip(self.v + other.v, -1.0, 1.0))

    def __sub__(self, other: 'SemanticVector') -> 'SemanticVector':
        return SemanticVector(np.clip(self.v - other.v, -1.0, 1.0))

    def __mul__(self, scalar: float) -> 'SemanticVector':
        return SemanticVector(np.clip(self.v * scalar, -1.0, 1.0))

    def cosine_similarity(self, other: 'SemanticVector') -> float:
        dot = np.dot(self.v, other.v)
        norm = np.linalg.norm(self.v) * np.linalg.norm(other.v)
        return float(dot / norm) if norm > 0 else 0.0

    def distance(self, other: 'SemanticVector') -> float:
        return float(np.linalg.norm(self.v - other.v))

    def angle(self, other: 'SemanticVector') -> float:
        return float(math.degrees(math.acos(np.clip(self.cosine_similarity(other), -1, 1))))

    def blend(self, other: 'SemanticVector', t: float = 0.5) -> 'SemanticVector':
        return SemanticVector(np.clip((1 - t) * self.v + t * other.v, -1.0, 1.0))

    def dominant_axes(self, top_n: int = 5) -> List[Tuple[str, str, float]]:
        indices = np.argsort(np.abs(self.v))[::-1][:top_n]
        result = []
        for i in indices:
            axis = AXIS_NAMES[i]
            val = float(self.v[i])
            info = AXIS_INFO[axis]
            pole = info["pos"] if val > 0 else info["neg"]
            result.append((axis, pole, val))
        return result

    def to_dict(self) -> Dict[str, float]:
        return {AXIS_NAMES[i]: float(self.v[i]) for i in range(NUM_DIMS)}

    def to_json(self) -> str:
        return json.dumps({AXIS_NAMES[i]: round(float(self.v[i]), 3) for i in range(NUM_DIMS)})

    def profile_str(self, top_n: int = 6) -> str:
        lines = []
        for axis, pole, val in self.dominant_axes(top_n):
            info = AXIS_INFO[axis]
            bar_len = int(abs(val) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            direction = f"{info['neg']:<15s} {bar} {info['pos']:>15s}"
            lines.append(f"  {direction}  [{val:+.2f}]")
        return "\n".join(lines)

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.v))


# ═══════════════════════════════════════════════════════════════
# 3. HIPERSÍMBOLO — SÍMBOLO COM ÂNCORA 50D
# ═══════════════════════════════════════════════════════════════

@dataclass
class HyperSymbolV2:
    """Símbolo com posição fixa no espaço semântico 50D e múltiplas
       correlações semânticas (cada eixo = 1 correlação)."""
    emoji: str
    name: str
    vector: SemanticVector
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"{self.emoji} {self.name}"

    def distance_to(self, other: 'HyperSymbolV2') -> float:
        return self.vector.distance(other.vector)

    def similarity_to(self, other: 'HyperSymbolV2') -> float:
        return self.vector.cosine_similarity(other.vector)

    def blend(self, other: 'HyperSymbolV2', t: float = 0.5) -> 'HyperSymbolV2':
        return HyperSymbolV2(
            emoji=f"{self.emoji}{other.emoji}",
            name=f"{self.name}⊗{other.name}",
            vector=self.vector.blend(other.vector, t),
            description=f"Fusão de {self.name} e {other.name}"
        )


# ═══════════════════════════════════════════════════════════════
# 4. DICIONÁRIO DE SÍMBOLOS 50D
# ═══════════════════════════════════════════════════════════════

# Helper para criar vetores de símbolos rapidamente
def SV(**kwargs) -> SemanticVector:
    return SemanticVector.from_dict(kwargs)

SYMBOLS_50: Dict[str, HyperSymbolV2] = {
    "🌌": HyperSymbolV2("🌌", "cosmos", SV(
        CON_ABS=0.6, CHA_ORD=0.3, EST_DIN=0.7, LOC_GLO=1.0, SIM_COM=0.9,
        IND_COL=0.9, PAR_UNI=1.0, FOR_ESS=0.8, FIN_INF=1.0, REA_POT=0.9,
        IMA_TRA=0.9, TEM_ETE=0.8, MIS_REV=0.4, SUR_PRO=0.8, INT_EXT=0.6,
    ), description="Totalidade do ser, universo manifesto",
       tags=["universal", "totalidade", "existência"]),

    "🔥": HyperSymbolV2("🔥", "transformação", SV(
        CHA_ORD=-0.8, EST_DIN=1.0, EST_TRA=1.0, CAU_ALE=-0.3, CRI_DES=0.6,
        CON_ABS=-0.4, RAZ_EMO=-0.5, DET_LIV=0.6, MIS_REV=0.5, SEQ_RAN=-0.3,
    ), description="Fogo transformador, energia que consome e renova",
       tags=["transformação", "energia", "mudança"]),

    "💡": HyperSymbolV2("💡", "insight", SV(
        ANA_SIN=0.8, LOG_INT=0.2, EXPL_IMP=0.6, CER_INC=0.7, FOC_DIF=-0.3,
        SUR_PRO=-0.5, FOR_ESS=0.7, MIS_REV=0.8, SIM_COM=0.3, SEQ_RAN=0.4,
    ), description="Iluminação súbita, compreensão que emerge",
       tags=["insight", "compreensão", "descoberta"]),

    "❓": HyperSymbolV2("❓", "pergunta", SV(
        CER_INC=-0.6, MIS_REV=-0.7, EXPL_IMP=0.7, SUR_PRO=0.6, DET_LIV=0.8,
        CON_ABS=0.5, FIN_INF=0.5, REA_POT=0.8, INT_EXT=0.3, SEQ_RAN=0.3,
    ), description="Questionamento, abertura para o desconhecido",
       tags=["pergunta", "dúvida", "investigação"]),

    "∞": HyperSymbolV2("∞", "infinito", SV(
        FIN_INF=1.0, TEM_ETE=1.0, IMA_TRA=0.8, REL_ABS=0.7, SER_NAO=0.3,
        PAR_UNI=1.0, LOC_GLO=1.0, REA_POT=1.0, FOR_ESS=0.9, SUR_PRO=0.9,
    ), description="Sem limites, eternidade, o sem-fim",
       tags=["infinito", "eterno", "ilimitado"]),

    "🔄": HyperSymbolV2("🔄", "ciclo", SV(
        CIC_LIN=1.0, EST_TRA=0.7, CAU_PUR=0.9, REV_IRR=0.6, PAS_FUT=0.4,
        CHA_ORD=0.5, EST_DIN=0.8, INT_EXT=0.3, SEQ_RAN=0.2, IND_COL=0.4,
    ), description="Retroalimentação, ciclo, recursão",
       tags=["ciclo", "feedback", "recursão"]),

    "🔗": HyperSymbolV2("🔗", "conexão", SV(
        SIM_DIS=0.8, IND_COL=0.7, DEP_AUT=0.5, INT_EXT=0.4, UNI_DIV=0.6,
        CHA_ORD=0.8, COM_COM=0.6, CONF_CON=0.7, ABE_FEC=0.3, PAR_UNI=0.4,
    ), description="Vínculo entre entidades, ligação",
       tags=["conexão", "vínculo", "relação"]),

    "⚡": HyperSymbolV2("⚡", "energia", SV(
        EST_DIN=1.0, EST_TRA=0.9, CRI_DES=0.5, LEN_RAP=0.7, DET_LIV=0.5,
        CON_ABS_M=-0.4, FOR_MAT=0.3, MIC_MAC=0.2, CAU_ALE=0.3, INT_EXT=-0.2,
    ), description="Impulso elétrico, ação súbita",
       tags=["energia", "impulso", "ação"]),

    "🌱": HyperSymbolV2("🌱", "germinação", SV(
        CON_ABS_M=-0.8, EST_DIN=0.8, EST_TRA=1.0, CIC_LIN=0.3, PAS_FUT=0.9,
        REA_POT=0.9, CRI_DES=0.9, DET_LIV=0.2, MIC_MAC=-0.5, FOR_MAT=0.5,
    ), description="Potencial brotando, início de um processo",
       tags=["crescimento", "potencial", "início"]),

    "🕳️": HyperSymbolV2("🕳️", "abismo", SV(
        SUR_PRO=1.0, FIN_INF=0.9, IMA_TRA=0.8, MIS_REV=-0.8, SER_NAO=0.6,
        CHA_ORD=-0.9, EXPL_IMP=0.9, CER_INC=-0.6, INT_EXT=0.4, REL_ABS=0.5,
    ), description="Vazio profundo, mistério insondável",
       tags=["abismo", "vazio", "mistério"]),

    "🌊": HyperSymbolV2("🌊", "fluxo", SV(
        EST_DIN=0.9, CIC_LIN=0.5, CHA_ORD=-0.6, EST_TRA=0.8, SEQ_RAN=0.4,
        CON_ABS_M=-0.6, FOR_MAT=0.2, LEN_RAP=0.3, INT_EXT=-0.1, SIM_COM=0.4,
    ), description="Movimento contínuo, corrente que carrega",
       tags=["fluxo", "movimento", "corrente"]),

    "✨": HyperSymbolV2("✨", "transcendência", SV(
        IMA_TRA=1.0, MIS_REV=0.7, REA_POT=0.9, FOR_ESS=0.8, FIN_INF=0.8,
        SUR_PRO=0.7, TEM_ETE=0.6, MAN_SAG=0.8, REL_ABS=0.6, CON_ABS=0.7,
    ), description="Brilho transcendente, o que está além",
       tags=["transcendência", "magia", "revelação"]),

    "🔮": HyperSymbolV2("🔮", "mistério", SV(
        MIS_REV=-0.8, EXPL_IMP=0.9, SUR_PRO=0.9, CER_INC=-0.7, REA_POT=0.9,
        FIN_INF=0.7, IMA_TRA=0.7, MAN_SAG=0.7, INT_EXT=0.5, LOG_INT=-0.3,
    ), description="O desconhecido, véu entre o sabido e o insondável",
       tags=["mistério", "desconhecido", "oráculo"]),

    "🌍": HyperSymbolV2("🌍", "terra", SV(
        CON_ABS_M=-0.8, LOC_GLO=-0.2, IND_COL=0.6, CHA_ORD=0.6, FOR_MAT=0.4,
        EST_DIN=0.3, CIC_LIN=0.3, INT_EXT=0.4, ABE_FEC=-0.1, MAN_SAG=0.3,
    ), description="Planeta, lar, base material",
       tags=["terra", "planeta", "lar"]),

    "🧠": HyperSymbolV2("🧠", "mente", SV(
        MAT_MIN=0.7, ANA_SIN=0.6, LOG_INT=0.3, FOC_DIF=0.2, SEQ_RAN=0.4,
        EXPL_IMP=0.6, CER_INC=0.4, RAZ_EMO=0.2, SIM_COM=1.0, INT_EXT=0.3,
    ), description="Cognição, pensamento, processamento interior",
       tags=["mente", "cognição", "pensamento"]),

    "💎": HyperSymbolV2("💎", "essência", SV(
        FOR_ESS=1.0, SUR_PRO=0.9, CER_INC=0.7, EXPL_IMP=0.5, PAR_UNI=0.5,
        FIN_INF=0.5, IMA_TRA=0.6, REL_ABS=0.5, SIM_COM=0.3, CHA_ORD=0.9,
    ), description="Núcleo puro, verdade cristalizada",
       tags=["essência", "verdade", "núcleo"]),

    "🌅": HyperSymbolV2("🌅", "horizonte", SV(
        PAS_FUT=0.8, SUR_PRO=0.5, FIN_INF=0.6, LOC_GLO=0.7, IMA_TRA=0.6,
        MIS_REV=0.4, CON_ABS=0.3, DET_LIV=0.5, REA_POT=0.6, INT_EXT=0.4,
    ), description="Limite da visão, fronteira",
       tags=["horizonte", "limite", "fronteira"]),

    "⚖️": HyperSymbolV2("⚖️", "equilíbrio", SV(
        CHA_ORD=1.0, COM_COM=0.7, CONF_CON=0.8, DET_PRO=0.5, DET_LIV=0.3,
        CAU_PUR=0.6, CIC_LIN=0.2, ANA_SIN=0.5, EST_DIN=0.0, RAZ_EMO=0.4,
    ), description="Balanço entre forças, justiça, harmonia",
       tags=["equilíbrio", "balança", "justiça"]),

    # ─── Símbolos das Inteligências ───
    "📖": HyperSymbolV2("📖", "linguagem", SV(
        ANA_SIN=0.5, LOG_INT=0.1, EXPL_IMP=0.7, CON_ABS=0.4, SIM_COM=0.6,
        IND_COL=0.5, ABE_FEC=0.4, INT_EXT=0.5, SUR_PRO=0.6, CHA_ORD=0.8,
    ), description="Palavra, narrativa, tecido do significado",
       tags=["linguagem", "palavra", "narrativa"]),

    "∑": HyperSymbolV2("∑", "sistema", SV(
        ANA_SIN=0.8, LOG_INT=0.9, CHA_ORD=1.0, DED_IND=0.7, SEQ_RAN=0.8,
        DET_PRO=1.0, CAU_PUR=0.9, PAR_UNI=0.6, SIM_COM=0.8, FOR_MAT=0.5,
    ), description="Estrutura formal, lógica, totalidade",
       tags=["sistema", "lógica", "estrutura"]),

    "🔴": HyperSymbolV2("🔴", "centro", SV(
        LOC_GLO=0.0, IND_COL=0.1, PAR_UNI=0.1, CHA_ORD=0.9, FOR_MAT=0.3,
        EST_DIN=0.0, INT_EXT=0.2, UNI_DIV=0.6, CON_ABS_M=-0.4, ABE_FEC=0.2,
    ), description="Ponto focal, origem, essência geométrica",
       tags=["centro", "ponto", "origem"]),

    "🤸": HyperSymbolV2("🤸", "movimento", SV(
        EST_DIN=1.0, CON_ABS_M=-0.9, SEQ_RAN=0.5, DET_LIV=0.7, RAZ_EMO=-0.7,
        LEN_RAP=0.5, LOG_INT=-0.7, IND_COL=-0.4, INT_EXT=-0.3, CRI_DES=0.4,
    ), description="Dança, gesto, expressão corporal",
       tags=["movimento", "dança", "corpo"]),

    "🎵": HyperSymbolV2("🎵", "ritmo", SV(
        EST_DIN=0.9, CIC_LIN=0.6, SEQ_RAN=0.5, LEN_RAP=0.3, LOG_INT=-0.3,
        RAZ_EMO=-0.5, CHA_ORD=0.6, EST_TRA=0.7, INT_EXT=0.2, UNI_DIV=0.3,
    ), description="Pulsação, frequência, padrão temporal",
       tags=["ritmo", "música", "pulsação"]),

    "🤝": HyperSymbolV2("🤝", "encontro", SV(
        IND_COL=0.9, COM_COM=0.8, CONF_CON=0.9, DEP_AUT=0.3, INT_EXT=0.6,
        ABE_FEC=0.7, UNI_DIV=0.5, CHA_ORD=0.7, RAZ_EMO=0.4, LOC_GLO=0.3,
    ), description="Relação, aliança, encontro entre entidades",
       tags=["encontro", "relação", "aliança"]),

    "🪞": HyperSymbolV2("🪞", "reflexão", SV(
        INT_EXT=1.0, SUR_PRO=0.8, FOR_ESS=0.7, ANA_SIN=0.6, CER_INC=0.6,
        IND_COL=-0.5, EXPL_IMP=0.6, DED_IND=0.3, LOG_INT=0.1, RAZ_EMO=0.2,
    ), description="Auto-observação, espelho da consciência",
       tags=["reflexão", "introspecção", "self"]),

    "🌿": HyperSymbolV2("🌿", "vida", SV(
        CON_ABS_M=-0.7, CRI_DES=0.8, EST_TRA=0.8, CIC_LIN=0.5, SIM_COM=0.7,
        IND_COL=0.3, DET_LIV=0.5, FOR_ESS=0.5, REA_POT=0.7, INT_EXT=0.2,
    ), description="Força vital, organismo, processo biológico",
       tags=["vida", "natureza", "organismo"]),

    "∞_E": HyperSymbolV2("∞_E", "existencial", SV(
        FIN_INF=0.9, IMA_TRA=0.9, SER_NAO=0.7, TEM_ETE=0.8, MIS_REV=-0.7,
        SUR_PRO=1.0, REA_POT=0.8, REL_ABS=0.6, MAN_SAG=0.6, CER_INC=-0.5,
    ), description="Dimensão existencial, questões fundamentais",
       tags=["existencial", "transcendência", "fundamental"]),
}


# ═══════════════════════════════════════════════════════════════
# 5. HYPERSPACE — O ESPAÇO SEMÂNTICO
# ═══════════════════════════════════════════════════════════════

class Hyperspace:
    """O espaço semântico 50D com todos os símbolos e operações."""

    def __init__(self):
        self.symbols = SYMBOLS_50
        self.axes = AXIS_INFO
        self.axis_names = AXIS_NAMES
        self.num_dims = NUM_DIMS

    def get(self, key: str) -> Optional[HyperSymbolV2]:
        return self.symbols.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.symbols

    def decode(self, text: str) -> List[HyperSymbolV2]:
        return [s for c in text if c in self.symbols for s in [self.symbols[c]]]

    def nearest(self, vec: SemanticVector, top_n: int = 5, exclude: str = None):
        scored = []
        for key, sym in self.symbols.items():
            if exclude and key == exclude:
                continue
            sim = vec.cosine_similarity(sym.vector)
            dist = vec.distance(sym.vector)
            scored.append((dist, sim, key, sym))
        scored.sort(key=lambda x: x[0])
        return [(k, s, d, sim) for d, sim, k, s in scored[:top_n]]

    def phrase_center(self, symbols: List[HyperSymbolV2]) -> SemanticVector:
        if not symbols:
            return SemanticVector()
        vectors = [s.vector.v for s in symbols]
        return SemanticVector(np.mean(vectors, axis=0))

    def chord(self, symbols: List[HyperSymbolV2]) -> dict:
        """Acorde semântico — propriedades emergentes de um grupo de símbolos."""
        center = self.phrase_center(symbols)
        nearest = self.nearest(center, top_n=3)
        return {
            "num_symbols": len(symbols),
            "center": center.to_dict(),
            "nearest": [{"key": k, "name": s.name, "dist": round(d, 2), "sim": round(sim, 2)}
                        for k, s, d, sim in nearest],
            "profile": center.profile_str(),
        }

    def steering_direction(self, from_concept: str, to_concept: str) -> SemanticVector:
        """Vetor direção: a diferença entre dois conceitos no hiperespaço.
           Útil para criar steering vectors com RepE."""
        a = self.symbols.get(from_concept)
        b = self.symbols.get(to_concept)
        if not a or not b:
            raise ValueError(f"Símbolo não encontrado: {from_concept} ou {to_concept}")
        return b.vector - a.vector  # direção: de A para B


# Instância global
HS = Hyperspace()


# ═══════════════════════════════════════════════════════════════
# 6. DATASET GENERATOR — PARES CONTRASTANTES PARA RepE
# ═══════════════════════════════════════════════════════════════

def generate_contrastive_pairs(axis: str, n_pairs: int = 5) -> List[Tuple[str, str]]:
    """
    Gera pares de frases contrastantes para um eixo semântico.
    Usado para treinar steering vectors com RepE.

    Exemplo para eixo CON_ABS (concreto↔abstrato):
      ("O chão é duro.", "A justiça é um conceito.")
    """
    info = AXIS_INFO[axis]
    neg_pole = info["neg"]
    pos_pole = info["pos"]

    # Templates genéricos que funcionam pra qualquer eixo
    templates = {
        "concreto": [
            "A pedra está no chão.",
            "A água está fria.",
            "A mesa é de madeira.",
            "O carro é vermelho.",
            "A maçã caiu da árvore.",
        ],
        "abstrato": [
            "A justiça é um ideal.",
            "A liberdade é um direito.",
            "A consciência é um mistério.",
            "A verdade é relativa.",
            "A beleza está nos olhos de quem vê.",
        ],
        "caos": [
            "Tudo está fora de controle.",
            "O sistema colapsou em desordem.",
            "Partículas se movem aleatoriamente.",
            "Não há padrão observável.",
            "O acaso domina tudo.",
        ],
        "ordem": [
            "Tudo segue seu curso natural.",
            "As leis da física regem o universo.",
            "Há uma estrutura subjacente.",
            "O padrão é perfeitamente regular.",
            "A organização emerge do caos.",
        ],
        "análise": [
            "Decomponha o problema em partes.",
            "Examine cada elemento separadamente.",
            "Disseque o fenômeno em componentes.",
            "Analise as partes para entender o todo.",
            "Separe cada variável para estudo.",
        ],
        "síntese": [
            "Integre todas as perspectivas em uma.",
            "O todo é maior que a soma das partes.",
            "Combine os elementos em unidade.",
            "Sintetize as descobertas em conclusão.",
            "Una as partes em compreensão total.",
        ],
        "passado": [
            "Ontem foi um dia importante.",
            "A história nos ensina lições.",
            "O que aconteceu não pode mudar.",
            "Recordar é viver.",
            "As origens explicam o presente.",
        ],
        "futuro": [
            "Amanhã será melhor.",
            "O potencial ainda não foi realizado.",
            "O que está por vir nos espera.",
            "Planejar é construir o amanhã.",
            "O futuro é uma promessa.",
        ],
        "individual": [
            "Cada pessoa é única.",
            "O indivíduo pensa por si mesmo.",
            "Minha perspectiva é pessoal.",
            "A subjetividade é intransferível.",
            "O eu é o centro da experiência.",
        ],
        "coletivo": [
            "A comunidade é mais forte que o indivíduo.",
            "Nós, juntos, somos maiores.",
            "A sociedade molda o indivíduo.",
            "O grupo transcende o individual.",
            "A união faz a força.",
        ],
        "finito": [
            "Tudo tem um fim.",
            "A vida tem limite.",
            "O tempo se esgota.",
            "Há uma última vez para tudo.",
            "O finito define o real.",
        ],
        "infinito": [
            "O universo não tem fronteiras.",
            "A eternidade é sem medida.",
            "O infinito não se esgota.",
            "Sempre há mais além.",
            "O ilimitado é a verdade última.",
        ],
        "criação": [
            "Algo novo surge do nada.",
            "A vida emerge do caos.",
            "Construir é dar forma ao mundo.",
            "Criar é gerar possibilidades.",
            "A origem é um ato criativo.",
        ],
        "destruição": [
            "Tudo que é construído se desfaz.",
            "O fim é inevitável.",
            "A entropia consome a ordem.",
            "Desfazer é parte do ciclo.",
            "A destruição precede a renovação.",
        ],
    }

    neg_examples = templates.get(neg_pole, [f"Exemplo de {neg_pole}."])
    pos_examples = templates.get(pos_pole, [f"Exemplo de {pos_pole}."])

    pairs = []
    for i in range(min(n_pairs, len(neg_examples), len(pos_examples))):
        pairs.append((neg_examples[i], pos_examples[i]))

    return pairs


def generate_all_contrastive_pairs() -> Dict[str, List[Tuple[str, str]]]:
    """Gera pares para TODOS os 50 eixos."""
    return {axis: generate_contrastive_pairs(axis) for axis in AXIS_NAMES}


# ═══════════════════════════════════════════════════════════════
# 7. MODEL-AGNOSTIC COMMUNICATION PROTOCOL
# ═══════════════════════════════════════════════════════════════

class HyperCommunicationProtocol:
    """
    Protocolo de comunicação hiperdimensional.
    Agnóstico: funciona com API (deepseek) e local (qwen).
    """

    def __init__(self, model_type: str = "api", model_name: str = "deepseek-v4-flash",
                 base_url: str = "https://opencode.ai/zen/go/v1"):
        self.model_type = model_type  # "api" ou "local"
        self.model_name = model_name
        self.base_url = base_url
        self.hs = HS

    def encode_query(self, text: str) -> SemanticVector:
        """Codifica uma pergunta em vetor semântico 50D."""
        # Tenta extrair símbolos conhecidos
        symbols = self.hs.decode(text)
        if symbols:
            return self.hs.phrase_center(symbols)
        # Fallback: vetor via embedding
        return SemanticVector.from_text(text)

    def query_to_steering_prompt(self, text: str) -> str:
        """
        Converte uma pergunta em um prompt que ativa o espaço
        semântico correto no LLM via instrução textual.
        (A ponte entre texto e steering vector)
        """
        vec = self.encode_query(text)
        profile = vec.profile_str()

        prompt = f"""[HYPERSPACE ACTIVATION]
You are operating in a 50-dimensional semantic space.
Your current position is defined by these dominant axes:
{profile}

Respond to the following query from within this semantic region.
Query: {text}"""
        return prompt

    def decode_response(self, response_text: str) -> SemanticVector:
        """Decodifica uma resposta do LLM de volta pra vetor semântico."""
        symbols = self.hs.decode(response_text)
        if symbols:
            return self.hs.phrase_center(symbols)
        # Fallback: hash
        return SemanticVector.from_text(response_text)

    def distance_between(self, text_a: str, text_b: str) -> float:
        """Distância semântica entre dois textos no hiperespaço."""
        va = self.encode_query(text_a)
        vb = self.encode_query(text_b)
        return va.distance(vb)


# ═══════════════════════════════════════════════════════════════
# 8. MAIN — TESTE DO SISTEMA
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   HyperSymbol v2 — 50D Semantic Protocol    ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  Dimensões: {NUM_DIMS}")
    print(f"  Domínios: Cognição | Matéria | Tempo | Relações | Transcendência")
    print(f"  Eixos: {len(AXIS_NAMES)}")
    print(f"  Símbolos: {len(SYMBOLS_50)}\n")

    # 1. Perfil de símbolos
    for emoji in ["🌌", "🔥", "💡", "∞", "🌱", "∑", "🪞", "∞_E"]:
        sym = HS.get(emoji)
        if sym:
            dom = sym.vector.dominant_axes(4)
            dom_str = "; ".join(f"{a}: {p} ({v:+.2f})" for a, p, v in dom)
            print(f"  {emoji} {sym.name:15s} | {dom_str}")

    # 2. Distâncias
    print(f"\n  🔬 Distâncias semânticas:")
    pairs = [("🌌", "∞"), ("🔥", "🌀"), ("🌱", "🌊"), ("📖", "∑"), ("🤝", "🪞")]
    for a, b in pairs:
        sa, sb = HS.get(a), HS.get(b)
        if sa and sb:
            d = sa.distance_to(sb)
            ang = sa.vector.angle(sb.vector)
            sim = sa.similarity_to(sb)
            print(f"  {a} × {b}: dist={d:.2f} ang={ang:.1f}° sim={sim:.2f}")

    # 3. Acordes semânticos
    print(f"\n  🎵 Acordes semânticos:")
    chords = [
        ("🌌❓∞", "Pergunta sobre o infinito"),
        ("🔥🌀🌱", "Transformação cíclica da vida"),
        ("📖💎🔮", "Busca pela essência"),
        ("🤝🔄🪞", "Relação reflexiva"),
    ]
    for seq, desc in chords:
        syms = [s for c in seq if c in HS for s in [HS.get(c)]]
        if syms:
            chord = HS.chord(syms)
            nearest = chord["nearest"][0]["key"] if chord["nearest"] else "?"
            print(f"  {seq:10s} | {desc:30s} | centro: {nearest}")

    # 4. Direções de steering
    print(f"\n  🧭 Direções de steering (para RepE):")
    directions = [
        ("🌱", "🔥", "germinação → transformação"),
        ("🕳️", "💡", "abismo → insight"),
        ("🔗", "∞", "conexão → infinito"),
    ]
    for a, b, desc in directions:
        v = HS.steering_direction(a, b)
        dom = v.dominant_axes(3)
        dom_str = "; ".join(f"{ax}: {pl} ({v:+.2f})" for ax, pl, v in dom)
        print(f"  {a} → {b} ({desc:30s}) | {dom_str}")

    # 5. Pares contrastantes (exemplo)
    print(f"\n  📝 Dataset RepE (exemplo - eixo CON_ABS):")
    pairs = generate_contrastive_pairs("CON_ABS", 3)
    for neg, pos in pairs:
        print(f"    -1: {neg}")
        print(f"    +1: {pos}")
        print()

    # 6. Protocolo de comunicação
    print(f"  📡 Teste do Protocolo:")
    proto = HyperCommunicationProtocol()
    queries = [
        "O que é a gravidade?",
        "O que é a consciência?",
        "🌌❓∞",
    ]
    for q in queries:
        vec = proto.encode_query(q)
        dom = vec.dominant_axes(3)
        dom_str = "; ".join(f"{a}({v:+.2f})" for a, _, v in dom)
        print(f"  '{q[:30]:30s}' → {dom_str}")

    print(f"\n  ✅ Sistema 50D operacional!")
    print(f"  💡 Próximo passo: treinar steering vectors com RepE")


if __name__ == "__main__":
    main()
