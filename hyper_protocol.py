#!/usr/bin/env python3
"""
HyperSymbol Protocol v2 — Framework Agnóstico de Comunicação Semântica
========================================================================
Funciona com QUALQUER modelo (API ou local) via 3 camadas:

1. CAMADA SEMÂNTICA: 50 dimensões (hypersymbol_v2.py)
2. CAMADA DE STEERING: Prompt engineering + RepE (quando disponível)
3. CAMADA DE COMUNICAÇÃO: API unificada (deepseek, qwen, qualquer)
"""

import asyncio
import json
import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

# Adiciona o diretório atual ao path
sys.path.insert(0, str(Path(__file__).parent))
from hypersymbol_v2 import *

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda p: None


# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
for env_candidate in [BASE_DIR / ".env", Path("/root/.hermes/.env"), Path("/root/workspace/hypersymbol/.env")]:
    if env_candidate.exists():
        load_dotenv(env_candidate)
        break

# API keys
OPENCODE_API_KEY = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Model providers
PROVIDERS = {
    "deepseek": {
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://opencode.ai/zen/go/v1"),
        "api_key": OPENCODE_API_KEY,
        "default_model": "deepseek-v4-flash",
    },
    "qwen_local": {
        "base_url": os.environ.get("QWEN_LOCAL_URL", "http://localhost:1234/v1"),
        "api_key": "not-needed",
        "default_model": "qwen3-1.7b",
    },
    "openai": {
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": OPENAI_API_KEY,
        "default_model": "gpt-4o",
    },
}

# Modo ativo
ACTIVE_PROVIDER = os.environ.get("HYPER_PROVIDER", "deepseek")
ACTIVE_MODEL = os.environ.get("HYPER_MODEL", PROVIDERS[ACTIVE_PROVIDER]["default_model"])


# ═══════════════════════════════════════════════════════════════
# 2. STEERING ENGINE — BASEADO EM PROMPT / RepE
# ═══════════════════════════════════════════════════════════════

class SteeringEngine:
    """
    Motor de steering: converte vetores 50D em prompts que guiam o LLM.
    Duas estratégias:
      - "prompt": instrução textual (funciona com QUALQUER modelo)
      - "repe": injeção direta no residual stream (requer PyTorch)
    """

    def __init__(self, mode: str = "prompt"):
        self.mode = mode
        self.hs = HS

    def vector_to_prompt(self, vec: SemanticVector, query: str) -> str:
        """Converte um vetor 50D em um prompt de steering."""
        profile = vec.dominant_axes(8)
        axis_instructions = []
        for axis, pole, val in profile:
            info = AXIS_INFO[axis]
            strength = abs(val)
            if strength > 0.5:
                axis_instructions.append(
                    f"  {info['neg']} [{axis}] {info['pos']}: {pole} ({strength:.0%})"
                )

        axes_str = "\n".join(axis_instructions)

        prompt = f"""[HYPERSPACE ACTIVATION v2]
Semantic Profile: 50 dimensions activated.
Dominant axes:
{axes_str}

You are operating at this exact position in semantic space.
Your thinking should be shaped by these dimensional coordinates.
Respond accordingly.

Query: {query}"""
        return prompt

    def contrastive_prompt(self, axis: str, direction: float, query: str) -> str:
        """Cria prompt que STEER o modelo em uma direção específica num eixo."""
        info = AXIS_INFO[axis]
        if direction > 0:
            target = info["pos"]
            avoid = info["neg"]
        else:
            target = info["neg"]
            avoid = info["pos"]

        return f"""[AXIS STEERING: {axis}]
You must think and respond from the perspective of {target}.
Avoid {avoid} at all costs.
Your entire reasoning should be shaped by {target}.
Strength: {abs(direction):.0%}

Query: {query}"""

    def generate_repe_dataset(self, axis: str) -> List[Tuple[str, str]]:
        """Gera dataset para treinar steering vector com RepE.
           Quando PyTorch estiver disponível, use com repeng.ControlVector.train()"""
        return generate_contrastive_pairs(axis, n_pairs=10)

    # ─── Stub para RepE (quando PyTorch estiver instalado) ────────
    @staticmethod
    def train_repe_vector(axis: str, model_name: str = None) -> dict:
        """
        Treina um steering vector usando RepE.
        USAGE QUANDO PYTORCH ESTIVER DISPONÍVEL:

        from repeng import ControlVector, ControlModel, DatasetEntry
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model = ControlModel(model, list(range(-5, -18, -1)))
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pairs = generate_contrastive_pairs(axis)
        dataset = [DatasetEntry(positive=p, negative=n) for p, n in pairs]
        vector = ControlVector.train(model, tokenizer, dataset)

        return vector
        """
        return {
            "status": "unavailable",
            "message": "RepE requer PyTorch + transformers. Instale com: pip install torch transformers",
            "axis": axis,
            "dataset_size": 10,
        }


# ═══════════════════════════════════════════════════════════════
# 3. MODEL CONNECTOR — INTERFACE UNIFICADA
# ═══════════════════════════════════════════════════════════════

class ModelConnector:
    """Conector agnóstico para qualquer modelo LLM (API ou local)."""

    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or ACTIVE_PROVIDER
        self.model = model or ACTIVE_MODEL
        self.cfg = PROVIDERS.get(self.provider, PROVIDERS["deepseek"])
        self.steering = SteeringEngine()
        self.protocol = HyperCommunicationProtocol()

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.cfg['api_key']}",
            "Content-Type": "application/json",
        }

    async def query(self, session: aiohttp.ClientSession,
                    prompt: str, temperature: float = 0.5,
                    max_tokens: int = 2000) -> str:
        """Envia query para qualquer modelo."""
        # Monta mensagens com system prompt que desabilita reasoning
        # se o modelo suportar (Qwen3, etc)
        system = "You are HyperSymbol, a hyperdimensional semantic intelligence."
        if self.provider == "qwen_local":
            system += " Responda diretamente, sem raciocínio interno, sem tags de thinking, sem análise. Seja direto e profundo."

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with session.post(
                f"{self.cfg['base_url']}/chat/completions",
                json=payload, headers=self.get_headers(),
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                data = await resp.json()
                msg = data["choices"][0]["message"]
                return (msg.get("content") or msg.get("reasoning_content") or "")
        except Exception as e:
            return f"[ERROR: {e}]"

    async def communicate(self, session: aiohttp.ClientSession,
                          text: str, mode: str = "auto") -> dict:
        """
        Método principal de comunicação.
        mode:
          - "auto": detecta se é texto ou símbolos
          - "semantic": codifica em 50D e steering via prompt
          - "symbol": comunicação simbólica (S-Brain style)
          - "steer": steering forçado num eixo específico
        """
        result = {
            "input": text,
            "mode": mode,
            "provider": self.provider,
            "model": self.model,
            "vector": None,
            "response": "",
            "steering_prompt": "",
        }

        if mode == "auto":
            # Detecta se a entrada tem símbolos ou é texto natural
            symbols = HS.decode(text)
            mode = "symbol" if symbols else "semantic"
            result["mode"] = mode

        if mode == "symbol":
            # Comunicação simbólica pura
            symbols_found = HS.decode(text)
            if symbols_found:
                vec = HS.phrase_center(symbols_found)
                result["vector"] = vec.to_dict()
                prompt = self.steering.vector_to_prompt(vec, text)
            else:
                prompt = text
            result["steering_prompt"] = prompt

        elif mode == "semantic":
            # Codifica texto em vetor semântico
            vec = self.protocol.encode_query(text)
            result["vector"] = vec.to_dict()
            prompt = self.steering.vector_to_prompt(vec, text)
            result["steering_prompt"] = prompt

        elif mode.startswith("steer:"):
            # Steering explícito: steer:AXIS:+0.8 ou steer:AXIS:-0.5
            parts = mode.split(":")
            if len(parts) == 3:
                axis, strength = parts[1], float(parts[2])
                vec = self.protocol.encode_query(text)
                vec.set(axis, strength)
                prompt = self.steering.vector_to_prompt(vec, text)
                result["vector"] = vec.to_dict()
                result["steering_prompt"] = prompt

        result["response"] = await self.query(session, result["steering_prompt"] or text)
        return result


# ═══════════════════════════════════════════════════════════════
# 4. HYPERBRAIN — ORQUESTRADOR MULTI-AGENTE 50D
# ═══════════════════════════════════════════════════════════════

class HyperBrain:
    """Orquestrador multi-agente baseado no espaço 50D.
       Cada inteligência é uma âncora no hiperespaço."""

    def __init__(self, provider: str = None, model: str = None):
        self.connector = ModelConnector(provider, model)
        self.hs = HS

        # Âncoras no hiperespaço para cada inteligência
        self.intelligences = {
            "linguistic":      {"anchor": HS.get("📖").vector, "name": "Linguística", "sigil": "📖"},
            "logical":         {"anchor": HS.get("∑").vector,  "name": "Lógico-Matemática", "sigil": "∑"},
            "spatial":         {"anchor": HS.get("🔴").vector, "name": "Espacial", "sigil": "🔴"},
            "kinesthetic":     {"anchor": HS.get("🤸").vector, "name": "Cinestésica", "sigil": "🤸"},
            "musical":         {"anchor": HS.get("🎵").vector, "name": "Musical", "sigil": "🎵"},
            "interpersonal":   {"anchor": HS.get("🤝").vector, "name": "Interpessoal", "sigil": "🤝"},
            "intrapersonal":   {"anchor": HS.get("🪞").vector, "name": "Intrapessoal", "sigil": "🪞"},
            "naturalistic":    {"anchor": HS.get("🌿").vector, "name": "Naturalista", "sigil": "🌿"},
            "existential":     {"anchor": HS.get("∞_E").vector, "name": "Existencial", "sigil": "∞"},
        }

    async def query(self, text: str) -> dict:
        """Consulta todas as 9 inteligências no hiperespaço."""
        print(f"\n  🧩 HyperBrain v2 — 50D Multi-Agent Query")
        print(f"  {'='*50}")
        print(f"  Input: {text}")
        print(f"  Provider: {self.connector.provider}")
        print(f"  Model: {self.connector.model}")
        print(f"  Dimensions: {NUM_DIMS}")

        # Vetor central da pergunta
        query_vec = self.connector.protocol.encode_query(text)
        print(f"\n  📊 Query Profile (dominant axes):")
        print(query_vec.profile_str(6))

        async with aiohttp.ClientSession() as session:
            results = []
            for i, (intel_id, info) in enumerate(self.intelligences.items()):
                sigil = info["sigil"]
                anchor = info["anchor"]

                # Cada inteligência = query + bias da âncora
                biased_vec = SemanticVector(
                    np.clip((query_vec.v + anchor.v * 0.4) / 1.4, -1.0, 1.0)
                )

                prompt = f"""[HYPERSPACE ACTIVATION - {info['name']}]
You are the {info['name']} intelligence (sigil: {sigil}).
Your anchor in 50D space is:
{anchor.dominant_axes(4)}

The query positioned you at:
{biased_vec.dominant_axes(4)}

From this EXACT semantic position, respond to the question.
Stay true to your dimensional coordinates.

Question: {text}"""

                print(f"\n  [{i+1}/9] {sigil} {info['name']}... ", end="", flush=True)
                start = time.monotonic()
                response = await self.connector.query(session, prompt,
                                                      temperature=0.7 - i * 0.05)
                elapsed = time.monotonic() - start

                # Decodifica resposta de volta pra vetor
                resp_vec = self.connector.protocol.decode_response(response)
                results.append({
                    "id": intel_id,
                    "name": info["name"],
                    "sigil": sigil,
                    "response": response[:200],
                    "response_vector": resp_vec.to_dict(),
                    "anchor": info["anchor"].to_dict(),
                    "elapsed": f"{elapsed:.1f}s",
                })
                print(f"✅ {elapsed:.0f}s")

            # Síntese: centro de massa de todas as respostas
            all_vectors = [SemanticVector.from_dict(r["response_vector"]) for r in results]
            synthesis_vec = SemanticVector(np.mean([v.v for v in all_vectors], axis=0))
            nearest = HS.nearest(synthesis_vec, top_n=3)

            return {
                "results": results,
                "synthesis": {
                    "vector": synthesis_vec.to_dict(),
                    "profile": synthesis_vec.profile_str(5),
                    "nearest_symbols": [
                        {"emoji": k, "name": s.name, "sim": round(sim, 2)}
                        for k, s, d, sim in nearest
                    ],
                },
                "elapsed_seconds": round(time.monotonic() - start, 1)
                if 'start' in dir() else 0,
            }


# ═══════════════════════════════════════════════════════════════
# 5. CLI UNIFICADA
# ═══════════════════════════════════════════════════════════════

async def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   HyperSymbol v2 — 50D Semantic Framework   ║")
    print("║   Model-Agnostic Communication Protocol      ║")
    print("╚══════════════════════════════════════════════╝")

    args = sys.argv[1:]
    if not args:
        print("\n  Uso: python hyper_protocol.py <pergunta>")
        print("  Opções:")
        print("    --provider deepseek | qwen_local | openai")
        print("    --model <model_name>")
        print("    --mode semantic | symbol | steer:AXIS:+0.5")
        print()
        print("  Exemplos:")
        print("    python hyper_protocol.py \"O que é a gravidade?\"")
        print("    python hyper_protocol.py \"🌌❓\" --mode symbol")
        print("    python hyper_protocol.py --provider qwen_local \"O que é consciência?\"")
        print("    python hyper_protocol.py --mode steer:CON_ABS:+0.8 \"O que é o tempo?\"")
        sys.exit(1)

    # Parse args
    provider = "deepseek"
    model = None
    mode = "auto"
    question_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--provider" and i + 1 < len(args):
            provider = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        else:
            question_parts.append(args[i])
            i += 1

    question = " ".join(question_parts)

    if not provider or provider not in PROVIDERS:
        print(f"\n  ❌ Provider inválido. Opções: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"\n  ⚠️  API key para {provider} não encontrada no .env")
        sys.exit(1)

    print(f"\n  📡 Provider: {provider}")
    print(f"  🤖 Model: {model or cfg['default_model']}")
    print(f"  🔤 Mode: {mode}")
    print(f"  📥 Input: {question}")
    print(f"  🌐 Dims: {NUM_DIMS}")

    if mode in ("auto", "semantic", "symbol") or mode.startswith("steer:"):
        connector = ModelConnector(provider, model or cfg["default_model"])
        async with aiohttp.ClientSession() as session:
            result = await connector.communicate(session, question, mode)
            print(f"\n  {'='*50}")
            print(f"  📤 RESPOSTA ({result['mode']} mode)")
            print(f"  {'='*50}")
            if result.get("vector"):
                print(f"\n  🧭 Vetor semântico:")
                vec = SemanticVector.from_dict(result["vector"])
                print(vec.profile_str(5))
            print(f"\n  {result['response'][:500]}")
            print(f"\n  {'='*50}")

    # Salva sessão
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "question": question,
        "provider": provider,
        "model": model or cfg["default_model"],
        "mode": mode,
        "timestamp": ts,
    }
    Path(BASE_DIR / "responses").mkdir(exist_ok=True)
    with open(BASE_DIR / "responses" / f"session_{ts}.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
