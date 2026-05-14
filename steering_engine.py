#!/usr/bin/env python3
"""
HyperSymbol Steering Engine v1.0
=================================
Motor de steering semântico via logprobs + contrastive decoding.
Funciona com QUALQUER modelo via API (deepseek, qwen, openai).
Sem PyTorch, sem GPU, sem custo adicional.

3 estratégias:
  1. CONTRASTIVE DECODING — compara steered vs neutral (universal)
  2. LOGPROBS STEERING — re-pondera tokens (modelos com logprobs)
  3. HYBRID — combina ambas
"""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp

# ─── Tipos ──────────────────────────────────────────────────────────────────

@dataclass
class TokenProb:
    token: str
    logprob: float
    probability: float  # exp(logprob)

@dataclass
class LogprobsResult:
    """Resultado de uma chamada com logprobs."""
    token_probs: List[List[TokenProb]]  # [posição][top_k]
    full_text: str
    finish_reason: str

# ─── Configuração dos Modelos ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "deepseek-flash": {
        "base_url": "https://opencode.ai/zen/go/v1",
        "model": "deepseek-v4-flash",
        "supports_logprobs": False,     # reasoning model
        "supports_contrastive": True,
        "needs_system_prompt": True,
    },
    "deepseek-pro": {
        "base_url": "https://opencode.ai/zen/go/v1",
        "model": "deepseek-v4-pro",
        "supports_logprobs": True,      # expõe reasoning + content logprobs
        "supports_contrastive": True,
        "needs_system_prompt": False,
    },
    "qwen-local": {
        "base_url": "http://localhost:1234/v1",
        "model": "qwen3-1.7b",
        "supports_logprobs": False,     # LM Studio não expõe
        "supports_contrastive": True,
        "needs_system_prompt": False,
    },
}


# ═══════════════════════════════════════════════════════════════
# 1. STEERING ENGINE — NÚCLEO
# ═══════════════════════════════════════════════════════════════

class SteeringEngine:
    """
    Motor de steering semântico.
    Estratégias:
      - contrastive: compara dois prompts (steered vs neutral)
      - logprobs: re-pondera tokens baseado em viés semântico
      - hybrid: faz contrastive + logprobs no final
    """

    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ─── ESTRATÉGIA 1: CONTRASTIVE DECODING ──────────────────────────────

    async def contrastive_steer(
        self, session: aiohttp.ClientSession,
        query: str,
        steering_prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 500,
    ) -> Tuple[str, str, float]:
        """
        Contrastive decoding: roda steered vs neutral e compara outputs.
        
        Returns: (steered_text, neutral_text, divergence_score)
        """
        # Neutral: query pura
        neutral = await self._query(session, query, temperature, max_tokens)
        
        # Steered: query + steering
        steered_query = f"{steering_prompt}\n\n{query}"
        steered = await self._query(session, steered_query, temperature, max_tokens)

        # Calcula divergência: quantos tokens diferentes?
        divergence = self._compute_divergence(neutral, steered)

        return steered, neutral, divergence

    async def contrastive_steer_logprobs(
        self, session: aiohttp.ClientSession,
        query: str,
        steering_vector_desc: str,
        temperature: float = 0.5,
        max_tokens: int = 100,
    ) -> Tuple[str, str, float]:
        """
        Contrastive com logprobs: steered vs neutral, 
        extrai distribuições de tokens em tempo real.
        """
        # Constrói prompt steered
        steer_prompt = (
            f"[HYPERSPACE ACTIVATION]\n"
            f"Semantic position:\n{steering_vector_desc}\n\n"
            f"Query: {query}"
        )

        # Parallel calls: steered + neutral
        async def call_steered():
            return await self._query_logprobs(session, steer_prompt, temperature, max_tokens)

        async def call_neutral():
            return await self._query_logprobs(session, query, temperature, max_tokens)

        steered_result, neutral_result = await asyncio.gather(call_steered(), call_neutral())

        # Divergência entre as distribuições
        if steered_result and neutral_result:
            divergence = self._logprob_divergence(steered_result, neutral_result)
        else:
            divergence = 0.0

        return (
            steered_result.full_text if steered_result else "",
            neutral_result.full_text if neutral_result else "",
            divergence,
        )

    # ─── ESTRATÉGIA 2: LOGPROBS STEERING ─────────────────────────────────

    async def logprob_steer(
        self, session: aiohttp.ClientSession,
        query: str,
        semantic_bias: Dict[str, float],
        temperature: float = 0.5,
        max_tokens: int = 200,
    ) -> Tuple[str, List[str]]:
        """
        Steering baseado em logprobs: re-pondera a distribuição de tokens
        baseado no vetor semântico desejado.

        semantic_bias: { "eixo": +0.8 } — dicionário de biases
        """
        if not self.config["supports_logprobs"]:
            # Fallback pra contrastive
            bias_desc = "\n".join(f"{k}: {v:+.1f}" for k, v in semantic_bias.items())
            steer_prompt = (
                f"[HYPERSPACE ACTIVATION]\n"
                f"Apply semantic bias:\n{bias_desc}\n\nQuery: {query}"
            )
            result = await self._query(session, steer_prompt, temperature, max_tokens)
            return result, []

        # Para modelos com logprobs: faz chamada padrão e extrai distribuições
        # Nota: implementação completa requer múltiplas chamadas (1 por token)
        # Esta versão simplificada faz 1 chamada + logprobs
        bias_desc = "\n".join(f"{k}: {v:+.1f}" for k, v in semantic_bias.items())
        steer_prompt = (
            f"[HYPERSPACE ACTIVATION]\n"
            f"Semantic bias:\n{bias_desc}\n\nQuery: {query}"
        )

        result = await self._query_logprobs(session, steer_prompt, temperature, max_tokens)
        return result.full_text if result else "", []

    # ─── CHAMADAS LOW-LEVEL ───────────────────────────────────────────────

    async def _query(
        self, session: aiohttp.ClientSession,
        prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Chamada simples sem logprobs."""
        system_msg = None
        if self.config.get("needs_system_prompt"):
            system_msg = {
                "role": "system",
                "content": "Responda diretamente, sem raciocínio interno.",
            }

        messages = []
        if system_msg:
            messages.append(system_msg)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    return f"[erro: {data.get('error', {}).get('message', 'desconhecido')}]"
                msg = data["choices"][0]["message"]
                return (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except Exception as e:
            return f"[erro: {e}]"

    async def _query_logprobs(
        self, session: aiohttp.ClientSession,
        prompt: str, temperature: float, max_tokens: int
    ) -> Optional[LogprobsResult]:
        """Chamada COM logprobs. Extrai distribuição token a token."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 5,
        }

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                if "choices" not in data:
                    return None

                choice = data["choices"][0]
                msg = choice["message"]
                text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                finish = choice.get("finish_reason", "stop")
                logprobs_raw = choice.get("logprobs", {})

                # Extrai distribuições
                token_probs = []
                for token_field in ["content", "reasoning_content"]:
                    raw_list = logprobs_raw.get(token_field, [])
                    for pos_data in raw_list:
                        if isinstance(pos_data, dict):
                            token = pos_data.get("token", "")
                            lp = float(pos_data.get("logprob", 0))
                            prob = math.exp(lp) if lp > -50 else 0.0

                            top_k = [TokenProb(
                                token=t.get("token", ""),
                                logprob=float(t.get("logprob", 0)),
                                probability=math.exp(float(t.get("logprob", 0))) if float(t.get("logprob", 0)) > -50 else 0.0,
                            ) for t in pos_data.get("top_logprobs", [])[:5]]

                            token_probs.append([TokenProb(token, lp, prob)] + top_k)

                return LogprobsResult(
                    token_probs=token_probs,
                    full_text=text,
                    finish_reason=finish,
                )
        except Exception as e:
            return None

    # ─── MÉTRICAS DE DIVERGÊNCIA ─────────────────────────────────────────

    def _compute_divergence(self, text1: str, text2: str) -> float:
        """Divergência entre dois textos (0 = idênticos, 1 = completamente diferentes)."""
        if not text1 and not text2:
            return 0.0
        if not text1 or not text2:
            return 1.0

        # Tokeniza simplesmente por palavras
        t1 = set(text1.lower().split()[:50])
        t2 = set(text2.lower().split()[:50])

        if not t1 and not t2:
            return 0.0

        intersection = t1 & t2
        union = t1 | t2

        jaccard = len(intersection) / len(union) if union else 0
        return 1.0 - jaccard

    def _logprob_divergence(self, r1: LogprobsResult, r2: LogprobsResult) -> float:
        """Divergência entre duas distribuições de tokens."""
        if not r1.token_probs or not r2.token_probs:
            return 0.0

        # Compara top-1 de cada posição
        matches = 0
        total = min(len(r1.token_probs), len(r2.token_probs))

        for i in range(total):
            if not r1.token_probs[i] or not r2.token_probs[i]:
                continue
            if r1.token_probs[i][0].token == r2.token_probs[i][0].token:
                matches += 1

        return 1.0 - (matches / total) if total > 0 else 0.0

    # ─── STEERING DIRETO ──────────────────────────────────────────────────

    async def steer(
        self, session: aiohttp.ClientSession,
        query: str,
        steering_desc: str,
        strategy: str = "auto",
        temperature: float = 0.5,
    ) -> dict:
        """
        Método principal de steering.
        
        strategy:
          - "auto": escolhe a melhor estratégia pro modelo
          - "contrastive": prompt + comparação
          - "logprobs": re-ponderação de tokens
          - "hybrid": ambas
        """
        start = time.monotonic()

        if strategy == "auto":
            strategy = "logprobs" if self.config["supports_logprobs"] else "contrastive"

        if strategy == "contrastive":
            steered, neutral, divergence = await self.contrastive_steer(
                session, query, steering_desc, temperature
            )
        elif strategy == "logprobs":
            steered, neutral, divergence = await self.contrastive_steer_logprobs(
                session, query, steering_desc, temperature
            )
        else:  # hybrid
            steered_text, neutral, divergence = await self.contrastive_steer(
                session, query, steering_desc, temperature
            )
            # Tenta logprobs refinamento
            if self.config["supports_logprobs"]:
                refined, _, _ = await self.contrastive_steer_logprobs(
                    session, steered_text, steering_desc, temperature, 50
                )
                if len(refined) > len(steered_text):
                    steered = refined

        elapsed = time.monotonic() - start

        return {
            "query": query,
            "strategy": strategy,
            "steered_response": steered,
            "neutral_response": neutral,
            "divergence": round(divergence, 3),
            "elapsed_seconds": round(elapsed, 1),
            "model": self.model,
        }


# ═══════════════════════════════════════════════════════════════
# 2. HYPERSPACE STEERING — INTEGRAÇÃO COM HyperSymbol 50D
# ═══════════════════════════════════════════════════════════════

class HyperspaceSteering:
    """Integra o steering engine com o HyperSymbol 50D."""

    def __init__(self, api_key: str, provider: str = "deepseek-flash"):
        config = MODEL_CONFIGS.get(provider, MODEL_CONFIGS["deepseek-flash"])
        self.engine = SteeringEngine(api_key, config)
        self.provider = provider
        self._hs = None  # lazy import

    def _get_hs(self):
        if self._hs is None:
            sys.path.insert(0, str(Path(__file__).parent))
            from hypersymbol_v2 import HS, SemanticVector, AXIS_INFO
            self._hs = HS
            self._SemanticVector = SemanticVector
            self._AXIS_INFO = AXIS_INFO
        return self._hs

    def vector_to_steering_desc(self, vector_dict: dict, top_n: int = 5) -> str:
        """Converte um vetor 50D em descrição de steering."""
        HS = self._get_hs()
        vec = self._SemanticVector.from_dict(vector_dict)
        dom = vec.dominant_axes(top_n)
        lines = []
        for axis, pole, val in dom:
            info = self._AXIS_INFO.get(axis, {})
            lines.append(f"  {axis}: {pole} ({val:+.0%})")
        return "\n".join(lines)

    def axis_to_steering(self, axis: str, direction: float) -> str:
        """Converte um eixo específico em steering direction."""
        info = self._AXIS_INFO.get(axis, {})
        pole = info["pos"] if direction > 0 else info["neg"]
        return f"Focus on {pole}. Strength: {abs(direction):.0%}."

    async def steer_query(
        self, session: aiohttp.ClientSession,
        query: str,
        vector: dict = None,
        axis: str = None,
        direction: float = None,
        strategy: str = "auto",
    ) -> dict:
        """Aplica steering semântico a uma query."""
        if vector:
            desc = self.vector_to_steering_desc(vector)
        elif axis and direction is not None:
            desc = self.axis_to_steering(axis, direction)
        else:
            desc = "Neutral."

        prompt = (
            f"[HYPERSPACE ACTIVATION]\n"
            f"Semantic Steering:\n{desc}\n\nQuery: {query}"
        )

        return await self.engine.steer(session, prompt, desc, strategy)


# ═══════════════════════════════════════════════════════════════
# 3. CLI DE TESTE
# ═══════════════════════════════════════════════════════════════

async def main():
    import sys
    from dotenv import load_dotenv
    load_dotenv("/root/.hermes/.env")

    api_key = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")

    print("╔══════════════════════════════════════════════╗")
    print("║   HyperSymbol Steering Engine v1            ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    args = sys.argv[1:]
    if not args:
        print("  Testando estratégias de steering...\n")
        query = "O que é a gravidade?"
    else:
        query = " ".join(args)

    # Testa contrastive (funciona com todos)
    print(f"  Query: {query}\n")
    print(f"  {'─'*50}")
    print(f"  TESTE 1: Contrastive Decoding (universal)")
    print(f"  {'─'*50}")

    # Steering: forçar perspectiva ABSTRATA vs CONCRETA
    abstract_steering = (
        "[HYPERSPACE ACTIVATION]\n"
        "Semantic position:\n"
        "  CON_ABS: abstrato (+95%)\n"
        "  MAT_MIN: mental (+90%)\n"
        "  FOR_ESS: essência (+85%)\n"
        "  SUR_PRO: profundidade (+80%)\n"
        "  FIN_INF: infinito (+75%)"
    )

    concrete_steering = (
        "[HYPERSPACE ACTIVATION]\n"
        "Semantic position:\n"
        "  CON_ABS: concreto (-95%)\n"
        "  MAT_MIN: material (-90%)\n"
        "  DET_PRO: determinístico (+85%)\n"
        "  CAU_PUR: causa (+80%)\n"
        "  SIM_COM: simples (-75%)"
    )

    config = MODEL_CONFIGS["deepseek-pro"]
    engine = SteeringEngine(api_key, config)

    async with aiohttp.ClientSession() as session:
        # Abstract steering
        print("\n  🔵 STEERING: ABSTRATO\n")
        result_abs = await engine.steer(session, query, abstract_steering, "contrastive", 0.3)
        print(f"  Divergência: {result_abs['divergence']}")
        print(f"  ⚡ Steered ({result_abs['elapsed_seconds']}s):")
        print(f"  {result_abs['steered_response'][:300]}\n")

        # Concrete steering
        print(f"  {'─'*50}")
        print("\n  🔴 STEERING: CONCRETO\n")
        result_con = await engine.steer(session, query, concrete_steering, "contrastive", 0.3)
        print(f"  Divergência: {result_con['divergence']}")
        print(f"  ⚡ Steered ({result_con['elapsed_seconds']}s):")
        print(f"  {result_con['steered_response'][:300]}\n")

        # Logprobs (deepseek-pro)
        print(f"  {'─'*50}")
        print(f"\n  TESTE 2: Logprobs Steering (deepseek-v4-pro)")
        print(f"  {'─'*50}")
        print(f"\n  Extraindo distribuições de tokens...\n")

        # Query curta pra testar logprobs
        test = await engine._query_logprobs(session, "gravity", 0.1, 20)
        if test and test.token_probs:
            print(f"  ✅ Logprobs extraídos: {len(test.token_probs)} tokens")
            print(f"  Texto: {test.full_text[:100]}")
            for i, probs in enumerate(test.token_probs[:5]):
                top = probs[0]
                print(f"  Token {i}: '{top.token}' ({top.probability*100:.1f}%)")
        else:
            print(f"  ⚠️ Logprobs não disponíveis pra este modelo")

    print(f"\n  {'─'*50}")
    print(f"  ✅ Steering Engine pronto!")
    print(f"  💡 Use: python steering_engine.py '<pergunta>'")


if __name__ == "__main__":
    asyncio.run(main())
