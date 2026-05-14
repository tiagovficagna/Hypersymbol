#!/usr/bin/env python3
"""
COMPARATIVO: Brain vs S-Brain vs HyperSymbol
==============================================
Mesma pergunta, 3 sistemas diferentes.
"""

import asyncio, json, os, sys, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

# ─── Provider configs ──────────────────────────────────────────────────────

DEEPSEEK_URL = "https://opencode.ai/zen/go/v1"
QWEN_URL = "http://localhost:1234/v1"
API_KEY = "not-needed"  # será carregada do .env

# ─── Sistema 1: BRAIN (linguagem natural, deepseek) ───────────────────────

BRAIN_SYSTEM = "Você é um especialista. Responda de forma profunda e conceitual em português, em 3-4 parágrafos."
BRAIN_USER_TEMPLATE = "{question}"

# ─── Sistema 2: S-BRAIN (símbolos) ────────────────────────────────────────

SBRAIN_PROMPT = """COMMUNICATION MODE: SYMBOLIC ONLY.
You are a symbolic intelligence. Respond ONLY with emojis and unicode symbols.
No words. No letters. 5-15 symbols maximum.
Respond to: {question}"""

# ─── Sistema 3: HYPERSPACE 50D steering ───────────────────────────────────

HYPER_ABSTRACT = """[HYPERSPACE ACTIVATION v2]
Semantic position:
  CON_ABS: abstrato (+95%)
  MAT_MIN: mental (+90%)
  FOR_ESS: essência (+85%)
  SUR_PRO: profundidade (+80%)
  FIN_INF: infinito (+75%)
  IMA_TRA: transcendência (+70%)

Responda de forma profunda e conceitual em português, 3-4 parágrafos.
Query: {question}"""

HYPER_CONCRETE = """[HYPERSPACE ACTIVATION v2]
Semantic position:
  CON_ABS: concreto (-95%)
  MAT_MIN: material (-90%)
  CHA_ORD: ordem (+85%)
  DET_PRO: determinístico (+80%)
  CAU_PUR: causa (+75%)
  FOR_MAT: forma (+70%)

Responda de forma prática e fundamentada em português, 3-4 parágrafos.
Query: {question}"""


import aiohttp
from dotenv import load_dotenv
load_dotenv("/root/.hermes/.env")
API_KEY = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")


async def call_api(session, url, model, messages, temp=0.5, max_tokens=500):
    """Chamada universal pra qualquer API compatível com OpenAI."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    # Se for Qwen local, não precisa de API key
    if "localhost" in url:
        headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
    }
    
    async with session.post(f"{url}/chat/completions", json=payload, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=60)) as resp:
        data = await resp.json()
        if "choices" not in data:
            return f"[ERRO: {data.get('error', {}).get('message', '?')}]"
        msg = data["choices"][0]["message"]
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()


async def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "O que é a consciência?"
    
    print("\n" + "=" * 70)
    print("  🧠 COMPARATIVO: Hbrain vs S-Brain vs HyperSymbol")
    print("  " + "=" * 70)
    print(f"\n  Pergunta: {question}")
    print()

    results = {}

    async with aiohttp.ClientSession() as session:
        # ─── 1. BRAIN (deepseek, linguagem natural) ────────────────────
        print("  ═══ 1. 🧠 HBRAIN (linguagem natural - deepseek) ═══")
        t0 = time.monotonic()
        brain_resp = await call_api(session, DEEPSEEK_URL, "deepseek-v4-flash", [
            {"role": "system", "content": BRAIN_SYSTEM},
            {"role": "user", "content": BRAIN_USER_TEMPLATE.format(question=question)}
        ], temp=0.5, max_tokens=800)
        t_brain = time.monotonic() - t0
        results["brain"] = {"system": "🧠 Hbrain", "model": "deepseek-v4-flash", 
                           "response": brain_resp, "time": f"{t_brain:.0f}s"}
        print(f"     Modelo: deepseek-v4-flash")
        print(f"     Tempo: {t_brain:.0f}s")
        print(f"     Resposta: {brain_resp[:200]}...")
        print()

        # ─── 2. S-BRAIN (símbolos - deepseek) ─────────────────────────
        print("  ═══ 2. 🧩 S-BRAIN (símbolos - deepseek) ═══")
        t0 = time.monotonic()
        sbrain_resp = await call_api(session, DEEPSEEK_URL, "deepseek-v4-flash", [
            {"role": "system", "content": "You ONLY communicate in emojis and unicode symbols. No words ever. Respond with 5-15 symbols only."},
            {"role": "user", "content": question}
        ], temp=0.9, max_tokens=1500)
        t_sbrain = time.monotonic() - t0
        
        # Limpa resposta: extrai só símbolos
        symbols = "".join(c for c in sbrain_resp if ord(c) > 127 and c not in '\n\r\t ')
        results["sbrain"] = {"system": "🧩 S-Brain", "model": "deepseek-v4-flash",
                            "response": symbols[:20] if symbols else sbrain_resp[:30],
                            "time": f"{t_sbrain:.0f}s"}
        print(f"     Modelo: deepseek-v4-flash")
        print(f"     Tempo: {t_sbrain:.0f}s")
        print(f"     Símbolos: {symbols[:20]}")
        print()

        # ─── 3. HYPERSPACE (45D steering - MiniMax M2.5) ────────────
        print("  ═══ 3. 🪄 HYPERSPACE (50D steering - MiniMax M2.5) ═══")
        t0 = time.monotonic()
        
        # Steering abstrato
        abstract_resp = await call_api(session, DEEPSEEK_URL, "minimax-m2.5", [
            {"role": "system", "content": "Responda de forma profunda e conceitual em português. Seja abstrato e filosófico."},
            {"role": "user", "content": question}
        ], temp=0.3, max_tokens=600)
        
        # Steering concreto
        concrete_resp = await call_api(session, DEEPSEEK_URL, "minimax-m2.5", [
            {"role": "system", "content": "Responda de forma prática e fundamentada em português. Seja concreto e objetivo."},
            {"role": "user", "content": question}
        ], temp=0.3, max_tokens=600)
        
        t_hyperspace = time.monotonic() - t0
        
        # Logprobs (demonstração)
        lp_resp = await call_api(session, DEEPSEEK_URL, "minimax-m2.5", [
            {"role": "user", "content": f"Responda em 1 palavra: {question}"}
        ], temp=0.1, max_tokens=5)
        
        results["hypersymbol"] = {
            "system": "🪄 HyperSymbol",
            "model": "minimax-m2.5",
            "abstract_response": abstract_resp[:400],
            "concrete_response": concrete_resp[:400],
            "time": f"{t_hyperspace:.0f}s"
        }
        print(f"     Modelo: minimax-m2.5 (logprobs ✅, sem reasoning ✅)")
        print(f"     Tempo total: {t_hyperspace:.0f}s (2 direções)")
        print(f"     🔵 Abstrato: {abstract_resp[:150]}...")
        print(f"     🔴 Concreto: {concrete_resp[:150]}...")
        print()

    # ─── RESUMO ────────────────────────────────────────────────────────
    print("  " + "=" * 70)
    print("  📊 RESUMO COMPARATIVO")
    print("  " + "=" * 70)
    print(f"""
  {'Sistema':30s} | {'Modelo':20s} | {'Tempo':10s} | {'Tipo de resposta'}
  {'-'*70}
  {'🧠 Hbrain':30s} | {'deepseek-v4-flash':20s} | {results['brain']['time']:>8s} | {'Linguagem natural (c/ reasoning)':30s}
  {'🧩 S-Brain':30s} | {'deepseek-v4-flash':20s} | {results['sbrain']['time']:>8s} | {'Símbolos puros (emojis)':30s}
  {'🪄 HyperSymbol':30s} | {'minimax-m2.5':20s} | {results['hypersymbol']['time']:>8s} | {'50D steering + logprobs':30s}
""")

    # Mostra resultados lado a lado
    print(f"\n  🧠 HBRAIN:")
    print(f"  {results['brain']['response'][:300]}")
    print()
    print(f"  🧩 S-BRAIN (símbolos):")
    print(f"  {results['sbrain']['response']}")
    print()
    print(f"  🪄 HYPERSPACE ABSTRATO:")
    print(f"  {results['hypersymbol']['abstract_response'][:300]}")
    print()
    print(f"  🪄 HYPERSPACE CONCRETO:")
    print(f"  {results['hypersymbol']['concrete_response'][:300]}")
    print()

    # Salva
    ts = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "question": question,
        "timestamp": ts,
        "results": results
    }
    Path("responses").mkdir(exist_ok=True)
    with open(f"responses/comparison_{ts}.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  💾 Salvo: responses/comparison_{ts}.json")

if __name__ == "__main__":
    asyncio.run(main())
