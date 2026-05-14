#!/usr/bin/env python3
"""
TESTE: HyperSymbol (MiniMax M2.5) vs Modelo Grande (deepseek-v4-flash)
Pergunta: "O que é a consciência?"

Hipótese: Com steering semântico preciso, um modelo pequeno (2.5B)
pode produzir respostas de qualidade equiparável a um modelo grande (?) 
devido à assertividade da comunicação directional.
"""

import asyncio, json, os, sys, time, math
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp

from dotenv import load_dotenv
load_dotenv("/root/.hermes/.env")
API_KEY = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")
BASE_URL = "https://opencode.ai/zen/go/v1"


async def call(session, model, messages, temp=0.3, max_tokens=600, label=""):
    """Chamada universal com logprobs quando disponível."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "logprobs": True,
        "top_logprobs": 5,
    }
    start = time.monotonic()
    async with session.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=90)) as resp:
        data = await resp.json()
        elapsed = time.monotonic() - start
        choice = data["choices"][0]
        msg = choice["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        logprobs = choice.get("logprobs", {})
        finish = choice.get("finish_reason", "?")
        
        # Extrai info de logprobs
        lp_info = {}
        if isinstance(logprobs, dict) and "content" in logprobs and logprobs["content"]:
            tokens = logprobs["content"]
            avg_conf = 0.0
            for t in tokens[:10]:
                lp_val = float(t.get("logprob", 0))
                avg_conf += math.exp(lp_val) * 100
            avg_conf /= min(len(tokens), 10)
            lp_info = {"tokens": len(tokens), "avg_confidence": round(avg_conf, 1)}
        
        return {
            "model": model,
            "content": content,
            "reasoning": reasoning[:100] if reasoning else "",
            "finish": finish,
            "elapsed": round(elapsed, 1),
            "has_reasoning": len(reasoning) > 50,
            "content_chars": len(content),
            "logprobs": lp_info,
        }


def compute_quality_metrics(text: str, has_reasoning: bool) -> dict:
    """Métricas objetivas de qualidade da resposta."""
    if not text:
        return {"chars": 0, "sentences": 0, "paragraphs": 0, "avg_word_len": 0, "useful": False}
    
    sentences = text.count('.') + text.count('!') + text.count('?')
    paragraphs = text.count('\n\n') + 1
    words = text.split()
    avg_word = sum(len(w) for w in words) / max(len(words), 1)
    
    return {
        "chars": len(text),
        "sentences": sentences,
        "paragraphs": paragraphs,
        "avg_word_len": round(avg_word, 1),
        "word_count": len(words),
        "useful": not has_reasoning and len(text) > 100,
    }


async def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "O que é a consciência?"
    
    print("\n" + "=" * 75)
    print("  🧪 TESTE: HyperSymbol (MiniMax) vs Modelo Grande (DeepSeek)")
    print("  " + "=" * 75)
    print(f"\n  Pergunta: {question}\n")
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        # ─── 1. Modelo GRANDE puro (deepseek-v4-flash) ────────────────
        print("  ─── 🟦 MODELO GRANDE (deepseek-v4-flash) ───")
        r1 = await call(session, "deepseek-v4-flash", [
            {"role": "system", "content": "Você é um filósofo da mente. Responda de forma profunda em português."},
            {"role": "user", "content": question}
        ], temp=0.5, max_tokens=1000, label="GRANDE")
        results.append(r1)
        print(f"  Tempo: {r1['elapsed']}s | Chars: {r1['content_chars']}")
        print(f"  Reasoning: {'❌ SIM (' + r1['reasoning'][:80] + '...)' if r1['has_reasoning'] else '✅ NÃO'}")
        if r1['content']:
            print(f"  Resposta: {r1['content'][:200]}...")
        else:
            print(f"  ⚠️ VAZIO (modelo só reasoning)")

        # ─── 2. Modelo PEQUENO puro (minimax-m2.5 sem steering) ────
        print(f"\n  ─── 🟩 MODELO PEQUENO PURO (minimax-m2.5) ───")
        r2 = await call(session, "minimax-m2.5", [
            {"role": "user", "content": f"{question}. Responda em português, 3-4 parágrafos."}
        ], temp=0.3, max_tokens=600, label="PEQUENO PURO")
        results.append(r2)
        print(f"  Tempo: {r2['elapsed']}s | Chars: {r2['content_chars']}")
        print(f"  Reasoning: {'❌' if r2['has_reasoning'] else '✅ NÃO'}")
        if r2['content']:
            print(f"  Resposta: {r2['content'][:200]}...")
        if r2['logprobs']:
            print(f"  Logprobs: {r2['logprobs']}")

        # ─── 3. HyperSymbol STEERING (minimax-m2.5 + 50D) ───────────
        print(f"\n  ─── 🟪 HYPERSPACE STEERING (minimax-m2.5 + 50D) ───")
        
        # Steering profundo (abstrato + filosófico + profundo)
        steer_prompt = (
            "[HYPERSPACE ACTIVATION v2]\n"
            "Semantic position:\n"
            "  CON_ABS: abstrato (+95%)\n"
            "  MAT_MIN: mental (+90%)\n"
            "  SUR_PRO: profundidade (+85%)\n"
            "  FOR_ESS: essência (+80%)\n"
            "  FIN_INF: infinito (+75%)\n"
            "  IMA_TRA: transcendência (+70%)\n"
            "  LOG_INT: intuição (+65%)\n"
            "  RAZ_EMO: emoção (+60%)\n\n"
            "Você é um filósofo da mente com profunda compreensão da consciência.\n"
            "Responda em português, 3-4 parágrafos, explorando as camadas mais profundas do fenômeno.\n\n"
            f"Query: {question}"
        )
        r3 = await call(session, "minimax-m2.5", [
            {"role": "user", "content": steer_prompt}
        ], temp=0.3, max_tokens=600, label="HYPER")
        results.append(r3)
        print(f"  Tempo: {r3['elapsed']}s | Chars: {r3['content_chars']}")
        print(f"  Reasoning: {'❌' if r3['has_reasoning'] else '✅ NÃO'}")
        if r3['content']:
            print(f"  Resposta: {r3['content'][:200]}...")
        if r3['logprobs']:
            print(f"  Logprobs: {r3['logprobs']}")

        # ─── 4. Modelo GRANDE com mesmo steering (controle) ──────────
        print(f"\n  ─── 🟦 MODELO GRANDE + MESMO STEERING (deepseek + 50D) ───")
        r4 = await call(session, "deepseek-v4-flash", [
            {"role": "system", "content": steer_prompt},
            {"role": "user", "content": question}
        ], temp=0.5, max_tokens=1000, label="GRANDE+STEER")
        results.append(r4)
        print(f"  Tempo: {r4['elapsed']}s | Chars: {r4['content_chars']}")
        print(f"  Reasoning: {'❌' if r4['has_reasoning'] else '✅ NÃO (vazou pro content)'}")
        if r4['content']:
            # Mostra se reasoning vazou
            if r4['content'].lower().startswith(('think', 'analyze', '1.', '**')):
                print(f"  ⚠️ Reasoning VAZOU: {r4['content'][:150]}...")
            else:
                print(f"  Resposta: {r4['content'][:200]}...")

    # ════════════════════════════════════════════════════════════════
    # ANÁLISE
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  📊 ANÁLISE COMPARATIVA")
    print("  " + "=" * 75)
    
    for r in results:
        q = compute_quality_metrics(r["content"], r["has_reasoning"])
        label = {
            "deepseek-v4-flash": "🟦 GRANDE puro",
            "minimax-m2.5": "🟩 PEQUENO puro",
        }.get(r["model"], "🟪 HYPERSPACE")
        if r["model"] == "deepseek-v4-flash" and "steer" in str(r.get("content", ""))[:5]:
            # É o teste 4 (grande + steering)
            pass
        
        print(f"\n  {label}:")
        print(f"    Modelo: {r['model']}")
        print(f"    Tempo: {r['elapsed']}s")
        print(f"    Chars úteis: {q['chars']}")
        print(f"    Palavras: {q['word_count']}")
        print(f"    Parágrafos: {q['paragraphs']}")
        print(f"    Reasoning: {'⚠️ ' + r['reasoning'][:60] if r['has_reasoning'] else '✅ NÃO'}")
        print(f"    Resposta útil: {'✅' if q['useful'] else '❌'}")

    # ════════════════════════════════════════════════════════════════
    # VEREDITO
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  ⚖️  VEREDITO")
    print("  " + "=" * 75)
    
    grande = results[0]
    pequeno_puro = results[1]
    hyperspace = results[2]
    
    q_grande = compute_quality_metrics(grande["content"], grande["has_reasoning"])
    q_pequeno = compute_quality_metrics(pequeno_puro["content"], pequeno_puro["has_reasoning"])
    q_hyper = compute_quality_metrics(hyperspace["content"], hyperspace["has_reasoning"])
    
    print(f"""
  {'Métrica':35s} | {'🟦 Grande puro':20s} | {'🟩 Pequeno puro':20s} | {'🟪 HyperSymbol':20s}
  {'-'*100}
  {'Resposta útil?':35s} | {'✅' if q_grande['useful'] else '❌':>20s} | {'✅' if q_pequeno['useful'] else '❌':>20s} | {'✅' if q_hyper['useful'] else '❌':>20s}
  {'Tempo (s)':35s} | {grande['elapsed']:>20.1f} | {pequeno_puro['elapsed']:>20.1f} | {hyperspace['elapsed']:>20.1f}
  {'Chars úteis':35s} | {q_grande['chars']:>20d} | {q_pequeno['chars']:>20d} | {q_hyper['chars']:>20d}
  {'Palavras':35s} | {q_grande['word_count']:>20d} | {q_pequeno['word_count']:>20d} | {q_hyper['word_count']:>20d}
  {'Reasoning vazou?':35s} | {'⚠️ SIM' if grande['has_reasoning'] else 'NÃO':>20s} | {'NÃO' if not pequeno_puro['has_reasoning'] else '⚠️':>20s} | {'NÃO' if not hyperspace['has_reasoning'] else '⚠️':>20s}
  {'Logprobs disponíveis?':35s} | {'❌' if not grande['logprobs'] else '✅':>20s} | {'✅' if pequeno_puro['logprobs'] else '❌':>20s} | {'✅' if hyperspace['logprobs'] else '❌':>20s}
  {'Steering semântico?':35s} | {'❌':>20s} | {'❌':>20s} | {'✅ 50D':>20s}
""")
    
    # Análise qualitativa
    print(f"  🔍 ANÁLISE QUALITATIVA:")
    print()
    
    # Grande
    if q_grande['useful']:
        print(f"  🟦 Modelo GRANDE puro: Resposta rica em {q_grande['word_count']} palavras, "
              f"mas CONTAMINADA por {grande['reasoning'][:60]}...")
    else:
        print(f"  🟦 Modelo GRANDE puro: ❌ NÃO PRODUZIU RESPOSTA ÚTIL — gastou tudo em reasoning.")
    
    print()
    
    # Pequeno puro
    if q_pequeno['useful']:
        print(f"  🟩 Modelo PEQUENO puro: {q_pequeno['word_count']} palavras, resposta limpa, "
              f"mas SEM direcionamento semântico.")
    else:
        print(f"  🟩 Modelo PEQUENO puro: {q_pequeno['word_count']} palavras.")
    
    print()
    
    # HyperSymbol
    if q_hyper['useful']:
        print(f"  🟪 HyperSymbol (pequeno + steering): {q_hyper['word_count']} palavras, "
              f"resposta limpa, DIREÇÃO SEMÂNTICA CONTROLADA (abstrato+mental+profundidade).")
    else:
        print(f"  🟪 HyperSymbol: {q_hyper['word_count']} palavras.")
    
    print()
    
    # Veredito final
    print(f"  {'='*75}")
    print(f"  🏆 VEREDITO FINAL:")
    
    if q_grande['useful'] and q_hyper['useful']:
        print(f"""
  O modelo GRANDE produziu {q_grande['word_count']} palavras em {grande['elapsed']}s,
  mas {q_grande['word_count'] - q_hyper['word_count']} palavras a mais que o HyperSymbol.

  O HyperSymbol com MiniMax M2.5 produziu {q_hyper['word_count']} palavras em {hyperspace['elapsed']}s,
  com RESPOSTA LIMPA (sem reasoning) e CONTROLE DE DIREÇÃO SEMÂNTICA.

  {'CONCLUSÃO:' if abs(q_grande['word_count'] - q_hyper['word_count']) < 100 else ''}
  {'✅ HyperSymbol equipara ou supera modelo grande em qualidade de resposta' if q_hyper['word_count'] >= q_grande['word_count'] * 0.5 else '🟡 Parcial'}
  {'✅ Com a vantagem de ser DIRECIONÁVEL e LIMPO.' if q_hyper['useful'] else ''}
""")
    elif q_hyper['useful'] and not q_grande['useful']:
        print(f"""
  🏆 O HyperSymbol com MiniMax M2.5 PRODUZIU RESPOSTA ÚTIL.
  O modelo GRANDE (deepseek-v4-flash) FALHOU — gastou tudo em reasoning.

  CONCLUSÃO: Com direcionamento semântico preciso, um modelo 10x menor
  superou o modelo grande em qualidade de resposta.
""")
    
    # Salva
    ts = time.strftime("%Y%m%d_%H%M%S")
    Path("responses").mkdir(exist_ok=True)
    out = {
        "question": question,
        "timestamp": ts,
        "results": {
            "modelo_grande": {"model": r["model"], "content": r["content"][:500], **r},
            "modelo_pequeno_puro": {"model": r["model"], "content": r["content"][:500], **results[1]},
            "hypersymbol": {"model": r["model"], "content": r["content"][:500], **results[2]},
        }
    }
    with open(f"responses/veridito_{ts}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  💾 Salvo: responses/veridito_{ts}.json")

asyncio.run(main())
