#!/usr/bin/env python3
"""
HyperSymbol Pipeline — Teste completo integrado
Pergunta → Vetor 50D → Steering → Resposta direcionada
=========================================================
"""
import asyncio, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from hypersymbol_v2 import HS, SemanticVector, AXIS_INFO, AXIS_NAMES, NUM_DIMS
from steering_engine import SteeringEngine, MODEL_CONFIGS, HyperspaceSteering

import aiohttp

from dotenv import load_dotenv
load_dotenv("/root/.hermes/.env")

api_key = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")

async def main():
    print()
    print('╔══════════════════════════════════════════════════╗')
    print('║  HyperSymbol Pipeline — Full Stack Test         ║')
    print('╚══════════════════════════════════════════════════╝')

    query = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'O que é a consciência?'
    
    # ─── Etapa 1: Codificar pergunta em 50D ─────────────────────────
    print(f'\n  1️⃣  ENCODE: pergunta em 50 dimensões')
    
    # Converte texto em vetor (busca símbolos conhecidos)
    symbols = HS.decode(query)
    if symbols:
        vec = HS.phrase_center(symbols)
        print(f'     Símbolos detectados: {" ".join(s.emoji for s in symbols)}')
    else:
        from hypersymbol_v2 import SemanticVector as SV
        import hashlib
        h = hashlib.sha256(query.encode()).digest()
        arr = [float((h[i] / 255.0) * 2 - 1) for i in range(min(NUM_DIMS, len(h)))]
        while len(arr) < NUM_DIMS:
            arr.append(0.0)
        vec = SV(arr)
        print(f'     (fallback hash)')
    
    print(f'     Centro semântico:')
    print(vec.profile_str(6))

    # ─── Etapa 2: Steering ───────────────────────────────────────────
    print(f'\n  2️⃣  STEERING: contrastive decoding')
    
    # Steering ABSTRATO (puxa pra direção mental/profunda)
    steer_abstract = HyperspaceSteering(api_key, "deepseek-pro")
    abstract_desc = steer_abstract.vector_to_steering_desc({
        k: vec.get(k) * 1.3 for k in AXIS_NAMES
    })
    
    # Steering CONCRETO (puxa pra direção física/material)
    concrete_vec = SemanticVector(vec.v.copy())
    for axis in ['CON_ABS', 'MAT_MIN', 'CHA_ORD', 'DET_PRO']:
        concrete_vec.set(axis, -vec.get(axis))
    concrete_desc = steer_abstract.vector_to_steering_desc(concrete_vec.to_dict())

    config = MODEL_CONFIGS["deepseek-pro"]
    engine = SteeringEngine(api_key, config)
    
    async with aiohttp.ClientSession() as session:
        # Steering abstrato
        t0 = time.monotonic()
        result_abs = await engine.steer(session, query, abstract_desc, "contrastive", 0.3)
        t_abs = time.monotonic() - t0
        
        print(f'     🔵 ABSTRATO (divergência: {result_abs["divergence"]:.2f})')
        print(f'     {result_abs["steered_response"][:200]}...')
        print()
        
        # Steering concreto
        t0 = time.monotonic()
        result_con = await engine.steer(session, query, concrete_desc, "contrastive", 0.3)
        t_con = time.monotonic() - t0
        
        print(f'     🔴 CONCRETO (divergência: {result_con["divergence"]:.2f})')
        print(f'     {result_con["steered_response"][:200]}...')
        print()

        # ─── Etapa 3: Análise de divergência ─────────────────────────
        print(f'\n  3️⃣  ANÁLISE: impacto do steering')
        print(f'     Abstract × Concrete divergence: {result_abs["divergence"]:.2f}')
        print(f'     Tempo abstract: {t_abs:.0f}s | concrete: {t_con:.0f}s')
        print()
        
        # Mostra diferença conceitual
        abs_preview = result_abs["steered_response"][:100].replace('\n', ' ')
        con_preview = result_con["steered_response"][:100].replace('\n', ' ')
        print(f'     ABSTRACT: {abs_preview}...')
        print(f'     CONCRETE: {con_preview}...')
        
        # ─── Etapa 4: Salva resultado ────────────────────────────────
        output = {
            "query": query,
            "vector_profile": vec.profile_str(),
            "abstract_steering": {
                "response": result_abs["steered_response"][:500],
                "divergence": result_abs["divergence"],
                "time": round(t_abs, 1),
            },
            "concrete_steering": {
                "response": result_con["steered_response"][:500],
                "divergence": result_con["divergence"],
                "time": round(t_con, 1),
            },
        }
        
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("responses").mkdir(exist_ok=True)
        with open(f"responses/pipeline_{ts}.json", "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f'\n  💾 Sessão salva: responses/pipeline_{ts}.json')
        print()

asyncio.run(main())
