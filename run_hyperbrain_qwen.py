#!/usr/bin/env python3
"""HyperBrain v2 — Quick 50D test with Qwen3 1.7B local"""
import asyncio, json, sys, os, time
sys.path.insert(0, '/root/workspace/hypersymbolv2')
sys.stdout.reconfigure(line_buffering=True)
os.chdir('/root/workspace/hypersymbolv2')

from hypersymbol_v2 import *
import aiohttp

HS = Hyperspace()
BASE_URL = 'http://localhost:1234/v1'
MODEL = 'qwen3-1.7b'

INTELLIGENCES = [
    ('📖', 'Linguística', '📖'),
    ('∑', 'Lógico-Matemática', '∑'),
    ('🔴', 'Espacial', '🔴'),
    ('🤸', 'Cinestésica', '🤸'),
    ('🎵', 'Musical', '🎵'),
    ('🤝', 'Interpessoal', '🤝'),
    ('🪞', 'Intrapessoal', '🪞'),
    ('🌿', 'Naturalista', '🌿'),
    ('∞', 'Existencial', '∞_E'),
]

async def query(session, prompt, temp=0.5):
    async with session.post(f'{BASE_URL}/chat/completions', json={
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': 'Responda diretamente, sem raciocínio interno, sem tags de thinking. Seja profundo e conceitual.'},
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 600,
        'temperature': temp,
    }, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        data = await resp.json()
        msg = data['choices'][0]['message']
        return (msg.get('content') or msg.get('reasoning_content') or '').strip()

async def main():
    question = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'como tornar a I.A. um ser sensiente?'
    
    print()
    print('╔══════════════════════════════════════════════╗')
    print('║   🧠 HyperBrain v2 — 50D  (Qwen3 1.7B)     ║')
    print('╚══════════════════════════════════════════════╝')
    print(f'  Input: {question}')
    print(f'  Model: {MODEL}')
    print(f'  Dims: 50\n')
    
    # Codifica pergunta
    from hyper_protocol import HyperCommunicationProtocol
    proto = HyperCommunicationProtocol()
    query_vec = proto.encode_query(question)
    
    print('  📊 Query Profile:')
    print(query_vec.profile_str(6))
    print()
    
    async with aiohttp.ClientSession() as session:
        results = []
        start = time.monotonic()
        
        for i, (sigil, name, key) in enumerate(INTELLIGENCES):
            sym = HS.get(key)
            anchor = sym.vector if sym else SemanticVector()
            biased = SemanticVector(np.clip((query_vec.v + anchor.v * 0.3) / 1.3, -1.0, 1.0))
            
            prompt = f'[HYPERSPACE - {name}]\nYou are the {name} intelligence (sigil: {sigil}).\nYour semantic position:\n{biased.dominant_axes(4)}\n\nFrom this exact position, answer: {question}\nBe concise (3-4 sentences).'
            
            print(f'  [{i+1}/9] {sigil} {name}... ', end='', flush=True)
            t0 = time.monotonic()
            resp = await query(session, prompt, 0.7 - i * 0.04)
            elapsed = time.monotonic() - t0
            print(f'✅ {elapsed:.0f}s | {len(resp)} chars')
            results.append({'sigil': sigil, 'name': name, 'resp': resp[:200], 'time': f'{elapsed:.0f}s'})
        
        total = time.monotonic() - start
        
        # Síntese
        print(f'\n  ⏳ Synthesis... ', end='', flush=True)
        ctx = '\n\n'.join(f'{r["sigil"]} {r["name"]}: {r["resp"]}' for r in results)
        syn = await query(session, f'[HYPERSPACE INTEGRATION]\nIntegrate these 9 perspectives into ONE concept (3-4 sentences).\n\n{ctx}\n\nSynthesis:', 0.3)
        print('✅\n')
        
        # Output
        print('╔══════════════════════════════════════════════╗')
        print('║        🧠  HYPERSPACE SYNTHESIS            ║')
        print('╚══════════════════════════════════════════════╝')
        print(f'\nInput: {question}')
        print(f'Time: {total:.0f}s | Provider: qwen3-1.7b (local)')
        print()
        
        for r in results:
            print(f'  {r["sigil"]} {r["name"]:20s} | {r["time"]} | {r["resp"][:100]}...')
        
        print(f'\n  {"─"*50}')
        print(f'\n  🧩 SYNTHESIS:\n  {syn[:800]}')
        print(f'\n  {"─"*50}')
        print(f'  9 intelligences | 50 dimensions | {total:.0f}s\n')

asyncio.run(main())
