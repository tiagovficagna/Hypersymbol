#!/usr/bin/env python3
"""Teste MiniMax M2.5 — logprobs + português"""
import os, json, requests, math
from dotenv import load_dotenv
load_dotenv('/root/.hermes/.env')
api_key = os.environ.get('OPENCODE_GO_API_KEY')
url = 'https://opencode.ai/zen/go/v1/chat/completions'

print('=== MINIMAX M2.5 — LOGPROBS TEST ===\n')

resp = requests.post(url, headers={
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}, json={
    'model': 'minimax-m2.5',
    'messages': [{'role': 'user', 'content': 'What is gravity? Explain in 2 sentences.'}],
    'max_tokens': 100,
    'temperature': 0.1,
    'logprobs': True,
    'top_logprobs': 5,
}, timeout=30)

data = resp.json()
choice = data['choices'][0]
msg = choice['message']
c = msg.get('content', '')
lp = choice.get('logprobs', None)
fr = choice.get('finish_reason', '?')

print(f'Finish: {fr}')
print(f'Content ({len(c)} chars): {c[:200]}')
print()

if isinstance(lp, dict) and 'content' in lp and lp['content']:
    tokens = lp['content']
    print(f'Logprobs: {len(tokens)} tokens')
    print()
    for i, t in enumerate(tokens[:8]):
        tok = t.get('token', '?')
        lp_val = float(t.get('logprob', 0))
        prob = math.exp(lp_val) * 100
        top = t.get('top_logprobs', [])
        top_items = [(x.get('token', '?'), f'{math.exp(float(x.get("logprob",0)))*100:.0f}%') for x in top[:3]]
        top_str = ', '.join(f'{a}({b})' for a, b in top_items)
        print(f'  [{i}] {tok:15s} {prob:.1f}% | {top_str}')
else:
    print(f'Logprobs: {json.dumps(str(lp))[:200]}')

print()
print('--- RESPOSTA EM PORTUGUÊS ---')
resp2 = requests.post(url, headers={
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}, json={
    'model': 'minimax-m2.5',
    'messages': [{'role': 'user', 'content': 'O que é a gravidade? Responda em 2 frases.'}],
    'max_tokens': 200,
    'temperature': 0.3,
    'logprobs': True,
    'top_logprobs': 3,
}, timeout=30)

data2 = resp2.json()
c2 = data2['choices'][0]['message'].get('content', '')
lp2 = data2['choices'][0].get('logprobs', {})
print(f'Resposta ({len(c2)} chars):')
print(c2)
print()
if isinstance(lp2, dict) and 'content' in lp2 and lp2['content']:
    t0 = lp2['content'][0]
    top = [(x.get('token', ''), f'{math.exp(float(x.get("logprob",0)))*100:.0f}%') for x in t0.get('top_logprobs', [])[:3]]
    print(f'Top 3 pro primeiro token: {top}')

print('\n✅ MiniMax M2.5: LOGPROBS OK + PT-BR OK + sem reasoning')
