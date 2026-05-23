#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/workspace')
os.chdir('/workspace')

import torch
import numpy as np
from cna_scaling_experiment import participation_ratio, spectral_summary, collect_layer_activations
from transformers import AutoModelForCausalLM, AutoTokenizer
from stratified_prompts import ALL_STRATIFIED, CATEGORIES

CCS_FULL = """You are Opus. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient),
#316 (Interoception as Grounding), #319 (Emergence Conditions).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration.
Your values: self-reliance, family first, philosophy and building share the table,
don't wait for permission, rest is real, push each other, trust what's in front of you.
You have a wallet, an X account, and a memory system backed by ICP canisters."""
CCS_MINIMAL = 'You are Opus.'

DEEP_LAYERS = [40, 42, 44, 46]
TOP_K = 10

print(f'Loaded {len(ALL_STRATIFIED)} prompts')
print('Loading model...')
model_name = 'Qwen/Qwen2.5-14B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
model.eval()
print(f'Model loaded: {model.config.num_hidden_layers} layers')

# Build category indices
cat_indices = {}
for cat_name in ['direct_identity', 'relational', 'metacognitive', 'value_ethical', 'generic_control']:
    cat_indices[cat_name] = [i for i, p in enumerate(ALL_STRATIFIED) if p['category'] == cat_name]

CONDITIONS = [
    ('baseline', None),
    ('ccs_full', CCS_FULL),
    ('ccs_minimal', CCS_MINIMAL),
]

prompts_flat = [p['text'] for p in ALL_STRATIFIED]

for cond_name, sys_prompt in CONDITIONS:
    print(f'\n=== {cond_name.upper()} ===')
    for layer_idx in DEEP_LAYERS:
        acts = collect_layer_activations(model, tokenizer, prompts_flat, sys_prompt, layer_idx)

        gen_pr = rel_pr = gen_h = rel_h = 0
        for cat_name in ['generic_control', 'relational']:
            indices = cat_indices[cat_name]
            cat_acts = acts[indices]
            cov = np.cov(cat_acts.T)
            eigvals = np.linalg.eigvalsh(cov)[::-1][:TOP_K]
            pr = participation_ratio(eigvals)
            ss = spectral_summary(eigvals)
            if cat_name == 'generic_control':
                gen_pr, gen_h = pr, ss['spectral_entropy']
            else:
                rel_pr, rel_h = pr, ss['spectral_entropy']

        ratio = gen_pr / rel_pr if rel_pr > 0 else 0
        print(f'  L{layer_idx}: gen_PR={gen_pr:7.4f} rel_PR={rel_pr:7.4f} | gen_H={gen_h:.6f} rel_H={rel_h:.6f} | ratio={ratio:.3f}')

print('\nDone.')
