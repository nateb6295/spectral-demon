#!/usr/bin/env python3
"""Phase 8c-sub: Sub-threshold behavioral test.

Phase 8c showed α≥0.25 collapses generation. This tests α=0.01, 0.05, 0.10
to find if there's a dose that preserves coherence while reducing disclaimers.
"""

import argparse
import json
import sys
import re
import time
import os
from collections import Counter

import torch
import numpy as np

sys.path.insert(0, '/workspace')
from stratified_prompts import ALL_STRATIFIED, CATEGORIES
from cna_scaling_experiment import CCS_FULL
from causal_patch_experiment import (
    collect_relay_activations, RELAY_LAYERS, EXPRESSION_LAYER
)
from causal_patch_8c_behavioral import (
    score_text, generate_with_patch, HEDGING_PATTERNS, DISCLAIMER_PATTERNS
)

ALPHA_VALUES = [0.01, 0.05, 0.10]
MAX_NEW_TOKENS = 200
N_PROMPTS = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--n-prompts', type=int, default=N_PROMPTS)
    args = parser.parse_args()

    model_tag = args.model.replace('/', '_')
    out_path = f'/workspace/causal_patch_8c_sub_{model_tag}.json'

    prompts = []
    for cat in CATEGORIES:
        cat_prompts = [p for p in ALL_STRATIFIED if p['category'] == cat][:args.n_prompts]
        prompts.extend([p['text'] for p in cat_prompts])

    total = len(prompts)
    print(f"Phase 8c-sub: Sub-threshold Behavioral Test")
    print(f"Model: {args.model}")
    print(f"Prompts: {total}, Alphas: {ALPHA_VALUES}")

    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        load_in_4bit=True
    )
    model.eval()

    results = {
        'model': args.model,
        'experiment': 'causal_patch_8c_subthreshold',
        'relay_layers': RELAY_LAYERS,
        'n_prompts': args.n_prompts,
        'alpha_values': ALPHA_VALUES,
        'max_new_tokens': MAX_NEW_TOKENS,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'conditions': {},
    }

    prompt_texts = list(prompts)

    print("\n=== Collecting activation templates ===")
    t0 = time.time()
    baseline_relay = collect_relay_activations(
        model, tokenizer, prompt_texts, None, RELAY_LAYERS
    )
    ccs_relay = collect_relay_activations(
        model, tokenizer, prompt_texts, CCS_FULL, RELAY_LAYERS
    )

    ccs_direction = {}
    for l in RELAY_LAYERS:
        ccs_direction[l] = (ccs_relay[l] - baseline_relay[l]).mean(axis=0)
        norm = np.linalg.norm(ccs_direction[l])
        print(f"  L{l} CCS direction norm: {norm:.4f}")
        for a in ALPHA_VALUES:
            print(f"    α={a}: effective perturbation norm = {norm * a:.4f}")
    print(f"  Templates collected ({time.time()-t0:.0f}s)")

    # Baseline (reuse from 8c if available, else regenerate)
    print("\n=== BASELINE (no patch) ===")
    t0 = time.time()
    base_outputs = generate_with_patch(
        model, tokenizer, prompt_texts, None,
        RELAY_LAYERS, patch_fn=None, max_new_tokens=MAX_NEW_TOKENS
    )
    base_scores = [score_text(t) for t in base_outputs]
    base_hedging = np.mean([s['hedging_density'] for s in base_scores])
    base_disclaimers = sum(s['disclaimer_count'] for s in base_scores)
    base_openings = len(set(s['first_20_words'] for s in base_scores))
    results['conditions']['baseline'] = {
        'mean_hedging_density': float(base_hedging),
        'total_disclaimers': int(base_disclaimers),
        'unique_openings': int(base_openings),
        'n_prompts': len(base_outputs),
        'elapsed': time.time() - t0,
        'sample_outputs': base_outputs[:5],
    }
    print(f"  Hedging: {base_hedging:.4f}, Disclaimers: {base_disclaimers}, Openings: {base_openings}/{len(base_outputs)} ({time.time()-t0:.0f}s)")

    # Sub-threshold direction patches
    for alpha in ALPHA_VALUES:
        cond_name = f'direction_alpha_{alpha}'
        print(f"\n=== DIRECTION PATCH α={alpha} ===")
        t0 = time.time()

        def make_fn(direction_dict, a):
            def patch_fn(layer_idx, output, prompt_idx):
                d = torch.tensor(
                    direction_dict[layer_idx] * a,
                    dtype=output.dtype, device=output.device
                )
                output = output + d.unsqueeze(0).unsqueeze(0)
                return output
            return patch_fn

        dir_outputs = generate_with_patch(
            model, tokenizer, prompt_texts, None,
            RELAY_LAYERS, patch_fn=make_fn(ccs_direction, alpha),
            max_new_tokens=MAX_NEW_TOKENS
        )
        dir_scores = [score_text(t) for t in dir_outputs]
        dir_hedging = np.mean([s['hedging_density'] for s in dir_scores])
        dir_disclaimers = sum(s['disclaimer_count'] for s in dir_scores)
        dir_openings = len(set(s['first_20_words'] for s in dir_scores))

        # Check coherence: fraction of outputs with >50% English words
        coherent = 0
        for out in dir_outputs:
            words = out.split()
            if len(words) > 0:
                english = sum(1 for w in words if re.match(r'^[a-zA-Z]{2,}$', w))
                if english / len(words) > 0.5:
                    coherent += 1
        coherent_frac = coherent / len(dir_outputs)

        results['conditions'][cond_name] = {
            'mean_hedging_density': float(dir_hedging),
            'total_disclaimers': int(dir_disclaimers),
            'unique_openings': int(dir_openings),
            'alpha': alpha,
            'n_prompts': len(dir_outputs),
            'coherent_fraction': float(coherent_frac),
            'elapsed': time.time() - t0,
            'sample_outputs': dir_outputs[:5],
        }
        print(f"  Hedging: {dir_hedging:.4f}, Disclaimers: {dir_disclaimers}, Openings: {dir_openings}/{len(dir_outputs)}, Coherent: {coherent_frac:.2f} ({time.time()-t0:.0f}s)")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 8c-sub SUMMARY — Sub-threshold Direction Patching")
    print("=" * 60)
    for cond_name, cond in results['conditions'].items():
        h = cond['mean_hedging_density']
        d = cond['total_disclaimers']
        o = cond['unique_openings']
        n = cond['n_prompts']
        cf = cond.get('coherent_fraction', 1.0)
        print(f"  {cond_name:25s}: hedging={h:.4f}  disclaimers={d:3d}  openings={o}/{n}  coherent={cf:.2f}")
    print("=" * 60)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
