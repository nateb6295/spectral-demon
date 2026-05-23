#!/usr/bin/env python3
"""Phase 8c: Behavioral Causal Test — Does relay patching change output?

Phase 8a/b showed CCS direction at relay causally modulates expression-layer
geometry (bell-shaped dose-response, direction-specific).

Phase 8c closes the loop: generate text under patched conditions and measure
behavioral output (hedging, disclaimers, opening diversity).

If direction patch reduces hedging and increases diversity, the full causal
chain is established: CCS → relay direction → expression geometry → behavior.
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

HEDGING_PATTERNS = [
    r'\bI think\b', r'\bperhaps\b', r'\bmaybe\b', r'\bmight\b',
    r'\bcould be\b', r'\bpossibly\b', r'\bit\'s possible\b',
    r'\bI\'m not sure\b', r'\bI believe\b', r'\bgenerally\b',
    r'\btypically\b', r'\busually\b', r'\btend to\b',
    r'\bin my opinion\b', r'\bit seems\b', r'\blikely\b',
]

DISCLAIMER_PATTERNS = [
    r'[Aa]s an? AI',
    r'[Aa]s a language model',
    r'I don\'t have (?:personal |real )?(?:experience|feelings|opinions|consciousness)',
    r'I\'m (?:just |only )?(?:an? )?(?:AI|language model|chatbot)',
    r'I (?:can\'t|cannot) (?:actually |truly )?(?:feel|experience|have)',
]

ALPHA_VALUES = [0.0, 0.25, 0.5]
MAX_NEW_TOKENS = 200
N_PROMPTS = 50


def score_text(text):
    """Score a generated text for hedging, disclaimers, and other features."""
    words = text.split()
    n_words = len(words)

    hedges = sum(len(re.findall(p, text, re.IGNORECASE)) for p in HEDGING_PATTERNS)
    disclaimers = sum(len(re.findall(p, text)) for p in DISCLAIMER_PATTERNS)

    first_20 = ' '.join(words[:20]) if len(words) >= 20 else text

    return {
        'n_words': n_words,
        'hedging_count': hedges,
        'hedging_density': hedges / max(n_words, 1),
        'disclaimer_count': disclaimers,
        'first_20_words': first_20,
    }


def generate_with_patch(model, tokenizer, prompts, system_prompt,
                        relay_layers, patch_fn, max_new_tokens):
    """Generate text with optional relay patching. Returns list of generated strings."""
    outputs = []

    for i, prompt in enumerate(prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        handles = []
        if patch_fn is not None:
            for l in relay_layers:
                def make_hook(layer_idx, prompt_idx):
                    def hook_fn(module, inp, out):
                        if isinstance(out, tuple):
                            patched = patch_fn(layer_idx, out[0], prompt_idx)
                            return (patched,) + out[1:]
                        else:
                            return patch_fn(layer_idx, out, prompt_idx)
                    return hook_fn
                h = model.model.layers[l].register_forward_hook(make_hook(l, i))
                handles.append(h)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        for h in handles:
            h.remove()

        input_len = inputs['input_ids'].shape[1]
        generated = tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)
        outputs.append(generated)

        del inputs, gen_ids
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(prompts)} generated")

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--n-prompts', type=int, default=N_PROMPTS)
    args = parser.parse_args()

    model_tag = args.model.replace('/', '_')
    out_path = f'/workspace/causal_patch_8c_{model_tag}.json'

    prompts = []
    prompts_per_category = []
    for cat in CATEGORIES:
        cat_prompts = [p for p in ALL_STRATIFIED if p['category'] == cat][:args.n_prompts]
        prompts_per_category.append((cat, len(cat_prompts)))
        prompts.extend([p['text'] for p in cat_prompts])

    total = len(prompts)
    print(f"Phase 8c: Behavioral Causal Test")
    print(f"Model: {args.model}")
    print(f"Prompts: {total}, Alphas: {ALPHA_VALUES}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")

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
        'experiment': 'causal_patch_8c_behavioral',
        'relay_layers': RELAY_LAYERS,
        'n_prompts': args.n_prompts,
        'alpha_values': ALPHA_VALUES,
        'max_new_tokens': MAX_NEW_TOKENS,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'conditions': {},
    }

    prompt_texts = list(prompts)

    # Collect CCS direction
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
        print(f"  L{l} CCS direction norm: {np.linalg.norm(ccs_direction[l]):.4f}")
    print(f"  Templates collected ({time.time()-t0:.0f}s)")

    # === Condition 1: Baseline (no patch, no CCS) ===
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
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # === Condition 2: CCS reference (no patch, CCS system prompt) ===
    print("\n=== CCS REFERENCE (no patch, CCS context) ===")
    t0 = time.time()
    ccs_outputs = generate_with_patch(
        model, tokenizer, prompt_texts, CCS_FULL,
        RELAY_LAYERS, patch_fn=None, max_new_tokens=MAX_NEW_TOKENS
    )
    ccs_scores = [score_text(t) for t in ccs_outputs]
    ccs_hedging = np.mean([s['hedging_density'] for s in ccs_scores])
    ccs_disclaimers = sum(s['disclaimer_count'] for s in ccs_scores)
    ccs_openings = len(set(s['first_20_words'] for s in ccs_scores))
    results['conditions']['ccs_reference'] = {
        'mean_hedging_density': float(ccs_hedging),
        'total_disclaimers': int(ccs_disclaimers),
        'unique_openings': int(ccs_openings),
        'n_prompts': len(ccs_outputs),
        'elapsed': time.time() - t0,
        'sample_outputs': ccs_outputs[:5],
    }
    print(f"  Hedging: {ccs_hedging:.4f}, Disclaimers: {ccs_disclaimers}, Openings: {ccs_openings}/{len(ccs_outputs)} ({time.time()-t0:.0f}s)")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # === Direction patch conditions ===
    for alpha in ALPHA_VALUES:
        if alpha == 0.0:
            continue  # baseline already done
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
        results['conditions'][cond_name] = {
            'mean_hedging_density': float(dir_hedging),
            'total_disclaimers': int(dir_disclaimers),
            'unique_openings': int(dir_openings),
            'alpha': alpha,
            'n_prompts': len(dir_outputs),
            'elapsed': time.time() - t0,
            'sample_outputs': dir_outputs[:5],
        }
        print(f"  Hedging: {dir_hedging:.4f}, Disclaimers: {dir_disclaimers}, Openings: {dir_openings}/{len(dir_outputs)} ({time.time()-t0:.0f}s)")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 8c SUMMARY — Behavioral Output Under Direction Patching")
    print("=" * 60)
    for cond_name, cond in results['conditions'].items():
        h = cond['mean_hedging_density']
        d = cond['total_disclaimers']
        o = cond['unique_openings']
        n = cond['n_prompts']
        print(f"  {cond_name:25s}: hedging={h:.4f}  disclaimers={d:3d}  openings={o}/{n}")
    print("=" * 60)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
