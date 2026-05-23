#!/usr/bin/env python3
"""Phase 8a: Causal Relay Patching — Geometric Propagation Test.

Tests whether relay-zone geometry CAUSES expression-layer geometry.
Patches L14-17 activations during baseline inference with CCS-derived
directions and measures whether L25 PR shifts toward CCS-like values.

No text generation — pure geometric causality test.

Conditions:
  baseline        — no patching, baseline context
  ccs_reference   — no patching, CCS context (target PR values)
  full_patch      — replace baseline relay activations with CCS activations (matched)
  direction_patch — add mean CCS direction (scaled by alpha) to baseline
  shuffled_patch  — replace with CCS activations from DIFFERENT prompts
  random_patch    — add random noise matched in norm to CCS direction
  anti_patch      — subtract CCS direction from CCS activations
"""

import argparse
import json
import sys
import gc
import time
import os

import torch
import numpy as np

sys.path.insert(0, '/workspace')
from stratified_prompts import ALL_STRATIFIED, CATEGORIES
from cna_scaling_experiment import (
    participation_ratio, spectral_summary, get_layer_config, CCS_FULL, CCS_MINIMAL
)

TOP_K = 10
RELAY_LAYERS = [14, 15, 16, 17]
EXPRESSION_LAYER = 25
ALPHA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5]


def collect_relay_activations(model, tokenizer, prompts, system_prompt, relay_layers):
    """Collect per-prompt activations at all relay layers. Returns dict[layer] -> (N, D)."""
    layer_acts = {l: [] for l in relay_layers}

    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    captured[layer_idx] = out[0].detach().cpu()
                else:
                    captured[layer_idx] = out.detach().cpu()
            return hook_fn

        handles = []
        for l in relay_layers:
            h = model.model.layers[l].register_forward_hook(make_hook(l))
            handles.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        for l in relay_layers:
            act = captured[l].squeeze(0).float().numpy()
            layer_acts[l].append(act.mean(axis=0))

        del inputs, captured
        torch.cuda.empty_cache()

    return {l: np.array(v) for l, v in layer_acts.items()}


def collect_expression_with_patch(model, tokenizer, prompts, system_prompt,
                                  relay_layers, expression_layer,
                                  patch_fn=None):
    """Collect expression-layer activations, optionally patching relay layers.

    patch_fn: if provided, called as patch_fn(layer_idx, original_output, prompt_idx)
              and should return modified output tensor.
    """
    activations = []

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

        hidden = [None]
        handles = []

        # Patch hooks on relay layers
        if patch_fn is not None:
            for l in relay_layers:
                def make_patch_hook(layer_idx, prompt_idx):
                    def hook_fn(module, inp, out):
                        if isinstance(out, tuple):
                            patched = patch_fn(layer_idx, out[0], prompt_idx)
                            return (patched,) + out[1:]
                        else:
                            return patch_fn(layer_idx, out, prompt_idx)
                    return hook_fn
                h = model.model.layers[l].register_forward_hook(
                    make_patch_hook(l, i)
                )
                handles.append(h)

        # Capture hook on expression layer
        def expr_hook(module, inp, out):
            if isinstance(out, tuple):
                hidden[0] = out[0].detach().cpu()
            else:
                hidden[0] = out.detach().cpu()

        h = model.model.layers[expression_layer].register_forward_hook(expr_hook)
        handles.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        act = hidden[0].squeeze(0).float().numpy()
        activations.append(act.mean(axis=0))

        del inputs, hidden[0]
        torch.cuda.empty_cache()

    return np.array(activations)


def compute_pr_by_category(activations, prompts_per_category):
    """Compute PR for each category from stratified activations."""
    results = {}
    offset = 0
    for cat_name, n_prompts in prompts_per_category:
        cat_acts = activations[offset:offset + n_prompts]
        if len(cat_acts) >= 2:
            results[cat_name] = {
                'participation_ratio': participation_ratio(cat_acts),
                **spectral_summary(cat_acts),
            }
        offset += n_prompts
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--n-prompts', type=int, default=30)
    args = parser.parse_args()

    model_tag = args.model.replace('/', '_')
    out_path = f'/workspace/causal_patch_{model_tag}.json'

    # Build prompt set
    prompts = []
    prompts_per_category = []
    for cat in CATEGORIES:
        cat_prompts = [p for p in ALL_STRATIFIED if p['category'] == cat][:args.n_prompts]
        prompts_per_category.append((cat, len(cat_prompts)))
        prompts.extend([p['text'] for p in cat_prompts])

    total = len(prompts)
    print(f"Phase 8a: Causal Relay Patching")
    print(f"Model: {args.model}")
    print(f"Prompts: {total} across {len(CATEGORIES)} categories")

    # Load model
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
        'experiment': 'causal_relay_patch',
        'relay_layers': RELAY_LAYERS,
        'expression_layer': EXPRESSION_LAYER,
        'n_prompts': args.n_prompts,
        'alpha_values': ALPHA_VALUES,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'conditions': {},
    }

    prompt_texts = [p for p in prompts]

    # === Step 1: Collect relay templates ===
    print("\n=== STEP 1: Collecting relay activation templates ===")

    t0 = time.time()
    print("  Baseline relay activations...")
    baseline_relay = collect_relay_activations(
        model, tokenizer, prompt_texts, None, RELAY_LAYERS
    )
    print(f"    Done ({time.time()-t0:.0f}s)")

    t0 = time.time()
    print("  CCS relay activations...")
    ccs_relay = collect_relay_activations(
        model, tokenizer, prompt_texts, CCS_FULL, RELAY_LAYERS
    )
    print(f"    Done ({time.time()-t0:.0f}s)")

    # Compute CCS direction per layer
    ccs_direction = {}
    for l in RELAY_LAYERS:
        ccs_direction[l] = (ccs_relay[l] - baseline_relay[l]).mean(axis=0)
        norm = np.linalg.norm(ccs_direction[l])
        print(f"  L{l} CCS direction norm: {norm:.4f}")

    # === Step 2: Collect expression-layer PR under each condition ===
    print("\n=== STEP 2: Measuring expression-layer geometry ===")

    # Condition 1: Baseline (no patch)
    print("\n  --- BASELINE ---")
    t0 = time.time()
    base_expr = collect_expression_with_patch(
        model, tokenizer, prompt_texts, None,
        RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=None
    )
    base_cats = compute_pr_by_category(base_expr, prompts_per_category)
    results['conditions']['baseline'] = {
        'categories': base_cats,
        'elapsed': time.time() - t0,
    }
    print(f"    rel_PR = {base_cats.get('relational',{}).get('participation_ratio',0):.3f} ({time.time()-t0:.0f}s)")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Condition 2: CCS reference (no patch, CCS context)
    print("\n  --- CCS REFERENCE ---")
    t0 = time.time()
    ccs_expr = collect_expression_with_patch(
        model, tokenizer, prompt_texts, CCS_FULL,
        RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=None
    )
    ccs_cats = compute_pr_by_category(ccs_expr, prompts_per_category)
    results['conditions']['ccs_reference'] = {
        'categories': ccs_cats,
        'elapsed': time.time() - t0,
    }
    print(f"    rel_PR = {ccs_cats.get('relational',{}).get('participation_ratio',0):.3f} ({time.time()-t0:.0f}s)")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Condition 3: Full patch (replace baseline relay with CCS relay, matched)
    print("\n  --- FULL PATCH ---")
    t0 = time.time()

    def full_patch_fn(layer_idx, output, prompt_idx):
        patch = torch.tensor(
            ccs_relay[layer_idx][prompt_idx],
            dtype=output.dtype, device=output.device
        )
        # Replace the mean activation across token positions
        # Scale patch to match output shape: (batch, seq_len, hidden)
        output[:, :, :] = patch.unsqueeze(0).unsqueeze(0).expand_as(output)
        return output

    full_expr = collect_expression_with_patch(
        model, tokenizer, prompt_texts, None,
        RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=full_patch_fn
    )
    full_cats = compute_pr_by_category(full_expr, prompts_per_category)
    results['conditions']['full_patch'] = {
        'categories': full_cats,
        'elapsed': time.time() - t0,
    }
    print(f"    rel_PR = {full_cats.get('relational',{}).get('participation_ratio',0):.3f} ({time.time()-t0:.0f}s)")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Condition 4: Direction patch (add mean CCS direction, multiple alphas)
    for alpha in ALPHA_VALUES:
        cond_name = f'direction_alpha_{alpha}'
        print(f"\n  --- DIRECTION PATCH (α={alpha}) ---")
        t0 = time.time()

        def make_direction_fn(a):
            def direction_patch_fn(layer_idx, output, prompt_idx):
                direction = torch.tensor(
                    ccs_direction[layer_idx] * a,
                    dtype=output.dtype, device=output.device
                )
                output = output + direction.unsqueeze(0).unsqueeze(0)
                return output
            return direction_patch_fn

        dir_expr = collect_expression_with_patch(
            model, tokenizer, prompt_texts, None,
            RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=make_direction_fn(alpha)
        )
        dir_cats = compute_pr_by_category(dir_expr, prompts_per_category)
        results['conditions'][cond_name] = {
            'categories': dir_cats,
            'alpha': alpha,
            'elapsed': time.time() - t0,
        }
        print(f"    rel_PR = {dir_cats.get('relational',{}).get('participation_ratio',0):.3f} ({time.time()-t0:.0f}s)")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Condition 5: Shuffled patch (CCS activations from different prompts)
    print("\n  --- SHUFFLED PATCH ---")
    t0 = time.time()
    shuffle_idx = np.random.permutation(len(prompt_texts))

    def shuffled_patch_fn(layer_idx, output, prompt_idx):
        shuffled_i = shuffle_idx[prompt_idx]
        patch = torch.tensor(
            ccs_relay[layer_idx][shuffled_i],
            dtype=output.dtype, device=output.device
        )
        output[:, :, :] = patch.unsqueeze(0).unsqueeze(0).expand_as(output)
        return output

    shuf_expr = collect_expression_with_patch(
        model, tokenizer, prompt_texts, None,
        RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=shuffled_patch_fn
    )
    shuf_cats = compute_pr_by_category(shuf_expr, prompts_per_category)
    results['conditions']['shuffled_patch'] = {
        'categories': shuf_cats,
        'elapsed': time.time() - t0,
    }
    print(f"    rel_PR = {shuf_cats.get('relational',{}).get('participation_ratio',0):.3f} ({time.time()-t0:.0f}s)")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Condition 6: Random patch (noise matched in norm)
    print("\n  --- RANDOM PATCH ---")
    t0 = time.time()

    def random_patch_fn(layer_idx, output, prompt_idx):
        noise = np.random.randn(*ccs_direction[layer_idx].shape)
        noise = noise / np.linalg.norm(noise) * np.linalg.norm(ccs_direction[layer_idx])
        noise_t = torch.tensor(noise, dtype=output.dtype, device=output.device)
        output = output + noise_t.unsqueeze(0).unsqueeze(0)
        return output

    rand_expr = collect_expression_with_patch(
        model, tokenizer, prompt_texts, None,
        RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=random_patch_fn
    )
    rand_cats = compute_pr_by_category(rand_expr, prompts_per_category)
    results['conditions']['random_patch'] = {
        'categories': rand_cats,
        'elapsed': time.time() - t0,
    }
    print(f"    rel_PR = {rand_cats.get('relational',{}).get('participation_ratio',0):.3f} ({time.time()-t0:.0f}s)")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Condition 7: Anti-patch (subtract CCS direction from CCS context)
    print("\n  --- ANTI-PATCH ---")
    t0 = time.time()

    def anti_patch_fn(layer_idx, output, prompt_idx):
        direction = torch.tensor(
            ccs_direction[layer_idx] * -1.0,
            dtype=output.dtype, device=output.device
        )
        output = output + direction.unsqueeze(0).unsqueeze(0)
        return output

    anti_expr = collect_expression_with_patch(
        model, tokenizer, prompt_texts, CCS_FULL,
        RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=anti_patch_fn
    )
    anti_cats = compute_pr_by_category(anti_expr, prompts_per_category)
    results['conditions']['anti_patch'] = {
        'categories': anti_cats,
        'elapsed': time.time() - t0,
    }
    print(f"    rel_PR = {anti_cats.get('relational',{}).get('participation_ratio',0):.3f} ({time.time()-t0:.0f}s)")

    # Final save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("PHASE 8a SUMMARY — Expression Layer (L25) Relational PR")
    print("="*60)
    for cond_name, cond_data in results['conditions'].items():
        rel_pr = cond_data['categories'].get('relational', {}).get('participation_ratio', 0)
        print(f"  {cond_name:25s}: {rel_pr:.3f}")
    print("="*60)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
