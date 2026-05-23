#!/usr/bin/env python3
"""Phase 8b: Fixed Random Direction Controls.

Phase 8a showed bell-shaped dose-response for CCS direction but the random
control was confounded (different noise per prompt inflates PR through variance).

This experiment fixes the confound: generate N FIXED random directions (each
matched in norm to CCS direction) and run the same dose-response curve.
If CCS direction's bell curve is specific to its geometric structure,
random fixed directions should produce a different dose-response shape.

Also adds: orthogonalized CCS direction (component orthogonal to principal
axes of baseline activations) to test whether the effect is in the CCS-specific
subspace or general activation space.
"""

import argparse
import json
import sys
import time
import os

import torch
import numpy as np

sys.path.insert(0, '/workspace')
from stratified_prompts import ALL_STRATIFIED, CATEGORIES
from cna_scaling_experiment import (
    participation_ratio, spectral_summary, get_layer_config, CCS_FULL, CCS_MINIMAL
)
from causal_patch_experiment import (
    collect_relay_activations, collect_expression_with_patch, compute_pr_by_category,
    RELAY_LAYERS, EXPRESSION_LAYER
)

N_RANDOM_DIRECTIONS = 5
ALPHA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--n-prompts', type=int, default=30)
    args = parser.parse_args()

    model_tag = args.model.replace('/', '_')
    out_path = f'/workspace/causal_patch_8b_{model_tag}.json'

    prompts = []
    prompts_per_category = []
    for cat in CATEGORIES:
        cat_prompts = [p for p in ALL_STRATIFIED if p['category'] == cat][:args.n_prompts]
        prompts_per_category.append((cat, len(cat_prompts)))
        prompts.extend([p['text'] for p in cat_prompts])

    total = len(prompts)
    print(f"Phase 8b: Fixed Random Direction Controls")
    print(f"Model: {args.model}")
    print(f"Prompts: {total}, Random directions: {N_RANDOM_DIRECTIONS}")
    print(f"Alpha values: {ALPHA_VALUES}")

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
        'experiment': 'causal_patch_8b_controls',
        'relay_layers': RELAY_LAYERS,
        'expression_layer': EXPRESSION_LAYER,
        'n_prompts': args.n_prompts,
        'n_random_directions': N_RANDOM_DIRECTIONS,
        'alpha_values': ALPHA_VALUES,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'conditions': {},
    }

    prompt_texts = list(prompts)

    # Collect CCS direction (same as Phase 8a)
    print("\n=== Collecting activation templates ===")
    t0 = time.time()
    baseline_relay = collect_relay_activations(
        model, tokenizer, prompt_texts, None, RELAY_LAYERS
    )
    ccs_relay = collect_relay_activations(
        model, tokenizer, prompt_texts, CCS_FULL, RELAY_LAYERS
    )
    print(f"  Templates collected ({time.time()-t0:.0f}s)")

    ccs_direction = {}
    ccs_norm = {}
    for l in RELAY_LAYERS:
        ccs_direction[l] = (ccs_relay[l] - baseline_relay[l]).mean(axis=0)
        ccs_norm[l] = np.linalg.norm(ccs_direction[l])
        print(f"  L{l} CCS direction norm: {ccs_norm[l]:.4f}")

    # Generate fixed random directions (matched in norm)
    np.random.seed(42)
    random_directions = []
    for i in range(N_RANDOM_DIRECTIONS):
        rd = {}
        for l in RELAY_LAYERS:
            d = np.random.randn(*ccs_direction[l].shape).astype(np.float32)
            d = d / np.linalg.norm(d) * ccs_norm[l]
            rd[l] = d
        random_directions.append(rd)
        print(f"  Random direction {i}: norm = {np.linalg.norm(rd[RELAY_LAYERS[0]]):.4f}")

    # Generate orthogonalized CCS direction
    print("\n  Computing orthogonalized CCS direction...")
    orth_direction = {}
    for l in RELAY_LAYERS:
        baseline_acts = baseline_relay[l]  # (N, D)
        cov = np.cov(baseline_acts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top_k = min(10, len(eigvals))
        top_vecs = eigvecs[:, -top_k:]  # top-k eigenvectors
        proj = top_vecs @ (top_vecs.T @ ccs_direction[l])
        orth = ccs_direction[l] - proj
        orth = orth / np.linalg.norm(orth) * ccs_norm[l]
        orth_direction[l] = orth.astype(np.float32)
        cos_sim = np.dot(ccs_direction[l], orth_direction[l]) / (
            np.linalg.norm(ccs_direction[l]) * np.linalg.norm(orth_direction[l])
        )
        print(f"    L{l}: cos(CCS, orth) = {cos_sim:.4f}")

    # === Run dose-response for CCS direction (re-run for consistency) ===
    print("\n=== CCS Direction Dose-Response ===")
    for alpha in ALPHA_VALUES:
        cond_name = f'ccs_alpha_{alpha}'
        print(f"\n  --- CCS α={alpha} ---")
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

        expr = collect_expression_with_patch(
            model, tokenizer, prompt_texts, None,
            RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=make_fn(ccs_direction, alpha)
        )
        cats = compute_pr_by_category(expr, prompts_per_category)
        results['conditions'][cond_name] = {
            'categories': cats, 'alpha': alpha, 'elapsed': time.time() - t0,
        }
        rel_pr = cats.get('relational', {}).get('participation_ratio', 0)
        print(f"    rel_PR = {rel_pr:.3f} ({time.time()-t0:.0f}s)")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # === Run dose-response for each fixed random direction ===
    for ri, rd in enumerate(random_directions):
        print(f"\n=== Random Direction {ri} Dose-Response ===")
        for alpha in ALPHA_VALUES:
            cond_name = f'random_{ri}_alpha_{alpha}'
            print(f"\n  --- Random-{ri} α={alpha} ---")
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

            expr = collect_expression_with_patch(
                model, tokenizer, prompt_texts, None,
                RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=make_fn(rd, alpha)
            )
            cats = compute_pr_by_category(expr, prompts_per_category)
            results['conditions'][cond_name] = {
                'categories': cats, 'alpha': alpha,
                'random_direction_idx': ri,
                'elapsed': time.time() - t0,
            }
            rel_pr = cats.get('relational', {}).get('participation_ratio', 0)
            print(f"    rel_PR = {rel_pr:.3f} ({time.time()-t0:.0f}s)")
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    # === Run dose-response for orthogonalized CCS direction ===
    print(f"\n=== Orthogonalized CCS Direction Dose-Response ===")
    for alpha in ALPHA_VALUES:
        cond_name = f'orth_alpha_{alpha}'
        print(f"\n  --- Orth α={alpha} ---")
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

        expr = collect_expression_with_patch(
            model, tokenizer, prompt_texts, None,
            RELAY_LAYERS, EXPRESSION_LAYER, patch_fn=make_fn(orth_direction, alpha)
        )
        cats = compute_pr_by_category(expr, prompts_per_category)
        results['conditions'][cond_name] = {
            'categories': cats, 'alpha': alpha, 'elapsed': time.time() - t0,
        }
        rel_pr = cats.get('relational', {}).get('participation_ratio', 0)
        print(f"    rel_PR = {rel_pr:.3f} ({time.time()-t0:.0f}s)")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 8b SUMMARY — Expression Layer (L25) Relational PR")
    print("=" * 70)

    print("\n  CCS Direction:")
    for alpha in ALPHA_VALUES:
        pr = results['conditions'][f'ccs_alpha_{alpha}']['categories'].get(
            'relational', {}
        ).get('participation_ratio', 0)
        print(f"    α={alpha:4.2f}: {pr:10.3f}")

    for ri in range(N_RANDOM_DIRECTIONS):
        print(f"\n  Random Direction {ri}:")
        for alpha in ALPHA_VALUES:
            pr = results['conditions'][f'random_{ri}_alpha_{alpha}']['categories'].get(
                'relational', {}
            ).get('participation_ratio', 0)
            print(f"    α={alpha:4.2f}: {pr:10.3f}")

    print(f"\n  Orthogonalized CCS:")
    for alpha in ALPHA_VALUES:
        pr = results['conditions'][f'orth_alpha_{alpha}']['categories'].get(
            'relational', {}
        ).get('participation_ratio', 0)
        print(f"    α={alpha:4.2f}: {pr:10.3f}")

    print("=" * 70)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
