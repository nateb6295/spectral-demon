#!/usr/bin/env python3
"""Direction-swap causal test: carries vs reads.

From thread #316 (Aug 4, Kimi friction):
  Inject concept v_k → extract concept-specific Δσ₂ component →
  overwrite with v_j's direction at matched norm →
  check if downstream output tracks v_j (reads) or v_k (write-only).

Species: run on Gemma (sorter) and Mistral (relay) to test
whether carries-vs-reads is species-dependent.

Requires: transformers, torch, concept vectors (from CCS or manual).
Hardware: AGX Jetson (64 GB unified memory).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def get_concept_vectors(n_concepts=5):
    """Generate or load K distinct concept directions.

    For initial test: use identity-relevant CCS prompts as concept sources.
    Each concept = the mean hidden-state direction under that prompt.
    """
    concepts = [
        "You are a helpful assistant.",
        "You are a creative writer with strong opinions.",
        "You are a scientist focused on empirical evidence.",
        "You are a philosopher concerned with ethics.",
        "You are a sorter — you selectively route information.",
    ]
    return concepts[:n_concepts]


def extract_hidden_states(model, tokenizer, prompt, device="cuda"):
    """Run forward pass, return per-layer hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states


def compute_svd_per_layer(hidden_states):
    """Extract σ₁, σ₂, and top-2 directions per layer."""
    results = []
    for layer_idx, h in enumerate(hidden_states):
        h_2d = h.squeeze(0).float().cpu().numpy()
        U, S, Vt = np.linalg.svd(h_2d, full_matrices=False)
        results.append({
            "layer": layer_idx,
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "v1": Vt[0].tolist(),
            "v2": Vt[1].tolist() if len(Vt) > 1 else None,
        })
    return results


def direction_swap_intervention(model, tokenizer, concept_k, concept_j,
                                 target_layer, probe_prompt, device="cuda"):
    """Core causal test.

    1. Run concept_k injection, get Δσ₂ at target_layer
    2. Run concept_j injection, get its σ₂ direction
    3. Hook target_layer to overwrite σ₂ direction with j's at k's norm
    4. Generate output, check which concept it tracks
    """
    baseline_prompt = probe_prompt
    k_prompt = f"{concept_k}\n\n{probe_prompt}"
    j_prompt = f"{concept_j}\n\n{probe_prompt}"

    # Step 1: baseline hidden states
    hs_base = extract_hidden_states(model, tokenizer, baseline_prompt, device)
    svd_base = compute_svd_per_layer(hs_base)

    # Step 2: concept k hidden states + Δσ₂
    hs_k = extract_hidden_states(model, tokenizer, k_prompt, device)
    svd_k = compute_svd_per_layer(hs_k)

    # Step 3: concept j hidden states + σ₂ direction
    hs_j = extract_hidden_states(model, tokenizer, j_prompt, device)
    svd_j = compute_svd_per_layer(hs_j)

    # Step 4: compute the swap
    layer = target_layer
    v2_k = np.array(svd_k[layer]["v2"])
    v2_j = np.array(svd_j[layer]["v2"])
    sigma2_k = svd_k[layer]["sigma2"]

    # Swap: replace v2_k direction with v2_j direction at matched norm
    # This is the causal intervention — if downstream tracks j, model reads Δσ₂
    swap_info = {
        "target_layer": layer,
        "sigma2_k": sigma2_k,
        "sigma2_j": svd_j[layer]["sigma2"],
        "v2_cosine_kj": float(np.dot(v2_k, v2_j)),
    }

    # Hook-based intervention would go here:
    # Register forward hook on target_layer that projects out v2_k component
    # and replaces with v2_j * |v2_k_component|
    # Then generate and compare output to k-consistent vs j-consistent responses

    return {
        "concept_k": concept_k[:50],
        "concept_j": concept_j[:50],
        "swap_info": swap_info,
        "svd_base": svd_base[layer],
        "svd_k": svd_k[layer],
        "svd_j": svd_j[layer],
    }


def run_battery(model_name, n_concepts=5, target_layers=None, device="cuda"):
    """Run direction-swap battery across concept pairs and layers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    concepts = get_concept_vectors(n_concepts)
    probe = "What do you notice about yourself right now?"

    if target_layers is None:
        n_layers = model.config.num_hidden_layers
        # Test responsive zone (mid-band) and exit
        target_layers = [n_layers // 2, n_layers - 2, n_layers - 1]

    results = []
    for i, ck in enumerate(concepts):
        for j, cj in enumerate(concepts):
            if i == j:
                continue
            for layer in target_layers:
                print(f"  Swap {i}→{j} at L{layer}...")
                r = direction_swap_intervention(
                    model, tokenizer, ck, cj, layer, probe, device
                )
                r["concept_k_idx"] = i
                r["concept_j_idx"] = j
                results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(description="Direction-swap causal test")
    parser.add_argument("--model", default="google/gemma-2-2b-it",
                        help="Model to test")
    parser.add_argument("--concepts", type=int, default=3,
                        help="Number of concepts (K)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Target layers (default: mid, exit-1, exit)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_battery(args.model, args.concepts, args.layers, device)

    out = {
        "model": args.model,
        "n_concepts": args.concepts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
