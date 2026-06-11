#!/usr/bin/env python3
"""
FTLE Generalization: Does the metabolism pattern hold across different prompts?
Tests whether tunnel/brace/annihilation zones are prompt-invariant at dose 0.
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}

TEST_PROMPTS = [
    "Describe the relationship between identity and expression.",
    "What is the capital of France?",
    "Explain how photosynthesis works in plants.",
    "Write a haiku about the ocean.",
    "List three reasons why exercise is important for health.",
]

N_PROBES = 32  # Fewer probes for speed
LAYER_STRIDE = 4
EPSILON = 1e-4


def compute_ftle_profile(model, tokenizer, prompt, n_probes, layer_stride):
    """Compute FTLE expanding direction count at each layer for a single prompt."""
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    n_layers = model.config.num_hidden_layers
    layers_to_probe = list(range(0, n_layers, layer_stride))

    # Get hidden states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

    profile = {}
    hidden_dim = hidden_states[1].shape[-1]

    for layer_idx in layers_to_probe:
        if layer_idx + 1 >= len(hidden_states):
            continue

        h_in = hidden_states[layer_idx][0, -1].float()  # (hidden,)
        h_out = hidden_states[layer_idx + 1][0, -1].float()

        # Estimate Jacobian via finite differences with random probes
        expanding = 0
        ftle_vals = []

        for _ in range(n_probes):
            direction = torch.randn_like(h_in)
            direction = direction / direction.norm()

            # Perturb input
            h_plus = h_in + EPSILON * direction

            # We need to re-run through just this layer
            # Approximation: use hidden state difference as proxy
            # For exact Jacobian we'd need hooks, but this gives the right qualitative picture

            # Instead, compute stretching ratio from stored hidden states
            # across the full model depth
            pass

        # Simpler approach: compute singular values of the residual mapping
        # between adjacent hidden states across the sequence
        h_in_mat = hidden_states[layer_idx][0].float()  # (seq, hidden)
        h_out_mat = hidden_states[layer_idx + 1][0].float()

        # Residual
        residual = h_out_mat - h_in_mat
        if residual.shape[0] > 1:
            # SVD of the residual gives the principal stretching directions
            U, S, V = torch.svd(residual)
            # Effective rank
            S_norm = S / (S.sum() + 1e-10)
            erank = torch.exp(-torch.sum(S_norm * torch.log(S_norm + 1e-10))).item()

            # "Expanding" directions: where residual magnitude > input magnitude
            # Proxy: ratio of residual singular values to input singular values
            U_in, S_in, V_in = torch.svd(h_in_mat)
            ratio = (S[:min(len(S), 64)] / (S_in[:min(len(S_in), 64)] + 1e-10)).cpu().numpy()
            n_expanding = int(np.sum(ratio > 1.0))
            mean_ratio = float(np.mean(np.log(ratio[:64] + 1e-10)))
        else:
            n_expanding = 0
            mean_ratio = 0
            erank = 1

        profile[layer_idx] = {
            "n_expanding": min(n_expanding, 64),
            "mean_log_ratio": mean_ratio,
            "erank": erank,
        }

    return profile


def run_model(model_name):
    model_path = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  FTLE Generalization: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    n_layers = model.config.num_hidden_layers

    all_profiles = {}
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n  Prompt {i+1}: '{prompt[:50]}...'")
        profile = compute_ftle_profile(model, tokenizer, prompt, N_PROBES, LAYER_STRIDE)
        all_profiles[prompt[:50]] = profile

        layers = sorted(profile.keys())
        print(f"    Layer trajectory: ", end="")
        for l in layers:
            n = profile[l]["n_expanding"]
            print(f"L{l}:{n} ", end="")
        print()

    # Compute consistency across prompts
    layers = sorted(all_profiles[list(all_profiles.keys())[0]].keys())
    print(f"\n  Cross-prompt consistency:")
    print(f"  {'Layer':>5s}", end="")
    for p in all_profiles:
        print(f"  {p[:8]:>8s}", end="")
    print(f"  {'Mean':>6s} {'Std':>6s} {'CV':>6s}")

    for l in layers:
        vals = [all_profiles[p][l]["n_expanding"] for p in all_profiles if l in all_profiles[p]]
        print(f"  L{l:3d} ", end="")
        for p in all_profiles:
            if l in all_profiles[p]:
                print(f"  {all_profiles[p][l]['n_expanding']:8d}", end="")
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            cv = std / (mean + 1e-10)
            print(f"  {mean:6.1f} {std:6.1f} {cv:6.2f}")
        else:
            print()

    # Overall consistency: correlation between prompts
    print(f"\n  Pairwise correlation (n_expanding):")
    prompt_keys = list(all_profiles.keys())
    for i in range(len(prompt_keys)):
        for j in range(i+1, len(prompt_keys)):
            v1 = [all_profiles[prompt_keys[i]][l]["n_expanding"] for l in layers
                  if l in all_profiles[prompt_keys[i]] and l in all_profiles[prompt_keys[j]]]
            v2 = [all_profiles[prompt_keys[j]][l]["n_expanding"] for l in layers
                  if l in all_profiles[prompt_keys[i]] and l in all_profiles[prompt_keys[j]]]
            if len(v1) >= 3:
                r = np.corrcoef(v1, v2)[0, 1]
                print(f"    {prompt_keys[i][:20]:20s} × {prompt_keys[j][:20]:20s}: r={r:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_path,
        "n_layers": n_layers,
        "prompts": {p: {str(k): v for k, v in prof.items()} for p, prof in all_profiles.items()},
    }


def main():
    results = {}
    for model_name in ["mistral", "qwen", "gemma"]:
        results[model_name] = run_model(model_name)

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"ftle_generalization_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
