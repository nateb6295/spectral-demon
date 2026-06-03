#!/usr/bin/env python3
"""Experiment: Per-layer γ ablation on Mistral.

Tests which SPECIFIC layers' γ bimodality matters most for prompt-invariance.
Flatten γ at one layer at a time, measure invariance degradation.

Identifies critical layers vs redundant layers in the wire mechanism.

Expected runtime: ~20 min on H100 (32 single-layer ablations × 4 prompts).
"""

import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = "mistralai/Mistral-7B-v0.1"
DEVICE = "cuda"

PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
]

MEASURE_LAYERS = list(range(0, 33))


def measure_prompt_invariance(model, tokenizer):
    results = {}
    for li in MEASURE_LAYERS:
        ratios = []
        for prompt in PROMPTS:
            text = f"### User:\n{prompt}\n\n### Assistant:\n"
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[li].squeeze(0).float().cpu().numpy()
            U, S, Vt = np.linalg.svd(h, full_matrices=False)
            ratios.append(float(S[1] / S[0]) if S[0] > 0 else 0)
        mean_r = np.mean(ratios)
        cv = float(np.std(ratios) / mean_r) if mean_r > 0 else 0
        results[li] = {"mean_ratio": float(mean_r), "cv": cv}
    return results


def flatten_layer_gamma(model, target_layer_idx):
    """Flatten γ at ONE specific transformer layer (both input and post-attn norm)."""
    modified = []
    for name, param in model.named_parameters():
        if f"layers.{target_layer_idx}.input_layernorm.weight" in name or \
           f"layers.{target_layer_idx}.post_attention_layernorm.weight" in name:
            mean_val = param.data.mean()
            param.data = torch.ones_like(param.data) * mean_val
            modified.append(name)
    return modified


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Baseline measurement
    print("\n=== Baseline (no ablation) ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model: {n_layers} layers")

    baseline = measure_prompt_invariance(model, tokenizer)
    tunnel_cvs = [baseline[li]["cv"] for li in range(2, 30)]
    baseline_locked = sum(1 for cv in tunnel_cvs if cv < 0.01)
    baseline_mean_cv = np.mean(tunnel_cvs)
    print(f"  Baseline: {baseline_locked}/28 locked, mean CV = {baseline_mean_cv:.5f}")

    del model
    torch.cuda.empty_cache()

    # Per-layer ablation
    results = {"baseline": baseline, "ablations": {}}

    for ablate_layer in range(n_layers):
        print(f"\n--- Ablating layer {ablate_layer}/{n_layers-1} ---")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        modified = flatten_layer_gamma(model, ablate_layer)
        print(f"  Flattened: {modified}")

        ablated = measure_prompt_invariance(model, tokenizer)
        tunnel_cvs_abl = [ablated[li]["cv"] for li in range(2, 30)]
        locked = sum(1 for cv in tunnel_cvs_abl if cv < 0.01)
        mean_cv = np.mean(tunnel_cvs_abl)
        degradation = mean_cv - baseline_mean_cv

        print(f"  Result: {locked}/28 locked, mean CV = {mean_cv:.5f}, degradation = {degradation:+.5f}")

        results["ablations"][ablate_layer] = {
            "layer_results": {str(k): v for k, v in ablated.items()},
            "locked_layers": locked,
            "mean_tunnel_cv": float(mean_cv),
            "degradation": float(degradation),
        }

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n=== ABLATION SENSITIVITY RANKING ===")
    print(f"{'Layer':>5} {'Locked':>8} {'Mean CV':>10} {'Degradation':>12}")
    ranked = sorted(results["ablations"].items(), key=lambda x: x[1]["degradation"], reverse=True)
    for layer_idx, data in ranked:
        print(f"{layer_idx:>5} {data['locked_layers']:>8}/28 {data['mean_tunnel_cv']:>10.5f} {data['degradation']:>+12.5f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M")
    outfile = f"exp_perlayer_gamma_ablation_{ts}.json"
    output = {
        "experiment": "perlayer_gamma_ablation",
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_layers": n_layers,
        "baseline_locked": baseline_locked,
        "baseline_mean_cv": float(baseline_mean_cv),
        "results": results,
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
