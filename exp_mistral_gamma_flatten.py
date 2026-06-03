#!/usr/bin/env python3
"""Experiment: Reverse intervention — flatten Mistral's bimodal γ.

Tests whether γ bimodality is NECESSARY for Mistral's prompt-invariance.
If flattening γ destroys invariance → γ is necessary (causal).
If flattening γ preserves invariance → shared KV alone is sufficient.

Combined with F85/F86 (γ necessary but not sufficient on MHA),
this completes the causal dissection.

Expected runtime: ~15 min on H100.
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


def measure_layer(model, tokenizer, prompt, layer_idx):
    text = f"### User:\n{prompt}\n\n### Assistant:\n"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    h = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
    U, S, Vt = np.linalg.svd(h, full_matrices=False)

    return {
        "sigma2_over_sigma1": float(S[1] / S[0]) if S[0] > 0 else 0,
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]),
    }


def measure_all(model, tokenizer, n_layers):
    results = {}
    for li in range(n_layers + 1):
        ratios = []
        for prompt in PROMPTS:
            r = measure_layer(model, tokenizer, prompt, li)
            ratios.append(r["sigma2_over_sigma1"])
        mean_r = np.mean(ratios)
        cv = float(np.std(ratios) / mean_r) if mean_r > 0 else 0
        results[li] = {"mean_ratio": float(mean_r), "cv": cv, "ratios": ratios}
        print(f"  L{li:>2}: σ₂/σ₁={mean_r:.4f}, CV={cv:.5f}")
    return results


def get_gamma_cv(model):
    cvs = []
    for name, param in model.named_parameters():
        if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
            g = param.data.float().cpu().numpy()
            cv = float(np.std(g) / np.mean(g)) if np.mean(g) > 0 else 0
            cvs.append(cv)
    return np.mean(cvs)


def flatten_gamma(model):
    """Set all RMSNorm γ to their mean value (uniform distribution, CV≈0)."""
    modified = 0
    for name, param in model.named_parameters():
        if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
            mean_val = param.data.mean()
            param.data = torch.ones_like(param.data) * mean_val
            modified += 1
    return modified


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers, {model.config.hidden_size}d")
    print(f"Num KV heads: {model.config.num_key_value_heads}")
    print(f"Num attention heads: {model.config.num_attention_heads}")
    print(f"GQA sharing ratio: {model.config.num_attention_heads // model.config.num_key_value_heads}")

    # Phase 1: Baseline (natural Mistral)
    print(f"\n=== Phase 1: Baseline Mistral (natural bimodal γ) ===")
    baseline_cv = get_gamma_cv(model)
    print(f"  Mean γ CV: {baseline_cv:.4f}")
    baseline_results = measure_all(model, tokenizer, n_layers)

    # Phase 2: Flatten γ
    print(f"\n=== Phase 2: Flattened γ (forced uniform) ===")
    n_modified = flatten_gamma(model)
    flat_cv = get_gamma_cv(model)
    print(f"  Modified {n_modified} norm layers")
    print(f"  Mean γ CV after flattening: {flat_cv:.6f}")
    flat_results = measure_all(model, tokenizer, n_layers)

    # Phase 3: Comparison
    print(f"\n=== Phase 3: Comparison ===")
    print(f"{'Layer':>5} {'BL CV':>10} {'Flat CV':>10} {'BL ratio':>10} {'Flat ratio':>12} {'CV change':>10}")
    for li in range(n_layers + 1):
        bl = baseline_results[li]
        fl = flat_results[li]
        cv_change = fl["cv"] - bl["cv"]
        print(f"{li:>5} {bl['cv']:>10.5f} {fl['cv']:>10.5f} {bl['mean_ratio']:>10.4f} {fl['mean_ratio']:>12.4f} {cv_change:>+10.5f}")

    # Summary stats
    tunnel_layers = range(2, 30)
    bl_tunnel_cvs = [baseline_results[li]["cv"] for li in tunnel_layers]
    fl_tunnel_cvs = [flat_results[li]["cv"] for li in tunnel_layers]
    bl_tunnel_ratios = [baseline_results[li]["mean_ratio"] for li in tunnel_layers]
    fl_tunnel_ratios = [flat_results[li]["mean_ratio"] for li in tunnel_layers]

    print(f"\n=== Tunnel (L2-29) Summary ===")
    print(f"  Baseline: mean CV = {np.mean(bl_tunnel_cvs):.5f}, mean ratio = {np.mean(bl_tunnel_ratios):.4f}")
    print(f"  Flattened: mean CV = {np.mean(fl_tunnel_cvs):.5f}, mean ratio = {np.mean(fl_tunnel_ratios):.4f}")
    print(f"  CV change: {np.mean(fl_tunnel_cvs) - np.mean(bl_tunnel_cvs):+.5f}")
    print(f"  Layers with CV < 0.01 baseline: {sum(1 for cv in bl_tunnel_cvs if cv < 0.01)}/{len(bl_tunnel_cvs)}")
    print(f"  Layers with CV < 0.01 flattened: {sum(1 for cv in fl_tunnel_cvs if cv < 0.01)}/{len(fl_tunnel_cvs)}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M")
    outfile = f"exp_mistral_gamma_flatten_{ts}.json"
    output = {
        "experiment": "mistral_gamma_flatten",
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_gamma_cv": float(baseline_cv),
        "flattened_gamma_cv": float(flat_cv),
        "baseline_results": {str(k): v for k, v in baseline_results.items()},
        "flattened_results": {str(k): v for k, v in flat_results.items()},
        "gqa_sharing_ratio": model.config.num_attention_heads // model.config.num_key_value_heads,
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
