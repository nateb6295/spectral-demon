#!/usr/bin/env python3
"""Experiment: Does instruction tuning modify γ bimodality?

Compares base vs instruct versions of the same model to test whether
IT installs, enhances, or leaves unchanged the γ bimodal distribution
that creates prompt-invariance.

Tests Mistral 7B v0.1 (base) vs Mistral 7B Instruct v0.1.

Expected runtime: ~10 min on H100.
"""

import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODELS = {
    "base": "mistralai/Mistral-7B-v0.1",
    "instruct": "mistralai/Mistral-7B-Instruct-v0.1",
}
DEVICE = "cuda"

PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
]


def get_gamma_stats(model):
    """Get per-layer γ statistics."""
    stats = []
    for name, param in model.named_parameters():
        if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
            g = param.detach().float().cpu().numpy()
            cv = float(np.std(g) / np.mean(g)) if np.mean(g) > 0 else 0
            stats.append({
                "name": name,
                "cv": cv,
                "mean": float(np.mean(g)),
                "std": float(np.std(g)),
                "min": float(np.min(g)),
                "max": float(np.max(g)),
                "max_min_ratio": float(np.max(g) / np.min(g)) if np.min(g) > 0 else float("inf"),
            })
    return stats


def measure_prompt_invariance(model, tokenizer, n_layers):
    results = {}
    for li in range(n_layers + 1):
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
        print(f"    L{li:>2}: σ₂/σ₁={mean_r:.4f}, CV={cv:.5f}")
    return results


def main():
    all_results = {}

    for label, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {label.upper()}: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
        n_layers = model.config.num_hidden_layers

        # γ stats
        print(f"\n  γ distribution:")
        gamma_stats = get_gamma_stats(model)
        cvs = [g["cv"] for g in gamma_stats]
        print(f"    Mean γ CV: {np.mean(cvs):.4f}")
        print(f"    Median γ CV: {np.median(cvs):.4f}")
        print(f"    γ CV range: [{np.min(cvs):.4f}, {np.max(cvs):.4f}]")

        # Prompt invariance
        print(f"\n  Prompt invariance:")
        pi_results = measure_prompt_invariance(model, tokenizer, n_layers)

        tunnel_cvs = [pi_results[li]["cv"] for li in range(2, 30)]
        locked = sum(1 for cv in tunnel_cvs if cv < 0.01)
        tunnel_ratios = [pi_results[li]["mean_ratio"] for li in range(2, 30)]

        print(f"\n  Summary:")
        print(f"    Locked layers (CV<0.01): {locked}/28")
        print(f"    Mean tunnel CV: {np.mean(tunnel_cvs):.5f}")
        print(f"    Mean tunnel σ₂/σ₁: {np.mean(tunnel_ratios):.4f}")

        all_results[label] = {
            "model": model_name,
            "gamma_stats": gamma_stats,
            "mean_gamma_cv": float(np.mean(cvs)),
            "pi_results": {str(k): v for k, v in pi_results.items()},
            "locked_layers": locked,
            "mean_tunnel_cv": float(np.mean(tunnel_cvs)),
            "mean_tunnel_ratio": float(np.mean(tunnel_ratios)),
        }

        del model
        torch.cuda.empty_cache()

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: BASE vs INSTRUCT")
    print(f"{'='*60}")

    b = all_results["base"]
    i = all_results["instruct"]

    print(f"\n  γ distribution:")
    print(f"    Base mean CV:     {b['mean_gamma_cv']:.4f}")
    print(f"    Instruct mean CV: {i['mean_gamma_cv']:.4f}")
    print(f"    IT changes γ CV by: {i['mean_gamma_cv'] - b['mean_gamma_cv']:+.4f}")

    # Per-layer γ CV comparison
    print(f"\n  Per-layer γ CV delta (instruct - base):")
    for bg, ig in zip(b["gamma_stats"], i["gamma_stats"]):
        delta = ig["cv"] - bg["cv"]
        if abs(delta) > 0.01:
            print(f"    {bg['name']}: {bg['cv']:.4f} -> {ig['cv']:.4f} (Δ={delta:+.4f})")

    print(f"\n  Prompt invariance:")
    print(f"    Base locked:     {b['locked_layers']}/28")
    print(f"    Instruct locked: {i['locked_layers']}/28")
    print(f"    Base tunnel CV:     {b['mean_tunnel_cv']:.5f}")
    print(f"    Instruct tunnel CV: {i['mean_tunnel_cv']:.5f}")
    print(f"    Base tunnel ratio:     {b['mean_tunnel_ratio']:.4f}")
    print(f"    Instruct tunnel ratio: {i['mean_tunnel_ratio']:.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M")
    outfile = f"exp_it_gamma_comparison_{ts}.json"
    output = {
        "experiment": "it_gamma_comparison",
        "models": MODELS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
