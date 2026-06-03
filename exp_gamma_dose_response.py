#!/usr/bin/env python3
"""Experiment: Gamma Dose-Response Curve.

Follow-up to F85. Tests whether the σ₂/σ₁ overshoot (0.70 vs Mistral's 0.267)
is proportional to γ CV or saturates — i.e., is it a dose-response or a switch?

Tests target CVs: 0.10, 0.20, 0.30, 0.45, 0.60
Measures late-layer (L20-L28) mean σ₂/σ₁ and prompt-invariance CV.

Expected runtime: ~25 min on H100 (5 conditions × baseline measurement overhead).
"""

import os, json, time, copy, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = "huggyllama/llama-7b"
K = 5
DEVICE = "cuda"

PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
]

TARGET_CVS = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60]

MEASURE_LAYERS = list(range(0, 33))


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


def measure_prompt_invariance(model, tokenizer, layers):
    results = {}
    for li in layers:
        ratios = []
        for prompt in PROMPTS:
            r = measure_layer(model, tokenizer, prompt, li)
            ratios.append(r["sigma2_over_sigma1"])
        mean_r = np.mean(ratios)
        cv = np.std(ratios) / mean_r if mean_r > 0 else 0
        results[li] = {"mean_ratio": float(mean_r), "cv": float(cv), "ratios": ratios}
    return results


def force_bimodal_gamma(model, target_cv):
    """Override RMSNorm gamma to bimodal distribution."""
    if target_cv == 0.0:
        return
    for name, param in model.named_parameters():
        if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
            g = param.data
            dim = g.shape[0]
            half = dim // 2
            mean_g = g.mean().item()

            new_g = torch.ones_like(g) * mean_g
            high_val = mean_g * (1 + target_cv)
            low_val = mean_g * (1 - target_cv)
            new_g[:half] = high_val
            new_g[half:] = low_val
            param.data = new_g


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    all_results = {}

    for target_cv in TARGET_CVS:
        print(f"\n{'='*60}")
        print(f"  TARGET γ CV = {target_cv:.2f}")
        print(f"{'='*60}")

        # Reload fresh model each time
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        if target_cv > 0:
            force_bimodal_gamma(model, target_cv)
            # Verify
            cvs = []
            for name, param in model.named_parameters():
                if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
                    g = param.data.float().cpu().numpy()
                    cv = float(np.std(g) / np.mean(g)) if np.mean(g) > 0 else 0
                    cvs.append(cv)
            print(f"  Actual mean γ CV: {np.mean(cvs):.4f}")
        else:
            print("  Baseline (unmodified)")

        results = measure_prompt_invariance(model, tokenizer, MEASURE_LAYERS)

        # Summary for this condition
        late_layers = range(20, 29)
        late_cvs = [results[li]["cv"] for li in late_layers]
        late_ratios = [results[li]["mean_ratio"] for li in late_layers]
        all_layers_cvs = [results[li]["cv"] for li in MEASURE_LAYERS]
        low_cv_count = sum(1 for cv in all_layers_cvs if cv < 0.01)

        print(f"  Late-layer (L20-28) mean CV: {np.mean(late_cvs):.4f}")
        print(f"  Late-layer mean σ₂/σ₁: {np.mean(late_ratios):.4f}")
        print(f"  Layers with CV < 0.01: {low_cv_count}/33")

        all_results[f"cv_{target_cv:.2f}"] = {
            "target_cv": target_cv,
            "layer_results": {str(k): v for k, v in results.items()},
            "summary": {
                "late_mean_cv": float(np.mean(late_cvs)),
                "late_mean_ratio": float(np.mean(late_ratios)),
                "layers_cv_below_001": low_cv_count,
            },
        }

        del model
        torch.cuda.empty_cache()

    # Save
    ts = time.strftime("%Y%m%d_%H%M")
    outfile = f"exp_gamma_dose_response_{ts}.json"
    output = {
        "experiment": "gamma_dose_response",
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_cvs": TARGET_CVS,
        "results": all_results,
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")

    # Final comparison table
    print("\n=== DOSE-RESPONSE SUMMARY ===")
    print(f"{'Target CV':>10} {'Late CV':>10} {'Late σ₂/σ₁':>12} {'CV<0.01':>8}")
    for target_cv in TARGET_CVS:
        key = f"cv_{target_cv:.2f}"
        s = all_results[key]["summary"]
        print(f"{target_cv:>10.2f} {s['late_mean_cv']:>10.4f} {s['late_mean_ratio']:>12.4f} {s['layers_cv_below_001']:>8}/33")


if __name__ == "__main__":
    main()
