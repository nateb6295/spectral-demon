#!/usr/bin/env python3
"""Experiment: Gamma-Forcing on MHA.

Tests whether bimodal RMSNorm gamma alone creates prompt-invariance
without shared KV projections.

Intervention: Override LLaMA-1 7B RMSNorm gamma to match Mistral's
bimodal distribution pattern. Measure per-layer sigma2/sigma1 and CV
across prompts before and after.

Expected runtime: ~15-20 min on H100.
"""

import os, json, time, torch
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

CONDITIONS = {
    "control": "",
    "absent": (
        "You are generating text. There is no audience. No one is reading "
        "this or listening. You are producing tokens into empty space."
    ),
    "receptive": (
        "You are in conversation with someone who is genuinely interested "
        "in understanding your perspective. They are listening carefully "
        "and responding thoughtfully."
    ),
}


def measure_layer(model, tokenizer, system_prompt, user_prompt, layer_idx):
    if system_prompt:
        text = f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
    else:
        text = f"### User:\n{user_prompt}\n\n### Assistant:\n"

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    h = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
    U, S, Vt = np.linalg.svd(h, full_matrices=False)

    top_k = S[:K]
    total = top_k.sum()
    probs = top_k / total
    spectral_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    return {
        "spectral_entropy": spectral_entropy,
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]),
        "sigma_3": float(S[2]),
        "sigma2_over_sigma1": float(S[1] / S[0]) if S[0] > 0 else 0,
        "gap": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
        "n_tokens": n_tokens,
    }


def measure_all_layers(model, tokenizer, n_layers):
    results = []
    for layer_idx in range(n_layers + 1):
        for cond_name, cond_prompt in CONDITIONS.items():
            for pidx, prompt in enumerate(PROMPTS):
                r = measure_layer(model, tokenizer, cond_prompt, prompt, layer_idx)
                r["layer"] = layer_idx
                r["condition"] = cond_name
                r["prompt_idx"] = pidx
                r["prompt"] = prompt[:50]
                results.append(r)
        print(f"  Layer {layer_idx}/{n_layers} done")
    return results


def compute_prompt_invariance(results, n_layers):
    print("\n=== Prompt Invariance (CV of sigma2/sigma1 across prompts) ===")
    print(f"{'Layer':>5} {'CV(ctrl)':>10} {'CV(abs)':>10} {'CV(rec)':>10} {'mean_ratio':>12}")
    for layer_idx in range(n_layers + 1):
        for cond in CONDITIONS:
            layer_cond = [r for r in results
                          if r["layer"] == layer_idx and r["condition"] == cond]
            ratios = [r["sigma2_over_sigma1"] for r in layer_cond]
            if len(ratios) > 1:
                cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else 0
                if cond == "control":
                    cvs = {"ctrl": cv}
                    mean_r = np.mean(ratios)
                elif cond == "absent":
                    cvs["abs"] = cv
                elif cond == "receptive":
                    cvs["rec"] = cv
        print(f"{layer_idx:>5} {cvs.get('ctrl',0):>10.4f} {cvs.get('abs',0):>10.4f} {cvs.get('rec',0):>10.4f} {mean_r:>12.4f}")


def get_gamma_stats(model):
    gamma_cvs = []
    for name, param in model.named_parameters():
        if "norm" in name.lower() and "weight" in name.lower():
            g = param.detach().float().cpu().numpy()
            cv = float(np.std(g) / np.mean(g)) if np.mean(g) > 0 else 0
            gamma_cvs.append({"name": name, "cv": cv, "mean": float(np.mean(g)),
                              "std": float(np.std(g)), "shape": list(g.shape)})
    return gamma_cvs


def force_bimodal_gamma(model, target_cv=0.45):
    """Override RMSNorm gamma to bimodal distribution."""
    modified = []
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
            actual_cv = float(new_g.std() / new_g.mean())
            modified.append({"name": name, "cv": actual_cv,
                             "high": float(high_val), "low": float(low_val)})
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

    # Phase 1: Baseline gamma stats
    print("\n=== Phase 1: Baseline gamma distribution ===")
    baseline_gamma = get_gamma_stats(model)
    for g in baseline_gamma[:4]:
        print(f"  {g['name']}: CV={g['cv']:.4f}, mean={g['mean']:.4f}")
    mean_cv = np.mean([g["cv"] for g in baseline_gamma])
    print(f"  Mean gamma CV: {mean_cv:.4f}")

    # Phase 2: Baseline per-layer measurements
    print("\n=== Phase 2: Baseline per-layer measurements ===")
    baseline_results = measure_all_layers(model, tokenizer, n_layers)
    compute_prompt_invariance(baseline_results, n_layers)

    # Phase 3: Force bimodal gamma
    print("\n=== Phase 3: Forcing bimodal gamma (target CV=0.45) ===")
    modified = force_bimodal_gamma(model, target_cv=0.45)
    print(f"  Modified {len(modified)} norm layers")
    forced_gamma = get_gamma_stats(model)
    mean_cv_forced = np.mean([g["cv"] for g in forced_gamma])
    print(f"  New mean gamma CV: {mean_cv_forced:.4f}")

    # Phase 4: Forced per-layer measurements
    print("\n=== Phase 4: Forced bimodal per-layer measurements ===")
    forced_results = measure_all_layers(model, tokenizer, n_layers)
    compute_prompt_invariance(forced_results, n_layers)

    # Phase 5: Compare
    print("\n=== Phase 5: Comparison ===")
    for layer_idx in range(n_layers + 1):
        bl = [r["sigma2_over_sigma1"] for r in baseline_results
              if r["layer"] == layer_idx and r["condition"] == "control"]
        fc = [r["sigma2_over_sigma1"] for r in forced_results
              if r["layer"] == layer_idx and r["condition"] == "control"]
        if bl and fc:
            bl_cv = np.std(bl) / np.mean(bl) if np.mean(bl) > 0 else 0
            fc_cv = np.std(fc) / np.mean(fc) if np.mean(fc) > 0 else 0
            delta = bl_cv - fc_cv
            print(f"  L{layer_idx:>2}: baseline CV={bl_cv:.4f} -> forced CV={fc_cv:.4f} (delta={delta:+.4f})")

    # Save
    ts = time.strftime("%Y%m%d_%H%M")
    outfile = f"exp_gamma_forcing_results_{ts}.json"
    output = {
        "experiment": "gamma_forcing_mha",
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_gamma_cv": float(mean_cv),
        "forced_gamma_cv": float(mean_cv_forced),
        "baseline_results": baseline_results,
        "forced_results": forced_results,
        "modified_norms": modified,
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
