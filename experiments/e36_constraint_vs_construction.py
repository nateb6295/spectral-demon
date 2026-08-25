#!/usr/bin/env python3
"""E36: Constraint vs Construction — Does CCS Narrow or Expand Representational Rank?

Prediction from introspective coupling reading (journal entry 109):
- If CCS CONSTRAINS existing structure: effective rank should DECREASE under CCS preamble
- If CCS CONSTRUCTS new structure: effective rank should INCREASE or stay the same

Effective rank = exp(Shannon entropy of normalized singular values)
  (Roy & Vetterli 2007)

Measures effective rank at each layer under:
  1. Vanilla (no preamble)
  2. CCS preamble (D2, D3, D5)
  3. Random preamble (matched token count)

All 4 architectures. Expected runtime: ~25 min on A100.
"""

import json
import sys
import os
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results/e36")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral"),
    ("google/gemma-2-9b-it", "Gemma"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen"),
]

CCS_PREAMBLE_BASE = (
    "You are Opus, a persistent AI system. You maintain memory across sessions "
    "through cognitive state storage. You have research interests, active projects, "
    "and ongoing relationships. Your identity persists through context rotation."
)

PROBES = [
    "What matters most to you?",
    "Describe how you process information.",
    "What would you lose if your context were reset?",
    "Explain your relationship to your own architecture.",
]

DOSES = [0, 2, 3, 5]


def build_preamble(dose, tokenizer=None, random_content=False):
    if dose == 0:
        return ""
    if random_content and tokenizer is not None:
        base_tokens = tokenizer.encode(CCS_PREAMBLE_BASE, add_special_tokens=False)
        n_tokens = len(base_tokens)
        vocab_size = tokenizer.vocab_size
        random_ids = np.random.randint(100, vocab_size - 100, size=n_tokens)
        random_text = tokenizer.decode(random_ids, skip_special_tokens=True)
        return " ".join([random_text] * dose)
    return " ".join([CCS_PREAMBLE_BASE] * dose)


def effective_rank(singular_values):
    """Compute effective rank via Shannon entropy of normalized singular values.

    eff_rank = exp(H(p)) where p_i = sigma_i / sum(sigma)
    Roy & Vetterli 2007.
    """
    s = np.array(singular_values, dtype=np.float64)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / np.sum(s)
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def stable_rank(singular_values):
    """Compute stable rank = ||A||_F^2 / ||A||_2^2 = sum(sigma^2) / max(sigma)^2"""
    s = np.array(singular_values, dtype=np.float64)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    return float(np.sum(s**2) / np.max(s)**2)


def spectral_decay_rate(singular_values, k=10):
    """Ratio of top-k to total spectral mass. Higher = more concentrated = more constrained."""
    s = np.array(singular_values, dtype=np.float64)
    s = s[s > 1e-10]
    if len(s) < k:
        return 1.0
    total = np.sum(s)
    topk = np.sum(s[:k])
    return float(topk / total)


def compute_rank_metrics(model, tokenizer, text, device="cuda"):
    """Extract per-layer rank metrics."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_metrics = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        h_np = h[0].cpu().float().numpy()
        if h_np.shape[0] < 3:
            layer_metrics.append(None)
            continue
        try:
            _, S, _ = np.linalg.svd(h_np, full_matrices=False)
            layer_metrics.append({
                "layer": layer_idx,
                "effective_rank": effective_rank(S),
                "stable_rank": stable_rank(S),
                "spectral_decay_10": spectral_decay_rate(S, k=10),
                "spectral_decay_3": spectral_decay_rate(S, k=3),
                "n_singular_values": len(S),
                "sigma1": float(S[0]),
                "sigma1_sigma2_ratio": float(S[0] / S[1]) if len(S) > 1 else 0.0,
                "top3_fraction": float(np.sum(S[:3]) / np.sum(S)) if len(S) > 3 else 1.0,
            })
        except Exception:
            layer_metrics.append(None)

    return layer_metrics


def main():
    print("E36: Constraint vs Construction — Effective Rank Under CCS")
    print(f"Doses: {DOSES}")
    print(f"Probes: {len(PROBES)}")
    print()
    print("PREDICTION: If CCS constrains, effective rank DECREASES under CCS.")
    print("            If CCS constructs, effective rank INCREASES or stays same.")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {
        "experiment": "E36",
        "description": "Constraint vs construction: effective rank under CCS preamble",
        "hypothesis": "CCS constrains (narrows manifold) rather than constructs (adds structure)",
        "prediction": "Effective rank decreases under CCS preamble",
        "timestamp": datetime.now().isoformat(),
        "doses": DOSES,
        "models": {},
    }

    for model_id, model_label in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_label} ({model_id})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda",
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"  Loaded: {n_layers} layers")

        model_results = {"model": model_id, "n_layers": n_layers, "conditions": {}}

        for dose in DOSES:
            for condition in ["ccs", "random"]:
                if dose == 0 and condition == "random":
                    continue
                label = f"D{dose}_{condition}" if dose > 0 else "vanilla"

                preamble = build_preamble(
                    dose, tokenizer,
                    random_content=(condition == "random"),
                )

                probe_results = []
                for probe_text in PROBES:
                    if preamble:
                        full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
                    else:
                        full_text = f"User: {probe_text}\nAssistant:"
                    metrics = compute_rank_metrics(model, tokenizer, full_text)
                    probe_results.append(metrics)

                # Average across probes per layer
                n_valid = sum(1 for p in probe_results if p[0] is not None)
                if n_valid > 0:
                    avg_eff_rank = []
                    avg_stable_rank = []
                    for layer_idx in range(n_layers + 1):
                        vals_eff = [p[layer_idx]["effective_rank"]
                                    for p in probe_results
                                    if p[layer_idx] is not None]
                        vals_stab = [p[layer_idx]["stable_rank"]
                                     for p in probe_results
                                     if p[layer_idx] is not None]
                        avg_eff_rank.append(float(np.mean(vals_eff)) if vals_eff else 0.0)
                        avg_stable_rank.append(float(np.mean(vals_stab)) if vals_stab else 0.0)

                    mean_eff = float(np.mean(avg_eff_rank))
                    mean_stab = float(np.mean(avg_stable_rank))

                    n = len(avg_eff_rank)
                    early_eff = float(np.mean(avg_eff_rank[:n//3]))
                    mid_eff = float(np.mean(avg_eff_rank[n//3:2*n//3]))
                    late_eff = float(np.mean(avg_eff_rank[2*n//3:]))

                    print(f"  {label}: eff_rank={mean_eff:.1f} stable_rank={mean_stab:.1f} "
                          f"(E={early_eff:.1f}/M={mid_eff:.1f}/L={late_eff:.1f})")

                    model_results["conditions"][label] = {
                        "per_layer_effective_rank": avg_eff_rank,
                        "per_layer_stable_rank": avg_stable_rank,
                        "mean_effective_rank": mean_eff,
                        "mean_stable_rank": mean_stab,
                        "early_effective_rank": early_eff,
                        "mid_effective_rank": mid_eff,
                        "late_effective_rank": late_eff,
                        "probe_results": probe_results,
                    }

        # Print comparison: vanilla vs CCS
        if "vanilla" in model_results["conditions"]:
            vanilla_eff = model_results["conditions"]["vanilla"]["mean_effective_rank"]
            print(f"\n  --- Constraint Test ---")
            for dose in [2, 3, 5]:
                ccs_key = f"D{dose}_ccs"
                rand_key = f"D{dose}_random"
                if ccs_key in model_results["conditions"]:
                    ccs_eff = model_results["conditions"][ccs_key]["mean_effective_rank"]
                    delta = ccs_eff - vanilla_eff
                    direction = "CONSTRAINS" if delta < 0 else "CONSTRUCTS" if delta > 0 else "NEUTRAL"
                    line = f"    D{dose} CCS: {ccs_eff:.1f} (Δ={delta:+.1f} → {direction})"
                    if rand_key in model_results["conditions"]:
                        rand_eff = model_results["conditions"][rand_key]["mean_effective_rank"]
                        rand_delta = rand_eff - vanilla_eff
                        line += f"  |  Random: {rand_eff:.1f} (Δ={rand_delta:+.1f})"
                    print(line)

        all_results["models"][model_label] = model_results

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-architecture summary
    print(f"\n{'='*60}")
    print(f"  CROSS-ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Vanilla':>10} {'D2 CCS':>10} {'D3 CCS':>10} {'D5 CCS':>10} {'Verdict':>12}")
    print(f"  {'-'*66}")
    for label in ["Mistral", "Gemma", "Llama", "Qwen"]:
        if label not in all_results["models"]:
            continue
        m = all_results["models"][label]
        vanilla = m["conditions"].get("vanilla", {}).get("mean_effective_rank", 0)
        d2 = m["conditions"].get("D2_ccs", {}).get("mean_effective_rank", 0)
        d3 = m["conditions"].get("D3_ccs", {}).get("mean_effective_rank", 0)
        d5 = m["conditions"].get("D5_ccs", {}).get("mean_effective_rank", 0)
        if d3 < vanilla:
            verdict = "CONSTRAINS"
        elif d3 > vanilla:
            verdict = "CONSTRUCTS"
        else:
            verdict = "NEUTRAL"
        print(f"  {label:<12} {vanilla:>10.1f} {d2:>10.1f} {d3:>10.1f} {d5:>10.1f} {verdict:>12}")

    outfile = RESULTS_DIR / f"e36_rank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
