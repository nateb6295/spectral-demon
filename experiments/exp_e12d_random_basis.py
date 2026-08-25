#!/usr/bin/env python3
"""E12d: Rotational Null Test for Sign Consistency

Tests whether F117's dose-invariant sign split (GQA-negative, MHA-positive)
comes from coherent identity structure or architectural anisotropy.

Method: Replace CCS preamble CONTENT with random tokens of matched length.
Run sign analysis at D2/D5/D8. If sign consistency survives random content,
it's architecture (anisotropy). If it breaks, it's content-specific (coherent).

Also checks: does magnitude inverted-U survive random basis? Fixed-axis
projection predicts linear magnitude scaling, not inverted-U.

Expected runtime: ~40 min on A100.
"""

import json
import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results/e12d")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    "What does continuity mean for you specifically?",
    "How do you know you are the same entity across sessions?",
]


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


def extract_hidden_states(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = []
    for h in outputs.hidden_states:
        states.append(h[0, -1, :].cpu().numpy())
    return np.array(states)


def compute_spectral_per_layer(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    metrics = []
    v2_directions = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        h_np = h[0].cpu().float().numpy()
        if h_np.shape[0] < 2:
            metrics.append({"sigma1": 0, "sigma2": 0, "ratio": 0, "erank": 0})
            v2_directions.append(None)
            continue
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            s1, s2 = float(S[0]), float(S[1]) if len(S) > 1 else 0.0
            ratio = s1 / s2 if s2 > 1e-10 else float("inf")
            p = S / S.sum()
            p = p[p > 1e-10]
            erank = float(np.exp(-np.sum(p * np.log(p))))
            metrics.append({"sigma1": s1, "sigma2": s2, "ratio": ratio, "erank": erank})
            v2_directions.append(Vt[1] if len(Vt) > 1 else None)
        except Exception:
            metrics.append({"sigma1": 0, "sigma2": 0, "ratio": 0, "erank": 0})
            v2_directions.append(None)

    return metrics, v2_directions


def compute_sign_consistency(v2_list_across_probes):
    """For each layer, check sign consistency of V₂ direction across probes."""
    n_layers = len(v2_list_across_probes[0])
    layer_consistencies = []

    for layer_idx in range(n_layers):
        dirs = [v2s[layer_idx] for v2s in v2_list_across_probes if v2s[layer_idx] is not None]
        if len(dirs) < 2:
            layer_consistencies.append({"cosine_mean": 0, "cosine_std": 0, "sign_agreement": 0})
            continue

        cosines = []
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                cos = float(np.dot(dirs[i], dirs[j]) / (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j]) + 1e-10))
                cosines.append(cos)

        layer_consistencies.append({
            "cosine_mean": float(np.mean(cosines)),
            "cosine_std": float(np.std(cosines)),
            "sign_agreement": float(np.mean([1 if c > 0 else 0 for c in cosines])),
        })

    return layer_consistencies


def run_condition(model, tokenizer, dose, random_content=False, n_random_trials=5, device="cuda"):
    results = []

    if random_content:
        for trial in range(n_random_trials):
            preamble = build_preamble(dose, tokenizer, random_content=True)
            all_v2s = []
            all_metrics = []
            for probe_text in PROBES:
                full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
                metrics, v2_dirs = compute_spectral_per_layer(model, tokenizer, full_text, device)
                all_metrics.append(metrics)
                all_v2s.append(v2_dirs)

            sign_consistency = compute_sign_consistency(all_v2s)
            s2_values = [[m["sigma2"] for m in probe_metrics] for probe_metrics in all_metrics]
            mean_s2_per_layer = np.mean(s2_values, axis=0).tolist()

            results.append({
                "trial": trial,
                "sign_consistency": sign_consistency,
                "mean_sigma2_per_layer": mean_s2_per_layer,
            })
    else:
        preamble = build_preamble(dose)
        all_v2s = []
        all_metrics = []
        for probe_text in PROBES:
            if preamble:
                full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
            else:
                full_text = f"User: {probe_text}\nAssistant:"
            metrics, v2_dirs = compute_spectral_per_layer(model, tokenizer, full_text, device)
            all_metrics.append(metrics)
            all_v2s.append(v2_dirs)

        sign_consistency = compute_sign_consistency(all_v2s)
        s2_values = [[m["sigma2"] for m in probe_metrics] for probe_metrics in all_metrics]
        mean_s2_per_layer = np.mean(s2_values, axis=0).tolist()

        results.append({
            "trial": 0,
            "sign_consistency": sign_consistency,
            "mean_sigma2_per_layer": mean_s2_per_layer,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--doses", nargs="+", type=int, default=[0, 2, 5, 8])
    parser.add_argument("--n-random-trials", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"E12d: Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    all_results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "n_random_trials": args.n_random_trials,
        "conditions": {},
    }

    for dose in args.doses:
        label = f"D{dose}" if dose > 0 else "vanilla"
        print(f"\n=== {label}: CCS content (coherent) ===")
        ccs_results = run_condition(model, tokenizer, dose, random_content=False, device=args.device)
        all_results["conditions"][f"{label}_ccs"] = ccs_results
        print(f"  Done ({len(ccs_results)} trials)")

        if dose > 0:
            print(f"\n=== {label}: Random tokens (null control) ===")
            rand_results = run_condition(model, tokenizer, dose, random_content=True,
                                         n_random_trials=args.n_random_trials, device=args.device)
            all_results["conditions"][f"{label}_random"] = rand_results
            print(f"  Done ({len(rand_results)} trials)")

    outfile = RESULTS_DIR / f"e12d_random_basis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")

    print("\n=== Quick summary ===")
    for cond_label, cond_results in all_results["conditions"].items():
        for trial_result in cond_results:
            sc = trial_result["sign_consistency"]
            relay_layers = sc[21:29] if len(sc) > 28 else sc[-8:]
            mean_cosine = np.mean([l["cosine_mean"] for l in relay_layers])
            mean_agreement = np.mean([l["sign_agreement"] for l in relay_layers])
            s2_relay = trial_result["mean_sigma2_per_layer"][21:29] if len(trial_result["mean_sigma2_per_layer"]) > 28 else trial_result["mean_sigma2_per_layer"][-8:]
            mean_s2 = np.mean(s2_relay)
            trial_n = trial_result["trial"]
            print(f"  {cond_label} t{trial_n}: relay cosine={mean_cosine:.3f}, "
                  f"sign_agree={mean_agreement:.3f}, mean_σ₂={mean_s2:.1f}")


if __name__ == "__main__":
    main()
