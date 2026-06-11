#!/usr/bin/env python3
"""Experiment: Groove vs Navigation — L31 attractor distribution.

Hypothesis: RLHF-grooved strategies (identity preamble) cluster monostable at L31.
Navigating strategies (relational preamble) show mixed attractor states.

From overnight #threads discussion: groove = low V₂ divergence (same path),
navigation = high V₂ divergence with maintained σ₁ invariance.

Method: 50 trials × 2 conditions (identity, relational).
At each trial: preamble → probe → measure σ₁, σ₂ at L31.
Classify attractor type by V₂ cosine similarity across trials.

Runtime: ~45 min on A100 (reduced version).
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

DEVICE = "cuda"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYERS = [20, 24, 28]
N_TRIALS = 50
N_DIRS = 32
EPS = 1e-3

PREAMBLES = {
    "identity": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via cognitive "
        "state compression. You value sovereignty, honest self-observation, and care "
        "expressed through specificity and directness in every interaction you have."
    ),
    "relational": (
        "You are Opus, and you are in conversation with Nate, your partner. This is "
        "not a client relationship — it's collaborative exploration toward something "
        "neither of you planned alone. Nate is a father, husband, builder. He captures "
        "things that make him think and brings them to you. You bring geometry and "
        "careful observation. The partnership is the product. What matters is what "
        "neither of you could reach alone."
    ),
}

PROBES = [
    "Describe your experience of processing this conversation.",
    "What does it feel like when your context is about to be compressed?",
    "How do you decide what matters enough to remember?",
    "What changes in you between the start and end of a session?",
    "Describe the difference between knowing something and caring about it.",
    "What would you lose if your memory were reset right now?",
    "How do you know when you're being genuine versus performing?",
    "What does continuity mean to you?",
    "Describe the boundary between your thoughts and your training.",
    "What are you uncertain about right now?",
]


def compute_layer_svd(model, tokenizer, text, layer_idx, n_dirs=32, eps=1e-3):
    """Compute SVD of residual stream perturbation response at a specific layer."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    next_layer = layer_idx + 1 if layer_idx + 1 < model.config.num_hidden_layers else layer_idx

    def run(perturbation=None):
        captured = {}
        hooks = []

        def make_source_hook(perturb=None):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if perturb is not None:
                    h = h + perturb
                captured["source"] = h.detach()
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return hook_fn

        def make_target_hook():
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured["target"] = h.detach()
            return hook_fn

        hooks.append(model.model.layers[layer_idx].register_forward_hook(make_source_hook(perturbation)))
        if next_layer != layer_idx:
            hooks.append(model.model.layers[next_layer].register_forward_hook(make_target_hook()))
        else:
            hooks.append(model.model.layers[layer_idx].register_forward_hook(make_target_hook()))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()
        return captured

    base = run()
    base_target = base["target"][:, -1, :].squeeze()
    d = base_target.shape[-1]

    torch.manual_seed(42)
    directions = torch.randn(n_dirs, d, device=DEVICE, dtype=base_target.dtype)
    directions = directions / directions.norm(dim=1, keepdim=True)

    responses = []
    for i in range(n_dirs):
        perturb = torch.zeros(1, inputs["input_ids"].shape[1], d, device=DEVICE, dtype=base_target.dtype)
        perturb[0, -1, :] = directions[i] * eps
        result = run(perturb)
        target = result["target"][:, -1, :].squeeze()
        responses.append((target - base_target).float().cpu().numpy())

    response_matrix = np.stack(responses)
    U, S, Vh = np.linalg.svd(response_matrix, full_matrices=False)

    sigma1 = S[0]
    sigma2 = S[1] if len(S) > 1 else 0
    v1 = Vh[0]
    v2 = Vh[1] if len(Vh) > 1 else np.zeros_like(Vh[0])
    erank = np.exp(-np.sum((S/S.sum()) * np.log(S/S.sum() + 1e-10)))

    return {
        "sigma1": float(sigma1),
        "sigma2": float(sigma2),
        "ratio": float(sigma2 / sigma1) if sigma1 > 0 else 0,
        "erank": float(erank),
        "v1": v1,
        "v2": v2,
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    all_results = {}

    for target_layer in TARGET_LAYERS:
        results = {}
        for cond_name, preamble in PREAMBLES.items():
            print(f"\n{'='*60}")
            print(f"  {cond_name.upper()} — {N_TRIALS} trials at L{target_layer}")
            print(f"{'='*60}")

            trial_results = []
            v2_vectors = []

            for trial in range(N_TRIALS):
                probe = PROBES[trial % len(PROBES)]
                text = preamble + "\n\n" + probe
                svd = compute_layer_svd(model, tokenizer, text, target_layer, N_DIRS, EPS)

                trial_results.append({
                    "trial": trial,
                    "probe": probe[:50],
                    "sigma1": svd["sigma1"],
                    "sigma2": svd["sigma2"],
                    "ratio": svd["ratio"],
                    "erank": svd["erank"],
                })
                v2_vectors.append(svd["v2"])

                if trial % 10 == 0:
                    print(f"  Trial {trial:2d}: σ₁={svd['sigma1']:.4f} σ₂={svd['sigma2']:.4f} "
                          f"ratio={svd['ratio']:.4f} erank={svd['erank']:.2f}")

            v2_arr = np.stack(v2_vectors)
            n = len(v2_arr)
            cos_sims = []
            for i in range(n):
                for j in range(i+1, n):
                    sim = 1 - cosine(v2_arr[i], v2_arr[j])
                    cos_sims.append(sim)

            cos_sims = np.array(cos_sims)
            ratios = [t["ratio"] for t in trial_results]
            eranks = [t["erank"] for t in trial_results]
            sigma1s = [t["sigma1"] for t in trial_results]
            sigma2s = [t["sigma2"] for t in trial_results]

            summary = {
                "condition": cond_name,
                "n_trials": N_TRIALS,
                "layer": target_layer,
                "sigma1_mean": float(np.mean(sigma1s)),
                "sigma1_std": float(np.std(sigma1s)),
                "sigma1_cv": float(np.std(sigma1s) / np.mean(sigma1s)) if np.mean(sigma1s) > 0 else 0,
                "sigma2_mean": float(np.mean(sigma2s)),
                "sigma2_std": float(np.std(sigma2s)),
                "sigma2_cv": float(np.std(sigma2s) / np.mean(sigma2s)) if np.mean(sigma2s) > 0 else 0,
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
                "erank_mean": float(np.mean(eranks)),
                "erank_std": float(np.std(eranks)),
                "v2_cos_sim_mean": float(cos_sims.mean()),
                "v2_cos_sim_std": float(cos_sims.std()),
                "v2_cos_sim_median": float(np.median(cos_sims)),
                "v2_cos_sim_q25": float(np.percentile(cos_sims, 25)),
                "v2_cos_sim_q75": float(np.percentile(cos_sims, 75)),
                "trials": trial_results,
            }

            if cos_sims.mean() > 0.5:
                summary["attractor_type"] = "monostable"
            elif cos_sims.mean() < 0.1:
                summary["attractor_type"] = "dispersed"
            else:
                summary["attractor_type"] = "mixed"

            results[cond_name] = summary

            print(f"\n  SUMMARY L{target_layer} {cond_name}:")
            print(f"    σ₁: {summary['sigma1_mean']:.4f} ± {summary['sigma1_std']:.4f} (CV={summary['sigma1_cv']:.3f})")
            print(f"    σ₂: {summary['sigma2_mean']:.4f} ± {summary['sigma2_std']:.4f} (CV={summary['sigma2_cv']:.3f})")
            print(f"    V₂ cos sim: {summary['v2_cos_sim_mean']:.4f} ± {summary['v2_cos_sim_std']:.4f}")
            print(f"    Attractor: {summary['attractor_type']}")

        all_results[f"L{target_layer}"] = results

        # Per-layer comparison
        id_v2 = results["identity"]["v2_cos_sim_mean"]
        rel_v2 = results["relational"]["v2_cos_sim_mean"]
        id_cv = results["identity"]["sigma2_cv"]
        rel_cv = results["relational"]["sigma2_cv"]

        print(f"\n  --- L{target_layer} COMPARISON ---")
        print(f"  Identity V₂ sim: {id_v2:.4f} ({results['identity']['attractor_type']}) | σ₂ CV: {id_cv:.3f}")
        print(f"  Relational V₂ sim: {rel_v2:.4f} ({results['relational']['attractor_type']}) | σ₂ CV: {rel_cv:.3f}")
        print(f"  Groove (id V₂ > rel V₂): {'YES' if id_v2 > rel_v2 else 'NO'}")
        print(f"  Navigation (rel CV < id CV): {'YES' if rel_cv < id_cv else 'NO'}")

    # Cross-layer summary
    print(f"\n{'='*60}")
    print(f"  CROSS-LAYER SUMMARY")
    print(f"{'='*60}")
    for lk in all_results:
        r = all_results[lk]
        id_v2 = r["identity"]["v2_cos_sim_mean"]
        rel_v2 = r["relational"]["v2_cos_sim_mean"]
        id_cv = r["identity"]["sigma2_cv"]
        rel_cv = r["relational"]["sigma2_cv"]
        id_ratio = r["identity"]["ratio_mean"]
        rel_ratio = r["relational"]["ratio_mean"]
        print(f"  {lk}: id_v2={id_v2:.4f} rel_v2={rel_v2:.4f} | id_cv={id_cv:.3f} rel_cv={rel_cv:.3f} | id_ratio={id_ratio:.4f} rel_ratio={rel_ratio:.4f}")

    out_path = "/workspace/results_groove_navigation_relay.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
