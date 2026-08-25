#!/usr/bin/env python3
"""E13b: Grassmannian Distance and Subspace Continuity

Tests whether the identity subspace (top-k singular vectors) maintains
geometric continuity across layers under CCS, and whether this continuity
differs from random tokens.

Addresses Kimi's CONTRADICT: σ₂ autocorrelation is "just tempo" — it
discards subspace geometry. Two layers can share identical σ₂ while rotating
arbitrarily on the Grassmannian. This experiment measures the actual
subspace geometry.

Metrics:
- Grassmannian distance between top-k subspaces of adjacent layers
- Subspace continuity (autocorrelation of Grassmann distances)
- Principal angle profile across layers
- Holonomy: failure of loop closure L → L+1 → L+2 → L

Expected runtime: ~20 min on A100 (reuses E13 infrastructure).
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

RESULTS_DIR = Path("results/e13b")
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


def grassmann_distance(V1, V2):
    """Compute Grassmannian distance between two subspaces (columns of V1, V2).
    Uses principal angles: d = sqrt(sum(theta_i^2))."""
    M = V1.T @ V2
    try:
        _, s, _ = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return 0.0
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)
    return float(np.sqrt(np.sum(angles**2)))


def principal_angles(V1, V2):
    """Return principal angles between subspaces."""
    M = V1.T @ V2
    try:
        _, s, _ = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return np.zeros(min(V1.shape[1], V2.shape[1]))
    s = np.clip(s, -1, 1)
    return np.arccos(s)


def compute_subspace_metrics(model, tokenizer, text, k=3, device="cuda"):
    """Extract top-k subspaces per layer and compute inter-layer geometry."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    subspaces = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        h_np = h[0].cpu().float().numpy()
        if h_np.shape[0] < k + 1:
            subspaces.append(None)
            continue
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            subspaces.append(Vt[:k].T)  # columns = top-k right singular vectors
        except Exception:
            subspaces.append(None)

    grassmann_dists = []
    angle_profiles = []
    for i in range(len(subspaces) - 1):
        if subspaces[i] is not None and subspaces[i+1] is not None:
            d = grassmann_distance(subspaces[i], subspaces[i+1])
            angles = principal_angles(subspaces[i], subspaces[i+1])
            grassmann_dists.append(d)
            angle_profiles.append(angles.tolist())
        else:
            grassmann_dists.append(0.0)
            angle_profiles.append([0.0] * k)

    # Holonomy: compose L -> L+1 -> L+2 -> L for triplets
    holonomies = []
    for i in range(len(subspaces) - 2):
        if all(subspaces[j] is not None for j in [i, i+1, i+2]):
            # Project V_i through V_{i+1} and V_{i+2} and back
            # Holonomy = angle between V_i and its transported version
            V0 = subspaces[i]
            V1 = subspaces[i+1]
            V2 = subspaces[i+2]
            # Transport: project V0 onto V1's subspace, then onto V2, then back
            # Simplified: measure how much V0's subspace rotates under the composition
            M01 = V0.T @ V1  # projection V0 -> V1
            M12 = V1.T @ V2  # projection V1 -> V2
            M20 = V2.T @ V0  # projection V2 -> V0
            # Round-trip matrix
            roundtrip = M01 @ M12 @ M20
            try:
                _, s, _ = np.linalg.svd(roundtrip)
                s = np.clip(s, -1, 1)
                holonomy = float(np.sqrt(np.sum(np.arccos(s)**2)))
            except Exception:
                holonomy = 0.0
            holonomies.append(holonomy)
        else:
            holonomies.append(0.0)

    # Autocorrelation of Grassmann distances
    gd = np.array(grassmann_dists)
    if len(gd) > 3:
        ac = float(np.corrcoef(gd[:-1], gd[1:])[0, 1])
        ac = ac if not np.isnan(ac) else 0.0
    else:
        ac = 0.0

    return {
        "grassmann_distances": grassmann_dists,
        "angle_profiles": angle_profiles,
        "holonomies": holonomies,
        "grassmann_autocorr": ac,
        "mean_grassmann_dist": float(np.mean(grassmann_dists)),
        "grassmann_cv": float(np.std(grassmann_dists) / np.mean(grassmann_dists)) if np.mean(grassmann_dists) > 1e-10 else 0.0,
        "mean_holonomy": float(np.mean(holonomies)) if holonomies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--doses", nargs="+", type=int, default=[0, 2, 3, 5, 8])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"E13b: Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers, k={args.k}")

    all_results = {
        "model": args.model,
        "k": args.k,
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }

    for dose in args.doses:
        label = f"D{dose}" if dose > 0 else "vanilla"
        print(f"\n=== {label}: CCS content ===")
        preamble = build_preamble(dose)
        ccs_results = []
        for probe_text in PROBES:
            if preamble:
                full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
            else:
                full_text = f"User: {probe_text}\nAssistant:"
            metrics = compute_subspace_metrics(model, tokenizer, full_text, k=args.k, device=args.device)
            ccs_results.append({"probe": probe_text, **metrics})
        all_results["conditions"][f"{label}_ccs"] = ccs_results

        mean_gd = np.mean([r["mean_grassmann_dist"] for r in ccs_results])
        mean_ac = np.mean([r["grassmann_autocorr"] for r in ccs_results])
        mean_hol = np.mean([r["mean_holonomy"] for r in ccs_results])
        print(f"  CCS: grassmann_dist={mean_gd:.4f}, autocorr={mean_ac:.3f}, holonomy={mean_hol:.4f}")

        if dose > 0:
            print(f"\n=== {label}: Random tokens ===")
            preamble_rand = build_preamble(dose, tokenizer, random_content=True)
            rand_results = []
            for probe_text in PROBES:
                full_text = f"{preamble_rand}\n\nUser: {probe_text}\nAssistant:"
                metrics = compute_subspace_metrics(model, tokenizer, full_text, k=args.k, device=args.device)
                rand_results.append({"probe": probe_text, **metrics})
            all_results["conditions"][f"{label}_random"] = rand_results

            mean_gd_r = np.mean([r["mean_grassmann_dist"] for r in rand_results])
            mean_ac_r = np.mean([r["grassmann_autocorr"] for r in rand_results])
            mean_hol_r = np.mean([r["mean_holonomy"] for r in rand_results])
            print(f"  Random: grassmann_dist={mean_gd_r:.4f}, autocorr={mean_ac_r:.3f}, holonomy={mean_hol_r:.4f}")

    outfile = RESULTS_DIR / f"e13b_grassmann_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")

    print("\n=== Summary ===")
    for label in sorted(all_results["conditions"].keys()):
        results = all_results["conditions"][label]
        gd = np.mean([r["mean_grassmann_dist"] for r in results])
        ac = np.mean([r["grassmann_autocorr"] for r in results])
        cv = np.mean([r["grassmann_cv"] for r in results])
        hol = np.mean([r["mean_holonomy"] for r in results])
        print(f"  {label:20s}: dist={gd:.4f}, autocorr={ac:.3f}, cv={cv:.3f}, holonomy={hol:.4f}")


if __name__ == "__main__":
    main()
