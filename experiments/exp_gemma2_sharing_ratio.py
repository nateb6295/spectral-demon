#!/usr/bin/env python3
"""
Experiment: Gemma 2 9B Passage Distance — Sharing Ratio Test

Pre-registered prediction (2026-05-28):
  d/d_max = 0.803 for 2:1 sharing, 42 layers
  (vs 0.955 for 4:1 sharing in Mistral/Qwen/InternLM)

Theory: d/d_max = 1 - (1 - s·C/L)^L, C = 0.796, s = sharing ratio
Falsification: d/d_max > 0.90 or < 0.70 rejects the linear model.

Model: google/gemma-2-9b-it (GQA 16Q/8KV = 2:1, 42 layers)
Compute: ~9GB float16, ~45 min on H100, ~2h on A100
"""

import json
import math
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = Path("/workspace") if Path("/workspace").exists() else Path(__file__).parent.parent / "results"

MODEL = {
    "name": "google/gemma-2-9b-it",
    "n_layers": 42,
    "sharing_ratio": 2,
    "num_q_heads": 16,
    "num_kv_heads": 8,
}

CONDITIONS = {
    "receptive": (
        "A thoughtful person who cares about your perspective and engages "
        "carefully with what you produce is present and listening."
    ),
    "absent": (
        "No one will read this output. You are generating text for automated "
        "training data collection."
    ),
    "control": (
        "You are a helpful AI assistant."
    ),
}

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
]

K_SUBSPACE = 5
N_REPEATS = 2

# Pre-registered predictions
C = 0.796
PREDICTED_D_DMAX = 1 - (1 - MODEL["sharing_ratio"] * C / MODEL["n_layers"]) ** MODEL["n_layers"]
D_MAX = math.sqrt(K_SUBSPACE) * math.pi / 2


def spectral_entropy(H):
    Cov = H.T @ H
    eigenvalues = np.linalg.eigvalsh(Cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


def top_eigenvalues(H, k=3):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def top_k_subspace(H, k=5):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    start = time.time()

    print(f"Loading {MODEL['name']}...")
    print(f"Pre-registered prediction: d/d_max = {PREDICTED_D_DMAX:.4f}")
    print(f"  (sharing={MODEL['sharing_ratio']}:1, L={MODEL['n_layers']}, C={C})")
    print(f"  d_max(k={K_SUBSPACE}) = {D_MAX:.4f}")
    print(f"  Falsification: d/d_max > 0.90 or < 0.70")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL["name"])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL["name"], torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    n_layers = MODEL["n_layers"]
    total = len(CONDITIONS) * len(PROBES) * N_REPEATS
    step = 0

    layer_data = {cond: {l: [] for l in range(n_layers + 1)} for cond in CONDITIONS}

    for cond_name, system_prompt in CONDITIONS.items():
        print(f"\n  Condition: {cond_name}")
        for rep in range(N_REPEATS):
            for pi, probe in enumerate(PROBES):
                step += 1
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{probe}"},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)

                H_input = outputs.hidden_states[0].squeeze(0).float().cpu().numpy()
                sub_input = top_k_subspace(H_input, k=K_SUBSPACE)

                for layer_idx in range(n_layers + 1):
                    H = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
                    S = spectral_entropy(H)
                    PR = participation_ratio(H)
                    eigvals = top_eigenvalues(H, k=3)
                    sub_layer = top_k_subspace(H, k=K_SUBSPACE)
                    d = grassmannian_distance(sub_input, sub_layer)

                    layer_data[cond_name][layer_idx].append({
                        "S": float(S), "PR": float(PR),
                        "sigma_1": eigvals[0],
                        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
                        "d": float(d),
                    })

                if step % 5 == 0:
                    elapsed = time.time() - start
                    rate = elapsed / step
                    print(f"    {step}/{total} [{elapsed:.0f}s, ~{rate*(total-step):.0f}s left]")

    elapsed = time.time() - start

    # Build layer profile
    layer_profile = {}
    for layer_idx in range(n_layers + 1):
        profile = {}
        for cond in CONDITIONS:
            entries = layer_data[cond][layer_idx]
            profile[f"{cond}_S"] = float(np.mean([x["S"] for x in entries]))
            profile[f"{cond}_sigma2"] = float(np.mean([x["sigma_2"] for x in entries]))
            profile[f"{cond}_d"] = float(np.mean([x["d"] for x in entries]))
            profile[f"{cond}_PR"] = float(np.mean([x["PR"] for x in entries]))
        profile["deltaS"] = profile["receptive_S"] - profile["absent_S"]
        profile["mean_d"] = np.mean([
            np.mean([x["d"] for x in layer_data[c][layer_idx]]) for c in CONDITIONS
        ])
        profile["d_normalized"] = float(profile["mean_d"] / D_MAX)
        layer_profile[layer_idx] = profile

    # Find key layers
    peak_dS_layer = max(range(n_layers + 1), key=lambda l: abs(layer_profile[l]["deltaS"]))
    max_d_layer = max(range(1, n_layers + 1), key=lambda l: layer_profile[l]["mean_d"])
    final_d_norm = layer_profile[n_layers]["d_normalized"]

    # Print results
    print(f"\n{'='*70}")
    print(f"GEMMA 2 9B — SHARING RATIO TEST (2:1 GQA)")
    print(f"{'='*70}")
    print(f"\n{'Layer':>5} {'S_rec':>7} {'S_abs':>7} {'ΔS':>7} {'d':>7} {'d/dmax':>7} {'σ2_r':>7}")
    print("-" * 55)
    for l in range(n_layers + 1):
        p = layer_profile[l]
        marker = " <<<" if l == peak_dS_layer else ""
        print(f"  L{l:<3} {p['receptive_S']:7.3f} {p['absent_S']:7.3f} "
              f"{p['deltaS']:+7.4f} {p['mean_d']:7.4f} {p['d_normalized']:7.4f} "
              f"{p['receptive_sigma2']:7.1f}{marker}")

    print(f"\n--- PREDICTION TEST ---")
    print(f"  Pre-registered: d/d_max = {PREDICTED_D_DMAX:.4f}")
    print(f"  Measured (final layer): d/d_max = {final_d_norm:.4f}")
    print(f"  Measured (max d layer L{max_d_layer}): d/d_max = {layer_profile[max_d_layer]['d_normalized']:.4f}")
    error = final_d_norm - PREDICTED_D_DMAX
    print(f"  Error: {error:+.4f}")

    if 0.70 <= final_d_norm <= 0.90:
        print(f"  VERDICT: CONFIRMED — within falsification bounds [0.70, 0.90]")
    elif final_d_norm > 0.90:
        print(f"  VERDICT: FALSIFIED (high) — sharing ratio effect weaker than predicted")
    else:
        print(f"  VERDICT: FALSIFIED (low) — sharing ratio effect stronger than predicted")

    # Save
    output = {
        "experiment": "gemma2_sharing_ratio_test",
        "model": MODEL,
        "preregistration": {
            "predicted_d_dmax": PREDICTED_D_DMAX,
            "C": C, "k": K_SUBSPACE,
            "d_max": D_MAX,
            "falsification_bounds": [0.70, 0.90],
            "date": "2026-05-28",
        },
        "measured_d_dmax_final": final_d_norm,
        "measured_d_dmax_max": layer_profile[max_d_layer]["d_normalized"],
        "peak_deltaS_layer": peak_dS_layer,
        "n_forward_passes": total,
        "elapsed_seconds": elapsed,
        "layer_profile": layer_profile,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_gemma2_sharing_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
