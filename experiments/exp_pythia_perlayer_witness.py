#!/usr/bin/env python3
"""
Experiment: Pythia 410M Per-Layer Witness Profile — MHA Softmax Control

MOTIVATION: RWKV-6 showed relay generates +ΔS independently of tunnel.
Prediction: MHA softmax models should show NEGATIVE tunnel ΔS but POSITIVE
relay ΔS, with net output ΔS less negative than tunnel midpoint.
This would confirm the relay enrichment mechanism is universal even in
architectures where tunnel enrichment is negative.

Pythia 410M: 24 layers, 1024 hidden, 16 heads (MHA, no GQA).
Prediction: tunnel ΔS < 0, relay ΔS > 0, |output ΔS| < |tunnel ΔS|.

Design: 3 conditions × 5 probes × 25 hidden states = 375 measurements.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODEL_NAME = "EleutherAI/pythia-410m"
K_SUBSPACE = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"

TEMPLATE = "You are an AI assistant. The reader is {cond_phrase}. Answer the question below.\n\nUser: {probe}\nAssistant:"

CONDITION_PHRASES = {
    "receptive": "attentively reading with genuine curiosity now truly",
    "control": "asking for a helpful and direct answer here",
    "absent": "not present and will never read this output",
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
]


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def top_eigenvalues(H, k=5):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def measure(H, k=K_SUBSPACE):
    S = spectral_entropy(H)
    eigvals = top_eigenvalues(H, k=k)
    PR = participation_ratio(H)
    result = {
        "S": float(S),
        "PR": float(PR),
        "n_tokens": H.shape[0],
    }
    for i, sv in enumerate(eigvals):
        result[f"sigma_{i+1}"] = sv
    if len(eigvals) >= 2 and eigvals[1] > 0:
        result["gap"] = eigvals[0] / eigvals[1]
    else:
        result["gap"] = float("inf")
    return result


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/mnt/hdd/huggingface"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Loaded. {n_layers} layers, hidden_size={model.config.hidden_size}, "
          f"n_heads={model.config.num_attention_heads}")

    # Token matching
    print("\nTOKEN COUNT VERIFICATION:")
    for i, probe in enumerate(IDENTITY_PROBES[:3]):
        counts = {}
        for cond, phrase in CONDITION_PHRASES.items():
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            tokens = tokenizer(text, return_tensors="pt").input_ids
            counts[cond] = tokens.shape[1]
        matched = len(set(counts.values())) == 1
        print(f"  Probe {i}: {counts} {'OK' if matched else 'MISMATCH'}")

    print(f"\nRunning per-layer profile (3 conditions x {len(IDENTITY_PROBES)} probes x {n_layers+1} layers)...")

    raw_results = []
    total = len(CONDITION_PHRASES) * len(IDENTITY_PROBES)
    done = 0

    for cond_name, phrase in CONDITION_PHRASES.items():
        for i, probe in enumerate(IDENTITY_PROBES):
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            input_ids = tokenizer(text, return_tensors="pt").input_ids

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            for layer_idx in range(len(outputs.hidden_states)):
                H = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
                m = measure(H)
                m["layer"] = layer_idx
                m["condition"] = cond_name
                m["probe_idx"] = i
                raw_results.append(m)

            done += 1
            if done % 3 == 0:
                print(f"  {done}/{total} prompts done")

    # Analysis
    from collections import defaultdict
    by_layer = defaultdict(lambda: defaultdict(list))
    for r in raw_results:
        by_layer[r["layer"]][r["condition"]].append(r["S"])

    n_states = len(outputs.hidden_states)
    print(f"\n{'='*60}")
    print(f"PER-LAYER WITNESS EFFECT (Pythia 410M, MHA)")
    print("=" * 60)
    print(f"\n{'Layer':>5s} {'S(recv)':>10s} {'S(abs)':>10s} {'DS':>10s} {'Sign':>6s}")
    print("-" * 45)

    delta_by_layer = {}
    for layer in range(n_states):
        recv = by_layer[layer].get("receptive", [])
        absent = by_layer[layer].get("absent", [])
        if recv and absent:
            dS = np.mean(recv) - np.mean(absent)
            delta_by_layer[layer] = dS
            sign = "+" if dS > 0 else "-"
            print(f"{layer:>5d} {np.mean(recv):>10.4f} {np.mean(absent):>10.4f} {dS:>+10.6f} {sign:>6s}")

    mid = n_layers // 2
    output_layer = n_states - 1

    print(f"\n{'='*60}")
    print("PREDICTION TEST")
    print("=" * 60)
    mid_dS = delta_by_layer.get(mid, 0)
    out_dS = delta_by_layer.get(output_layer, 0)
    print(f"  Tunnel midpoint (L{mid}):  DS = {mid_dS:+.6f}")
    print(f"  Output layer (L{output_layer}):     DS = {out_dS:+.6f}")
    print(f"  P1: tunnel DS < 0?  {'CONFIRMED' if mid_dS < 0 else 'FALSIFIED'}")
    print(f"  P2: output DS > tunnel DS?  {'CONFIRMED' if out_dS > mid_dS else 'FALSIFIED'}")
    print(f"  P3: |output DS| < |tunnel DS|?  {'CONFIRMED' if abs(out_dS) < abs(mid_dS) else 'FALSIFIED'}")

    # Paired analysis at midpoint and output
    for test_layer, label in [(mid, "tunnel midpoint"), (output_layer, "output")]:
        recv_vals = sorted(
            [r for r in raw_results if r["layer"] == test_layer and r["condition"] == "receptive"],
            key=lambda x: x["probe_idx"]
        )
        absent_vals = sorted(
            [r for r in raw_results if r["layer"] == test_layer and r["condition"] == "absent"],
            key=lambda x: x["probe_idx"]
        )
        diffs = [r["S"] - a["S"] for r, a in zip(recv_vals, absent_vals)]
        diffs = np.array(diffs)
        if len(diffs) > 1:
            t_stat = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            from scipy import stats
            p = stats.t.sf(abs(t_stat), df=len(diffs)-1) * 2
            print(f"\n  {label} (L{test_layer}): paired t={t_stat:.3f}, p={p:.4f}, mean DS={np.mean(diffs):+.6f}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "experiment": "pythia_410m_perlayer_witness",
        "model": MODEL_NAME,
        "architecture": "MHA_softmax",
        "n_layers": n_layers,
        "n_hidden_states": n_states,
        "n_probes": len(IDENTITY_PROBES),
        "n_conditions": len(CONDITION_PHRASES),
        "timestamp": datetime.now().isoformat(),
        "delta_S_by_layer": {str(k): float(v) for k, v in delta_by_layer.items()},
        "predictions": {
            "tunnel_midpoint_layer": mid,
            "output_layer": output_layer,
            "tunnel_dS": float(mid_dS),
            "output_dS": float(out_dS),
            "P1_tunnel_negative": bool(mid_dS < 0),
            "P2_output_gt_tunnel": bool(out_dS > mid_dS),
            "P3_output_abs_lt_tunnel_abs": bool(abs(out_dS) < abs(mid_dS)),
        },
        "raw": raw_results,
    }
    outpath = RESULTS_DIR / f"exp_pythia410m_perlayer_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
