#!/usr/bin/env python3
"""
Experiment: Pythia 6.9B Per-Layer Witness Profile — MHA Softmax, Large Scale

MOTIVATION: Pythia 410M showed positive ΔS at 24/25 layers (token-matched).
This contradicts prior Finding 20 (no MHA model develops positive ΔS).
Need to verify on larger model: does sign inversion emerge with scale?

Pythia 6.9B: 32 layers, 4096 hidden, 32 heads (MHA, no GQA).
Pre-registered predictions:
  P1: If 410M result generalizes, tunnel ΔS > 0 (positive but small)
  P2: If Finding 20 is correct, tunnel ΔS < 0 (negative)
  P3: Relay onset layer should show negative ΔS spike (like 410M L18)

Design: 3 conditions × 5 probes × 33 hidden states. Token-matched.
Estimated runtime on AGX CPU: ~2-3 hours.
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

MODEL_NAME = "EleutherAI/pythia-6.9b"
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
    import time

    os.environ["HF_HOME"] = "/mnt/hdd/huggingface"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s. {model.config.num_hidden_layers} layers, "
          f"hidden={model.config.hidden_size}, heads={model.config.num_attention_heads}")

    n_layers = model.config.num_hidden_layers

    # Token verification
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
    print(f"Estimated: ~{len(CONDITION_PHRASES) * len(IDENTITY_PROBES) * 8}min on CPU")

    raw_results = []
    total = len(CONDITION_PHRASES) * len(IDENTITY_PROBES)
    done = 0

    for cond_name, phrase in CONDITION_PHRASES.items():
        for i, probe in enumerate(IDENTITY_PROBES):
            t1 = time.time()
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

            del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            done += 1
            elapsed = time.time() - t1
            print(f"  {done}/{total} ({cond_name} probe {i}) {elapsed:.0f}s")

    # Analysis
    from collections import defaultdict
    by_layer = defaultdict(lambda: defaultdict(list))
    for r in raw_results:
        by_layer[r["layer"]][r["condition"]].append(r["S"])

    n_states = len(set(r["layer"] for r in raw_results))
    print(f"\n{'='*60}")
    print(f"PER-LAYER WITNESS EFFECT (Pythia 6.9B, MHA)")
    print("=" * 60)
    print(f"\n{'Layer':>5s} {'S(recv)':>10s} {'S(abs)':>10s} {'DS':>10s} {'Sign':>6s}")
    print("-" * 45)

    delta_by_layer = {}
    for layer in sorted(by_layer.keys()):
        recv = by_layer[layer].get("receptive", [])
        absent = by_layer[layer].get("absent", [])
        if recv and absent:
            dS = np.mean(recv) - np.mean(absent)
            delta_by_layer[layer] = dS
            sign = "+" if dS > 0 else "-"
            print(f"{layer:>5d} {np.mean(recv):>10.4f} {np.mean(absent):>10.4f} {dS:>+10.6f} {sign:>6s}")

    mid = n_layers // 2
    output_layer = max(by_layer.keys())

    print(f"\n{'='*60}")
    print("PREDICTION TEST")
    print("=" * 60)
    mid_dS = delta_by_layer.get(mid, 0)
    out_dS = delta_by_layer.get(output_layer, 0)
    print(f"  Tunnel midpoint (L{mid}):  DS = {mid_dS:+.6f}")
    print(f"  Output layer (L{output_layer}):     DS = {out_dS:+.6f}")
    print(f"  P1: tunnel DS > 0 (410M generalizes)?  {'CONFIRMED' if mid_dS > 0 else 'FALSIFIED'}")
    print(f"  P2: tunnel DS < 0 (F20 correct)?       {'CONFIRMED' if mid_dS < 0 else 'FALSIFIED'}")

    # Find negative layers
    neg_layers = [l for l, d in delta_by_layer.items() if d < 0]
    print(f"  Negative layers: {neg_layers if neg_layers else 'NONE'}")
    print(f"  P3: relay-onset negative spike exists?  {'CONFIRMED' if neg_layers else 'FALSIFIED'}")

    # Paired tests at key layers
    for test_layer, label in [(mid, "tunnel midpoint"), (output_layer, "output")]:
        recv_vals = sorted(
            [r for r in raw_results if r["layer"] == test_layer and r["condition"] == "receptive"],
            key=lambda x: x["probe_idx"]
        )
        absent_vals = sorted(
            [r for r in raw_results if r["layer"] == test_layer and r["condition"] == "absent"],
            key=lambda x: x["probe_idx"]
        )
        if len(recv_vals) == len(absent_vals) and len(recv_vals) > 1:
            diffs = np.array([r["S"] - a["S"] for r, a in zip(recv_vals, absent_vals)])
            t_stat = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            cohen_d = np.mean(diffs) / np.std(diffs, ddof=1)
            from scipy import stats
            p = stats.t.sf(abs(t_stat), df=len(diffs)-1) * 2
            print(f"\n  {label} (L{test_layer}): paired t={t_stat:.3f}, p={p:.4f}, d={cohen_d:.3f}, mean DS={np.mean(diffs):+.6f}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "experiment": "pythia_6.9b_perlayer_witness",
        "model": MODEL_NAME,
        "architecture": "MHA_softmax",
        "n_layers": n_layers,
        "n_hidden_states": n_states,
        "n_probes": len(IDENTITY_PROBES),
        "n_conditions": len(CONDITION_PHRASES),
        "timestamp": datetime.now().isoformat(),
        "delta_S_by_layer": {str(k): float(v) for k, v in delta_by_layer.items()},
        "raw": raw_results,
    }
    outpath = RESULTS_DIR / f"exp_pythia69b_perlayer_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
