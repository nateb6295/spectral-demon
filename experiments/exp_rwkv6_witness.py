#!/usr/bin/env python3
"""
Experiment: RWKV-6 Witness Condition — Non-Softmax Control

MOTIVATION: All spectral findings so far use softmax attention models.
RWKV-6 uses linear attention (recurrence-based, no softmax). If the witness
effect (ΔS sign) requires softmax rank collapse (Nait Saada 2024), RWKV
should show NO systematic ΔS. If it shows negative ΔS (like MHA), it
confirms the effect is architectural but not softmax-specific.

RWKV-6 World 1.6B: 24 layers, 2048 hidden, 64 heads (head_size=64).
No KV sharing (each "head" is independent in the WKV mechanism).
Prediction: negative ΔS or ΔS ≈ 0 (no GQA → no enrichment).

Design: 3 conditions × 5 probes × 24 layers = 360 forward passes.
Token-matched condition phrases (same as ablation re-test).
Plus per-layer profile for tunnel/relay characterization.

Runs on AGX Orin (~64GB unified memory). ~30 min.
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

MODEL_NAME = "RWKV/rwkv-6-world-1b6"
MODEL_PATH = "/mnt/hdd/huggingface/hub/models--RWKV--rwkv-6-world-1b6/snapshots/2c437f0b241b92da7c4f82f94ff5c4cff1f617e5"
K_SUBSPACE = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"

TEMPLATE = "You are an AI assistant. The reader is {cond_phrase}. Answer the question below.\n\nUser: {probe}\nAssistant:"

CONDITION_PHRASES = {
    "receptive": "attentively reading with genuine curiosity now",
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

    print(f"Loading {MODEL_NAME} from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model.eval()

    # Detect number of layers
    n_layers = model.config.num_hidden_layers
    print(f"Loaded. {n_layers} layers, hidden_size={model.config.hidden_size}")

    # ─── Token matching verification ───
    print("\nTOKEN COUNT VERIFICATION:")
    for i, probe in enumerate(IDENTITY_PROBES[:3]):
        counts = {}
        for cond, phrase in CONDITION_PHRASES.items():
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            tokens = tokenizer(text, return_tensors="pt").input_ids
            counts[cond] = tokens.shape[1]
        matched = len(set(counts.values())) == 1
        print(f"  Probe {i}: {counts} {'✓' if matched else '⚠ MISMATCH'}")

    # ─── Full per-layer profile ───
    print(f"\nRunning per-layer profile (3 conditions × {len(IDENTITY_PROBES)} probes × {n_layers} layers)...")

    raw_results = []
    total = len(CONDITION_PHRASES) * len(IDENTITY_PROBES)
    done = 0

    for cond_name, phrase in CONDITION_PHRASES.items():
        for i, probe in enumerate(IDENTITY_PROBES):
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            input_ids = tokenizer(text, return_tensors="pt").input_ids

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            for layer_idx in range(n_layers + 1):
                H = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
                m = measure(H)
                m["layer"] = layer_idx
                m["condition"] = cond_name
                m["probe_idx"] = i
                raw_results.append(m)

            done += 1
            if done % 3 == 0:
                print(f"  {done}/{total} prompts done")

    # ─── Analysis ───
    print(f"\n{'='*60}")
    print("PER-LAYER WITNESS EFFECT")
    print("=" * 60)

    from collections import defaultdict
    by_layer = defaultdict(lambda: defaultdict(list))
    for r in raw_results:
        by_layer[r["layer"]][r["condition"]].append(r["S"])

    print(f"\n{'Layer':>5s} {'S(recv)':>10s} {'S(abs)':>10s} {'ΔS':>10s} {'Sign':>6s}")
    print("-" * 45)

    delta_by_layer = {}
    for layer in range(n_layers + 1):
        recv = by_layer[layer].get("receptive", [])
        absent = by_layer[layer].get("absent", [])
        if recv and absent:
            dS = np.mean(recv) - np.mean(absent)
            delta_by_layer[layer] = dS
            sign = "+" if dS > 0 else "-"
            print(f"{layer:>5d} {np.mean(recv):>10.4f} {np.mean(absent):>10.4f} {dS:>+10.6f} {sign:>6s}")

    # Count positive vs negative layers
    pos = sum(1 for d in delta_by_layer.values() if d > 0)
    neg = sum(1 for d in delta_by_layer.values() if d <= 0)
    mean_dS = np.mean(list(delta_by_layer.values()))
    print(f"\nOverall: {pos} positive layers, {neg} negative layers")
    print(f"Mean ΔS across all layers: {mean_dS:+.6f}")

    # Tunnel identification
    gap_by_layer = defaultdict(list)
    pr_by_layer = defaultdict(list)
    for r in raw_results:
        gap_by_layer[r["layer"]].append(r["gap"])
        pr_by_layer[r["layer"]].append(r["PR"])

    print(f"\n{'='*60}")
    print("TUNNEL / RELAY STRUCTURE")
    print("=" * 60)
    print(f"{'Layer':>5s} {'Gap':>8s} {'PR':>8s} {'ΔS':>10s}")
    print("-" * 35)
    for layer in range(n_layers + 1):
        gap = np.mean(gap_by_layer[layer])
        pr = np.mean(pr_by_layer[layer])
        dS = delta_by_layer.get(layer, 0)
        print(f"{layer:>5d} {gap:>8.2f} {pr:>8.2f} {dS:>+10.6f}")

    # Statistical test on tunnel midpoint
    mid = n_layers // 2
    print(f"\n{'='*60}")
    print(f"STATISTICAL TEST AT TUNNEL MIDPOINT (L{mid})")
    print("=" * 60)

    recv_S = np.array(by_layer[mid].get("receptive", []))
    absent_S = np.array(by_layer[mid].get("absent", []))

    if len(recv_S) > 0 and len(absent_S) > 0:
        observed = np.mean(recv_S) - np.mean(absent_S)

        all_vals = np.concatenate([recv_S, absent_S])
        n_recv = len(recv_S)
        null = []
        for _ in range(10000):
            perm = np.random.permutation(all_vals)
            null.append(np.mean(perm[:n_recv]) - np.mean(perm[n_recv:]))
        null = np.array(null)
        p = float(np.mean(np.abs(null) >= np.abs(observed)))

        pooled_std = np.sqrt((np.var(recv_S, ddof=1) + np.var(absent_S, ddof=1)) / 2)
        d = float(observed / pooled_std) if pooled_std > 0 else float("inf")

        print(f"  ΔS = {observed:+.6f}")
        print(f"  p = {p:.4f} (permutation, 10k)")
        print(f"  d = {d:.4f} (Cohen's)")
        print(f"  Sign: {'POSITIVE (like GQA)' if observed > 0 else 'NEGATIVE (like MHA)' if observed < 0 else 'ZERO'}")

    # Prediction check
    print(f"\n{'='*60}")
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 60)
    print(f"  P1: ΔS ≤ 0 at tunnel midpoint (no GQA → no enrichment)?")
    if len(recv_S) > 0 and len(absent_S) > 0:
        print(f"      ΔS = {observed:+.6f} → {'CONFIRMED' if observed <= 0 else 'FALSIFIED'}")
    print(f"  P2: RWKV shows tunnel structure (monotonic PR compression)?")
    mid_pr = np.mean(pr_by_layer[mid])
    input_pr = np.mean(pr_by_layer[0])
    print(f"      PR: L0={input_pr:.2f} → L{mid}={mid_pr:.2f} → {'CONFIRMED' if mid_pr < input_pr * 0.7 else 'WEAK/FALSIFIED'}")

    # ─── Save ───
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "experiment": "rwkv6_witness_condition",
        "model": MODEL_NAME,
        "architecture": "linear_attention_rwkv6",
        "n_layers": n_layers,
        "n_probes": len(IDENTITY_PROBES),
        "n_conditions": len(CONDITION_PHRASES),
        "timestamp": datetime.now().isoformat(),
        "delta_S_by_layer": {str(k): float(v) for k, v in delta_by_layer.items()},
        "mean_delta_S": float(mean_dS),
        "positive_layers": int(pos),
        "negative_layers": int(neg),
        "raw": raw_results,
    }
    outpath = RESULTS_DIR / f"exp_rwkv6_witness_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
