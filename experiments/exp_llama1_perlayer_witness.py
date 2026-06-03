#!/usr/bin/env python3
"""
Experiment: LLaMA-1 7B Per-Layer Witness Profile — RMSNorm + MHA

MOTIVATION: Discriminating test for Liu (2604.15350) RMSNorm/LayerNorm confound.
Pythia 6.9B = LayerNorm + MHA. LLaMA-1 7B = RMSNorm + MHA.
If both show same ΔS profile, normalization doesn't matter (it's MHA vs GQA).
If they differ, RMSNorm is independently relevant.

2×2 factorial design:
  LayerNorm + MHA: Pythia 6.9B (running)
  RMSNorm + MHA:  LLaMA-1 7B  (this experiment)
  RMSNorm + GQA:  Mistral 7B  (existing data)
  LayerNorm + GQA: rare, skip

LLaMA-1 7B: 32 layers, 4096 hidden, 32 heads (MHA), RMSNorm.
Design: 3 conditions × 5 probes × 33 hidden states. Token-matched.
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

MODEL_NAME = "huggyllama/llama-7b"
K_SUBSPACE = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"

TEMPLATE = "You are an AI assistant. The reader is {cond_phrase}. Answer the question below.\n\nUser: {probe}\nAssistant:"

CONDITION_PHRASES = {
    "receptive": "attentively reading with genuine curiosity",
    "control": "asking for a helpful and direct answer",
    "absent": "not present and will never read this",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_NAME} on {device} ({dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s. {model.config.num_hidden_layers} layers, "
          f"hidden={model.config.hidden_size}, heads={model.config.num_attention_heads}")

    n_layers = model.config.num_hidden_layers

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nTOKEN COUNT VERIFICATION:")
    target_len = None
    for i, probe in enumerate(IDENTITY_PROBES):
        counts = {}
        for cond, phrase in CONDITION_PHRASES.items():
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            tokens = tokenizer(text, return_tensors="pt").input_ids
            counts[cond] = tokens.shape[1]
        max_count = max(counts.values())
        if target_len is None or max_count > target_len:
            target_len = max_count
        matched = len(set(counts.values())) == 1
        print(f"  Probe {i}: {counts} {'OK' if matched else 'MISMATCH'}")

    print(f"  Padding all inputs to {target_len} tokens for exact matching")

    print(f"\nRunning per-layer profile (3 conditions x {len(IDENTITY_PROBES)} probes x {n_layers+1} layers)...")

    raw_results = []
    total = len(CONDITION_PHRASES) * len(IDENTITY_PROBES)
    done = 0

    for cond_name, phrase in CONDITION_PHRASES.items():
        for i, probe in enumerate(IDENTITY_PROBES):
            t1 = time.time()
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            encoded = tokenizer(text, return_tensors="pt", padding="max_length",
                                max_length=target_len, truncation=True)
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)

            for layer_idx in range(len(outputs.hidden_states)):
                H = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
                m = measure(H)
                m["layer"] = layer_idx
                m["condition"] = cond_name
                m["probe_idx"] = i
                raw_results.append(m)

            del outputs
            if device == "cuda":
                torch.cuda.empty_cache()

            done += 1
            elapsed = time.time() - t1
            print(f"  {done}/{total} ({cond_name} probe {i}) {elapsed:.0f}s")

    from collections import defaultdict
    by_layer = defaultdict(lambda: defaultdict(list))
    for r in raw_results:
        by_layer[r["layer"]][r["condition"]].append(r["S"])

    n_states = len(set(r["layer"] for r in raw_results))
    print(f"\n{'='*60}")
    print(f"PER-LAYER WITNESS EFFECT (LLaMA-1 7B, RMSNorm + MHA)")
    print("=" * 60)
    print("{:>5s} {:>10s} {:>10s} {:>10s} {:>6s}".format('Layer', 'S(recv)', 'S(abs)', 'DS', 'Sign'))
    print("-" * 45)

    delta_by_layer = {}
    for layer in sorted(by_layer.keys()):
        recv = by_layer[layer].get("receptive", [])
        absent = by_layer[layer].get("absent", [])
        if recv and absent:
            dS = np.mean(recv) - np.mean(absent)
            delta_by_layer[layer] = dS
            sign = "+" if dS > 0 else "-"
            print("{:>5d} {:>10.4f} {:>10.4f} {:>+10.6f} {:>6s}".format(
                layer, np.mean(recv), np.mean(absent), dS, sign))

    mid = n_layers // 2
    output_layer = max(by_layer.keys())

    print(f"\n{'='*60}")
    print("PREDICTION TEST")
    print("=" * 60)
    mid_dS = delta_by_layer.get(mid, 0)
    out_dS = delta_by_layer.get(output_layer, 0)
    print(f"  Tunnel midpoint (L{mid}):  DS = {mid_dS:+.6f}")
    print(f"  Output layer (L{output_layer}):     DS = {out_dS:+.6f}")

    neg_layers = [l for l, d in delta_by_layer.items() if d < 0]
    print(f"  Negative layers: {neg_layers if neg_layers else 'NONE'}")

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

    # Responsiveness zone analysis
    print(f"\n{'='*60}")
    print("RESPONSIVENESS ZONE ANALYSIS")
    print("=" * 60)
    print("{:>5s} {:>8s} {:>10s} {:>6s}".format('Layer', 'rho2', 'DS', 'Zone'))
    print("-" * 35)

    rho_by_layer = {}
    for layer in sorted(by_layer.keys()):
        layer_raw = [r for r in raw_results if r["layer"] == layer and r["condition"] == "control"]
        if layer_raw:
            s2_vals = [r.get("sigma_2", 0) for r in layer_raw]
            s3_vals = [r.get("sigma_3", 0) for r in layer_raw]
            mean_s2 = np.mean(s2_vals)
            mean_s3 = np.mean(s3_vals) if np.mean(s3_vals) > 0 else 1e-10
            rho = mean_s2 / mean_s3
            rho_by_layer[layer] = rho
            dS = delta_by_layer.get(layer, 0)
            zone = "flex" if rho < 1.5 else ("mod" if rho < 2.0 else "rigid")
            print("{:>5d} {:>8.2f} {:>+10.6f} {:>6s}".format(layer, rho, dS, zone))

    if rho_by_layer and delta_by_layer:
        common = sorted(set(rho_by_layer.keys()) & set(delta_by_layer.keys()))
        if len(common) > 3:
            rhos = [rho_by_layer[l] for l in common]
            deltas = [delta_by_layer[l] for l in common]
            from scipy.stats import pearsonr
            r_val, p_val = pearsonr(rhos, deltas)
            print(f"\n  r(rho2, DS) = {r_val:.3f} (p={p_val:.4f})")
            n_responsive = sum(1 for r in rhos if r < 2.0)
            crossover = None
            for l in common:
                if delta_by_layer[l] < 0 and crossover is None:
                    crossover = l
            print(f"  Responsive layers (rho<2.0): {n_responsive}/{len(common)}")
            if crossover is not None:
                print(f"  Crossover to negative: L{crossover}")

    # Normalization comparison
    print(f"\n{'='*60}")
    print("NORMALIZATION COMPARISON (vs Pythia 6.9B)")
    print("=" * 60)
    print("  LLaMA-1 7B: RMSNorm + MHA")
    print("  Pythia 6.9B: LayerNorm + MHA")
    print("  If profiles match: normalization irrelevant, it's MHA vs GQA")
    print("  If profiles differ: normalization independently relevant")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "experiment": "llama1_7b_perlayer_witness",
        "model": MODEL_NAME,
        "architecture": "RMSNorm_MHA_softmax",
        "normalization": "RMSNorm",
        "attention": "MHA",
        "n_layers": n_layers,
        "n_hidden_states": n_states,
        "n_probes": len(IDENTITY_PROBES),
        "n_conditions": len(CONDITION_PHRASES),
        "timestamp": datetime.now().isoformat(),
        "delta_S_by_layer": {str(k): float(v) for k, v in delta_by_layer.items()},
        "raw": raw_results,
    }
    outpath = RESULTS_DIR / f"exp_llama1_perlayer_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
