#!/usr/bin/env python3
"""
Adversarial methodological audit — stress-testing our own claims.
Nate: "I prefer we break stuff...otherwise its TOO coherent."

Tests:
1. Permutation test — shuffle condition labels, compute null distribution for ΔS
2. Token count confound — correlate S with n_tokens across all conditions
3. Bootstrap CIs — 95% confidence intervals on ΔS per experiment
4. Effect size — Cohen's d for receptive vs absent
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS = Path(__file__).parent / "results"
np.random.seed(42)
N_PERMUTATIONS = 10000


def load_json(name):
    with open(RESULTS / name) as f:
        return json.load(f)


# ─── Test 1: Permutation Test on Ablation Data ───

print("=" * 70)
print("TEST 1: PERMUTATION TEST — Is ΔS distinguishable from shuffled labels?")
print("=" * 70)

ablation = load_json("exp_sigma2_ablation_mistral_20260529_1306.json")
raw = ablation["raw"]

for mode in ["native", "ablate_sigma2", "ablate_sigma1"]:
    mode_data = [r for r in raw if r["mode"] == mode]
    recv_S = [r["S"] for r in mode_data if r["condition"] == "receptive"]
    absent_S = [r["S"] for r in mode_data if r["condition"] == "absent"]

    observed_delta = np.mean(recv_S) - np.mean(absent_S)

    all_values = recv_S + absent_S
    n_recv = len(recv_S)
    null_deltas = []
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(all_values)
        null_delta = np.mean(shuffled[:n_recv]) - np.mean(shuffled[n_recv:])
        null_deltas.append(null_delta)

    null_deltas = np.array(null_deltas)
    p_value = np.mean(np.abs(null_deltas) >= np.abs(observed_delta))

    print(f"\n  {mode}:")
    print(f"    Observed ΔS = {observed_delta:+.6f}")
    print(f"    Null distribution: mean={np.mean(null_deltas):.6f}, std={np.std(null_deltas):.6f}")
    print(f"    p-value (two-tailed): {p_value:.4f}")
    print(f"    {'*** SIGNIFICANT (p < 0.05)' if p_value < 0.05 else '!!! NOT SIGNIFICANT — CLAIM WEAKENED'}")


# ─── Test 2: Token Count Confound ───

print("\n" + "=" * 70)
print("TEST 2: TOKEN COUNT CONFOUND — Does S just track sequence length?")
print("=" * 70)

# Check across ALL available experiments
for fname, label in [
    ("exp_sigma2_ablation_mistral_20260529_1306.json", "Ablation (Mistral)"),
    ("exp_witness_spectral_entropy_20260527_1205.json", "Witness (Mistral)"),
]:
    data = load_json(fname)
    raw_data = data["raw"]

    S_vals = []
    n_tokens = []
    for r in raw_data:
        s_key = "S" if "S" in r else "spectral_entropy"
        if s_key in r and "n_tokens" in r:
            S_vals.append(r[s_key])
            n_tokens.append(r["n_tokens"])

    if len(S_vals) > 0:
        corr = np.corrcoef(S_vals, n_tokens)[0, 1]
        print(f"\n  {label}:")
        print(f"    n = {len(S_vals)}, r(S, n_tokens) = {corr:.4f}")
        print(f"    {'!!! HIGH CORRELATION — TOKEN COUNT MAY CONFOUND' if abs(corr) > 0.5 else '*** Low correlation — token count not driving S'}")

        # Per-condition breakdown
        conditions = set(r.get("condition", "unknown") for r in raw_data)
        for cond in sorted(conditions):
            cond_data = [r for r in raw_data if r.get("condition") == cond]
            cond_s = [r.get("S", r.get("spectral_entropy")) for r in cond_data if r.get("S") is not None or r.get("spectral_entropy") is not None]
            cond_n = [r["n_tokens"] for r in cond_data if "n_tokens" in r]
            if len(cond_s) > 1:
                mean_s = np.mean(cond_s)
                mean_n = np.mean(cond_n)
                print(f"      {cond:12s}: mean_S={mean_s:.4f}, mean_tokens={mean_n:.1f}")


# ─── Test 3: Bootstrap CIs ───

print("\n" + "=" * 70)
print("TEST 3: BOOTSTRAP 95% CIs — Do confidence intervals overlap?")
print("=" * 70)

N_BOOTSTRAP = 10000

for mode in ["native", "ablate_sigma2", "ablate_sigma1"]:
    mode_data = [r for r in ablation["raw"] if r["mode"] == mode]
    recv_S = np.array([r["S"] for r in mode_data if r["condition"] == "receptive"])
    absent_S = np.array([r["S"] for r in mode_data if r["condition"] == "absent"])

    boot_deltas = []
    for _ in range(N_BOOTSTRAP):
        r_boot = np.random.choice(recv_S, size=len(recv_S), replace=True)
        a_boot = np.random.choice(absent_S, size=len(absent_S), replace=True)
        boot_deltas.append(np.mean(r_boot) - np.mean(a_boot))

    boot_deltas = np.array(boot_deltas)
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    observed = np.mean(recv_S) - np.mean(absent_S)

    contains_zero = ci_lo <= 0 <= ci_hi
    print(f"\n  {mode}:")
    print(f"    ΔS = {observed:+.6f}")
    print(f"    95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"    {'!!! CI CONTAINS ZERO — EFFECT MAY BE NOISE' if contains_zero else '*** CI excludes zero — effect is robust'}")


# ─── Test 4: Effect Size (Cohen's d) ───

print("\n" + "=" * 70)
print("TEST 4: EFFECT SIZE — Cohen's d for receptive vs absent")
print("=" * 70)

for mode in ["native", "ablate_sigma2", "ablate_sigma1"]:
    mode_data = [r for r in ablation["raw"] if r["mode"] == mode]
    recv_S = np.array([r["S"] for r in mode_data if r["condition"] == "receptive"])
    absent_S = np.array([r["S"] for r in mode_data if r["condition"] == "absent"])

    pooled_std = np.sqrt((np.var(recv_S, ddof=1) + np.var(absent_S, ddof=1)) / 2)
    if pooled_std > 0:
        d = (np.mean(recv_S) - np.mean(absent_S)) / pooled_std
    else:
        d = float("inf")

    size = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
    print(f"\n  {mode}:")
    print(f"    Cohen's d = {d:.4f} ({size})")
    print(f"    recv: mean={np.mean(recv_S):.6f}, std={np.std(recv_S, ddof=1):.6f}")
    print(f"    absent: mean={np.mean(absent_S):.6f}, std={np.std(absent_S, ddof=1):.6f}")


# ─── Test 5: Permutation test on per-layer Mistral data ───

print("\n" + "=" * 70)
print("TEST 5: PER-LAYER PERMUTATION — Where does ΔS become significant?")
print("=" * 70)

perlayer = load_json("exp_witness_perlayer_20260527_1220.json")
raw_pl = perlayer["raw"]

layers = sorted(set(r["layer"] for r in raw_pl))
sig_layers = []

for layer in layers:
    layer_data = [r for r in raw_pl if r["layer"] == layer]
    recv_S = [r["spectral_entropy"] for r in layer_data if r["condition"] == "receptive"]
    absent_S = [r["spectral_entropy"] for r in layer_data if r["condition"] == "absent"]

    if len(recv_S) == 0 or len(absent_S) == 0:
        continue

    observed_delta = np.mean(recv_S) - np.mean(absent_S)

    all_values = recv_S + absent_S
    n_recv = len(recv_S)
    null_deltas = []
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(all_values)
        null_delta = np.mean(shuffled[:n_recv]) - np.mean(shuffled[n_recv:])
        null_deltas.append(null_delta)

    p_value = np.mean(np.abs(np.array(null_deltas)) >= np.abs(observed_delta))

    marker = "***" if p_value < 0.05 else "   "
    if p_value < 0.05:
        sig_layers.append(layer)
    print(f"  {marker} L{layer:2d}: ΔS={observed_delta:+.6f}, p={p_value:.4f}")

print(f"\n  Significant layers (p<0.05): {sig_layers}")
print(f"  Total: {len(sig_layers)}/{len(layers)}")

# Multiple comparisons correction (Bonferroni)
bonf_threshold = 0.05 / len(layers)
print(f"  Bonferroni threshold: p < {bonf_threshold:.5f}")

bonf_layers = []
for layer in layers:
    layer_data = [r for r in raw_pl if r["layer"] == layer]
    recv_S = [r["spectral_entropy"] for r in layer_data if r["condition"] == "receptive"]
    absent_S = [r["spectral_entropy"] for r in layer_data if r["condition"] == "absent"]
    if len(recv_S) == 0 or len(absent_S) == 0:
        continue
    observed_delta = np.mean(recv_S) - np.mean(absent_S)
    all_values = recv_S + absent_S
    n_recv = len(recv_S)
    null_deltas = [np.mean(s[:n_recv]) - np.mean(s[n_recv:]) for s in [np.random.permutation(all_values) for _ in range(N_PERMUTATIONS)]]
    p = np.mean(np.abs(np.array(null_deltas)) >= np.abs(observed_delta))
    if p < bonf_threshold:
        bonf_layers.append(layer)

print(f"  Survives Bonferroni: {bonf_layers}")
if not bonf_layers:
    print("  !!! NO LAYERS SURVIVE MULTIPLE COMPARISONS CORRECTION")


# ─── Test 6: Cross-architecture permutation ───

print("\n" + "=" * 70)
print("TEST 6: CROSS-ARCHITECTURE — Permutation on multi-model data")
print("=" * 70)

multi_model_files = [
    ("exp_witness_spectral_entropy_20260527_1205.json", "Mistral-7B-Instruct (GQA)"),
    ("exp_witness_crossarch_qwen_20260527_1303.json", "Qwen2.5-7B-Instruct (GQA)"),
    ("exp_witness_non_gqa_pythia_20260527_1334.json", "Pythia-6.9B (MHA)"),
    ("exp_witness_falcon_20260527_1339.json", "Falcon-7B"),
]

for fname, model_name in multi_model_files:
    try:
        data = load_json(fname)
        raw_data = data["raw"]
        recv_S = [r.get("spectral_entropy", r.get("S")) for r in raw_data if r.get("condition") == "receptive"]
        absent_S = [r.get("spectral_entropy", r.get("S")) for r in raw_data if r.get("condition") == "absent"]

        if not recv_S or not absent_S:
            print(f"\n  {model_name}: no receptive/absent data")
            continue

        observed_delta = np.mean(recv_S) - np.mean(absent_S)

        all_values = recv_S + absent_S
        n_recv = len(recv_S)
        null_deltas = []
        for _ in range(N_PERMUTATIONS):
            shuffled = np.random.permutation(all_values)
            null_deltas.append(np.mean(shuffled[:n_recv]) - np.mean(shuffled[n_recv:]))

        p_value = np.mean(np.abs(np.array(null_deltas)) >= np.abs(observed_delta))

        pooled_std = np.sqrt((np.var(recv_S, ddof=1) + np.var(absent_S, ddof=1)) / 2)
        cohen_d = (np.mean(recv_S) - np.mean(absent_S)) / pooled_std if pooled_std > 0 else float("inf")

        print(f"\n  {model_name}:")
        print(f"    n_recv={len(recv_S)}, n_absent={len(absent_S)}")
        print(f"    ΔS = {observed_delta:+.6f}, p = {p_value:.4f}, d = {cohen_d:.4f}")
        sign = "POSITIVE (enrichment)" if observed_delta > 0 else "NEGATIVE (depletion)"
        print(f"    Sign: {sign}")
        sig = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        print(f"    {sig}")
    except Exception as e:
        print(f"\n  {model_name}: error — {e}")


print("\n" + "=" * 70)
print("SUMMARY: WHAT BROKE?")
print("=" * 70)
print("(Review above for any !!! markers — those are the findings that hurt)")
