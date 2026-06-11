#!/usr/bin/env python3
"""V₂ coherence trajectory analysis across the relay zone.

Loads all five-condition results (instruct/base × identity/neutral probes)
and computes:
1. Rank trajectories for each condition across layers
2. Bootstrap CIs on rank ordering at each layer
3. Displacement test: is relational's rank trajectory significantly different
   between base and instruct?
4. Trajectory stability: does the displacement hold through L30?

No GPU required — pure analysis of existing results.
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path(__file__).parent.parent / "results"
CONDITIONS = ["identity", "relational", "generic", "denial", "contradictory"]

def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def extract_coherence_by_layer(data, layers=None):
    if layers is None:
        layers = sorted(data.keys(), key=lambda x: int(x.replace("L", "")))
    result = {}
    for layer_key in layers:
        if layer_key not in data:
            continue
        layer_num = int(layer_key.replace("L", ""))
        result[layer_num] = {}
        for cond in CONDITIONS:
            if cond in data[layer_key]:
                entry = data[layer_key][cond]
                result[layer_num][cond] = {
                    "v2_coherence": entry.get("v2_cos_sim_mean", None),
                    "v2_std": entry.get("v2_cos_sim_std", None),
                    "sigma2_mean": entry.get("sigma2_mean", None),
                    "ratio_mean": entry.get("ratio_mean", None),
                    "ratio_std": entry.get("ratio_std", None),
                    "n_trials": entry.get("n_trials", 0),
                    "trial_ratios": [t["ratio"] for t in entry.get("trials", [])],
                }
    return result

def rank_conditions(layer_data, metric="v2_coherence"):
    vals = {}
    for cond in CONDITIONS:
        if cond in layer_data and layer_data[cond][metric] is not None:
            vals[cond] = layer_data[cond][metric]
    sorted_conds = sorted(vals.keys(), key=lambda c: vals[c], reverse=True)
    return {c: i + 1 for i, c in enumerate(sorted_conds)}, vals

def bootstrap_ranks(layer_data, n_bootstrap=10000, metric="ratio"):
    trial_key = "trial_ratios"
    cond_trials = {}
    for cond in CONDITIONS:
        if cond in layer_data and layer_data[cond][trial_key]:
            cond_trials[cond] = np.array(layer_data[cond][trial_key])

    if len(cond_trials) < 2:
        return {}

    n_trials = min(len(v) for v in cond_trials.values())
    rank_counts = {c: np.zeros(len(cond_trials)) for c in cond_trials}

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        means = {}
        for cond, trials in cond_trials.items():
            idx = rng.integers(0, len(trials), n_trials)
            means[cond] = np.mean(trials[idx])
        sorted_conds = sorted(means.keys(), key=lambda c: means[c], reverse=True)
        for rank, cond in enumerate(sorted_conds):
            rank_counts[cond][rank] += 1

    return {c: counts / n_bootstrap for c, counts in rank_counts.items()}

def displacement_test(base_data, instruct_data, layer, condition="relational"):
    if layer not in base_data or layer not in instruct_data:
        return None
    if condition not in base_data[layer] or condition not in instruct_data[layer]:
        return None

    base_trials = base_data[layer][condition]["trial_ratios"]
    inst_trials = instruct_data[layer][condition]["trial_ratios"]

    if not base_trials or not inst_trials:
        return None

    base_arr = np.array(base_trials)
    inst_arr = np.array(inst_trials)
    diff = np.mean(inst_arr) - np.mean(base_arr)
    pooled_std = np.sqrt(np.var(base_arr) / len(base_arr) + np.var(inst_arr) / len(inst_arr))
    if pooled_std == 0:
        return None
    z = diff / pooled_std
    return {"diff": float(diff), "z": float(z), "base_mean": float(np.mean(base_arr)),
            "inst_mean": float(np.mean(inst_arr)), "base_std": float(np.std(base_arr)),
            "inst_std": float(np.std(inst_arr))}


def main():
    print("=" * 70)
    print("V₂ COHERENCE TRAJECTORY ANALYSIS — RELAY DISPLACEMENT")
    print("=" * 70)

    instruct_shallow = load_results("results_groove_five_conditions.json")
    instruct_matched = load_results("results_groove_five_mistral_instruct_matched.json")
    instruct_L30 = load_results("results_groove_five_instruct_identity_L30.json")
    base_shallow = load_results("results_groove_five_mistral_base.json")
    base_deep = load_results("results_groove_five_base_deep.json")
    neutral_instruct = load_results("results_groove_five_neutral_probes_instruct.json")
    neutral_base = load_results("results_groove_five_neutral_probes.json")
    neutral_instruct_deep = load_results("results_groove_five_neutral_instruct_deep.json")
    neutral_base_deep = load_results("results_groove_five_neutral_base_deep.json")

    datasets = {
        "instruct_identity": [],
        "base_identity": [],
        "instruct_neutral": [],
        "base_neutral": [],
    }

    for name, sources in [
        ("instruct_identity", [instruct_shallow, instruct_matched, instruct_L30]),
        ("base_identity", [base_shallow, base_deep]),
        ("instruct_neutral", [neutral_instruct, neutral_instruct_deep]),
        ("base_neutral", [neutral_base, neutral_base_deep]),
    ]:
        merged = {}
        for src in sources:
            if src is None:
                continue
            extracted = extract_coherence_by_layer(src)
            for layer, data in extracted.items():
                if layer not in merged:
                    merged[layer] = data
                else:
                    for cond, vals in data.items():
                        if cond not in merged[layer]:
                            merged[layer][cond] = vals
        datasets[name] = merged

    # --- SECTION 1: Rank trajectories ---
    for dataset_name, data in datasets.items():
        if not data:
            continue
        layers = sorted(data.keys())
        print(f"\n{'─' * 50}")
        print(f"  {dataset_name.upper()} — Rank Trajectory (by σ₂/σ₁ ratio)")
        print(f"{'─' * 50}")
        print(f"  {'Layer':<8}", end="")
        for cond in CONDITIONS:
            print(f"{cond[:6]:>10}", end="")
        print()

        for layer in layers:
            ranks, vals = rank_conditions(data[layer], metric="ratio_mean")
            print(f"  L{layer:<6}", end="")
            for cond in CONDITIONS:
                if cond in ranks:
                    print(f"  {ranks[cond]}({vals[cond]:.4f})" if cond in vals else f"  {ranks[cond]}(---)", end="")
                else:
                    print(f"{'---':>10}", end="")
            print()

    # --- SECTION 2: V₂ coherence trajectories ---
    for dataset_name, data in datasets.items():
        if not data:
            continue
        layers = sorted(data.keys())
        print(f"\n{'─' * 50}")
        print(f"  {dataset_name.upper()} — V₂ Coherence Trajectory")
        print(f"{'─' * 50}")
        print(f"  {'Layer':<8}", end="")
        for cond in CONDITIONS:
            print(f"{cond[:6]:>10}", end="")
        print()

        for layer in layers:
            ranks, vals = rank_conditions(data[layer], metric="v2_coherence")
            print(f"  L{layer:<6}", end="")
            for cond in CONDITIONS:
                if cond in data[layer] and data[layer][cond]["v2_coherence"] is not None:
                    v = data[layer][cond]["v2_coherence"]
                    r = ranks.get(cond, "?")
                    print(f"  {r}({v:.3f})", end="")
                else:
                    print(f"{'---':>10}", end="")
            print()

    # --- SECTION 3: Bootstrap rank confidence ---
    print(f"\n{'=' * 70}")
    print("BOOTSTRAP RANK CONFIDENCE (10k resamples, σ₂/σ₁ ratio)")
    print(f"{'=' * 70}")

    for dataset_name in ["instruct_identity", "base_identity"]:
        data = datasets[dataset_name]
        if not data:
            continue
        key_layers = [l for l in [20, 22, 24, 28, 30] if l in data]
        for layer in key_layers:
            rank_probs = bootstrap_ranks(data[layer])
            if not rank_probs:
                continue
            print(f"\n  {dataset_name} L{layer}:")
            for cond in CONDITIONS:
                if cond in rank_probs:
                    probs = rank_probs[cond]
                    modal_rank = np.argmax(probs) + 1
                    conf = probs[modal_rank - 1]
                    ranks_str = " ".join(f"R{i+1}:{p:.1%}" for i, p in enumerate(probs) if p > 0.01)
                    print(f"    {cond:>14}: modal={modal_rank} ({conf:.1%})  [{ranks_str}]")

    # --- SECTION 4: Displacement test ---
    print(f"\n{'=' * 70}")
    print("DISPLACEMENT TEST (base vs instruct, relational condition)")
    print(f"{'=' * 70}")

    base_data = datasets["base_identity"]
    inst_data = datasets["instruct_identity"]
    test_layers = [l for l in [20, 22, 24, 28, 30] if l in base_data and l in inst_data]

    for layer in test_layers:
        for cond in CONDITIONS:
            result = displacement_test(base_data, inst_data, layer, cond)
            if result:
                sig = "***" if abs(result["z"]) > 3.29 else "**" if abs(result["z"]) > 2.58 else "*" if abs(result["z"]) > 1.96 else ""
                print(f"  L{layer} {cond:>14}: base={result['base_mean']:.4f}±{result['base_std']:.4f}  "
                      f"inst={result['inst_mean']:.4f}±{result['inst_std']:.4f}  "
                      f"diff={result['diff']:+.4f}  z={result['z']:+.2f} {sig}")

    # --- SECTION 5: Trajectory stability (L28→L30) ---
    print(f"\n{'=' * 70}")
    print("TRAJECTORY STABILITY: L28 → L30")
    print(f"{'=' * 70}")

    for dataset_name in ["instruct_identity", "base_identity"]:
        data = datasets[dataset_name]
        if 28 not in data or 30 not in data:
            print(f"  {dataset_name}: L28 or L30 data missing")
            continue

        print(f"\n  {dataset_name}:")
        ranks_28, vals_28 = rank_conditions(data[28], metric="v2_coherence")
        ranks_30, vals_30 = rank_conditions(data[30], metric="v2_coherence")

        for cond in CONDITIONS:
            r28 = ranks_28.get(cond, "?")
            r30 = ranks_30.get(cond, "?")
            v28 = vals_28.get(cond, 0)
            v30 = vals_30.get(cond, 0)
            stability = "HOLDS" if r28 == r30 else f"SHIFTS {r28}→{r30}"
            print(f"    {cond:>14}: L28=rank {r28} ({v28:.3f}) → L30=rank {r30} ({v30:.3f})  [{stability}]")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
