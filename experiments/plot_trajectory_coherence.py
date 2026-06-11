#!/usr/bin/env python3
"""Visualize V₂ coherence rank trajectories with bootstrap CIs.

Two panels:
  Left: Rank trajectory across layers for each condition (instruct vs base)
  Right: Bootstrap rank probability at key layers (L22, L28)

Uses data from trajectory_coherence.py analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path(__file__).parent.parent / "results"
CONDITIONS = ["identity", "relational", "generic", "denial", "contradictory"]
COLORS = {
    "identity": "#2196F3",
    "relational": "#E91E63",
    "generic": "#4CAF50",
    "denial": "#FF9800",
    "contradictory": "#9C27B0",
}
COND_SHORT = {"identity": "ID", "relational": "REL", "generic": "GEN",
              "denial": "DEN", "contradictory": "CON"}

def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def extract_v2_coherence(data):
    result = {}
    for layer_key in data:
        layer_num = int(layer_key.replace("L", ""))
        result[layer_num] = {}
        for cond in CONDITIONS:
            if cond in data[layer_key]:
                entry = data[layer_key][cond]
                result[layer_num][cond] = entry.get("v2_cos_sim_mean", None)
    return result

def extract_trial_ratios(data):
    result = {}
    for layer_key in data:
        layer_num = int(layer_key.replace("L", ""))
        result[layer_num] = {}
        for cond in CONDITIONS:
            if cond in data[layer_key]:
                entry = data[layer_key][cond]
                trials = entry.get("trials", [])
                if trials:
                    result[layer_num][cond] = np.array([t["ratio"] for t in trials])
    return result

def bootstrap_ranks(trial_data, n_bootstrap=10000):
    cond_trials = {c: trial_data[c] for c in CONDITIONS if c in trial_data}
    if len(cond_trials) < 2:
        return {}
    n_trials = min(len(v) for v in cond_trials.values())
    rank_counts = {c: np.zeros(len(cond_trials)) for c in cond_trials}
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        means = {c: np.mean(rng.choice(t, n_trials)) for c, t in cond_trials.items()}
        for rank, cond in enumerate(sorted(means, key=means.get, reverse=True)):
            rank_counts[cond][rank] += 1
    return {c: counts / n_bootstrap for c, counts in rank_counts.items()}

def merge_datasets(*sources):
    merged = {}
    for src in sources:
        if src is None:
            continue
        coh = extract_v2_coherence(src)
        for layer, data in coh.items():
            if layer not in merged:
                merged[layer] = {}
            for cond, val in data.items():
                if cond not in merged[layer] or val is not None:
                    merged[layer][cond] = val
    return merged

def merge_trial_datasets(*sources):
    merged = {}
    for src in sources:
        if src is None:
            continue
        trials = extract_trial_ratios(src)
        for layer, data in trials.items():
            if layer not in merged:
                merged[layer] = {}
            for cond, val in data.items():
                if cond not in merged[layer]:
                    merged[layer][cond] = val
    return merged

def main():
    inst_shallow = load_results("results_groove_five_conditions.json")
    inst_matched = load_results("results_groove_five_mistral_instruct_matched.json")
    inst_L30 = load_results("results_groove_five_instruct_identity_L30.json")
    base_shallow = load_results("results_groove_five_mistral_base.json")
    base_deep = load_results("results_groove_five_base_deep.json")

    inst_coh = merge_datasets(inst_shallow, inst_matched, inst_L30)
    base_coh = merge_datasets(base_shallow, base_deep)
    inst_trials = merge_trial_datasets(inst_shallow, inst_matched, inst_L30)
    base_trials = merge_trial_datasets(base_shallow, base_deep)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("V₂ Coherence Rank Trajectories — Relay Displacement", fontsize=14, fontweight="bold")

    # Panel 1: Instruct rank trajectory
    ax = axes[0]
    inst_layers = sorted(inst_coh.keys())
    for cond in CONDITIONS:
        vals = []
        layers_used = []
        for l in inst_layers:
            if cond in inst_coh[l] and inst_coh[l][cond] is not None:
                sorted_conds = sorted([c for c in CONDITIONS if c in inst_coh[l] and inst_coh[l][c] is not None],
                                      key=lambda c: inst_coh[l][c], reverse=True)
                rank = sorted_conds.index(cond) + 1
                vals.append(rank)
                layers_used.append(l)
        if vals:
            ax.plot(layers_used, vals, 'o-', color=COLORS[cond], label=COND_SHORT[cond],
                    linewidth=2, markersize=6)
    ax.set_ylabel("Rank (1=highest coherence)", fontsize=11)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title("Instruct × Identity Probes", fontsize=12)
    ax.set_ylim(5.5, 0.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper left", fontsize=9)
    ax.axvline(x=28, color="gray", linestyle="--", alpha=0.5, label="L28")
    ax.grid(True, alpha=0.3)

    # Panel 2: Base rank trajectory
    ax = axes[1]
    base_layers = sorted(base_coh.keys())
    for cond in CONDITIONS:
        vals = []
        layers_used = []
        for l in base_layers:
            if cond in base_coh[l] and base_coh[l][cond] is not None:
                sorted_conds = sorted([c for c in CONDITIONS if c in base_coh[l] and base_coh[l][c] is not None],
                                      key=lambda c: base_coh[l][c], reverse=True)
                rank = sorted_conds.index(cond) + 1
                vals.append(rank)
                layers_used.append(l)
        if vals:
            ax.plot(layers_used, vals, 'o-', color=COLORS[cond], label=COND_SHORT[cond],
                    linewidth=2, markersize=6)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title("Base × Identity Probes", fontsize=12)
    ax.set_ylim(5.5, 0.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper left", fontsize=9)
    ax.axvline(x=22, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 3: Bootstrap rank probabilities at L22 (base) and L28 (instruct)
    ax = axes[2]
    bar_width = 0.15
    x_pos = np.arange(5)

    # L28 instruct bootstrap
    if 28 in inst_trials:
        boot_28 = bootstrap_ranks(inst_trials[28])
        for i, cond in enumerate(CONDITIONS):
            if cond in boot_28:
                rank1_prob = boot_28[cond][0]
                ax.bar(i - bar_width/2, rank1_prob, bar_width, color=COLORS[cond],
                       alpha=0.9, label=f"Inst L28" if i == 0 else "")

    # L22 base bootstrap
    if 22 in base_trials:
        boot_22 = bootstrap_ranks(base_trials[22])
        for i, cond in enumerate(CONDITIONS):
            if cond in boot_22:
                rank1_prob = boot_22[cond][0]
                ax.bar(i + bar_width/2, rank1_prob, bar_width, color=COLORS[cond],
                       alpha=0.4, edgecolor=COLORS[cond], linewidth=1.5,
                       label=f"Base L22" if i == 0 else "")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([COND_SHORT[c] for c in CONDITIONS], fontsize=10)
    ax.set_ylabel("P(Rank 1)", fontsize=11)
    ax.set_title("Bootstrap: P(Rank 1)\nSolid=Inst L28, Faded=Base L22", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = RESULTS_DIR / "trajectory_coherence_bootstrap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
