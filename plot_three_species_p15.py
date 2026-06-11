#!/usr/bin/env python3
"""Plot three-species P15 comparison: P2 disruption bar chart + dose-response curve."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent / "results"

def load_summary(path):
    with open(path) as f:
        d = json.load(f)
    return d

def plot_three_species():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    species = {
        "Potter\n(Gemma 27B)": BASE / "exp_selfref_vs_relational_20260606_0211.json",
        "Goldsmith\n(Mistral 7B)": BASE / "exp_selfref_vs_relational_mistral_20260611_1154.json",
        "Painter\n(Phi 3.5)": BASE / "exp_selfref_vs_relational_phi_20260611_1202.json",
    }

    colors_r = "#2196F3"
    colors_s = "#FF9800"

    # Panel 1: P2 disruption (log scale)
    ax = axes[0]
    names = list(species.keys())
    p2_r = []
    p2_s = []
    for name, path in species.items():
        d = load_summary(path)
        p2_r.append(d["summary"]["relational"]["p2_disruption"])
        p2_s.append(d["summary"]["self_ref"]["p2_disruption"])

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, p2_r, w, label="Relational", color=colors_r, alpha=0.8)
    ax.bar(x + w/2, p2_s, w, label="Self-ref", color=colors_s, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("P2 Disruption (log)")
    ax.set_title("Preamble Removal Disruption")
    ax.legend(fontsize=8)

    for i, (r, s) in enumerate(zip(p2_r, p2_s)):
        ax.annotate(f"{r:.4f}", (i - w/2, r), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{s:.4f}", (i + w/2, s), ha="center", va="bottom", fontsize=7)

    # Panel 2: S/R ratio
    ax = axes[1]
    ratios = [s/r if r > 0 else 0 for r, s in zip(p2_r, p2_s)]
    colors_bar = [colors_r if r < 1 else "#F44336" for r in ratios]
    bars = ax.bar(x, ratios, 0.5, color=colors_bar, alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("S/R Disruption Ratio")
    ax.set_title("Self-ref / Relational Ratio")
    ax.set_ylim(0.6, 1.5)

    for i, r in enumerate(ratios):
        label = f"{r:.3f}"
        if r > 1:
            label += "\n(reversed)"
        ax.annotate(label, (i, r), ha="center", va="bottom", fontsize=8)

    # Panel 3: Tunnel erank
    ax = axes[2]
    erank_r = []
    erank_s = []
    for name, path in species.items():
        d = load_summary(path)
        rel_entries = d.get("relational", [])
        sr_entries = d.get("self_ref", [])
        rel_eranks = [e.get("tunnel_erank", 0) for e in rel_entries if e.get("tunnel_erank") is not None]
        sr_eranks = [e.get("tunnel_erank", 0) for e in sr_entries if e.get("tunnel_erank") is not None]
        erank_r.append(max(rel_eranks) if rel_eranks else 0)
        erank_s.append(max(sr_eranks) if sr_eranks else 0)

    ax.bar(x - w/2, erank_r, w, label="Relational", color=colors_r, alpha=0.8)
    ax.bar(x + w/2, erank_s, w, label="Self-ref", color=colors_s, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Max Tunnel Erank")
    ax.set_title("Spectral Complexity Ceiling")
    ax.legend(fontsize=8)

    for i, (r, s) in enumerate(zip(erank_r, erank_s)):
        if r > 0:
            ax.annotate(f"{r:.0f}", (i - w/2, r), ha="center", va="bottom", fontsize=7)
        if s > 0:
            ax.annotate(f"{s:.0f}", (i + w/2, s), ha="center", va="bottom", fontsize=7)

    fig.suptitle("F124: Three Identity Strategies — Cross-Architecture P15", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = BASE / "three_species_p15_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

if __name__ == "__main__":
    plot_three_species()
