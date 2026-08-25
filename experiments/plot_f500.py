#!/usr/bin/env python3
"""Plot F500 trajectory-divergent epsilon_c results.

Usage:
    python3 plot_f500.py /root/results/f500_trajectory_divergent.json
    python3 plot_f500.py  # reads from default path
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_PATH = "/root/results/f500_trajectory_divergent.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def load_data(path):
    with open(path) as f:
        return json.load(f)


def ec_val(v):
    """Convert epsilon_c to numeric, using 5.5 for >5.0x (censored)."""
    if v is None:
        return 5.5
    return v


def plot_vulnerability_heatmap(species_data, species_name, outdir):
    """Heatmap of epsilon_c across (KV_group × layer_band) for each condition."""
    bands = ["early", "mid", "late"]
    n_groups = species_data["n_groups"]
    groups = [f"KV{i}" for i in range(n_groups)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, cond in enumerate(["trajectory", "generic"]):
        cond_data = species_data["conditions"].get(cond)
        if not cond_data:
            axes[idx].text(0.5, 0.5, f"No {cond} data yet", ha="center", va="center",
                           transform=axes[idx].transAxes, fontsize=14)
            axes[idx].set_title(f"{cond.upper()}")
            continue

        matrix = np.zeros((n_groups, len(bands)))
        for bi, band in enumerate(bands):
            band_data = cond_data["bands"].get(band, {}).get("groups", {})
            for gi in range(n_groups):
                ec = band_data.get(f"kv{gi}", {}).get("epsilon_c")
                matrix[gi, bi] = ec_val(ec)

        im = axes[idx].imshow(matrix, cmap="RdYlGn", vmin=1.5, vmax=5.5,
                              aspect="auto", interpolation="nearest")
        axes[idx].set_xticks(range(len(bands)))
        axes[idx].set_xticklabels([f"{b}\n{species_data['layer_bands'][b]}" for b in bands])
        axes[idx].set_yticks(range(n_groups))
        axes[idx].set_yticklabels(groups)
        axes[idx].set_title(f"{cond.upper()}", fontsize=13, fontweight="bold")

        for gi in range(n_groups):
            for bi in range(len(bands)):
                val = matrix[gi, bi]
                txt = f">{5.0:.1f}" if val > 5.0 else f"{val:.2f}"
                color = "white" if val < 2.5 else "black"
                axes[idx].text(bi, gi, txt, ha="center", va="center",
                              fontsize=9, color=color, fontweight="bold")

    axes[0].set_ylabel("KV Group")
    fig.suptitle(f"F500 — {species_name} Vulnerability (ε_c)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=axes, label="ε_c (×baseline)", shrink=0.8)
    plt.tight_layout()
    outpath = outdir / f"f500_{species_name.lower()}_heatmap.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


def plot_trajectory_dependence(species_data, species_name, outdir):
    """Bar chart of Δε_c (trajectory - generic) per group per band."""
    td = species_data.get("trajectory_dependence")
    if not td:
        print(f"  No trajectory-dependence data for {species_name}")
        return

    bands = ["early", "mid", "late"]
    n_groups = species_data["n_groups"]
    groups = [f"KV{i}" for i in range(n_groups)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for bi, band in enumerate(bands):
        band_td = td.get(band, {})
        deltas = []
        colors = []
        for gi in range(n_groups):
            d = band_td.get(f"kv{gi}", {}).get("delta")
            if d is None:
                deltas.append(0)
                colors.append("gray")
            else:
                deltas.append(d)
                colors.append("steelblue" if d > 0 else "coral")

        axes[bi].bar(range(n_groups), deltas, color=colors, edgecolor="black", linewidth=0.5)
        axes[bi].axhline(0, color="black", linewidth=0.8)
        axes[bi].axhline(0.25, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[bi].axhline(-0.25, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[bi].set_xticks(range(n_groups))
        axes[bi].set_xticklabels(groups, rotation=45)
        axes[bi].set_title(f"{band.upper()}", fontsize=12, fontweight="bold")
        axes[bi].set_xlabel("KV Group")

    axes[0].set_ylabel("Δε_c (trajectory - generic)")
    fig.suptitle(f"F500 — {species_name} Trajectory Dependence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    outpath = outdir / f"f500_{species_name.lower()}_trajectory_dep.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


def plot_migration(species_data, species_name, outdir):
    """Line plot showing epsilon_c migration across depth for each KV group."""
    bands = ["early", "mid", "late"]
    n_groups = species_data["n_groups"]
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in ["trajectory", "generic"]:
        cond_data = species_data["conditions"].get(cond)
        if not cond_data:
            continue
        style = "-o" if cond == "trajectory" else "--s"
        for gi in range(n_groups):
            vals = []
            for band in bands:
                ec = cond_data["bands"].get(band, {}).get("groups", {}).get(f"kv{gi}", {}).get("epsilon_c")
                vals.append(ec_val(ec))
            label = f"KV{gi} ({cond[:4]})" if cond == "trajectory" else None
            alpha = 0.9 if cond == "trajectory" else 0.4
            ax.plot(range(len(bands)), vals, style, color=cmap(gi), alpha=alpha,
                    linewidth=2, markersize=6, label=label)

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_ylabel("ε_c (×baseline)")
    ax.set_xlabel("Layer Band")
    ax.axhline(5.0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_title(f"F500 — {species_name} Vulnerability Migration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    outpath = outdir / f"f500_{species_name.lower()}_migration.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    data = load_data(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for species in ["llama", "gemma", "qwen"]:
        if species not in data:
            continue
        print(f"\n=== {species.upper()} ===")
        sdata = data[species]
        plot_vulnerability_heatmap(sdata, species.capitalize(), OUT_DIR)
        plot_trajectory_dependence(sdata, species.capitalize(), OUT_DIR)
        plot_migration(sdata, species.capitalize(), OUT_DIR)


if __name__ == "__main__":
    main()
