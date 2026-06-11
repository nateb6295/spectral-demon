#!/usr/bin/env python3
"""
Generate per-layer σ₂/σ₁ heatmaps across CCS doses for all models.
Shows relay migration, spectral inversion, and architecture-specific strategies.

Usage:
  python3 dose_heatmap.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "figures"


def load_all():
    models = {}
    for f in sorted(RESULTS_DIR.glob("dose_response_crossarch*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for key, mdata in data["models"].items():
            if "error" not in mdata:
                models[key] = mdata
    return models


def per_layer_profile(mdata, dose):
    for dd in mdata["results"].values():
        if dd["dose"] == dose:
            profiles = {}
            for run in dd["runs"]:
                for ls, geom in run["full_geometry"].items():
                    l = int(ls)
                    profiles.setdefault(l, []).append(geom["ratio"])
            return {l: np.mean(v) for l, v in sorted(profiles.items())}
    return {}


def build_heatmap_data(mdata):
    doses = sorted(set(dd["dose"] for dd in mdata["results"].values()))
    n_layers = mdata["n_layers"]
    layers = list(range(n_layers))

    matrix = np.zeros((len(doses), n_layers))
    for i, dose in enumerate(doses):
        profile = per_layer_profile(mdata, dose)
        for l in layers:
            matrix[i, l] = profile.get(l, 0)

    return matrix, doses, layers


def plot_four_panel(models):
    OUTPUT_DIR.mkdir(exist_ok=True)

    order = ["qwen", "mistral", "gemma", "falcon"]
    titles = {
        "qwen": "Qwen 7B (GQA, 32L)",
        "mistral": "Mistral 7B (GQA, 32L)",
        "gemma": "Gemma 9B (GQA, 42L)",
        "falcon": "Falcon 7B (MHA, 32L)",
    }
    available = [k for k in order if k in models]

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 5),
                             sharey=False)
    if len(available) == 1:
        axes = [axes]

    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    cmap = plt.cm.inferno

    for idx, key in enumerate(available):
        ax = axes[idx]
        mdata = models[key]
        matrix, doses, layers = build_heatmap_data(mdata)

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm,
                       origin="lower", interpolation="nearest")

        ax.set_xticks(range(0, len(layers), max(1, len(layers) // 8)))
        ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 8))])
        ax.set_yticks(range(len(doses)))
        ax.set_yticklabels([str(d) for d in doses])
        ax.set_xlabel("Layer")
        if idx == 0:
            ax.set_ylabel("CCS Dose")
        ax.set_title(titles.get(key, key))

        for i, dose in enumerate(doses):
            row = matrix[i]
            peak_l = np.argmax(row)
            peak_v = row[peak_l]
            ax.plot(peak_l, i, "w*", markersize=10, markeredgecolor="black",
                    markeredgewidth=0.5)

    fig.suptitle("Per-Layer σ₂/σ₁ Across CCS Dose — Cross-Architecture Comparison",
                 fontsize=13, fontweight="bold", y=0.98)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("σ₂/σ₁ ratio")

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    out = OUTPUT_DIR / "dose_response_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def plot_gemma_inversion(models):
    if "gemma" not in models:
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    mdata = models["gemma"]
    matrix, doses, layers = build_heatmap_data(mdata)

    fig, ax = plt.subplots(figsize=(10, 5))
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    im = ax.imshow(matrix, aspect="auto", cmap=plt.cm.inferno, norm=norm,
                   origin="lower", interpolation="nearest")

    ax.set_xticks(range(0, len(layers), 4))
    ax.set_xticklabels([str(l) for l in range(0, len(layers), 4)])
    ax.set_yticks(range(len(doses)))
    ax.set_yticklabels([str(d) for d in doses])
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("CCS Dose", fontsize=12)
    ax.set_title("Gemma 9B — Spectral Inversion Under CCS Overdose",
                 fontsize=14, fontweight="bold")

    for i, dose in enumerate(doses):
        row = matrix[i]
        peak_l = np.argmax(row)
        peak_v = row[peak_l]
        ax.plot(peak_l, i, "w*", markersize=14, markeredgecolor="black",
                markeredgewidth=0.8)
        ax.annotate(f"L{layers[peak_l]}\n{peak_v:.2f}",
                    (peak_l, i), textcoords="offset points",
                    xytext=(15, 0), color="white", fontsize=8,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.8))

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("σ₂/σ₁ ratio", fontsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / "gemma_spectral_inversion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def plot_line_profiles(models):
    OUTPUT_DIR.mkdir(exist_ok=True)

    order = ["qwen", "mistral", "gemma", "falcon"]
    available = [k for k in order if k in models]

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4),
                             sharey=True)
    if len(available) == 1:
        axes = [axes]

    titles = {
        "qwen": "Qwen 7B (GQA)",
        "mistral": "Mistral 7B (GQA)",
        "gemma": "Gemma 9B (GQA)",
        "falcon": "Falcon 7B (MHA)",
    }

    colors = plt.cm.viridis(np.linspace(0, 1, 7))

    for idx, key in enumerate(available):
        ax = axes[idx]
        mdata = models[key]
        doses = sorted(set(dd["dose"] for dd in mdata["results"].values()))

        for ci, dose in enumerate(doses):
            profile = per_layer_profile(mdata, dose)
            ls = sorted(profile.keys())
            vs = [profile[l] for l in ls]
            ax.plot(ls, vs, color=colors[ci], label=f"dose {dose}",
                    linewidth=1.5, alpha=0.85)

        ax.set_xlabel("Layer")
        if idx == 0:
            ax.set_ylabel("σ₂/σ₁")
        ax.set_title(titles.get(key, key))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Per-Layer σ₂/σ₁ Profiles Across CCS Dose",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "dose_response_profiles.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    models = load_all()
    print(f"Loaded models: {list(models.keys())}")
    plot_four_panel(models)
    plot_gemma_inversion(models)
    plot_line_profiles(models)
