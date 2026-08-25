#!/usr/bin/env python3
"""Five-condition V₂ coherence trajectories across relay zone.

Panel A: Mistral (sorting — differentiate→reconverge with reversal)
Panel B: Gemma (compression — tight spread, no reversal)
Panel C: Qwen (sharpening — identity dominant throughout) [if data available]

Run after experiment results are copied to spectral-demon/results/.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = os.path.expanduser("~/chronicle/spectral-demon/results")

COLORS = {
    "identity": "#2196F3",
    "relational": "#FF5722",
    "generic": "#9E9E9E",
    "denial": "#673AB7",
    "contradictory": "#FF9800",
}

CONDITION_ORDER = ["identity", "relational", "generic", "denial", "contradictory"]

def load_data():
    data = {}

    mistral_path = os.path.join(RESULTS_DIR, "results_groove_five_conditions.json")
    if os.path.exists(mistral_path):
        with open(mistral_path) as f:
            raw = json.load(f)
        data["Mistral-7B\n(MHA — Sorting)"] = {
            "layers": [20, 24, 28],
            "results": raw,
        }

    crossarch_path = os.path.join(RESULTS_DIR, "results_groove_five_conditions_crossarch.json")
    if os.path.exists(crossarch_path):
        with open(crossarch_path) as f:
            raw = json.load(f)
        if "gemma" in raw:
            data["Gemma-2-9B\n(GQA — Equalizing)"] = {
                "layers": [18, 24, 30],
                "results": raw["gemma"],
            }
        if "qwen" in raw:
            data["Qwen-2.5-7B\n(GQA — Selection)"] = {
                "layers": [10, 16, 22],
                "results": raw["qwen"],
            }

    return data


def plot(data):
    n_panels = len(data)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (title, d) in zip(axes, data.items()):
        layers = d["layers"]
        results = d["results"]

        for cond in CONDITION_ORDER:
            v2_values = []
            for l in layers:
                key = f"L{l}"
                if key in results and cond in results[key]:
                    v2_values.append(results[key][cond]["v2_cos_sim_mean"])
                else:
                    v2_values.append(np.nan)

            ax.plot(layers, v2_values, 'o-', color=COLORS[cond], label=cond,
                    linewidth=2, markersize=6, alpha=0.85)

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("V₂ Coherence\n(cross-trial cosine similarity)", fontsize=10)
    axes[-1].legend(loc='upper left', fontsize=8, framealpha=0.9)

    fig.suptitle("Five-Condition V₂ Coherence Through the Relay Zone",
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "fig_five_condition_trajectories.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    data = load_data()
    if not data:
        print("No data found. Run experiments first.")
    else:
        print(f"Found data for: {list(data.keys())}")
        plot(data)
