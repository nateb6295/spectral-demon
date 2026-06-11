#!/usr/bin/env python3
"""Plot 2×2 probe-type comparison: condition trajectories across layers."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/home/nate-agx/chronicle/spectral-demon/results"
CONDITIONS = ["identity", "relational", "generic", "denial", "contradictory"]
COLORS = {
    "identity": "#2196F3",
    "relational": "#E91E63",
    "generic": "#4CAF50",
    "denial": "#FF9800",
    "contradictory": "#9C27B0",
}

def load(path):
    with open(path) as f:
        return json.load(f)

def get_trajectories(data, layers):
    trajectories = {}
    for cond in CONDITIONS:
        vals = []
        for layer in layers:
            lk = f"L{layer}"
            if lk in data:
                vals.append(data[lk][cond]["v2_cos_sim_mean"])
            else:
                vals.append(None)
        trajectories[cond] = vals
    return trajectories

def plot_panel(ax, trajectories, layers, title):
    for cond in CONDITIONS:
        vals = trajectories[cond]
        valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
        if valid:
            ls, vs = zip(*valid)
            ax.plot(ls, vs, 'o-', color=COLORS[cond], label=cond, linewidth=2, markersize=5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('V₂ coherence')
    ax.grid(True, alpha=0.3)

def main():
    base_id = load(f"{RESULTS_DIR}/results_groove_five_mistral_base.json")
    base_ne = load(f"{RESULTS_DIR}/results_groove_five_neutral_probes.json")
    inst_id = load(f"{RESULTS_DIR}/results_groove_five_mistral_instruct_matched.json")
    inst_ne = load(f"{RESULTS_DIR}/results_groove_five_neutral_probes_instruct.json")

    # Try to load deep layers
    try:
        inst_ne_deep = load(f"{RESULTS_DIR}/results_groove_five_neutral_instruct_deep.json")
        has_deep = True
    except FileNotFoundError:
        has_deep = False

    layers_shallow = [10, 16, 22]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: base × identity
    t = get_trajectories(base_id, layers_shallow)
    plot_panel(axes[0, 0], t, layers_shallow, 'Base × Identity Probes')

    # Top right: base × neutral
    t = get_trajectories(base_ne, layers_shallow)
    plot_panel(axes[0, 1], t, layers_shallow, 'Base × Neutral Probes')

    # Bottom left: instruct × identity
    t = get_trajectories(inst_id, layers_shallow)
    plot_panel(axes[1, 0], t, layers_shallow, 'Instruct × Identity Probes')

    # Bottom right: instruct × neutral (+ deep if available)
    if has_deep:
        layers_full = [10, 16, 22, 24, 28, 30]
        merged = {}
        for cond in CONDITIONS:
            vals = []
            for layer in layers_full:
                lk = f"L{layer}"
                if layer <= 22 and lk in inst_ne:
                    vals.append(inst_ne[lk][cond]["v2_cos_sim_mean"])
                elif layer > 22 and lk in inst_ne_deep:
                    vals.append(inst_ne_deep[lk][cond]["v2_cos_sim_mean"])
                else:
                    vals.append(None)
            merged[cond] = vals
        plot_panel(axes[1, 1], merged, layers_full, 'Instruct × Neutral Probes (+ deep)')
    else:
        t = get_trajectories(inst_ne, layers_shallow)
        plot_panel(axes[1, 1], t, layers_shallow, 'Instruct × Neutral Probes')

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.98))

    plt.suptitle('V₂ Coherence Trajectories: 2×2 Probe-Type Control\n(Mistral-7B)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/probe_type_2x2.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {RESULTS_DIR}/probe_type_2x2.png")

if __name__ == "__main__":
    main()
