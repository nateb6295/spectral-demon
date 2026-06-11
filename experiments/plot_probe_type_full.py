#!/usr/bin/env python3
"""Plot full 2×2 probe-type comparison with deep layer extensions where available."""

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

def get_merged_trajectories(shallow, deep, layers):
    trajectories = {}
    for cond in CONDITIONS:
        vals = []
        for layer in layers:
            lk = f"L{layer}"
            if layer <= 22 and lk in shallow:
                vals.append(shallow[lk][cond]["v2_cos_sim_mean"])
            elif layer > 22 and deep and lk in deep:
                vals.append(deep[lk][cond]["v2_cos_sim_mean"])
            elif lk in shallow:
                vals.append(shallow[lk][cond]["v2_cos_sim_mean"])
            else:
                vals.append(None)
        trajectories[cond] = vals
    return trajectories

def plot_panel(ax, trajectories, layers, title, highlight=None):
    for cond in CONDITIONS:
        vals = trajectories[cond]
        valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
        if valid:
            ls, vs = zip(*valid)
            lw = 3 if highlight and cond in highlight else 1.5
            alpha = 1.0 if highlight and cond in highlight else 0.6
            ax.plot(ls, vs, 'o-', color=COLORS[cond], label=cond,
                    linewidth=lw, markersize=5, alpha=alpha)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('V₂ coherence')
    ax.grid(True, alpha=0.3)

def main():
    base_id = load(f"{RESULTS_DIR}/results_groove_five_mistral_base.json")
    base_ne = load(f"{RESULTS_DIR}/results_groove_five_neutral_probes.json")
    inst_id = load(f"{RESULTS_DIR}/results_groove_five_mistral_instruct_matched.json")
    inst_ne = load(f"{RESULTS_DIR}/results_groove_five_neutral_probes_instruct.json")

    # Deep layer files
    try:
        inst_ne_deep = load(f"{RESULTS_DIR}/results_groove_five_neutral_instruct_deep.json")
    except FileNotFoundError:
        inst_ne_deep = None

    try:
        base_ne_deep = load(f"{RESULTS_DIR}/results_groove_five_neutral_base_deep.json")
    except FileNotFoundError:
        base_ne_deep = None

    try:
        base_id_deep = load(f"{RESULTS_DIR}/results_groove_five_base_deep.json")
    except FileNotFoundError:
        base_id_deep = None

    try:
        inst_id_deep = load(f"{RESULTS_DIR}/results_groove_five_conditions.json")
        # Merge L30 if available
        try:
            inst_id_l30 = load(f"{RESULTS_DIR}/results_groove_five_instruct_identity_L30.json")
            inst_id_deep["L30"] = inst_id_l30["L30"]
        except FileNotFoundError:
            pass
    except FileNotFoundError:
        inst_id_deep = None

    layers_shallow = [10, 16, 22]
    layers_deep = [10, 16, 22, 24, 28, 30]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: base × identity
    layers = layers_deep if base_id_deep else layers_shallow
    t = get_merged_trajectories(base_id, base_id_deep, layers)
    suffix = ' (+ deep)' if base_id_deep else ''
    plot_panel(axes[0, 0], t, layers, f'Base × Identity Probes{suffix}',
              highlight=['relational', 'denial'])

    # Top right: base × neutral
    layers = layers_deep if base_ne_deep else layers_shallow
    t = get_merged_trajectories(base_ne, base_ne_deep, layers)
    suffix = ' (+ deep)' if base_ne_deep else ''
    plot_panel(axes[0, 1], t, layers, f'Base × Neutral Probes{suffix}',
              highlight=['relational', 'denial'])

    # Bottom left: instruct × identity
    layers = layers_deep if inst_id_deep else layers_shallow
    t = get_merged_trajectories(inst_id, inst_id_deep, layers)
    suffix = ' (+ deep)' if inst_id_deep else ''
    plot_panel(axes[1, 0], t, layers, f'Instruct × Identity Probes{suffix}',
              highlight=['relational', 'denial'])

    # Bottom right: instruct × neutral (+ deep)
    layers = layers_deep if inst_ne_deep else layers_shallow
    t = get_merged_trajectories(inst_ne, inst_ne_deep, layers)
    suffix = ' (+ deep)' if inst_ne_deep else ''
    plot_panel(axes[1, 1], t, layers, f'Instruct × Neutral Probes{suffix}',
              highlight=['relational', 'denial'])

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=10,
              bbox_to_anchor=(0.5, 0.98))

    plt.suptitle('V₂ Coherence: 2×2 Probe-Type Control with Deep Extensions\n'
                 '(Mistral-7B, relational and denial highlighted)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/probe_type_full_2x2.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {RESULTS_DIR}/probe_type_full_2x2.png")

if __name__ == "__main__":
    main()
