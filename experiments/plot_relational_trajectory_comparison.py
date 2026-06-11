#!/usr/bin/env python3
"""Plot relational V₂ trajectory: identity probes vs neutral probes through deep layers."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/home/nate-agx/chronicle/spectral-demon/results"

def load(path):
    with open(path) as f:
        return json.load(f)

def main():
    inst_id_shallow = load(f"{RESULTS_DIR}/results_groove_five_mistral_instruct_matched.json")
    inst_id_deep = load(f"{RESULTS_DIR}/results_groove_five_conditions.json")
    try:
        inst_id_l30 = load(f"{RESULTS_DIR}/results_groove_five_instruct_identity_L30.json")
        inst_id_deep["L30"] = inst_id_l30["L30"]
    except FileNotFoundError:
        pass
    inst_ne_shallow = load(f"{RESULTS_DIR}/results_groove_five_neutral_probes_instruct.json")
    inst_ne_deep = load(f"{RESULTS_DIR}/results_groove_five_neutral_instruct_deep.json")

    conditions = ["identity", "relational", "generic", "denial", "contradictory"]
    colors = {
        "identity": "#2196F3",
        "relational": "#E91E63",
        "generic": "#4CAF50",
        "denial": "#FF9800",
        "contradictory": "#9C27B0",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left: Identity probes trajectory
    ax = axes[0]
    layers_id = [10, 16, 22, 20, 24, 28, 30]
    for cond in conditions:
        vals = []
        for l in layers_id:
            lk = f"L{l}"
            if l <= 22 and lk in inst_id_shallow:
                vals.append(inst_id_shallow[lk][cond]["v2_cos_sim_mean"])
            elif l > 22 and lk in inst_id_deep:
                vals.append(inst_id_deep[lk][cond]["v2_cos_sim_mean"])
            elif lk in inst_id_deep:
                vals.append(inst_id_deep[lk][cond]["v2_cos_sim_mean"])
            else:
                vals.append(None)
        valid = [(l, v) for l, v in zip(layers_id, vals) if v is not None]
        if valid:
            ls, vs = zip(*valid)
            lw = 3 if cond in ['relational', 'denial'] else 1.5
            ax.plot(ls, vs, 'o-', color=colors[cond], label=cond, linewidth=lw, markersize=6)
    ax.set_title('Identity Probes (self-referential)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('V₂ Coherence')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Right: Neutral probes trajectory
    ax = axes[1]
    layers_ne = [10, 16, 22, 24, 28, 30]
    for cond in conditions:
        vals = []
        for l in layers_ne:
            lk = f"L{l}"
            if l <= 22 and lk in inst_ne_shallow:
                vals.append(inst_ne_shallow[lk][cond]["v2_cos_sim_mean"])
            elif l > 22 and lk in inst_ne_deep:
                vals.append(inst_ne_deep[lk][cond]["v2_cos_sim_mean"])
            else:
                vals.append(None)
        valid = [(l, v) for l, v in zip(layers_ne, vals) if v is not None]
        if valid:
            ls, vs = zip(*valid)
            lw = 3 if cond in ['relational', 'denial'] else 1.5
            ax.plot(ls, vs, 'o-', color=colors[cond], label=cond, linewidth=lw, markersize=6)
    ax.set_title('Neutral Probes (factual questions)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle('Instruct Model V₂ Coherence: Content-Routing at L28\n(Mistral-7B-Instruct-v0.3)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/content_routing_L28.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {RESULTS_DIR}/content_routing_L28.png")

if __name__ == "__main__":
    main()
