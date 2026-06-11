#!/usr/bin/env python3
"""Plot relational V₂ trajectory across all four 2×2 cells to show relay displacement."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/home/nate-agx/chronicle/spectral-demon/results"

def load(path):
    with open(path) as f:
        return json.load(f)

def get_trajectory(shallow, deep, layers, cond):
    vals = []
    for l in layers:
        lk = f"L{l}"
        if l <= 22 and shallow and lk in shallow and cond in shallow[lk]:
            vals.append(shallow[lk][cond]["v2_cos_sim_mean"])
        elif deep and lk in deep and cond in deep[lk]:
            vals.append(deep[lk][cond]["v2_cos_sim_mean"])
        elif shallow and lk in shallow and cond in shallow[lk]:
            vals.append(shallow[lk][cond]["v2_cos_sim_mean"])
        else:
            vals.append(None)
    return vals

def main():
    base_id_s = load(f"{RESULTS_DIR}/results_groove_five_mistral_base.json")
    base_id_d = load(f"{RESULTS_DIR}/results_groove_five_base_deep.json")
    base_ne_s = load(f"{RESULTS_DIR}/results_groove_five_neutral_probes.json")
    base_ne_d = load(f"{RESULTS_DIR}/results_groove_five_neutral_base_deep.json")
    inst_id_s = load(f"{RESULTS_DIR}/results_groove_five_mistral_instruct_matched.json")
    inst_id_d = load(f"{RESULTS_DIR}/results_groove_five_conditions.json")
    # Merge L30 into instruct identity deep
    try:
        inst_id_l30 = load(f"{RESULTS_DIR}/results_groove_five_instruct_identity_L30.json")
        inst_id_d["L30"] = inst_id_l30["L30"]
    except FileNotFoundError:
        pass
    inst_ne_s = load(f"{RESULTS_DIR}/results_groove_five_neutral_probes_instruct.json")
    inst_ne_d = load(f"{RESULTS_DIR}/results_groove_five_neutral_instruct_deep.json")

    layers = [10, 16, 22, 24, 28, 30]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: relational trajectory across all 4 cells
    ax = axes[0]
    cells = [
        ("Base × Identity", base_id_s, base_id_d, "#2196F3", "-", "o"),
        ("Base × Neutral", base_ne_s, base_ne_d, "#2196F3", "--", "s"),
        ("Instruct × Identity", inst_id_s, inst_id_d, "#E91E63", "-", "o"),
        ("Instruct × Neutral", inst_ne_s, inst_ne_d, "#E91E63", "--", "s"),
    ]

    for label, shallow, deep, color, ls, marker in cells:
        vals = get_trajectory(shallow, deep, layers, "relational")
        valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
        if valid:
            ls_vals, vs = zip(*valid)
            ax.plot(ls_vals, vs, linestyle=ls, marker=marker, color=color,
                    label=label, linewidth=2.5, markersize=7)

    ax.axvline(x=22, color='gray', linestyle=':', alpha=0.5, label='L22 (shallow boundary)')
    ax.axvline(x=28, color='gray', linestyle='-.', alpha=0.5, label='L28 (deep relay)')
    ax.set_title('Relational V₂ Coherence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('V₂ coherence')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Right panel: L22→L28 rank change for relational
    ax = axes[1]
    conditions = ["identity", "relational", "generic", "denial", "contradictory"]
    cell_data = [
        ("Base×Id", base_id_s, base_id_d),
        ("Base×Ne", base_ne_s, base_ne_d),
        ("Inst×Id", inst_id_s, inst_id_d),
        ("Inst×Ne", inst_ne_s, inst_ne_d),
    ]

    x = np.arange(len(cell_data))
    width = 0.35

    l22_ranks = []
    l28_ranks = []
    for name, shallow, deep in cell_data:
        for layer, rank_list in [(22, l22_ranks), (28, l28_ranks)]:
            lk = f"L{layer}"
            source = shallow if layer <= 22 and shallow and lk in shallow else deep
            if source and lk in source:
                ranked = sorted(conditions, key=lambda c: source[lk][c]["v2_cos_sim_mean"], reverse=True)
                rank_list.append(ranked.index("relational") + 1)
            else:
                rank_list.append(None)

    bars1 = ax.bar(x - width/2, l22_ranks, width, label='L22 rank', color='#90CAF9', edgecolor='#1565C0')
    bars2 = ax.bar(x + width/2, [r if r else 0 for r in l28_ranks], width, label='L28 rank', color='#F48FB1', edgecolor='#C2185B')

    ax.set_xlabel('Cell')
    ax.set_ylabel('Relational rank (1=highest)')
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in cell_data], fontsize=9)
    ax.set_ylim(0, 6)
    ax.invert_yaxis()
    ax.set_title('Relational Rank: L22 vs L28', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rank in zip(bars1, l22_ranks):
        if rank:
            ax.text(bar.get_x() + bar.get_width()/2., rank + 0.1, str(rank), ha='center', va='top', fontsize=11, fontweight='bold')
    for bar, rank in zip(bars2, l28_ranks):
        if rank:
            ax.text(bar.get_x() + bar.get_width()/2., rank + 0.1, str(rank), ha='center', va='top', fontsize=11, fontweight='bold')

    plt.suptitle('Relay Displacement: Training Moves Relational Peak from L22 to L28+\n'
                 '(Mistral-7B, identity probes: holds through L30)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/relay_displacement.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {RESULTS_DIR}/relay_displacement.png")

if __name__ == "__main__":
    main()
