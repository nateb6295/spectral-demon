#!/usr/bin/env python3
"""Generate a combined 4-panel wire mechanism summary figure."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    # Load all data
    with open(RESULTS_DIR / "exp_gamma_v2_depth_profile.json") as f:
        depth_data = json.load(f)
    with open(RESULTS_DIR / "gamma_spectrum_shape.json") as f:
        shape_data = json.load(f)
    with open(RESULTS_DIR / "exp_channel_identity.json") as f:
        channel_data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Depth profile (γ-V₂ correlation)
    ax = axes[0, 0]
    layers = sorted(set(v['layer'] for v in depth_data.values()))
    conditions = ['receptive', 'absent', 'control']
    colors = {'receptive': '#2196F3', 'absent': '#F44336', 'control': '#9E9E9E'}
    for cond in conditions:
        rs = [depth_data[f'L{l}_{cond}']['pearson_r'] for l in layers]
        ax.plot(layers, rs, '-o', markersize=2, color=colors[cond],
                label=cond, alpha=0.8, linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.axvspan(13, 13.5, alpha=0.3, color='gold')
    ax.set_ylabel('r(|γ|, |V₂|)')
    ax.set_title('A. γ-V₂ Correlation Depth Profile')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(-0.5, 0.35)
    ax.set_xlabel('Layer')

    # Panel B: Bimodality (Ashman D)
    ax = axes[0, 1]
    shape_layers = sorted(int(k[1:]) for k in shape_data.keys())
    ashman = [shape_data[f'L{l}']['bimodality']['ashman_D'] for l in shape_layers]
    zone_colors = []
    for l in shape_layers:
        z = shape_data[f'L{l}']['zone']
        zone_colors.append({'early': '#FFB74D', 'tunnel': '#2196F3',
                           'transition': '#9C27B0', 'relay': '#4CAF50'}[z])
    ax.bar(shape_layers, ashman, color=zone_colors, alpha=0.8, width=1.8)
    ax.axhline(y=2.0, color='red', linewidth=1.5, linestyle='--', label='D=2 threshold')
    ax.set_ylabel("Ashman's D")
    ax.set_title('B. γ Bimodality: Two Channel Populations')
    ax.legend(fontsize=7)
    ax.set_xlabel('Layer')

    # Panel C: Channel identity Jaccard
    ax = axes[1, 0]
    tunnel_layers = list(range(10, 27))
    jaccards = []
    for i in range(len(tunnel_layers) - 1):
        l1, l2 = tunnel_layers[i], tunnel_layers[i+1]
        key = f"L{l1}_L{l2}"
        if key in channel_data['jaccard_matrix']:
            jaccards.append((l1, channel_data['jaccard_matrix'][key]))

    if jaccards:
        xs, ys = zip(*jaccards)
        bar_colors = ['gold' if x in [12, 13] else '#2196F3' for x in xs]
        ax.bar(xs, ys, color=bar_colors, alpha=0.8, width=0.8)
        ax.axhline(y=0.5, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_ylabel('Jaccard (adjacent layers)')
        ax.set_title('C. Channel Population Persistence')
        ax.set_xlabel('Source Layer')
        ax.annotate('L13\nreshuffle', xy=(12.5, 0.52), fontsize=8,
                    ha='center', va='bottom', color='goldenrod', fontweight='bold')

    # Panel D: V₂ service/highway ratio
    ax = axes[1, 1]
    v2_data = channel_data['v2_crossref']
    test_layers = [13, 16, 18, 20]
    width = 0.25
    for i, cond in enumerate(['receptive', 'absent', 'control']):
        ratios = [v2_data[f'L{l}_{cond}']['ratio_service_highway'] for l in test_layers]
        positions = [j + (i - 1) * width for j in range(len(test_layers))]
        ax.bar(positions, ratios, width=width, color=colors[cond], alpha=0.7, label=cond)

    ax.axhline(y=1.0, color='black', linewidth=1, linestyle='--', label='equal loading')
    ax.set_xticks(range(len(test_layers)))
    ax.set_xticklabels([f'L{l}' for l in test_layers])
    ax.set_ylabel('V₂ service/highway ratio')
    ax.set_title('D. V₂ Preferentially Loads Service Road')
    ax.legend(fontsize=7)
    ax.annotate('L13:\nuncommitted', xy=(0, 1.0), fontsize=8,
                ha='center', va='top', color='goldenrod')
    ax.annotate('2.3× on\nservice road', xy=(2, 2.4), fontsize=8,
                ha='center', va='bottom', color='#2196F3')

    plt.tight_layout()
    out = FIG_DIR / "wire_mechanism_summary.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
