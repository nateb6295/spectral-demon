#!/usr/bin/env python3
"""Visualize γ spectrum shape evolution across depth."""

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
    with open(RESULTS_DIR / "gamma_spectrum_shape.json") as f:
        data = json.load(f)

    layers = sorted(int(k[1:]) for k in data.keys())
    skews = [data[f'L{l}']['shape']['skewness'] for l in layers]
    kurts = [data[f'L{l}']['shape']['kurtosis'] for l in layers]
    ashman = [data[f'L{l}']['bimodality']['ashman_D'] for l in layers]
    dips = [data[f'L{l}']['bimodality']['dip_statistic'] for l in layers]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    zone_colors = []
    for l in layers:
        z = data[f'L{l}']['zone']
        zone_colors.append({'early': '#FFB74D', 'tunnel': '#2196F3',
                           'transition': '#9C27B0', 'relay': '#4CAF50'}[z])

    axes[0].bar(layers, skews, color=zone_colors, alpha=0.8, width=1.8)
    axes[0].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_ylabel('Skewness')
    axes[0].set_title('γ Distribution Shape Across Depth (Qwen2.5-3B)')
    axes[0].annotate('Right-skewed:\nfew channels dominate',
                     xy=(2, 3.5), fontsize=8, ha='center')
    axes[0].annotate('Left-skewed:\nlow-γ channels\noutnumber',
                     xy=(25, -1.0), fontsize=8, ha='center')

    axes[1].bar(layers, ashman, color=zone_colors, alpha=0.8, width=1.8)
    axes[1].axhline(y=2.0, color='red', linewidth=1.5, linestyle='--',
                    label='Bimodality threshold (D=2)')
    axes[1].set_ylabel("Ashman's D")
    axes[1].legend(fontsize=8)
    axes[1].annotate('TWO POPULATIONS', xy=(18, 2.8), fontsize=9,
                     ha='center', fontweight='bold', color='red')

    axes[2].bar(layers, kurts, color=zone_colors, alpha=0.8, width=1.8)
    axes[2].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    axes[2].set_ylabel('Excess Kurtosis')
    axes[2].set_xlabel('Layer')
    axes[2].annotate('Leptokurtic:\nheavy tails', xy=(0, 22), fontsize=8, ha='center')
    axes[2].annotate('Platykurtic:\nuniform-ish', xy=(18, -2), fontsize=8, ha='center')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FFB74D', label='Early'),
                       Patch(facecolor='#2196F3', label='Tunnel'),
                       Patch(facecolor='#9C27B0', label='Transition'),
                       Patch(facecolor='#4CAF50', label='Relay')]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "gamma_spectrum_evolution.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
