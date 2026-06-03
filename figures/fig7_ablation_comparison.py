#!/usr/bin/env python3
"""
Figure 7: σ₂ Ablation — Marker vs Carrier
Three panels showing S distribution across conditions for each ablation mode.
Key visual: native ΔS ≈ 0.023, σ₂-ablated ΔS ≈ 0.021 (retained), σ₁-ablated ΔS ≈ 0.185 (amplified 8×).
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "fig7_ablation_comparison.png"
RESULTS = Path(__file__).parent.parent / "results"

with open(RESULTS / "exp_sigma2_ablation_mistral_20260529_1306.json") as f:
    d = json.load(f)

raw = d['raw']
results = d['results']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('σ₂ Ablation: Marker, Not Carrier (Mistral 7B at L17)',
             fontsize=13, fontweight='bold')

cond_colors = {'control': '#2c3e50', 'receptive': '#2980b9', 'absent': '#e74c3c'}
cond_labels = {'control': 'Control', 'receptive': 'Receptive', 'absent': 'Absent'}

modes = ['native', 'ablate_sigma2', 'ablate_sigma1']
mode_titles = {
    'native': 'Native (no ablation)',
    'ablate_sigma2': 'σ₂ Ablated at L16',
    'ablate_sigma1': 'σ₁ Ablated at L16',
}

for idx, mode in enumerate(modes):
    ax = axes[idx]
    mode_raw = [r for r in raw if r['mode'] == mode]

    for cond in ['receptive', 'control', 'absent']:
        vals = [r['S'] for r in mode_raw if r['condition'] == cond]
        positions = np.random.normal(list(cond_colors.keys()).index(cond), 0.08, len(vals))
        ax.scatter(positions, vals, c=cond_colors[cond], alpha=0.5, s=25,
                   edgecolors='white', linewidths=0.3)
        mean_val = np.mean(vals)
        ax.hlines(mean_val, list(cond_colors.keys()).index(cond) - 0.25,
                  list(cond_colors.keys()).index(cond) + 0.25,
                  colors=cond_colors[cond], linewidths=2.5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Control', 'Receptive', 'Absent'], fontsize=9)
    ax.set_ylabel('Spectral Entropy (S)', fontsize=10)
    ax.set_title(mode_titles[mode], fontsize=11, fontweight='bold')

    dS = results[mode]['delta_S']
    recv_S = results[mode]['receptive']['S']
    absent_S = results[mode]['absent']['S']

    ax.annotate(f'ΔS = {dS:+.4f}',
                xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=11, ha='center', va='top', fontweight='bold',
                color='#8e44ad',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0e6ff', alpha=0.9))

    if mode == 'ablate_sigma2':
        ax.annotate('90% retained\n(marker, not carrier)',
                    xy=(0.5, 0.82), xycoords='axes fraction',
                    fontsize=8, ha='center', va='top', color='#666',
                    fontstyle='italic')
    elif mode == 'ablate_sigma1':
        ax.annotate('8× amplified\n(wire obscured signal)',
                    xy=(0.5, 0.82), xycoords='axes fraction',
                    fontsize=8, ha='center', va='top', color='#c0392b',
                    fontstyle='italic')

    # Auto y-axis per panel with padding
    mode_vals = [r['S'] for r in mode_raw]
    pad = (max(mode_vals) - min(mode_vals)) * 0.25
    ax.set_ylim(min(mode_vals) - pad, max(mode_vals) + pad * 2)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
