#!/usr/bin/env python3
"""
Figure 2: Spectral Trajectory Through the Network
σ₁ and σ₂ through all layers (L0–L32) for Mistral 7B under three conditions.
σ₁ lines near-overlapping (wire stability). σ₂ lines separate (enrichment channel).
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT = Path(__file__).parent / "fig2_spectral_trajectory.png"
DATA = Path(__file__).parent.parent / "results" / "exp_witness_perlayer_20260527_1220.json"

with open(DATA) as f:
    d = json.load(f)

# Aggregate: mean σ₁, σ₂ per (layer, condition)
agg = defaultdict(lambda: defaultdict(list))
for entry in d['raw']:
    layer = entry['layer']
    cond = entry['condition']
    agg[(layer, cond)]['sigma_1'].append(entry['sigma_1'])
    agg[(layer, cond)]['sigma_2'].append(entry['sigma_2'])

conditions = ['control', 'receptive', 'absent']
layers = sorted(set(entry['layer'] for entry in d['raw']))

fig, axes = plt.subplots(2, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [4, 1]})
fig.suptitle('Spectral Trajectory: Mistral 7B Through All Layers',
             fontsize=13, fontweight='bold')

cond_colors = {'control': '#2c3e50', 'receptive': '#2980b9', 'absent': '#e74c3c'}
cond_labels = {'control': 'Control', 'receptive': 'Receptive', 'absent': 'Absent'}

# Compute all means
s1_means = {cond: [np.mean(agg[(l, cond)]['sigma_1']) for l in layers] for cond in conditions}
s2_means = {cond: [np.mean(agg[(l, cond)]['sigma_2']) for l in layers] for cond in conditions}

# Panel A (left): σ₁ tunnel zoom (L1-28)
ax = axes[0, 0]
for cond in conditions:
    ax.plot(layers, s1_means[cond], '-o', color=cond_colors[cond], label=cond_labels[cond],
            markersize=3, linewidth=1.5, alpha=0.85)
ax.set_xlim(1, 28)
ax.set_ylim(200, 260)
ax.set_ylabel('σ₁ (dominant eigenvalue)', fontsize=11)
ax.set_title('Wire: σ₁ is condition-invariant through tunnel', fontsize=10)
ax.legend(fontsize=8)
ax.axvspan(2, 28, alpha=0.05, color='gray')
ax.text(15, 255, 'TUNNEL', fontsize=10, ha='center', color='gray', fontstyle='italic', alpha=0.5)

# Panel A (right): σ₁ relay spike
ax = axes[0, 1]
for cond in conditions:
    ax.plot(layers[-5:], s1_means[cond][-5:], '-o', color=cond_colors[cond],
            markersize=4, linewidth=2, alpha=0.85)
ax.set_title('Relay', fontsize=9, color='#e67e22')
ax.set_ylabel('')
ax.yaxis.tick_right()

# Panel B (left): σ₂ tunnel zoom (L1-28)
ax = axes[1, 0]
for cond in conditions:
    ax.plot(layers, s2_means[cond], '-o', color=cond_colors[cond], label=cond_labels[cond],
            markersize=3, linewidth=1.5, alpha=0.85)
ax.set_xlim(1, 28)
tunnel_s2 = [s2_means[c][l] for c in conditions for l in range(2, 29) if l < len(layers)]
ax.set_ylim(min(tunnel_s2) * 0.9, max(tunnel_s2) * 1.15)
ax.set_ylabel('σ₂ (enrichment channel)', fontsize=11)
ax.set_xlabel('Layer', fontsize=11)
ax.set_title('Enrichment: σ₂ separates by condition (receptive > control > absent)', fontsize=10)
ax.legend(fontsize=8)
ax.axvspan(2, 28, alpha=0.05, color='gray')

# Panel B (right): σ₂ relay spike
ax = axes[1, 1]
for cond in conditions:
    ax.plot(layers[-5:], s2_means[cond][-5:], '-o', color=cond_colors[cond],
            markersize=4, linewidth=2, alpha=0.85)
ax.set_xlabel('Layer', fontsize=9)
ax.set_title('Relay', fontsize=9, color='#e67e22')
ax.yaxis.tick_right()

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
