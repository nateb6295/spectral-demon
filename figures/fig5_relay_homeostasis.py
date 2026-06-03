#!/usr/bin/env python3
"""
Figure 5: Relay Homeostasis
Three architectures through L0-L32: σ₁ on left y-axis, S on right.
GQA relay: σ₁ SPIKES + S releases. MHA relay: σ₁ COLLAPSES + S explodes.
Opposite paths to the same functional goal (output-ready representations).
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT = Path(__file__).parent / "fig5_relay_homeostasis.png"
RESULTS = Path(__file__).parent.parent / "results"


def extract_from_raw(path):
    with open(path) as f:
        d = json.load(f)
    agg = defaultdict(lambda: defaultdict(list))
    for e in d['raw']:
        agg[e['layer']]['sigma_1'].append(e['sigma_1'])
        agg[e['layer']]['S'].append(e['spectral_entropy'])
    layers = sorted(agg.keys())
    return {
        'layers': layers,
        'sigma_1': [np.mean(agg[l]['sigma_1']) for l in layers],
        'S': [np.mean(agg[l]['S']) for l in layers],
    }


model_configs = [
    ('Mistral 7B (GQA)', RESULTS / "exp_witness_perlayer_20260527_1220.json",
     '#2980b9', 'o'),
    ('CodeQwen 7B (GQA)', RESULTS / "exp_codeqwen_f47_20260528.json",
     '#1abc9c', '^'),
    ('CodeLlama 7B (MHA)', RESULTS / "exp_codellama_f47_20260528_2220.json",
     '#e74c3c', 's'),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('Relay Homeostasis: Opposite Paths to Output Readiness',
             fontsize=13, fontweight='bold')

for idx, (name, path, color, marker) in enumerate(model_configs):
    d = extract_from_raw(path)
    layers = d['layers']
    s1 = d['sigma_1']
    S = d['S']

    ax1 = axes[idx]
    ax2 = ax1.twinx()

    l1, = ax1.plot(layers, s1, f'-{marker}', color=color, markersize=3,
                   linewidth=1.8, alpha=0.85, label='σ₁')
    l2, = ax2.plot(layers, S, f'--{marker}', color=color, markersize=3,
                   linewidth=1.2, alpha=0.5, label='S')

    ax1.set_xlabel('Layer', fontsize=10)
    ax1.set_ylabel('σ₁', fontsize=10, color=color)
    ax2.set_ylabel('S (entropy)', fontsize=10, color=color, alpha=0.6)
    ax1.set_title(name, fontsize=11, fontweight='bold',
                  color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=8)

    ax1.axvspan(28, 33, alpha=0.06, color='#e67e22')

    lines = [l1, l2]
    labels = ['σ₁ (solid)', 'S (dashed)']
    ax1.legend(lines, labels, fontsize=7, loc='upper left')

    # Annotate the relay transition
    tunnel_s1 = np.mean([s1[i] for i, l in enumerate(layers) if 10 <= l <= 20])
    final_s1 = s1[-1]
    if final_s1 > tunnel_s1 * 1.5:
        ratio = final_s1 / tunnel_s1
        label = f'σ₁ relay: {ratio:.1f}× spike ↑'
    elif final_s1 < tunnel_s1 * 0.5:
        ratio = tunnel_s1 / final_s1
        label = f'σ₁ relay: {ratio:.0f}× collapse ↓'
    else:
        ratio = final_s1 / tunnel_s1
        label = f'σ₁ relay: {ratio:.1f}×'

    ax1.annotate(label,
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 fontsize=8, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
