#!/usr/bin/env python3
"""Three-panel figure: relay strategies across architectures.
Each panel: L_last σ₂/σ₁ (x) vs gen_H (y) with condition labels."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

conditions = ['identity', 'relational', 'generic', 'denial', 'contradictory', 'random']
labels_short = ['Id', 'Rel', 'Gen', 'Den', 'Con', 'Rnd']
colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']

# Data from cross-architecture experiments
mistral_ratio = [0.490, 0.635, 0.422, 0.345, 0.569, 0.580]
mistral_h     = [0.785, 0.591, 0.615, 0.703, 0.931, 0.887]

qwen_ratio = [0.479, 0.474, 0.429, 0.448, 0.484, 0.449]
qwen_h     = [0.906, 0.816, 0.597, 0.590, 0.885, 0.703]

gemma_ratio = [0.677, 0.698, 0.712, 0.697, 0.705, 0.506]
gemma_h     = [0.561, 0.792, 0.250, 0.219, 0.691, 0.338]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

models = [
    ('Mistral-7B (33 layers)', mistral_ratio, mistral_h, 0.290, 0.855),
    ('Qwen-2.5-7B (28 layers)', qwen_ratio, qwen_h, 0.055, 0.940),
    ('Gemma-2-9B (42 layers)', gemma_ratio, gemma_h, 0.035, 0.155),
]

strategies = ['Differentiating', 'Compressing', 'Equalizing']

for idx, (title, ratios, hs, spread, r_excl) in enumerate(models):
    ax = axes[idx]

    # Fit line excluding random (last point)
    x_fit = np.array(ratios[:5])
    y_fit = np.array(hs[:5])
    if np.std(x_fit) > 0.001:
        z = np.polyfit(x_fit, y_fit, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(ratios) - 0.02, max(ratios) + 0.02, 100)
        ax.plot(x_line, p(x_line), '--', color='#BDBDBD', linewidth=1.5, zorder=1)

    for i, (r, h) in enumerate(zip(ratios, hs)):
        marker = 'o' if i < 5 else 's'
        ax.scatter(r, h, c=colors[i], s=120, zorder=3, marker=marker,
                   edgecolors='white', linewidth=1.5)
        offset_x, offset_y = 0.008, 0.025
        if labels_short[i] == 'Rel':
            offset_y = -0.04
        ax.annotate(labels_short[i], (r, h),
                    xytext=(r + offset_x, h + offset_y),
                    fontsize=9, fontweight='bold', color=colors[i])

    ax.set_title(f'{title}\n{strategies[idx]} relay', fontsize=11, fontweight='bold')
    ax.set_xlabel('L_last σ₂/σ₁ ratio', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Generation entropy (H)', fontsize=10)

    ax.text(0.05, 0.95, f'spread={spread:.3f}\nr_excl={r_excl:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.1, 1.05)

plt.tight_layout()
plt.savefig('/home/nate-agx/chronicle/spectral-demon/figures/three_relay_strategies.png',
            dpi=150, bbox_inches='tight')
print("Saved: spectral-demon/figures/three_relay_strategies.png")
