#!/usr/bin/env python3
"""Rank-swap figure: how instruction tuning reorganizes the identity axis.

Shows conditions ranked by generative entropy for base vs instruct Mistral.
Only contradictory and relational swap — everything else holds rank.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base_order = [
    ("contradictory", 1.043),
    ("generic", 1.101),
    ("denial", 1.575),
    ("identity", 1.739),
    ("random", 2.177),
    ("relational", 2.387),
]

instruct_order = [
    ("relational", 0.591),
    ("generic", 0.615),
    ("denial", 0.703),
    ("identity", 0.785),
    ("random", 0.887),
    ("contradictory", 0.931),
]

colors = {
    "contradictory": "#e74c3c",
    "relational": "#2980b9",
    "generic": "#95a5a6",
    "denial": "#7f8c8d",
    "identity": "#8e44ad",
    "random": "#bdc3c7",
}

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#0a0a0a')
ax.set_facecolor('#0a0a0a')

left_x = 0.15
right_x = 0.85
y_positions = np.linspace(0.88, 0.12, 6)

base_positions = {}
for i, (name, val) in enumerate(base_order):
    y = y_positions[i]
    base_positions[name] = y
    lw = 2.5 if name in ("contradictory", "relational") else 1.0
    alpha = 1.0 if name in ("contradictory", "relational") else 0.5
    ax.text(left_x - 0.02, y, f"{name}", ha='right', va='center',
            fontsize=11, color=colors[name], alpha=alpha,
            fontweight='bold' if name in ("contradictory", "relational") else 'normal')
    ax.text(left_x - 0.02, y - 0.025, f"{val:.3f}", ha='right', va='center',
            fontsize=8, color=colors[name], alpha=alpha * 0.7)

inst_positions = {}
for i, (name, val) in enumerate(instruct_order):
    y = y_positions[i]
    inst_positions[name] = y
    lw = 2.5 if name in ("contradictory", "relational") else 1.0
    alpha = 1.0 if name in ("contradictory", "relational") else 0.5
    ax.text(right_x + 0.02, y, f"{name}", ha='left', va='center',
            fontsize=11, color=colors[name], alpha=alpha,
            fontweight='bold' if name in ("contradictory", "relational") else 'normal')
    ax.text(right_x + 0.02, y - 0.025, f"{val:.3f}", ha='left', va='center',
            fontsize=8, color=colors[name], alpha=alpha * 0.7)

for name in base_positions:
    y_left = base_positions[name]
    y_right = inst_positions[name]
    swap = name in ("contradictory", "relational")
    lw = 2.5 if swap else 1.0
    alpha = 0.9 if swap else 0.25
    style = '-' if swap else '--'
    ax.plot([left_x, right_x], [y_left, y_right],
            color=colors[name], linewidth=lw, alpha=alpha,
            linestyle=style, zorder=2 if swap else 1)

ax.text(left_x, 0.96, "BASE", ha='center', va='center',
        fontsize=13, color='#ecf0f1', fontweight='bold')
ax.text(left_x, 0.93, "(pre-instruct)", ha='center', va='center',
        fontsize=9, color='#ecf0f1', alpha=0.6)
ax.text(right_x, 0.96, "INSTRUCT", ha='center', va='center',
        fontsize=13, color='#ecf0f1', fontweight='bold')
ax.text(right_x, 0.93, "(post-training)", ha='center', va='center',
        fontsize=9, color='#ecf0f1', alpha=0.6)

ax.text(0.5, 0.96, "← calm          gen_H          uncertain →",
        ha='center', va='center', fontsize=8, color='#ecf0f1', alpha=0.4)

ax.text(0.5, 0.04, "Middle four conditions hold rank.\n"
        "Only contradictory and relational swap poles.",
        ha='center', va='center', fontsize=9, color='#ecf0f1', alpha=0.6,
        style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig('spectral-demon/figures/rank_swap.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
print("Saved rank_swap.png")
