#!/usr/bin/env python3
"""
Figure 1: The Step Function
d/d_max at tunnel endpoint for 9 architectures.
MHA cluster at ~55%, GQA cluster at 91-96%.
The MHA→GQA transition produces 9× more change than all within-GQA variation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "fig1_step_function.png"

# Values from paper findings (F50, F52-54)
# Format: (name, sharing_ratio, d_norm, architecture)
models = [
    ("Pythia 6.9B", 1, 0.549, "MHA"),
    ("LLaMA 1 7B", 1, 0.553, "MHA"),
    ("Falcon 7B", 1, 0.557, "MHA"),
    ("CodeLlama 7B", 1, 0.551, "MHA"),
    ("Gemma 2 9B", 2, 0.914, "GQA"),
    ("Mistral 7B", 4, 0.950, "GQA"),
    ("Qwen 2.5 7B", 4, 0.962, "GQA"),
    ("CodeQwen 7B", 4, 0.955, "GQA"),
    ("InternLM 7B", 4, 0.959, "GQA"),
    ("Qwen 2.5 3B", 8, 0.956, "GQA"),
]

fig, ax = plt.subplots(figsize=(10, 5))

mha_models = [(m[0], m[2]) for m in models if m[3] == "MHA"]
gqa_models = [(m[0], m[1], m[2]) for m in models if m[3] == "GQA"]

x_mha = np.arange(len(mha_models))
x_gqa = np.arange(len(gqa_models)) + len(mha_models) + 0.8

# MHA bars
bars_mha = ax.bar(x_mha, [m[1] for m in mha_models], 0.7,
                  color='#e74c3c', alpha=0.85, label='MHA (s=1)')

# GQA bars colored by sharing ratio
s_colors = {2: '#f39c12', 4: '#2980b9', 8: '#8e44ad'}
for i, (name, s, d) in enumerate(gqa_models):
    ax.bar(x_gqa[i], d, 0.7, color=s_colors[s], alpha=0.85)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.85, label='MHA (s=1)'),
    Patch(facecolor='#f39c12', alpha=0.85, label='GQA s=2'),
    Patch(facecolor='#2980b9', alpha=0.85, label='GQA s=4'),
    Patch(facecolor='#8e44ad', alpha=0.85, label='GQA s=8'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='center left')

# Labels
all_names = [m[0] for m in mha_models] + [m[0] for m in gqa_models]
all_x = list(x_mha) + list(x_gqa)
ax.set_xticks(all_x)
ax.set_xticklabels(all_names, rotation=35, ha='right', fontsize=8)

ax.set_ylabel('d/d_max (passage distance fraction)', fontsize=11)
ax.set_title('The Step Function: MHA→GQA Transition Produces 9× Within-GQA Variation',
             fontsize=12, fontweight='bold')

# Horizontal bands
mha_mean = np.mean([m[1] for m in mha_models])
gqa_mean = np.mean([m[2] for m in gqa_models])
ax.axhline(y=mha_mean, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(y=gqa_mean, color='#2980b9', linewidth=1, linestyle='--', alpha=0.5)

# Annotation: gap size
mid_y = (mha_mean + gqa_mean) / 2
ax.annotate('', xy=(len(mha_models) - 0.2, gqa_mean),
            xytext=(len(mha_models) - 0.2, mha_mean),
            arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=2))
gap = gqa_mean - mha_mean
ax.text(len(mha_models) - 0.1, mid_y, f'  Δ = {gap:.2f}\n  (9× within-GQA)',
        fontsize=9, va='center', fontweight='bold', color='#2c3e50')

ax.set_ylim(0, 1.05)
ax.set_xlim(-0.5, max(all_x) + 0.8)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
