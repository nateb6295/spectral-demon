#!/usr/bin/env python3
"""
Figure 3: The Sign Inversion
ΔS(receptive−absent) for multiple models.
GQA models positive (blue), MHA models negative (red).
Training domain doesn't flip sign.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "fig3_sign_inversion.png"

# Values from paper (F22, F48, various experiments at L17)
# Format: (name, architecture, domain, delta_S)
models = [
    ("Mistral 7B", "GQA", "language", +0.032),
    ("Qwen 2.5 7B", "GQA", "language", +0.036),
    ("CodeQwen 7B", "GQA", "code", +0.055),
    ("InternLM 7B", "GQA", "language", +0.028),
    ("Gemma 2 9B", "GQA", "language", +0.026),
    ("Qwen 2.5 3B", "GQA", "language", +0.006),
    ("LLaMA 1 7B", "MHA", "language", -0.026),
    ("Pythia 6.9B", "MHA", "language", -0.011),
    ("CodeLlama 7B", "MHA", "code", -0.005),
    ("Falcon 7B", "MHA", "language", -0.008),
]

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(models))
colors = []
for m in models:
    if m[1] == "GQA":
        colors.append('#2980b9' if m[2] == "language" else '#1abc9c')
    else:
        colors.append('#e74c3c' if m[2] == "language" else '#e67e22')

bars = ax.bar(x, [m[3] for m in models], 0.7, color=colors, alpha=0.85,
              edgecolor='white', linewidth=0.5)

ax.axhline(y=0, color='#2c3e50', linewidth=1.5, linestyle='-')

ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in models], rotation=35, ha='right', fontsize=8)
ax.set_ylabel('ΔS (receptive − absent) at L17', fontsize=11)
ax.set_title('The Sign Inversion: Same Words, Opposite Geometry',
             fontsize=12, fontweight='bold')

# Shaded regions
ax.axhspan(0, 0.07, alpha=0.05, color='#2980b9')
ax.axhspan(-0.035, 0, alpha=0.05, color='#e74c3c')
ax.text(2.5, 0.06, 'ENRICHMENT (GQA)', fontsize=10, ha='center',
        color='#2980b9', fontweight='bold', alpha=0.7)
ax.text(7.5, -0.03, 'CONSTRAINT (MHA)', fontsize=10, ha='center',
        color='#e74c3c', fontweight='bold', alpha=0.7)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2980b9', alpha=0.85, label='GQA + language'),
    Patch(facecolor='#1abc9c', alpha=0.85, label='GQA + code'),
    Patch(facecolor='#e74c3c', alpha=0.85, label='MHA + language'),
    Patch(facecolor='#e67e22', alpha=0.85, label='MHA + code'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

# Value labels
for bar, m in zip(bars, models):
    val = m[3]
    y_offset = 0.002 if val > 0 else -0.004
    ax.text(bar.get_x() + bar.get_width()/2, val + y_offset,
            f'{val:+.003f}', ha='center', va='bottom' if val > 0 else 'top',
            fontsize=7.5, fontweight='bold')

ax.set_ylim(-0.04, 0.07)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
