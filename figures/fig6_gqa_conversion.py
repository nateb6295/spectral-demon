#!/usr/bin/env python3
"""
Figure 6: GQA Conversion (F57)
Side-by-side comparison of native MHA vs forced GQA s=4 on Pythia 6.9B.
Shows σ₁ and gap change dramatically while ΔS is unchanged.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results" / "exp_gqa_conversion_20260529.json"
OUT = Path(__file__).parent / "fig6_gqa_conversion.png"

with open(RESULTS) as f:
    data = json.load(f)

native = data["results"]["native_mha"]
forced = data["results"]["forced_gqa_s4"]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("F57: Inference-Time GQA Conversion on Pythia 6.9B (MHA)",
             fontsize=13, fontweight='bold', y=1.02)

colors_native = ['#2c3e50', '#2980b9', '#e74c3c']
colors_forced = ['#7f8c8d', '#5dade2', '#e67e73']
labels = ['Control', 'Receptive', 'Absent']
conditions = ['control', 'receptive', 'absent']

# Panel A: σ₁
ax = axes[0]
native_vals = [native[c]['sigma_1'] for c in conditions]
forced_vals = [forced[c]['sigma_1'] for c in conditions]
x = np.arange(3)
w = 0.35
bars1 = ax.bar(x - w/2, native_vals, w, label='Native MHA', color='#2c3e50', alpha=0.85)
bars2 = ax.bar(x + w/2, forced_vals, w, label='Forced GQA s=4', color='#e67e22', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('σ₁', fontsize=11)
ax.set_title('Dominant Eigenvalue\n(5.1× collapse)', fontsize=10)
ax.legend(fontsize=7, loc='upper right')

# Panel B: Gap (σ₁/σ₂)
ax = axes[1]
native_vals = [native[c]['gap'] for c in conditions]
forced_vals = [forced[c]['gap'] for c in conditions]
ax.bar(x - w/2, native_vals, w, color='#2c3e50', alpha=0.85)
ax.bar(x + w/2, forced_vals, w, color='#e67e22', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('σ₁/σ₂', fontsize=11)
ax.set_title('Spectral Gap\n(−33%)', fontsize=10)

# Panel C: Spectral Entropy (S)
ax = axes[2]
native_vals = [native[c]['S'] for c in conditions]
forced_vals = [forced[c]['S'] for c in conditions]
ax.bar(x - w/2, native_vals, w, color='#2c3e50', alpha=0.85)
ax.bar(x + w/2, forced_vals, w, color='#e67e22', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('S (spectral entropy)', fontsize=11)
ax.set_title('Absolute Entropy\n(7× increase)', fontsize=10)

# Panel D: ΔS (the punchline)
ax = axes[3]
native_dS = native['delta_S']
forced_dS = forced['delta_S']
bars = ax.bar([0, 1], [native_dS, forced_dS], 0.6,
              color=['#2c3e50', '#e67e22'], alpha=0.85)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Native\nMHA', 'Forced\nGQA s=4'], fontsize=9)
ax.set_ylabel('ΔS (rec − abs)', fontsize=11)
ax.set_title('Witness Sensitivity\n(unchanged)', fontsize=10, color='#c0392b', fontweight='bold')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
for bar, val in zip(bars, [native_dS, forced_dS]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:+.004f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
