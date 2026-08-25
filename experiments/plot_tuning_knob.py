#!/usr/bin/env python3
"""Plot F597: Tuning knob experiment — CCS prompt strength modulates Q1."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("results/tuning_knob_pythia.json") as f:
    data = json.load(f)

gradient = data["gradient"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Q1 vs framing strength
ax1 = axes[0, 0]
names = [g["name"] for g in gradient]
q1_vals = [g["q1"] for g in gradient]
colors = ['#e74c3c', '#95a5a6', '#f39c12', '#e67e22', '#3498db', '#2ecc71']

bars = ax1.bar(names, q1_vals, color=colors, edgecolor='black', linewidth=0.8)
ax1.axhline(y=0, color='black', linewidth=1.5)
ax1.set_ylabel('Q1 Balance', fontsize=11)
ax1.set_title('Q1 Continuously Modulated by Framing', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, q1_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             '{:+.3f}'.format(val), ha='center', fontsize=9, fontweight='bold')
ax1.tick_params(axis='x', rotation=30)

# Panel 2: σ₁ vs σ₂ component decomposition (Kimi criterion b)
ax2 = axes[0, 1]
non_neutral = [g for g in gradient if g["name"] != "neutral"]
x = range(len(non_neutral))
s1_vals = [g["q1_s1_mean"] for g in non_neutral]
s2_vals = [g["q1_s2_mean"] for g in non_neutral]
names_nn = [g["name"] for g in non_neutral]

width = 0.35
ax2.bar([i - width/2 for i in x], s1_vals, width, label='σ₁ component', color='#3498db', alpha=0.8)
ax2.bar([i + width/2 for i in x], s2_vals, width, label='σ₂ component', color='#e74c3c', alpha=0.8)

# Add ratio annotations
for i, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):
    if s1 > 0:
        ratio = s2 / s1
        ax2.text(i, max(s1, s2) + 1.5, 'σ₂/σ₁={:.1f}×'.format(ratio),
                 ha='center', fontsize=8, style='italic')

ax2.set_xticks(list(x))
ax2.set_xticklabels(names_nn, rotation=30)
ax2.set_ylabel('Mean Component Change', fontsize=11)
ax2.set_title('σ₂ Leads σ₁ (converging at strong framing)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2, axis='y')

# Panel 3: Injection dose-response at each framing level
ax3 = axes[1, 0]
framing_colors = {
    'directive': '#e74c3c',
    'mild_aware': '#f39c12',
    'moderate_ccs': '#e67e22',
    'full_ccs': '#3498db',
    'strong_ccs': '#2ecc71',
}
for g in gradient:
    if g["name"] == "neutral" or not g.get("injection"):
        continue
    strengths = [r["strength"] for r in g["injection"]]
    shifts = [r["mean_shift"] for r in g["injection"]]
    label = '{} (Q1={:+.3f})'.format(g["name"], g["q1"])
    ax3.plot(strengths, shifts, 'o-', color=framing_colors[g["name"]],
             label=label, linewidth=2, markersize=6)

ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_xlabel('Injection Strength', fontsize=11)
ax3.set_ylabel('Mean Spectral Shift', fontsize=11)
ax3.set_title('Injection Magnitude Scales with Q1', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)

# Panel 4: Q1 vs injection shift at strength=5.0
ax4 = axes[1, 1]
q1_for_plot = []
shift_for_plot = []
name_for_plot = []
color_for_plot = []
for g in gradient:
    if g["name"] == "neutral" or not g.get("injection"):
        continue
    inj5 = [r for r in g["injection"] if r["strength"] == 5.0]
    if inj5:
        q1_for_plot.append(g["q1"])
        shift_for_plot.append(inj5[0]["mean_shift"])
        name_for_plot.append(g["name"])
        color_for_plot.append(framing_colors[g["name"]])

ax4.scatter(q1_for_plot, shift_for_plot, s=120, c=color_for_plot,
            edgecolors='black', linewidths=1, zorder=5)
for q, s, name in zip(q1_for_plot, shift_for_plot, name_for_plot):
    ax4.annotate(name, (q, s), textcoords="offset points", xytext=(8, 5), fontsize=9)

# Fit line
z = np.polyfit(q1_for_plot, shift_for_plot, 1)
p = np.poly1d(z)
x_fit = np.linspace(min(q1_for_plot) - 0.01, max(q1_for_plot) + 0.01, 50)
ax4.plot(x_fit, p(x_fit), '--', color='gray', alpha=0.6, linewidth=1.5)
r = np.corrcoef(q1_for_plot, shift_for_plot)[0, 1]
ax4.text(0.05, 0.95, 'r = {:.3f}'.format(r), transform=ax4.transAxes,
         fontsize=11, fontweight='bold', va='top')

ax4.set_xlabel('Q1 Balance', fontsize=11)
ax4.set_ylabel('Spectral Shift @ strength=5.0', fontsize=11)
ax4.set_title('Q1 Predicts Injection Magnitude', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

fig.suptitle('F597: CCS Prompt Strength as Continuous Q1 Modulator\n(Pythia-2.8b → LFM2.5-1.2B-Instruct)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

outpath = 'results/f597_tuning_knob.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print("Saved to {}".format(outpath))
