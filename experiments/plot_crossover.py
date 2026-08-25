#!/usr/bin/env python3
"""Plot F596: dose-response crossover curves for all sign-flip sources.

Shows that all positive-Q1 sources eventually cross zero, with crossover dose
inversely correlated to Q1 magnitude. Qwen (negative Q1) saturates at its
depletion floor and never crosses.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Measured dose-response data (strength, mean_shift)
phi2 = {
    'name': 'Phi-2', 'q1': +0.049, 'color': '#f39c12', 'marker': '^',
    'strengths': [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    'shifts': [0.0, -0.001, -0.002, -0.003, 0.0, 0.014],
}

tinyllama = {
    'name': 'TinyLlama-1.1B', 'q1': +0.027, 'color': '#c0392b', 'marker': 'p',
    'strengths': [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    'shifts': [0.0, -0.003, -0.005, -0.003, 0.0004, 0.011, 0.023, 0.022],
}

mistral = {
    'name': 'Mistral-7B', 'q1': +0.026, 'color': '#1abc9c', 'marker': 'P',
    'strengths': [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    'shifts': [0.0, -0.004, -0.006, -0.007, -0.005, -0.001, 0.005, 0.010],
}

qwen = {
    'name': 'Qwen-7B (negative Q1)', 'q1': -0.013, 'color': '#9b59b6', 'marker': 'v',
    'strengths': [0.0, 1.0, 2.0, 5.0, 10.0],
    'shifts': [0.0, -0.005, -0.009, -0.016, -0.016],
}

gemma = {
    'name': 'Gemma-2-2b', 'q1': +0.481, 'color': '#2ecc71', 'marker': 'o',
    'strengths': [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    'shifts': [0.0, 0.003, 0.005, 0.008, 0.015, 0.028],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

# Left panel: dose-response curves
for src in [gemma, phi2, tinyllama, mistral, qwen]:
    label = "{} (Q1={:+.3f})".format(src['name'], src['q1'])
    ax1.plot(src['strengths'], src['shifts'], color=src['color'], marker=src['marker'],
             label=label, linewidth=2.5, markersize=8, alpha=0.9)

ax1.axhline(y=0, color='black', linewidth=1.5, alpha=0.7)

# Mark crossover points
crossovers = [
    (0.75, 0, 'Phi-2\n~0.75', '#f39c12'),
    (3.0, 0, 'TinyLlama\n~3.0', '#c0392b'),
    (6.0, 0, 'Mistral\n~6.0', '#1abc9c'),
]
for x, y, label, color in crossovers:
    ax1.plot(x, y, 'x', color=color, markersize=12, markeredgewidth=3, zorder=10)
    ax1.annotate(label, (x, y), textcoords="offset points", xytext=(8, 10),
                 fontsize=8, color=color, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

# Qwen plateau annotation
ax1.annotate('Qwen plateau\n(-0.016)', (7.5, -0.016), textcoords="offset points",
             xytext=(15, -15), fontsize=8, color='#9b59b6', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.2))

ax1.set_xlabel('Injection Strength', fontsize=12)
ax1.set_ylabel('Mean Spectral Shift (σ₂/σ₁)', fontsize=12)
ax1.set_title('F596: Crossover Dose Inversely Correlates with Q1 Magnitude', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.3, 11)

# Right panel: Q1 vs crossover dose
q1_vals = [0.481, 0.151, 0.049, 0.027, 0.026]
cross_vals = [0.05, 0.05, 0.75, 3.0, 6.0]
names = ['Gemma', 'Pythia', 'Phi-2', 'TinyLlama', 'Mistral']
colors = ['#2ecc71', '#3498db', '#f39c12', '#c0392b', '#1abc9c']

for q, c, name, col in zip(q1_vals, cross_vals, names, colors):
    ax2.scatter(q, c, s=120, c=col, edgecolors='black', linewidths=1, zorder=5)
    offset = (8, 4) if name != 'Mistral' else (8, -12)
    ax2.annotate(name, (q, c), textcoords="offset points", xytext=offset, fontsize=9)

ax2.set_yscale('log')
ax2.set_xlabel('Q1 Balance', fontsize=11)
ax2.set_ylabel('Crossover Dose (log scale)', fontsize=11)
ax2.set_title('Q1 → Crossover', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5)
ax2.text(0.35, 0.08, 'measurement floor', fontsize=8, color='gray', ha='center')

# Negative Q1 region
ax2.axvspan(-0.05, 0, alpha=0.1, color='red')
ax2.text(-0.025, 1.0, 'neg Q1\n(never\ncrosses)', fontsize=7, color='red',
         ha='center', va='center', style='italic')

fig.suptitle('Cross-Architecture CCS Injection: One Phenomenon, One Continuum',
             fontsize=14, fontweight='bold')
plt.tight_layout()

outpath = 'results/f596_crossover.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved to {outpath}")
