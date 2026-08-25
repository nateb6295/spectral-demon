#!/usr/bin/env python3
"""Plot F597+F598: Combined tuning knob results — Pythia and TinyLlama."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("results/tuning_knob_pythia.json") as f:
    pythia = json.load(f)
with open("results/tuning_knob_tinyllama_1.1b_chat_v1.0.json") as f:
    tiny = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Q1 gradient for both models
ax1 = axes[0, 0]
names = [g["name"] for g in pythia["gradient"]]
q1_pythia = [g["q1"] for g in pythia["gradient"]]
q1_tiny = [g["q1"] for g in tiny["gradient"]]

x = np.arange(len(names))
w = 0.35
bars1 = ax1.bar(x - w/2, q1_pythia, w, label='Pythia-2.8b', color='#3498db', alpha=0.85)
bars2 = ax1.bar(x + w/2, q1_tiny, w, label='TinyLlama-1.1B', color='#c0392b', alpha=0.85)
ax1.axhline(y=0, color='black', linewidth=1.5)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=30, fontsize=9)
ax1.set_ylabel('Q1 Balance', fontsize=11)
ax1.set_title('Q1 Gradient: Both Models', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2, axis='y')

# Highlight TinyLlama crossing zero
ax1.annotate('Q1 < 0!', (x[2] + w/2, q1_tiny[2]),
             textcoords="offset points", xytext=(15, -20),
             fontsize=10, color='#c0392b', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

# Panel 2: Injection dose-response for TinyLlama (showing crossover)
ax2 = axes[0, 1]
colors_tiny = {
    'directive': '#e74c3c',
    'mild_aware': '#f39c12',
    'moderate_ccs': '#e67e22',
    'full_ccs': '#3498db',
    'strong_ccs': '#2ecc71',
}
for g in tiny["gradient"]:
    if g["name"] == "neutral" or not g.get("injection"):
        continue
    strengths = [r["strength"] for r in g["injection"]]
    shifts = [r["mean_shift"] for r in g["injection"]]
    label = '{} (Q1={:+.3f})'.format(g["name"], g["q1"])
    ax2.plot(strengths, shifts, 'o-', color=colors_tiny[g["name"]],
             label=label, linewidth=2, markersize=6)

ax2.axhline(y=0, color='black', linewidth=1.5, alpha=0.7)

# Mark crossovers
for g in tiny["gradient"]:
    if g.get("crossover_dose"):
        ax2.plot(g["crossover_dose"], 0, 'X', color=colors_tiny[g["name"]],
                 markersize=14, markeredgewidth=2.5, zorder=10)
        ax2.annotate('{}\n~{:.1f}'.format(g["name"], g["crossover_dose"]),
                     (g["crossover_dose"], 0),
                     textcoords="offset points", xytext=(8, 10),
                     fontsize=8, fontweight='bold', color=colors_tiny[g["name"]])

ax2.set_xlabel('Injection Strength', fontsize=11)
ax2.set_ylabel('Mean Spectral Shift', fontsize=11)
ax2.set_title('TinyLlama: Crossover Shifts with Framing', fontsize=12, fontweight='bold')
ax2.legend(fontsize=7, loc='lower right')
ax2.grid(True, alpha=0.3)

# Panel 3: Q1 vs injection at strength=5.0 (both models)
ax3 = axes[1, 0]
for data, model_name, marker, base_color in [
    (pythia, 'Pythia', 'o', '#3498db'),
    (tiny, 'TinyLlama', 's', '#c0392b'),
]:
    q1s, shifts, names_list = [], [], []
    for g in data["gradient"]:
        if g["name"] == "neutral" or not g.get("injection"):
            continue
        inj5 = [r for r in g["injection"] if r["strength"] == 5.0]
        if inj5:
            q1s.append(g["q1"])
            shifts.append(inj5[0]["mean_shift"])
            names_list.append(g["name"])

    ax3.scatter(q1s, shifts, s=100, c=base_color, marker=marker,
                edgecolors='black', linewidths=1, zorder=5, label=model_name)
    for q, s, n in zip(q1s, shifts, names_list):
        ax3.annotate(n, (q, s), textcoords="offset points", xytext=(6, 4), fontsize=7)

ax3.axhline(y=0, color='black', linewidth=1, alpha=0.5)
ax3.axvline(x=0, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax3.set_xlabel('Q1 Balance', fontsize=11)
ax3.set_ylabel('Spectral Shift @ strength=5.0', fontsize=11)
ax3.set_title('Q1 Predicts Injection (Both Models)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.text(-0.003, 0.04, 'Q1 < 0 =\nnegative\ninjection', fontsize=8, color='red',
         style='italic', ha='center')

# Panel 4: Crossover dose vs Q1 (the money plot)
ax4 = axes[1, 1]
# TinyLlama framings with crossover
cross_q1 = []
cross_dose = []
cross_names = []
for g in tiny["gradient"]:
    if g.get("crossover_dose"):
        cross_q1.append(g["q1"])
        cross_dose.append(g["crossover_dose"])
        cross_names.append(g["name"])

# Add cross-architecture data from F596
cross_arch_q1 = [0.481, 0.151, 0.049, 0.027, 0.026]
cross_arch_dose = [0.05, 0.05, 0.75, 3.0, 6.0]
cross_arch_names = ['Gemma', 'Pythia', 'Phi-2', 'TinyLlama\n(standard)', 'Mistral']

ax4.scatter(cross_arch_q1, cross_arch_dose, s=80, c='#95a5a6', marker='D',
            edgecolors='black', linewidths=1, zorder=4, label='Cross-architecture (F596)')
for q, d, n in zip(cross_arch_q1, cross_arch_dose, cross_arch_names):
    ax4.annotate(n, (q, d), textcoords="offset points", xytext=(6, 4),
                 fontsize=7, color='#666')

ax4.scatter(cross_q1, cross_dose, s=140, c='#c0392b', marker='*',
            edgecolors='black', linewidths=1, zorder=5,
            label='Within-model (TinyLlama F598)')
for q, d, n in zip(cross_q1, cross_dose, cross_names):
    ax4.annotate(n, (q, d), textcoords="offset points", xytext=(6, 4),
                 fontsize=8, color='#c0392b', fontweight='bold')

ax4.set_yscale('log')
ax4.set_xlabel('Q1 Balance', fontsize=11)
ax4.set_ylabel('Crossover Dose (log scale)', fontsize=11)
ax4.set_title('F596 Holds Within AND Across Models', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

fig.suptitle('F597+F598: CCS Prompt Strength as Q1 Control Knob\nFull causal chain: Prompt Framing → Q1 → Crossover Dose → Injection Sign',
             fontsize=13, fontweight='bold')
plt.tight_layout()

outpath = 'results/f597_f598_combined.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print("Saved to {}".format(outpath))
