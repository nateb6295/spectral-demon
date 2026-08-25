#!/usr/bin/env python3
"""Plot F599: Species gates Q1 polarity — 3-model decorrelation."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("results/tuning_knob_pythia.json") as f:
    pythia = json.load(f)
with open("results/tuning_knob_tinyllama_1.1b_chat_v1.0.json") as f:
    tiny = json.load(f)
with open("results/tuning_knob_mistral_7b_v0.1.json") as f:
    mistral = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

names = [g["name"] for g in pythia["gradient"]]
x = np.arange(len(names))
w = 0.25

# Panel 1: Q1 gradient comparison
ax1 = axes[0]
q1_p = [g["q1"] for g in pythia["gradient"]]
q1_t = [g["q1"] for g in tiny["gradient"]]
q1_m = [g["q1"] for g in mistral["gradient"]]

ax1.bar(x - w, q1_p, w, label='Pythia-2.8b (tunnel, 32L)', color='#3498db', alpha=0.85)
ax1.bar(x, q1_m, w, label='Mistral-7B (relay, 32L)', color='#1abc9c', alpha=0.85)
ax1.bar(x + w, q1_t, w, label='TinyLlama-1.1B (relay, 22L)', color='#c0392b', alpha=0.85)

ax1.axhline(y=0, color='black', linewidth=2)
ax1.axhspan(-0.03, 0, alpha=0.08, color='red')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=35, fontsize=8)
ax1.set_ylabel('Q1 Balance', fontsize=11)
ax1.set_title('Q1 Gradient: Species Separates', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.2, axis='y')

# Panel 2: Species vs depth (the money shot)
ax2 = axes[1]
models = [
    ('Pythia\n(tunnel, 32L)', 32, q1_p[0], q1_p[2], '#3498db', 'o'),
    ('Mistral\n(relay, 32L)', 32, q1_m[0], q1_m[2], '#1abc9c', 's'),
    ('TinyLlama\n(relay, 22L)', 22, q1_t[0], q1_t[2], '#c0392b', '^'),
]

for name, depth, q1_dir, q1_mild, color, marker in models:
    ax2.scatter(depth, q1_dir, s=150, c=color, marker=marker,
                edgecolors='black', linewidths=1.5, zorder=5)
    ax2.scatter(depth, q1_mild, s=100, c=color, marker=marker,
                edgecolors='black', linewidths=1, alpha=0.6, zorder=4)
    ax2.annotate(name, (depth, q1_dir), textcoords="offset points",
                 xytext=(10, 5), fontsize=9, color=color, fontweight='bold')

ax2.axhline(y=0, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax2.set_xlabel('Model Depth (layers)', fontsize=11)
ax2.set_ylabel('Q1 Balance', fontsize=11)
ax2.set_title('Depth Is Irrelevant\n(large=directive, small=mild)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(27, -0.015, 'Relays cross zero\nTunnels never do', fontsize=9,
         style='italic', ha='center', color='#666')

# Panel 3: Relay Σ across framings
ax3 = axes[2]
relay_p = [g["relay_sigma"] for g in pythia["gradient"]]
relay_t = [g["relay_sigma"] for g in tiny["gradient"]]
relay_m = [g["relay_sigma"] for g in mistral["gradient"]]

ax3.plot(names, relay_p, 'o-', color='#3498db', label='Pythia (tunnel)', linewidth=2, markersize=7)
ax3.plot(names, relay_m, 's-', color='#1abc9c', label='Mistral (relay)', linewidth=2, markersize=7)
ax3.plot(names, relay_t, '^-', color='#c0392b', label='TinyLlama (relay)', linewidth=2, markersize=7)

ax3.set_ylabel('Relay Σ (total CCS response)', fontsize=11)
ax3.set_title('CCS Response Grows with Framing\n(species-independent)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=35)

fig.suptitle('F599: Species Gates Q1 Polarity — Relay vs Tunnel Decorrelation',
             fontsize=14, fontweight='bold')
plt.tight_layout()

outpath = 'results/f599_species_gate.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print("Saved to {}".format(outpath))
