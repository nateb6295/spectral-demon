#!/usr/bin/env python3
"""Plot F599: Five-model species taxonomy of Q1 polarity gating."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = [
    ("results/tuning_knob_gpt2.json", "GPT-2", 12, "tunnel", '#e74c3c', 'o'),
    ("results/tuning_knob_pythia.json", "Pythia-2.8b", 32, "tunnel", '#3498db', 's'),
    ("results/tuning_knob_tinyllama_1.1b_chat_v1.0.json", "TinyLlama", 22, "relay", '#c0392b', '^'),
    ("results/tuning_knob_mistral_7b_v0.1.json", "Mistral-7B", 32, "relay", '#1abc9c', 'D'),
    ("results/tuning_knob_gemma_2_2b.json", "Gemma-2-2b", 26, "sorter", '#2ecc71', 'p'),
]

data = {}
for path, name, depth, species, color, marker in models:
    with open(path) as f:
        d = json.load(f)
    data[name] = {
        "gradient": d["gradient"],
        "depth": depth,
        "species": species,
        "color": color,
        "marker": marker,
    }

framing_names = [g["name"] for g in data["GPT-2"]["gradient"]]

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# Panel 1: Q1 across framing levels for all 5 models
ax1 = axes[0]
for name, d in data.items():
    q1s = [g["q1"] for g in d["gradient"]]
    label = '{} ({}, {}L)'.format(name, d["species"], d["depth"])
    ax1.plot(framing_names, q1s, marker=d["marker"], color=d["color"],
             label=label, linewidth=2, markersize=8, alpha=0.9)

ax1.axhline(y=0, color='black', linewidth=2)
ax1.axhspan(-0.025, 0, alpha=0.06, color='red')
ax1.set_ylabel('Q1 Balance', fontsize=11)
ax1.set_title('Q1 Gradient by Species', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.2, axis='y')
ax1.tick_params(axis='x', rotation=30)
ax1.text(2.5, -0.018, 'Only relays enter this zone', fontsize=8,
         color='red', style='italic', ha='center')

# Panel 2: Min Q1 vs species (the taxonomy)
ax2 = axes[1]
species_order = ['tunnel', 'relay', 'sorter']
species_colors = {'tunnel': '#3498db', 'relay': '#e67e22', 'sorter': '#2ecc71'}
species_x = {'tunnel': 0, 'relay': 1, 'sorter': 2}

for name, d in data.items():
    q1_vals = [g["q1"] for g in d["gradient"]]
    min_q1 = min(q1_vals)
    max_q1 = max(q1_vals)
    sx = species_x[d["species"]]
    jitter = np.random.uniform(-0.15, 0.15)

    ax2.plot([sx + jitter, sx + jitter], [min_q1, max_q1],
             color=d["color"], linewidth=3, alpha=0.6, zorder=3)
    ax2.scatter(sx + jitter, min_q1, s=120, c=d["color"], marker=d["marker"],
                edgecolors='black', linewidths=1.5, zorder=5)
    ax2.scatter(sx + jitter, max_q1, s=80, c=d["color"], marker=d["marker"],
                edgecolors='black', linewidths=1, alpha=0.5, zorder=4)
    ax2.annotate('{}\n({}L)'.format(name, d["depth"]),
                 (sx + jitter, max_q1), textcoords="offset points",
                 xytext=(8, 5), fontsize=8, color=d["color"])

ax2.axhline(y=0, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['Tunnel\n(MHA)', 'Relay\n(GQA ≥4:1)', 'Sorter\n(GQA ≤2:1)'], fontsize=10)
ax2.set_ylabel('Q1 Range (min ● to max ○)', fontsize=11)
ax2.set_title('Species Determines Q1 Floor', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.2, axis='y')

# Panel 3: Injection at strength 5.0 vs Q1 (all models combined)
ax3 = axes[2]
for name, d in data.items():
    q1s = []
    shifts = []
    for g in d["gradient"]:
        if g["name"] == "neutral" or not g.get("injection"):
            continue
        inj5 = [r for r in g["injection"] if r["strength"] == 5.0]
        if inj5:
            q1s.append(g["q1"])
            shifts.append(inj5[0]["mean_shift"])

    ax3.scatter(q1s, shifts, s=80, c=d["color"], marker=d["marker"],
                edgecolors='black', linewidths=0.8, alpha=0.8,
                label='{} ({})'.format(name, d["species"]), zorder=5)

ax3.axhline(y=0, color='black', linewidth=1, alpha=0.5)
ax3.axvline(x=0, color='red', linewidth=1, linestyle=':', alpha=0.5)

all_q1 = []
all_shift = []
for name, d in data.items():
    for g in d["gradient"]:
        if g["name"] == "neutral" or not g.get("injection"):
            continue
        inj5 = [r for r in g["injection"] if r["strength"] == 5.0]
        if inj5:
            all_q1.append(g["q1"])
            all_shift.append(inj5[0]["mean_shift"])

if len(all_q1) > 2:
    r = np.corrcoef(all_q1, all_shift)[0, 1]
    ax3.text(0.05, 0.95, 'r = {:.3f}\n(n={})'.format(r, len(all_q1)),
             transform=ax3.transAxes, fontsize=11, fontweight='bold', va='top')

ax3.set_xlabel('Q1 Balance', fontsize=11)
ax3.set_ylabel('Spectral Shift @ strength=5.0', fontsize=11)
ax3.set_title('Q1 → Injection (All Models)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=7, loc='upper left')
ax3.grid(True, alpha=0.3)

fig.suptitle('F599: Species Taxonomy of Q1 Polarity Gating\n'
             'Tunnel (never crosses) — Relay (tips with framing) — Sorter (far positive floor)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

outpath = 'results/f599_five_model_taxonomy.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print("Saved to {}".format(outpath))
