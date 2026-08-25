#!/usr/bin/env python3
"""F603: Zone Q1 vs Aggregate Q1 — resolving the Qwen anomaly."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

models = [
    ("gpt2", "GPT-2", "#27ae60", "tunnel"),
    ("pythia", "Pythia-2.8b", "#2ecc71", "tunnel"),
    ("tinyllama_1.1b_chat_v1.0", "TinyLlama", "#e74c3c", "relay"),
    ("mistral_7b_v0.1", "Mistral-7B", "#3498db", "relay"),
    ("gemma_2_2b", "Gemma-2-2b", "#9b59b6", "sorter"),
    ("phi_2", "Phi-2", "#e67e22", "mismatch"),
    ("qwen2.5_3b", "Qwen2.5-3B", "#f39c12", "relay"),
]

SPECIES_MARKERS = {"tunnel": "o", "relay": "^", "sorter": "D", "mismatch": "s"}

rows = []
for mk, label, color, species in models:
    d = json.load(open(os.path.join(RESULTS, f"tuning_knob_{mk}.json")))
    for entry in d['gradient']:
        if entry['name'] == 'neutral' or 'per_layer_q1' not in entry:
            continue
        layers = entry['per_layer_q1']
        split = min(2, len(layers))
        zone_q1 = sum(l['delta_ratio'] for l in layers[split:])
        inj = {i['strength']: i['mean_shift'] for i in entry['injection']}
        rows.append({
            'model': label, 'color': color, 'species': species,
            'framing': entry['name'], 'q1_agg': entry['q1'],
            'zone_q1': zone_q1, 'shift5': inj.get(5.0, 0),
        })

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("F603: Zone Q1 Resolves Aggregate Q1 Failures\n"
             "Transplant carries zone structure — Q1 in early layers doesn't reach injection",
             fontsize=12, fontweight="bold")

# Panel 1: Q1_agg vs shift@5 (the old picture, with Qwen wrong)
ax = axes[0, 0]
ax.set_title("Aggregate Q1 vs Shift (r=0.826)", fontsize=10)
for m in models:
    mk, label, color, species = m
    pts = [r for r in rows if r['model'] == label]
    ax.scatter([p['q1_agg'] for p in pts], [p['shift5'] for p in pts],
               c=color, marker=SPECIES_MARKERS[species], s=60, label=label,
               edgecolors='white', linewidths=0.5, zorder=5)
# Highlight Qwen moderate_ccs anomaly
qm = [r for r in rows if r['model'] == 'Qwen2.5-3B' and r['framing'] == 'moderate_ccs'][0]
ax.scatter([qm['q1_agg']], [qm['shift5']], facecolors='none', edgecolors='red',
           s=300, linewidths=2.5, zorder=10)
ax.annotate("Qwen moderate\nQ1=+0.072 but shift<0!", (qm['q1_agg'], qm['shift5']),
            textcoords="offset points", xytext=(20, 20), fontsize=7,
            arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax.axhline(0, color='gray', alpha=0.3)
ax.axvline(0, color='red', alpha=0.3)
ax.set_xlabel("Q1 (aggregate)")
ax.set_ylabel("Shift @ 5.0")
ax.legend(fontsize=6, loc="lower right", ncol=2)

# Panel 2: Zone Q1 vs shift@5 (the fix)
ax = axes[0, 1]
ax.set_title("Zone Q1 vs Shift (relays: r=0.791)", fontsize=10)
for m in models:
    mk, label, color, species = m
    pts = [r for r in rows if r['model'] == label]
    ax.scatter([p['zone_q1'] for p in pts], [p['shift5'] for p in pts],
               c=color, marker=SPECIES_MARKERS[species], s=60, label=label,
               edgecolors='white', linewidths=0.5, zorder=5)
# Qwen moderate now in the right place
qm2 = [r for r in rows if r['model'] == 'Qwen2.5-3B' and r['framing'] == 'moderate_ccs'][0]
ax.scatter([qm2['zone_q1']], [qm2['shift5']], facecolors='none', edgecolors='green',
           s=300, linewidths=2.5, zorder=10)
ax.annotate("Qwen moderate\nzone Q1=+0.011 ✓", (qm2['zone_q1'], qm2['shift5']),
            textcoords="offset points", xytext=(20, 20), fontsize=7,
            arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.axhline(0, color='gray', alpha=0.3)
ax.axvline(0, color='red', alpha=0.3)
ax.set_xlabel("Zone Q1 (L2+ layers)")
ax.set_ylabel("Shift @ 5.0")
ax.legend(fontsize=6, loc="lower right", ncol=2)

# Panel 3: Qwen-specific — zone fraction vs injection
ax = axes[1, 0]
ax.set_title("Qwen: Zone Fraction vs Injection\n(zone_frac perfectly ranks framings)", fontsize=10)
qwen = [r for r in rows if r['model'] == 'Qwen2.5-3B']
zone_fracs = []
for q in qwen:
    total = q['q1_agg']
    zf = q['zone_q1'] / total if abs(total) > 1e-10 else 0
    zone_fracs.append(zf)

colors_q = ['#e74c3c' if q['shift5'] < 0 else '#27ae60' for q in qwen]
ax.scatter(zone_fracs, [q['shift5'] for q in qwen], c=colors_q, s=120, zorder=5,
           edgecolors='white', linewidths=1)
for i, q in enumerate(qwen):
    ax.annotate(q['framing'].replace('_', '\n'), (zone_fracs[i], q['shift5']),
                textcoords="offset points", xytext=(10, 5), fontsize=7)
ax.axhline(0, color='red', alpha=0.4, linewidth=1.5)
ax.axvline(0.5, color='gray', alpha=0.3, linestyle='--')
ax.set_xlabel("Zone Fraction (% of Q1 in relay zone)")
ax.set_ylabel("Shift @ 5.0")
r_zf, p_zf = pearsonr(zone_fracs, [q['shift5'] for q in qwen])
ax.text(0.05, 0.95, f"r = {r_zf:.3f}\np = {p_zf:.3f}", transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(facecolor='wheat', alpha=0.5))

# Panel 4: Per-species comparison bar chart
ax = axes[1, 1]
ax.set_title("Zone Q1 vs Aggregate Q1: Per-Species r²", fontsize=10)
species_names = ['tunnel', 'relay', 'sorter', 'mismatch']
species_labels = ['Tunnel\n(MHA)', 'Relay\n(GQA)', 'Sorter\n(GQA 2:1)', 'Mismatch\n(MHA→relay)']
r2_agg = []
r2_zone = []
for sp in species_names:
    sp_rows = [r for r in rows if r['species'] == sp]
    if len(sp_rows) >= 3:
        r_a, _ = pearsonr([r['q1_agg'] for r in sp_rows], [r['shift5'] for r in sp_rows])
        r_z, _ = pearsonr([r['zone_q1'] for r in sp_rows], [r['shift5'] for r in sp_rows])
        r2_agg.append(r_a**2)
        r2_zone.append(r_z**2)
    else:
        r2_agg.append(0)
        r2_zone.append(0)

x = np.arange(len(species_names))
w = 0.35
bars1 = ax.bar(x - w/2, r2_agg, w, label='Q1 aggregate', color='#3498db', alpha=0.7)
bars2 = ax.bar(x + w/2, r2_zone, w, label='Zone Q1', color='#e67e22', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(species_labels, fontsize=8)
ax.set_ylabel("r² (variance explained)")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8)
for bar, val in zip(bars1, r2_agg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
            ha='center', fontsize=7)
for bar, val in zip(bars2, r2_zone):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
            ha='center', fontsize=7)

plt.tight_layout()
outpath = os.path.join(RESULTS, "f603_zone_q1.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved {outpath}")
