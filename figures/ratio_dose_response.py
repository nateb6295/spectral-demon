#!/usr/bin/env python3
"""R27 figure: Dose-response of zone gain ratios — therapeutic window = ratio crossover."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("spectral-demon/results/error_prop_gemma-2-2b.json") as f:
    data = json.load(f)

doses = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D10']
dose_nums = [2, 3, 4, 5, 6, 7, 8, 10]
zones = {'Broad (L0-L10)': (0, 11), 'Selective (L11-L20)': (11, 21), 'Sharp (L21-L25)': (21, 26)}
colors = {'Broad (L0-L10)': '#457b9d', 'Selective (L11-L20)': '#2a9d8f', 'Sharp (L21-L25)': '#e63946'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
fig.suptitle("CCS Dose-Response: Selectivity Decay Across Three Regimes (Gemma-2B sorter)",
             fontsize=13, fontweight='bold')

for zname, (lo, hi) in zones.items():
    means = []
    for dk in doses:
        if dk not in data['ledger']:
            means.append(float('nan'))
            continue
        layers = data['ledger'][dk]['layers']
        ratios = []
        for L in range(lo, hi):
            lk = str(L)
            if lk not in layers:
                continue
            ld = layers[lk]
            s1r, s2r = ld['sigma1_ref'], ld['sigma2_ref']
            s1c, s2c = ld['sigma1_ccs'], ld['sigma2_ccs']
            e_ref = s1r**2 + s2r**2
            ds1 = (s1c**2 - s1r**2) / e_ref * 100
            ds2 = (s2c**2 - s2r**2) / e_ref * 100
            ratio = ds1 / ds2 if abs(ds2) > 0.01 else float('nan')
            ratios.append(ratio)
        valid = [r for r in ratios if not np.isnan(r)]
        means.append(np.mean(valid) if valid else float('nan'))

    ax1.plot(dose_nums, means, 'o-', color=colors[zname], linewidth=2.5,
             markersize=8, label=zname, zorder=3)

ax1.axhline(1.0, color='gray', linewidth=1.5, linestyle='--', alpha=0.7, label='Equal gain (ratio=1)')
ax1.axvspan(1.5, 3.5, alpha=0.06, color='green', zorder=0)
ax1.axvspan(4.5, 8.5, alpha=0.06, color='#e9c46a', zorder=0)
ax1.axvspan(9.0, 11, alpha=0.06, color='#e63946', zorder=0)
ax1.text(2.5, 0.3, 'therapeutic', fontsize=8, color='green', alpha=0.7, ha='center')
ax1.text(6.5, 0.3, 'transition', fontsize=8, color='#b8860b', alpha=0.7, ha='center')
ax1.text(10, 0.3, 'overdose', fontsize=8, color='#e63946', alpha=0.7, ha='center')
ax1.set_xlabel("Dose (number of CCS turns)", fontsize=11)
ax1.set_ylabel("Mean zone gain ratio", fontsize=11)
ax1.set_title("Zone ratio trajectories", fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.set_xticks(dose_nums)
ax1.set_xticklabels([f'D{d}' for d in dose_nums])
ax1.set_ylim(0, 15)

# Panel 2: s2-dominant layer fraction in broad zone
s2_fracs = []
for dk in doses:
    if dk not in data['ledger']:
        s2_fracs.append(0)
        continue
    layers = data['ledger'][dk]['layers']
    s2_dom = 0
    total = 0
    for L in range(0, 11):
        lk = str(L)
        if lk not in layers:
            continue
        ld = layers[lk]
        s1r, s2r = ld['sigma1_ref'], ld['sigma2_ref']
        s1c, s2c = ld['sigma1_ccs'], ld['sigma2_ccs']
        e_ref = s1r**2 + s2r**2
        ds1 = (s1c**2 - s1r**2) / e_ref * 100
        ds2 = (s2c**2 - s2r**2) / e_ref * 100
        ratio = ds1 / ds2 if abs(ds2) > 0.01 else float('nan')
        if not np.isnan(ratio):
            total += 1
            if ratio < 1.0:
                s2_dom += 1
    s2_fracs.append(100 * s2_dom / total if total > 0 else 0)

bar_colors = ['#2a9d8f' if f > 50 else '#e76f51' if f < 30 else '#e9c46a' for f in s2_fracs]
ax2.bar(dose_nums, s2_fracs, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.7)
ax2.axhline(50, color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax2.set_xlabel("Dose (number of CCS turns)", fontsize=11)
ax2.set_ylabel("% broad-zone layers with ratio < 1.0", fontsize=11)
ax2.set_title("Individual-signal dominance in early layers", fontsize=11, fontweight='bold')
ax2.set_xticks(dose_nums)
ax2.set_xticklabels([f'D{d}' for d in dose_nums])
ax2.set_ylim(0, 100)

for i, (d, f) in enumerate(zip(dose_nums, s2_fracs)):
    ax2.text(d, f + 2, f'{f:.0f}%', ha='center', fontsize=10, fontweight='bold')

out = "spectral-demon/figures/ratio_dose_response_r27.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
