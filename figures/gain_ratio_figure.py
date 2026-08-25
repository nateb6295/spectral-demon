#!/usr/bin/env python3
"""R24 figure: σ₁/σ₂ gain ratio across depth — selective amplifier reframe."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("spectral-demon/results/error_prop_gemma-2-2b.json") as f:
    gemma = json.load(f)
with open("spectral-demon/results/error_prop_Qwen2.5-7B-Instruct.json") as f:
    qwen = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
fig.suptitle("CCS Selective Amplification — σ₁/σ₂ Gain Ratio by Depth (D2)",
             fontsize=13, fontweight='bold')

for ax, (name, data, color) in zip(axes, [
    ("Gemma-2B (sorter, GQA 2:1)", gemma, "#e63946"),
    ("Qwen-7B (relay, GQA 7:1)", qwen, "#457b9d"),
]):
    d2 = data["ledger"]["D2"]["layers"]
    layers = sorted(d2.keys(), key=int)
    L = [int(x) for x in layers]
    ratios = []
    for x in layers:
        s1r, s2r = d2[x]["sigma1_ref"], d2[x]["sigma2_ref"]
        s1c, s2c = d2[x]["sigma1_ccs"], d2[x]["sigma2_ccs"]
        e_ref = s1r**2 + s2r**2
        ds1 = (s1c**2 - s1r**2) / e_ref * 100
        ds2 = (s2c**2 - s2r**2) / e_ref * 100
        ratios.append(ds1 / ds2 if ds2 > 0.01 else float('nan'))

    ax.bar(L, ratios, color=color, alpha=0.7, edgecolor=color)
    ax.axhline(1.0, color='gray', linewidth=1, linestyle='--', label='Equal gain')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel("Layer")
    ax.set_ylabel("σ₁²/σ₂² gain ratio" if ax == axes[0] else "")

    if "Gemma" in name:
        ax.axvspan(-0.5, 10.5, alpha=0.06, color='blue', label='Broad zone')
        ax.axvspan(10.5, 20.5, alpha=0.06, color='green', label='Selective zone')
        ax.axvspan(20.5, max(L) + 0.5, alpha=0.06, color='orange', label='Sharp σ₁ zone')
        ax.legend(fontsize=8, loc='upper left')
    else:
        ax.legend(fontsize=8)

out = "spectral-demon/figures/gain_ratio_r24.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
