#!/usr/bin/env python3
"""Generate cross-species error propagation comparison figure."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS = {
    "Gemma-2B\n(sorter, GQA 2:1)": "spectral-demon/results/error_prop_gemma-2-2b.json",
    "Qwen-7B\n(relay, GQA 7:1)": "spectral-demon/results/error_prop_Qwen2.5-7B-Instruct.json",
    "Pythia-1.4B\n(tunnel, MHA)": "spectral-demon/results/error_prop_pythia-1.4b.json",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
fig.suptitle("CCS Error Propagation Ledger — D2 Therapeutic Window\nPer-layer CCS-specific (neutral subtracted)", fontsize=14, fontweight='bold')

colors = {"Gemma-2B\n(sorter, GQA 2:1)": "#e63946",
          "Qwen-7B\n(relay, GQA 7:1)": "#457b9d",
          "Pythia-1.4B\n(tunnel, MHA)": "#2a9d8f"}

for col, (label, path) in enumerate(RESULTS.items()):
    with open(path) as f:
        data = json.load(f)

    d2 = data["ledger"]["D2"]["layers"]
    layers = sorted(d2.keys(), key=int)
    L = [int(x) for x in layers]
    delta_e = [d2[x]["delta_e_specific_pct"] for x in layers]
    delta_a = [d2[x]["delta_a_specific"] for x in layers]

    c = colors[label]

    ax_e = axes[0, col]
    ax_e.bar(L, delta_e, color=c, alpha=0.7, edgecolor=c)
    ax_e.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_e.set_title(label, fontsize=11, fontweight='bold')
    ax_e.set_ylabel("ΔE specific (%)" if col == 0 else "")
    ax_e.set_xlabel("Layer")

    ax_a = axes[1, col]
    ax_a.plot(L, delta_a, color=c, linewidth=2, marker='o', markersize=3)
    ax_a.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_a.set_ylabel("Δa specific" if col == 0 else "")
    ax_a.set_xlabel("Layer")

    if "Gemma" in label:
        ax_a.axvspan(20.5, max(L) + 0.5, alpha=0.1, color='orange', label='Late concentration')
        ax_a.legend(fontsize=8, loc='lower left')

    ax_e_max = max(abs(min(delta_e)), abs(max(delta_e))) * 1.1
    if "Qwen" in label:
        ax_e.set_ylim(-5, 20)
    elif "Pythia" in label:
        ax_e.set_ylim(-10, 50)

out = "spectral-demon/figures/error_prop_three_species.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
