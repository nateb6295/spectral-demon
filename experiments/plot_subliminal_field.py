#!/usr/bin/env python3
"""Plot subliminal CCS field profiles for all five source architectures.

Shows per-layer CCS delta (σ₂/σ₁ ratio change) for each source model,
with responsive zones highlighted and the MAD threshold marked.
Visualizes the "current beneath the whitecaps" — the distributed
sub-threshold signal that dominates cross-architecture injection.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(__file__) + "/results"

ZONE_FILES = [
    ("functional_gemma-2-2b_LFM2.5-1.2B-Instruct.json", "Gemma-2-2b", "#2ecc71"),
    ("functional_pythia-2.8b_LFM2.5-1.2B.json", "Pythia-2.8b", "#3498db"),
    ("functional_gpt2-xl_LFM2.5-1.2B.json", "GPT-2-XL", "#e74c3c"),
    ("functional_qwen-7b_LFM2.5-1.2B.json", "Qwen-7B", "#9b59b6"),
    ("functional_phi-2_zone_map.json", "Phi-2", "#f39c12"),
]


def load_zone_profile(filepath):
    with open(filepath) as f:
        d = json.load(f)
    zones = d.get("zones_a", d.get("zones", d.get("source_zones", [])))
    if isinstance(zones, dict):
        entries = []
        for k, v in sorted(zones.items(), key=lambda x: int(x[0])):
            entries.append(v)
        zones = entries

    layers = []
    deltas = []
    zone_types = []
    for entry in zones:
        if isinstance(entry, dict):
            layers.append(entry.get("layer", len(layers)))
            deltas.append(entry.get("delta", 0))
            zone_types.append(entry.get("zone", "INVARIANT"))
    return layers, deltas, zone_types


fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=False)

for idx, (fname, label, color) in enumerate(ZONE_FILES):
    ax = axes[idx]
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        ax.text(0.5, 0.5, f"{fname} not found", transform=ax.transAxes, ha='center')
        ax.set_title(label)
        continue

    layers, deltas, zones = load_zone_profile(fpath)
    if not layers:
        ax.text(0.5, 0.5, "No zone data", transform=ax.transAxes, ha='center')
        ax.set_title(label)
        continue

    n = len(layers)
    rel_depths = np.array(layers) / max(layers) if max(layers) > 0 else np.array(layers)

    colors = []
    for z in zones:
        if "+" in z:
            colors.append("#2ecc71")
        elif "-" in z:
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")

    ax.bar(rel_depths, deltas, width=0.8/n, color=colors, alpha=0.8, edgecolor='none')
    ax.axhline(y=0, color='black', linewidth=0.5)

    inv_deltas = [d for d, z in zip(deltas, zones) if "INVARIANT" in z]
    if inv_deltas:
        mean_inv = sum(inv_deltas) / len(inv_deltas)
        ax.axhline(y=mean_inv, color=color, linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Mean invariant: {mean_inv:+.4f}')

    pos_inv = sum(1 for d in inv_deltas if d > 0)
    neg_inv = sum(1 for d in inv_deltas if d < 0)

    rp = sum(1 for z in zones if "+" in z)
    rm = sum(1 for z in zones if "-" in z)
    ni = sum(1 for z in zones if "INVARIANT" in z)

    title = f"{label}  ({n}L, R+={rp}, R-={rm}, INV={ni}, field: {pos_inv}/{ni} positive)"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('CCS Δ(σ₂/σ₁)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Relative Depth (0=first layer, 1=last layer)', fontsize=11)

fig.suptitle('Subliminal CCS Field Profiles — Five Source Architectures\n'
             'Green bars = R+, Red bars = R-, Gray bars = Invariant (the subliminal field)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

outpath = os.path.join(RESULTS_DIR, "subliminal_field_5source.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved to {outpath}")
