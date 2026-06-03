#!/usr/bin/env python3
"""
Figure 8: Gradient Model — Scale Compresses the Responsive Niche
Dual panel: Pythia 410M (wide responsive zone) vs 6.9B (compressed responsive zone).
Top: per-layer ΔS with responsive zone shading.
Bottom: ρ₂ (σ₂/σ₃ ratio) showing crossover to rigid zone.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT = Path(__file__).parent / "fig8_gradient_model.png"
RESULTS = Path(__file__).parent.parent / "results"

def load_and_analyze(filepath):
    data = json.load(open(filepath))
    delta = data["delta_S_by_layer"]
    n_layers = data["n_layers"]

    sigma_by_layer = defaultdict(lambda: defaultdict(list))
    for r in data["raw"]:
        sigma_by_layer[r["layer"]][r["condition"]].append(
            (r.get("sigma_2", 0), r.get("sigma_3", 0))
        )

    layers, ds_vals, rhos = [], [], []
    for layer_str in sorted(delta.keys(), key=int):
        layer = int(layer_str)
        if not (2 <= layer <= n_layers - 4):
            continue
        layers.append(layer)
        ds_vals.append(delta[layer_str])
        all_s2 = [s[0] for cond in sigma_by_layer[layer].values() for s in cond]
        all_s3 = [s[1] for cond in sigma_by_layer[layer].values() for s in cond]
        if all_s2 and all_s3 and np.mean(all_s3) > 0:
            rhos.append(np.mean(all_s2) / np.mean(all_s3))
        else:
            rhos.append(1.0)

    return np.array(layers), np.array(ds_vals), np.array(rhos)

# Load data
p410_files = sorted(RESULTS.glob("exp_pythia410m_perlayer_*.json"))
p69b_files = sorted(RESULTS.glob("exp_pythia69b_perlayer_*.json"))

if not p410_files or not p69b_files:
    print("Need both 410M and 6.9B results. Available:")
    print(f"  410M: {[f.name for f in p410_files]}")
    print(f"  6.9B: {[f.name for f in p69b_files]}")
    exit(1)

l410, ds410, rho410 = load_and_analyze(str(p410_files[-1]))
l69b, ds69b, rho69b = load_and_analyze(str(p69b_files[-1]))

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
fig.suptitle('Scale Compresses the Responsive Niche: Gradient Model (Scenario C)',
             fontsize=13, fontweight='bold', y=0.98)

for col, (layers, ds, rho, label, n_layers) in enumerate([
    (l410, ds410, rho410, 'Pythia 410M (LayerNorm+MHA)', 24),
    (l69b, ds69b, rho69b, 'Pythia 6.9B (LayerNorm+MHA)', 32),
]):
    # Normalize to fractional depth
    frac = layers / n_layers

    # Top panel: ΔS per layer
    ax = axes[0, col]
    responsive = rho < 2.0
    ax.fill_between(frac, -0.02, 0.1, where=responsive,
                    alpha=0.15, color='#27ae60', label='Responsive zone')
    ax.fill_between(frac, -0.02, 0.1, where=~responsive,
                    alpha=0.1, color='#e74c3c', label='Rigid zone')
    ax.bar(frac, ds, width=0.02, color=['#27ae60' if r else '#e74c3c' for r in responsive],
           alpha=0.8, edgecolor='none')
    ax.axhline(0, color='#34495e', linewidth=0.8, linestyle='--')
    ax.set_ylabel('ΔS (receptive − absent)', fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_ylim(-0.02, max(ds) * 1.2)

    n_resp = responsive.sum()
    n_total = len(responsive)
    ax.text(0.98, 0.95, f'Responsive: {n_resp}/{n_total} ({100*n_resp/n_total:.0f}%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Bottom panel: ρ₂ with threshold line
    ax = axes[1, col]
    ax.plot(frac, rho, 'o-', color='#2c3e50', markersize=4, linewidth=1.5)
    ax.axhline(2.0, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7, label='ρ₂ = 2.0 threshold')
    ax.fill_between(frac, 0, 2.0, alpha=0.08, color='#27ae60')
    ax.fill_between(frac, 2.0, max(rho) * 1.1, alpha=0.06, color='#e74c3c')
    ax.set_ylabel('ρ₂ = σ₂/σ₃', fontsize=10)
    ax.set_xlabel('Fractional depth', fontsize=10)
    ax.set_ylim(0.8, max(rho) * 1.1)

    # Mark crossover
    for i in range(len(rho) - 1):
        if rho[i] < 2.0 and rho[i + 1] >= 2.0:
            cross_frac = (frac[i] + frac[i + 1]) / 2
            ax.axvline(cross_frac, color='#8e44ad', linewidth=1.5, linestyle=':',
                       alpha=0.7)
            ax.text(cross_frac + 0.01, max(rho) * 0.95, f'Crossover\nL{layers[i+1]}',
                    fontsize=8, color='#8e44ad', va='top')
            break

axes[0, 0].legend(loc='upper right', fontsize=8)
axes[1, 0].legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(str(OUT), dpi=200, bbox_inches='tight')
print(f"Saved: {OUT}")
