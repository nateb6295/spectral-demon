#!/usr/bin/env python3
"""Fig 10: LLaMA-1 vs Pythia 6.9B — Liu confound resolution.
Shows per-layer ΔS and ρ₂ for both RMSNorm+MHA and LayerNorm+MHA,
demonstrating normalization type doesn't change the gradient."""
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from glob import glob

results_dir = Path(__file__).parent.parent / "results"

def load_latest(pattern):
    files = sorted(glob(str(results_dir / pattern)))
    if not files:
        print(f"No files matching {pattern}")
        return None
    return json.load(open(files[-1]))

def get_rho2_by_layer(raw_data, condition='receptive'):
    layers = defaultdict(list)
    for entry in raw_data:
        if entry['condition'] == condition:
            s2, s3 = entry.get('sigma_2', 0), entry.get('sigma_3', 0)
            if s3 > 0:
                layers[entry['layer']].append(s2/s3)
    return {l: np.mean(v) for l, v in layers.items()}

pythia = load_latest("exp_pythia69b_perlayer_*.json")
llama1 = load_latest("exp_llama1_perlayer_*.json")

if not pythia:
    sys.exit("Missing Pythia 6.9B results")
if not llama1:
    sys.exit("Missing LLaMA-1 results — run after experiment completes")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

models = [
    (axes[0], pythia, "Pythia 6.9B (LayerNorm + MHA)", 33),
    (axes[1], llama1, "LLaMA-1 7B (RMSNorm + MHA)", 33),
]

stats = {}
for ax, data, title, n_layers in models:
    rho2 = get_rho2_by_layer(data['raw'])
    layers_sorted = sorted(rho2.keys())
    ds_vals = [data['delta_S_by_layer'].get(str(l), 0) for l in layers_sorted]
    rho_vals = [rho2[l] for l in layers_sorted]
    depth_pct = [l / (n_layers - 1) * 100 for l in layers_sorted]

    colors = ['#2196F3' if r < 2.0 else '#F44336' for r in rho_vals]

    ax.bar(depth_pct, ds_vals, width=100/(n_layers*1.2), color=colors, alpha=0.7, zorder=3)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Depth (%)', fontsize=11)
    ax.set_ylabel('ΔS (receptive − absent)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 103)

    ax2 = ax.twinx()
    ax2.plot(depth_pct, rho_vals, 'k-', alpha=0.5, linewidth=1.5, zorder=2)
    ax2.axhline(y=2.0, color='#FF9800', linewidth=2, linestyle='--', alpha=0.8, label='ρ₂ = 2.0')
    ax2.set_ylabel('ρ₂ = σ₂/σ₃', fontsize=11, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0.5, 4.5)
    ax2.legend(loc='upper right', fontsize=9)

    n_resp = sum(1 for r in rho_vals if r < 2.0)
    tunnel_layers = [i for i, l in enumerate(layers_sorted) if l < n_layers - 1]
    tunnel_ds = [ds_vals[i] for i in tunnel_layers]
    tunnel_rho = [rho_vals[i] for i in tunnel_layers]
    if len(tunnel_rho) > 2:
        from scipy.stats import pearsonr
        r_val, p_val = pearsonr(tunnel_rho, tunnel_ds)
    else:
        r_val, p_val = 0, 1

    crossover = None
    for i, l in enumerate(layers_sorted):
        if rho_vals[i] >= 2.0 and l > 0:
            crossover = l
            break

    info = f'Responsive: {n_resp}/{len(rho_vals)}\nr(ρ₂,ΔS) = {r_val:.3f}\nCrossover: L{crossover}'
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

    stats[title] = {'n_resp': n_resp, 'total': len(rho_vals),
                     'r': r_val, 'crossover': crossover,
                     'tunnel_mean': np.mean(tunnel_ds)}

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2196F3', alpha=0.7, label='Responsive (ρ₂ < 2.0)'),
    Patch(facecolor='#F44336', alpha=0.7, label='Rigid (ρ₂ > 2.0)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Liu Confound Resolution: Normalization ≠ Attention Architecture',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(str(Path(__file__).parent / 'fig10_liu_confound.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Saved fig10_liu_confound.png")
print()
for name, s in stats.items():
    print(f"{name}:")
    print(f"  Responsive: {s['n_resp']}/{s['total']} ({s['n_resp']/s['total']:.0%})")
    print(f"  r(ρ₂, ΔS) = {s['r']:.3f}")
    print(f"  Crossover: L{s['crossover']}")
    print(f"  Tunnel mean ΔS: {s['tunnel_mean']:.4f}")
    print()

r_diff = abs(stats[list(stats.keys())[0]]['r'] - stats[list(stats.keys())[1]]['r'])
if r_diff < 0.15:
    print("VERDICT: Gradient shapes MATCH (|Δr| < 0.15)")
    print("→ Normalization does NOT modulate the sensitivity gradient")
    print("→ Liu confound FULLY resolved: GQA/MHA is the sole factor")
elif r_diff < 0.30:
    print("VERDICT: Gradient shapes SIMILAR (0.15 < |Δr| < 0.30)")
    print("→ Normalization is a secondary modulator")
else:
    print("VERDICT: Gradient shapes DIFFER (|Δr| > 0.30)")
    print("→ Both factors contribute independently")
