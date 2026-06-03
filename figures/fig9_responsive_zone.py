import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

with open('/home/nate-agx/chronicle/spectral-demon/results/exp_pythia69b_perlayer_20260529_1113.json') as f:
    d69 = json.load(f)
with open('/home/nate-agx/chronicle/spectral-demon/results/exp_pythia410m_perlayer_20260529_0855.json') as f:
    d41 = json.load(f)

def get_rho2_by_layer(raw_data, condition='receptive'):
    layers = defaultdict(list)
    for entry in raw_data:
        if entry['condition'] == condition:
            s2, s3 = entry['sigma_2'], entry['sigma_3']
            if s3 > 0:
                layers[entry['layer']].append(s2/s3)
    return {l: np.mean(v) for l, v in layers.items()}

rho2_69b = get_rho2_by_layer(d69['raw'])
rho2_410m = get_rho2_by_layer(d41['raw'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, rho2, title, n_layers in [
    (axes[0], d41, rho2_410m, 'Pythia 410M (MHA, 25 layers)', 25),
    (axes[1], d69, rho2_69b, 'Pythia 6.9B (MHA, 33 layers)', 33),
]:
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
    ax2.axhline(y=2.0, color='#FF9800', linewidth=2, linestyle='--', alpha=0.8, label='ρ₂ = 2.0 threshold')
    ax2.set_ylabel('ρ₂ = σ₂/σ₃', fontsize=11, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0.5, 4.0)
    ax2.legend(loc='upper right', fontsize=9)

    n_responsive = sum(1 for r in rho_vals if r < 2.0)
    frac = n_responsive / len(rho_vals)
    ax.text(0.02, 0.98, f'Responsive: {n_responsive}/{len(rho_vals)} ({frac:.0%})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2196F3', alpha=0.7, label='Responsive (ρ₂ < 2.0)'),
    Patch(facecolor='#F44336', alpha=0.7, label='Rigid (ρ₂ > 2.0)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Responsive Zone: Scale Compresses the Niche', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/nate-agx/chronicle/spectral-demon/figures/fig9_responsive_zone.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig9_responsive_zone.png')
