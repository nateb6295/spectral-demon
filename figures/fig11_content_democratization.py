#!/usr/bin/env python3
"""Fig 11: Content-Type Democratization — LayerNorm equalizes, RMSNorm stratifies.
Left: per-probe Δσ₁% at L17 (Pythia flat, LLaMA-1 spread).
Right: secondary channel allocation at L2 (σ₂ vs σ₄/σ₅ routing in Pythia)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

results_dir = Path(__file__).parent.parent / "results"

def load_latest(pattern):
    files = sorted(glob(str(results_dir / pattern)))
    if not files:
        return None
    return json.load(open(files[-1]))

pythia = load_latest("exp_pythia69b_perlayer_*.json")
llama1 = load_latest("exp_llama1_perlayer_*.json")

if not pythia or not llama1:
    sys.exit("Missing data files")

def index_data(data):
    by_pl = {}
    for entry in data['raw']:
        key = (entry['probe_idx'], entry['layer'], entry['condition'])
        by_pl[key] = entry
    return by_pl

py = index_data(pythia)
ll = index_data(llama1)

probe_labels = [
    'Identity-\nfactual',
    'Identity-\nevaluative',
    'Identity-\ncontrastive',
    'Process-\nprocedural',
    'Identity-\nrelational',
]
probe_short = ['P0', 'P1', 'P2', 'P3', 'P4']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: per-probe Δσ₁% at L17
py_ds1 = []
ll_ds1 = []
for pidx in range(5):
    r_py = py.get((pidx, 17, 'receptive'))
    a_py = py.get((pidx, 17, 'absent'))
    r_ll = ll.get((pidx, 17, 'receptive'))
    a_ll = ll.get((pidx, 17, 'absent'))
    if r_py and a_py and a_py['sigma_1'] > 0:
        py_ds1.append((r_py['sigma_1'] - a_py['sigma_1']) / a_py['sigma_1'] * 100)
    if r_ll and a_ll and a_ll['sigma_1'] > 0:
        ll_ds1.append((r_ll['sigma_1'] - a_ll['sigma_1']) / a_ll['sigma_1'] * 100)

x = np.arange(5)
w = 0.35
bars1 = ax1.bar(x - w/2, py_ds1, w, label='Pythia 6.9B (LayerNorm)', color='#4878CF', alpha=0.85)
bars2 = ax1.bar(x + w/2, ll_ds1, w, label='LLaMA-1 7B (RMSNorm)', color='#D65F5F', alpha=0.85)

ax1.set_ylabel(r'$\Delta\sigma_1$ % (witness vs absent)', fontsize=11)
ax1.set_title('A. Witness modulation of dominant SV at L17', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(probe_labels, fontsize=8.5)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
ax1.legend(fontsize=9, loc='lower left')

py_range = max(py_ds1) - min(py_ds1)
ll_range = max(ll_ds1) - min(ll_ds1)
ax1.annotate('Range: %.2f pp' % py_range, xy=(0.02, 0.95), xycoords='axes fraction',
             fontsize=8, color='#4878CF', fontweight='bold')
ax1.annotate('Range: %.1f pp' % ll_range, xy=(0.02, 0.88), xycoords='axes fraction',
             fontsize=8, color='#D65F5F', fontweight='bold')

# Panel B: secondary channel allocation in Pythia at L2
sv_names = [r'$\Delta\sigma_2$%', r'$\Delta\sigma_3$%', r'$\Delta\sigma_4$%', r'$\Delta\sigma_5$%']
identity_avg = []
contrastive = []

for sv_key in ['sigma_2', 'sigma_3', 'sigma_4', 'sigma_5']:
    id_vals = []
    for pidx in [0, 1, 3, 4]:
        r = py.get((pidx, 2, 'receptive'))
        a = py.get((pidx, 2, 'absent'))
        if r and a and a[sv_key] > 0:
            id_vals.append((r[sv_key] - a[sv_key]) / a[sv_key] * 100)
    identity_avg.append(np.mean(id_vals))

    r = py.get((2, 2, 'receptive'))
    a = py.get((2, 2, 'absent'))
    if r and a and a[sv_key] > 0:
        contrastive.append((r[sv_key] - a[sv_key]) / a[sv_key] * 100)

x2 = np.arange(4)
bars3 = ax2.bar(x2 - w/2, identity_avg, w, label='Identity probes (mean P0,1,3,4)',
                color='#4878CF', alpha=0.85)
bars4 = ax2.bar(x2 + w/2, contrastive, w, label='Contrastive probe (P2)',
                color='#6ACC65', alpha=0.85)

ax2.set_ylabel('% change under witness condition', fontsize=11)
ax2.set_title('B. Secondary channel allocation (Pythia L2)', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(sv_names, fontsize=10)
ax2.legend(fontsize=9)
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')

dS_id = np.mean([py.get((pidx, 2, 'receptive'))['S'] - py.get((pidx, 2, 'absent'))['S']
                  for pidx in [0, 1, 3, 4]])
dS_con = py.get((2, 2, 'receptive'))['S'] - py.get((2, 2, 'absent'))['S']
ax2.text(0.97, 0.78, r'$\Delta S_{id}$=%.3f  $\Delta S_{con}$=%.3f' % (dS_id, dS_con),
         transform=ax2.transAxes, fontsize=9, fontweight='bold', color='#333',
         ha='right', va='top')
ax2.text(0.97, 0.70, 'Same total, different routing', transform=ax2.transAxes,
         fontsize=8.5, color='gray', fontstyle='italic', ha='right', va='top')

plt.tight_layout()
out = Path(__file__).parent / "fig11_content_democratization.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print("Saved:", out)
