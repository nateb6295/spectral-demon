#!/usr/bin/env python3
"""
Figure 4: Default Witness Gradient (F47)
σ₂ trajectories through all layers for CodeQwen 7B (GQA) and CodeLlama 7B (MHA).
Top: raw σ₂ per condition. Bottom: (recv − absent) σ₂ difference per layer.
GQA shows signed enrichment; MHA shows near-zero difference (both deviate from control equally).
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "fig4_witness_gradient.png"
RESULTS = Path(__file__).parent.parent / "results"

with open(RESULTS / "exp_codeqwen_f47_20260528.json") as f:
    gqa = json.load(f)
with open(RESULTS / "exp_codellama_f47_20260528_2220.json") as f:
    mha = json.load(f)

gqa_grad = gqa['f47_gradient']
mha_grad = mha['f47_gradient']

gqa_layers = [g['layer'] for g in gqa_grad]
mha_layers = [g['layer'] for g in mha_grad]

cond_colors = {'receptive': '#2980b9', 'control': '#2c3e50', 'absent': '#e74c3c'}

fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 2]})
fig.suptitle('Default Witness Gradient (F47): σ₂ Trajectory by Condition',
             fontsize=13, fontweight='bold')

# Panel A: CodeQwen (GQA) σ₂ trajectories
ax = axes[0, 0]
for key, label in [('recv_s2', 'Receptive'), ('ctrl_s2', 'Control'), ('absent_s2', 'Absent')]:
    cond = label.lower()
    vals = [g[key] for g in gqa_grad]
    ax.plot(gqa_layers, vals, '-o', color=cond_colors[cond], label=label,
            markersize=2.5, linewidth=1.5, alpha=0.85)
ax.set_ylabel('σ₂', fontsize=11)
ax.set_title('CodeQwen 7B (GQA, s=4)', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.axvspan(2, 28, alpha=0.04, color='gray')
ax.text(15, ax.get_ylim()[1] * 0.85, 'TUNNEL', fontsize=9, ha='center',
        color='gray', fontstyle='italic', alpha=0.5)

# Panel B: CodeLlama (MHA) σ₂ trajectories
ax = axes[0, 1]
for key, label in [('recv_s2', 'Receptive'), ('ctrl_s2', 'Control'), ('absent_s2', 'Absent')]:
    cond = label.lower()
    vals = [g[key] for g in mha_grad]
    ax.plot(mha_layers, vals, '-o', color=cond_colors[cond], label=label,
            markersize=2.5, linewidth=1.5, alpha=0.85)
ax.set_ylabel('σ₂', fontsize=11)
ax.set_title('CodeLlama 7B (MHA, s=1)', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.axvspan(2, 28, alpha=0.04, color='gray')
ax.text(15, ax.get_ylim()[1] * 0.85, 'TUNNEL', fontsize=9, ha='center',
        color='gray', fontstyle='italic', alpha=0.5)

# Panel C: Normalized σ₂ difference (recv - absent) / ctrl — tunnel zoom
# Normalize by control σ₂ to show relative enrichment
diff_gqa_norm = [(g['recv_s2'] - g['absent_s2']) / g['ctrl_s2'] if g['ctrl_s2'] > 0.1 else 0
                 for g in gqa_grad]
diff_mha_norm = [(g['recv_s2'] - g['absent_s2']) / g['ctrl_s2'] if g['ctrl_s2'] > 0.1 else 0
                 for g in mha_grad]

# Tunnel zoom (L1-28) for both
tunnel_gqa = [(l, d) for l, d in zip(gqa_layers, diff_gqa_norm) if 1 <= l <= 28]
tunnel_mha = [(l, d) for l, d in zip(mha_layers, diff_mha_norm) if 1 <= l <= 28]

ax = axes[1, 0]
tl_gqa, td_gqa = zip(*tunnel_gqa)
ax.bar(tl_gqa, td_gqa, color=['#2980b9' if d > 0 else '#e74c3c' for d in td_gqa],
       alpha=0.7, width=0.8)
ax.axhline(y=0, color='#2c3e50', linewidth=1, linestyle='-')
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('(σ₂_recv − σ₂_abs) / σ₂_ctrl', fontsize=9)
ax.set_title('GQA Tunnel: Consistent enrichment (recv > abs)', fontsize=10, color='#2980b9')
mean_gqa = np.mean(td_gqa)
ax.axhline(y=mean_gqa, color='#2980b9', linewidth=1, linestyle='--', alpha=0.5)
ax.text(25, mean_gqa + 0.01, f'mean={mean_gqa:+.3f}', fontsize=8, color='#2980b9')

ax = axes[1, 1]
tl_mha, td_mha = zip(*tunnel_mha)
ax.bar(tl_mha, td_mha, color=['#2980b9' if d > 0 else '#e74c3c' for d in td_mha],
       alpha=0.7, width=0.8)
ax.axhline(y=0, color='#2c3e50', linewidth=1, linestyle='-')
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('(σ₂_recv − σ₂_abs) / σ₂_ctrl', fontsize=9)
ax.set_title('MHA Tunnel: Near-zero (recv ≈ abs, both ≠ ctrl)', fontsize=10, color='#e74c3c')
mean_mha = np.mean(td_mha)
ax.axhline(y=mean_mha, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.5)
ax.text(25, mean_mha + 0.01, f'mean={mean_mha:+.3f}', fontsize=8, color='#e74c3c')

# Match y-axes on bottom row
y_max = max(max(abs(d) for d in td_gqa), max(abs(d) for d in td_mha)) * 1.3
axes[1, 0].set_ylim(-y_max, y_max)
axes[1, 1].set_ylim(-y_max, y_max)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT}")
