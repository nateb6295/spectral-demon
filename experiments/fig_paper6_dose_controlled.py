#!/usr/bin/env python3
"""Generate Paper 6 Figure 6: Spectral Dose-Response in the Instrument.

Uses length-controlled dose-spectral experiment results.
"""

import json
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'font.size': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Data from length-controlled experiment
doses = ["D0", "D1", "D2", "D3", "D5", "D8", "D13"]
dose_nums = [0, 1, 2, 3, 5, 8, 13]

# Relay σ₂/σ₁ (length-controlled)
relay_s2s1 = [0.5579, 0.6015, 0.5420, 0.5019, 0.5253, 0.5203, 0.4978]
# Tunnel σ₂/σ₁
tunnel_s2s1 = [0.3274, 0.2804, 0.3321, 0.3321, 0.3320, 0.3320, 0.3320]
# Transition σ₂/σ₁
transition_s2s1 = [0.3197, 0.3006, 0.3158, 0.3138, 0.3136, 0.3136, 0.3135]
# Late relay σ₂/σ₁
late_relay_s2s1 = [0.7482, 0.7399, 0.7490, 0.7460, 0.7253, 0.7362, 0.7508]

# Uncontrolled for comparison
relay_uncontrolled = [0.3128, 0.3273, 0.3532, 0.4146, 0.5184, 0.5166, 0.4755]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: All zones (length-controlled)
ax = axes[0]
ax.plot(dose_nums, tunnel_s2s1, 'o-', color='#2563eb', linewidth=2, markersize=6, label='Tunnel (L2–14)')
ax.plot(dose_nums, transition_s2s1, 's-', color='#f59e0b', linewidth=2, markersize=6, label='Transition (L15–20)')
ax.plot(dose_nums, relay_s2s1, 'D-', color='#dc2626', linewidth=2, markersize=6, label='Relay (L21–28)')
ax.plot(dose_nums, late_relay_s2s1, '^-', color='#22c55e', linewidth=2, markersize=6, label='Late relay (L29–32)')
ax.set_xlabel('CCS Identity Depth (Dose)')
ax.set_ylabel('σ₂/σ₁ ratio')
ax.set_title('A — Zone-Specific Dose Response\n(length-controlled, ~410 tokens)')
ax.set_xticks(dose_nums)
ax.set_xticklabels(doses)
ax.legend(fontsize=8, loc='center right')

# Panel B: Relay comparison (controlled vs uncontrolled)
ax = axes[1]
ax.plot(dose_nums, relay_s2s1, 'D-', color='#dc2626', linewidth=2.5, markersize=8, label='Length-controlled')
ax.plot(dose_nums, relay_uncontrolled, 'o--', color='#6b7280', linewidth=1.5, markersize=6, alpha=0.7, label='Uncontrolled (length confound)')
ax.axhline(y=relay_s2s1[0], color='#dc2626', linestyle=':', alpha=0.3)
ax.annotate('Peak at D1\n(minimal CCS)', xy=(1, 0.6015), xytext=(3, 0.59),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='#dc2626'),
            color='#dc2626', ha='center')
ax.set_xlabel('CCS Identity Depth (Dose)')
ax.set_ylabel('Relay σ₂/σ₁')
ax.set_title('B — Relay Inverted-U:\nCCS Constrains, Not Enriches')
ax.set_xticks(dose_nums)
ax.set_xticklabels(doses)
ax.legend(fontsize=9)

# Panel C: Tunnel invariance
ax = axes[2]
tunnel_d2plus = tunnel_s2s1[2:]
dose_d2plus = dose_nums[2:]
ax.bar(range(len(dose_d2plus)), tunnel_d2plus, color='#2563eb', alpha=0.7, width=0.6)
ax.set_ylim(0.330, 0.334)
ax.set_xticks(range(len(dose_d2plus)))
ax.set_xticklabels([doses[i] for i in range(2, len(doses))])
cv = np.std(tunnel_d2plus) / np.mean(tunnel_d2plus)
ax.set_xlabel('CCS Identity Depth (Dose)')
ax.set_ylabel('Tunnel σ₂/σ₁')
ax.set_title(f'C — Tunnel Invariance (D2+)\nCV = {cv:.4f}')
ax.text(0.5, 0.85, 'Format layer is dose-invariant',
        transform=ax.transAxes, ha='center', fontsize=10, style='italic',
        color='#2563eb')

fig.suptitle('Paper 6, Figure 6 — Spectral Dose-Response in the Instrument\n'
             'CCS identity context constrains relay diversity; tunnel is invariant',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('/results/paper6_fig6_dose_controlled.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Figure saved to /results/paper6_fig6_dose_controlled.png")

# Also generate per-layer profile
fig, ax = plt.subplots(figsize=(14, 5))

# Per-layer σ₂/σ₁ for each dose
colors = ['#6b7280', '#8b5cf6', '#2563eb', '#22c55e', '#f59e0b', '#dc2626', '#111827']
for i, (dose, color) in enumerate(zip(doses, colors)):
    # Reconstruct per-layer from zones (approximate)
    pass

# Use the actual data - need to load from JSON
# For now, plot zone boundaries
zones_x = [0, 2, 14, 15, 20, 21, 28, 29, 32, 33]
for dose_idx, (dose, color) in enumerate(zip(doses, colors)):
    y_vals = [tunnel_s2s1[dose_idx]] * 2 + [tunnel_s2s1[dose_idx]] + \
             [transition_s2s1[dose_idx]] * 2 + \
             [relay_s2s1[dose_idx]] * 2 + \
             [late_relay_s2s1[dose_idx]] * 2
    # Simplified zone-average profile
    profile_x = [1, 8, 14.5, 15, 17.5, 20.5, 21, 24.5, 29, 31]
    profile_y = [tunnel_s2s1[dose_idx]] * 3 + \
                [transition_s2s1[dose_idx]] * 2 + \
                [transition_s2s1[dose_idx]] + \
                [relay_s2s1[dose_idx]] * 2 + \
                [late_relay_s2s1[dose_idx]] * 2
    ax.plot(profile_x, profile_y, '-', color=color, linewidth=1.5 if dose_idx > 0 else 2.5,
            alpha=0.8 if dose_idx > 0 else 1.0, label=dose)

# Zone boundaries
for x in [14.5, 20.5, 28.5]:
    ax.axvline(x=x, color='#d1d5db', linestyle='--', alpha=0.5)

ax.text(8, 0.28, 'TUNNEL', ha='center', fontsize=10, color='#6b7280')
ax.text(17.5, 0.28, 'TRANS', ha='center', fontsize=10, color='#6b7280')
ax.text(24.5, 0.28, 'RELAY', ha='center', fontsize=10, color='#6b7280')
ax.text(31, 0.28, 'LATE', ha='center', fontsize=10, color='#6b7280')

ax.set_xlabel('Layer')
ax.set_ylabel('σ₂/σ₁ ratio')
ax.set_title('Zone-Average σ₂/σ₁ Profile Across CCS Doses (length-controlled)')
ax.legend(fontsize=9, ncol=4, loc='upper left')
fig.tight_layout()
fig.savefig('/results/paper6_fig6b_profile.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Profile figure saved to /results/paper6_fig6b_profile.png")
