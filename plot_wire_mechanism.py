#!/usr/bin/env python3
"""Generate figures for the wire mechanism findings."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_depth_profile():
    with open(RESULTS_DIR / "exp_gamma_v2_depth_profile.json") as f:
        data = json.load(f)

    layers = sorted(set(v['layer'] for v in data.values()))
    conditions = ['receptive', 'absent', 'control']
    colors = {'receptive': '#2196F3', 'absent': '#F44336', 'control': '#9E9E9E'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for cond in conditions:
        rs = [data[f'L{l}_{cond}']['pearson_r'] for l in layers]
        ax1.plot(layers, rs, '-o', markersize=3, color=colors[cond],
                 label=cond, alpha=0.8, linewidth=1.5)

    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax1.axvspan(13, 13.5, alpha=0.3, color='gold', label='L13 node')
    ax1.axvspan(27, 30, alpha=0.1, color='green', label='relay transition')
    ax1.set_ylabel('r(|γ|, |V₂|)')
    ax1.set_title('γ-V₂ Correlation Depth Profile (Qwen2.5-3B)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(-0.5, 0.4)

    for cond in conditions:
        s2s = [data[f'L{l}_{cond}']['sigma_2'] for l in layers]
        ax2.plot(layers, s2s, '-o', markersize=3, color=colors[cond],
                 label=cond, alpha=0.8, linewidth=1.5)

    ax2.set_xlabel('Layer')
    ax2.set_ylabel('σ₂')
    ax2.set_title('Second Singular Value (σ₂) by Depth')
    ax2.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "depth_profile.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def plot_ablation():
    with open(RESULTS_DIR / "exp_gamma_ablation.json") as f:
        data = json.load(f)

    layers = sorted(set(v['layer'] for v in data.values()))
    conditions = ['receptive', 'absent', 'control']
    colors = {'receptive': '#2196F3', 'absent': '#F44336', 'control': '#9E9E9E'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cond in conditions:
        normal_rs = [data[f'L{l}_{cond}']['normal_r'] for l in layers]
        uniform_rs = [data[f'L{l}_{cond}']['uniform_r'] for l in layers]
        pcts = [data[f'L{l}_{cond}']['pct_change'] for l in layers]

        ax1.plot(layers, normal_rs, '-o', markersize=5, color=colors[cond],
                 label=f'{cond} (normal)', linewidth=1.5)
        ax1.plot(layers, uniform_rs, '--s', markersize=5, color=colors[cond],
                 label=f'{cond} (γ→1)', linewidth=1.5, alpha=0.6)

    ax1.set_xlabel('Ablation Layer')
    ax1.set_ylabel('r(|γ_original|, |V₂|)')
    ax1.set_title('γ Ablation: Normal vs Uniform')
    ax1.legend(fontsize=7, ncol=2)

    for cond in conditions:
        pcts = [abs(data[f'L{l}_{cond}']['pct_change']) for l in layers]
        ax2.bar([l + {'receptive': -0.25, 'absent': 0, 'control': 0.25}[cond]
                 for l in layers], pcts, width=0.25, color=colors[cond],
                label=cond, alpha=0.7)

    ax2.set_xlabel('Ablation Layer')
    ax2.set_ylabel('|% Change|')
    ax2.set_title('Effect Size: |Δr/r| by Layer')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "ablation_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def plot_spectral_circuit():
    with open(RESULTS_DIR / "exp_gamma_v2_depth_profile.json") as f:
        data = json.load(f)

    layers = sorted(set(v['layer'] for v in data.values()))
    mean_rs = [np.mean([data[f'L{l}_{c}']['pearson_r']
                        for c in ['receptive', 'absent', 'control']]) for l in layers]

    fig, ax = plt.subplots(figsize=(14, 4))

    pos_mask = np.array(mean_rs) >= 0
    neg_mask = ~pos_mask

    ax.fill_between(layers, mean_rs, 0, where=pos_mask,
                    color='#4CAF50', alpha=0.3, label='Highway (σ₂ on high-γ)')
    ax.fill_between(layers, mean_rs, 0, where=neg_mask,
                    color='#2196F3', alpha=0.3, label='Service road (σ₂ on low-γ)')
    ax.plot(layers, mean_rs, 'k-', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.annotate('L0: On highway\nr=+0.28', xy=(0, 0.28), fontsize=7,
                ha='center', va='bottom')
    ax.annotate('L1: Exit ramp\nr=-0.20', xy=(1, -0.20), fontsize=7,
                ha='center', va='top')
    ax.annotate('L13: Reset node\nr≈0', xy=(13, 0.03), fontsize=7,
                ha='center', va='bottom')
    ax.annotate('L16: Peak wire\nr=-0.40', xy=(16, -0.40), fontsize=7,
                ha='center', va='top')
    ax.annotate('L27: Cliff', xy=(27, -0.13), fontsize=7,
                ha='center', va='top')
    ax.annotate('L30: Return\nto highway', xy=(30, 0.05), fontsize=7,
                ha='center', va='bottom')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean r(|γ|, |V₂|)')
    ax.set_title('The Spectral Circuit: σ₂ Routing Through the γ Landscape')
    ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "spectral_circuit.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating wire mechanism figures...")
    plot_depth_profile()
    plot_ablation()
    plot_spectral_circuit()
    print("All figures generated.")
