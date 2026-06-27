#!/usr/bin/env python3
"""Bootstrap rank CI visualization for trajectory stability (F132/trajectory data).

Shows how σ₁/σ₂ coupling rank (ratio) and V₂ drift evolve over 100 conversation turns
across conditions (persistent/fresh_reset/no_preamble), with bootstrap confidence intervals.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
DATA_FILE = RESULTS_DIR / 'exp_trajectory_stability_20260603_1745.json'
OUT_FILE = RESULTS_DIR / 'trajectory_coherence_bootstrap_ci.png'

N_BOOT = 1000
LAYERS = ['L18', 'L23', 'L27', 'L31']
CONDITIONS = ['persistent', 'fresh_reset', 'no_preamble']
COND_COLORS = {'persistent': '#d62728', 'fresh_reset': '#2ca02c', 'no_preamble': '#7f7f7f'}
COND_LABELS = {'persistent': 'Persistent context', 'fresh_reset': 'Fresh reset each turn', 'no_preamble': 'No preamble'}


def bootstrap_ci(values, n_boot=N_BOOT, ci=95):
    """Bootstrap confidence interval for the mean."""
    arr = np.array(values)
    boot_means = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(arr), lo, hi


def rolling_bootstrap(series, window=10, n_boot=N_BOOT):
    """Rolling window bootstrap CI across turns."""
    n = len(series)
    means, los, his = [], [], []
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        chunk = series[start:end]
        m, lo, hi = bootstrap_ci(chunk, n_boot)
        means.append(m)
        los.append(lo)
        his.append(hi)
    return np.array(means), np.array(los), np.array(his)


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    fig, axes = plt.subplots(len(LAYERS), 2, figsize=(14, 3.2 * len(LAYERS)), sharex=True)
    fig.suptitle('Trajectory Stability: Bootstrap 95% CIs\nMistral-7B, 100 turns × 3 conditions',
                 fontsize=13, fontweight='bold', y=0.995)

    for row, layer in enumerate(LAYERS):
        ax_drift = axes[row, 0]
        ax_ratio = axes[row, 1]

        for cond in CONDITIONS:
            turns_data = data['results'][cond]
            turns = [t['turn'] for t in turns_data]
            drift_key = f'{layer}_drift'
            ratio_key = f'{layer}_ratio'

            drift_vals = [t[drift_key] for t in turns_data]
            ratio_vals = [t[ratio_key] for t in turns_data]

            # Bootstrap CIs with rolling window
            d_mean, d_lo, d_hi = rolling_bootstrap(drift_vals, window=8)
            r_mean, r_lo, r_hi = rolling_bootstrap(ratio_vals, window=8)

            color = COND_COLORS[cond]
            label = COND_LABELS[cond]

            ax_drift.plot(turns, d_mean, color=color, linewidth=1.5, label=label)
            ax_drift.fill_between(turns, d_lo, d_hi, color=color, alpha=0.15)

            ax_ratio.plot(turns, r_mean, color=color, linewidth=1.5, label=label)
            ax_ratio.fill_between(turns, r_lo, r_hi, color=color, alpha=0.15)

        ax_drift.set_ylabel(f'{layer}\nV₂ drift (cos)', fontsize=10)
        ax_ratio.set_ylabel(f'{layer}\nσ₁/σ₂ ratio', fontsize=10)
        ax_drift.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        if row == 0:
            ax_drift.legend(fontsize=8, loc='upper right')
            ax_drift.set_title('V₂ direction drift', fontsize=11)
            ax_ratio.set_title('σ₁/σ₂ coupling ratio', fontsize=11)

    axes[-1, 0].set_xlabel('Conversation turn', fontsize=10)
    axes[-1, 1].set_xlabel('Conversation turn', fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUT_FILE}')

    # Print summary stats for each condition × layer
    print('\n--- Summary: turn 90-99 means ---')
    for cond in CONDITIONS:
        turns_data = data['results'][cond]
        late = turns_data[90:]
        print(f'\n{cond}:')
        for layer in LAYERS:
            drift_vals = [t[f'{layer}_drift'] for t in late]
            ratio_vals = [t[f'{layer}_ratio'] for t in late]
            print(f'  {layer}: drift={np.mean(drift_vals):.4f}±{np.std(drift_vals):.4f}  ratio={np.mean(ratio_vals):.2f}±{np.std(ratio_vals):.2f}')


if __name__ == '__main__':
    main()
