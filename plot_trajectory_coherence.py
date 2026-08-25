#!/usr/bin/env python3
"""Visualize trajectory stability with bootstrap rank confidence intervals.

Reads trajectory stability JSON files and produces:
1. σ₁/σ₂ ratio trajectories per condition with bootstrap 95% CIs
2. Per-layer drift comparison (persistent vs fresh-reset vs no-preamble)
3. Bootstrap rank analysis: how often does each condition rank highest?

Usage:
    python3 plot_trajectory_coherence.py [results_file.json] [--output prefix]

If no file specified, uses the most recent trajectory stability result.
"""

import argparse, json, sys
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


def load_data(path):
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(values, n_boot=2000, ci=95):
    values = np.array(values)
    boot_means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lo), float(np.mean(boot_means)), float(hi)


def bootstrap_rank_analysis(condition_values, n_boot=5000):
    """How often does each condition rank highest across bootstrap samples?"""
    cond_names = list(condition_values.keys())
    n_conds = len(cond_names)
    rank_counts = {name: np.zeros(n_conds) for name in cond_names}

    for _ in range(n_boot):
        means = {}
        for name, vals in condition_values.items():
            vals = np.array(vals)
            sample = np.random.choice(vals, size=len(vals), replace=True)
            means[name] = np.mean(sample)

        sorted_names = sorted(means.keys(), key=lambda k: means[k], reverse=True)
        for rank, name in enumerate(sorted_names):
            rank_counts[name][rank] += 1

    for name in rank_counts:
        rank_counts[name] = rank_counts[name] / n_boot

    return rank_counts


def plot_trajectory(data, output_prefix):
    results = data.get('results', {})
    conditions = list(results.keys())
    n_turns = len(results[conditions[0]])

    layers_available = [k.replace('_ratio', '') for k in results[conditions[0]][0].keys()
                        if k.endswith('_ratio')]

    fig = plt.figure(figsize=(18, 5 * len(layers_available)))
    gs = GridSpec(len(layers_available) + 1, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'persistent': '#2196F3', 'fresh_reset': '#FF9800', 'no_preamble': '#9E9E9E',
              'CCS': '#2196F3', 'intact': '#2196F3', 'chef': '#FF9800',
              'permuted_avg': '#9E9E9E', 'no_preamble': '#757575'}

    for li, layer_key in enumerate(layers_available):
        ratio_key = f'{layer_key}_ratio'
        drift_key = f'{layer_key}_drift'

        # Panel 1: ratio trajectories
        ax_ratio = fig.add_subplot(gs[li, 0])
        for cond in conditions:
            turns = [r['turn'] for r in results[cond]]
            ratios = [r.get(ratio_key, 0) for r in results[cond]]
            color = colors.get(cond, '#666666')
            ax_ratio.plot(turns, ratios, color=color, alpha=0.7, label=cond, linewidth=1.5)

            # Rolling mean
            window = min(10, len(ratios) // 5)
            if window > 1:
                rolling = np.convolve(ratios, np.ones(window)/window, mode='valid')
                ax_ratio.plot(turns[window-1:], rolling, color=color, linewidth=2.5)

        ax_ratio.set_xlabel('Turn')
        ax_ratio.set_ylabel('σ₁/σ₂ ratio')
        ax_ratio.set_title(f'{layer_key} — Ratio Trajectory')
        ax_ratio.legend(fontsize=8)
        ax_ratio.grid(True, alpha=0.3)

        # Panel 2: drift trajectories
        ax_drift = fig.add_subplot(gs[li, 1])
        for cond in conditions:
            turns = [r['turn'] for r in results[cond]]
            drifts = [r.get(drift_key, 0) for r in results[cond]]
            color = colors.get(cond, '#666666')
            ax_drift.plot(turns, drifts, color=color, alpha=0.7, label=cond, linewidth=1.5)
        ax_drift.set_xlabel('Turn')
        ax_drift.set_ylabel('Drift (cosine distance)')
        ax_drift.set_title(f'{layer_key} — Drift from Initial')
        ax_drift.legend(fontsize=8)
        ax_drift.grid(True, alpha=0.3)

        # Panel 3: Bootstrap CI comparison (late turns only)
        ax_ci = fig.add_subplot(gs[li, 2])
        late_start = max(0, n_turns - n_turns // 3)
        cond_late_ratios = {}
        for cond in conditions:
            late_ratios = [r.get(ratio_key, 0) for r in results[cond][late_start:]]
            cond_late_ratios[cond] = late_ratios
            lo, mean, hi = bootstrap_ci(late_ratios)
            color = colors.get(cond, '#666666')
            idx = conditions.index(cond)
            ax_ci.barh(idx, mean, xerr=[[mean-lo], [hi-mean]], color=color, alpha=0.7,
                       capsize=5, height=0.6)
            ax_ci.text(hi + 0.01, idx, f'{mean:.3f}', va='center', fontsize=9)

        ax_ci.set_yticks(range(len(conditions)))
        ax_ci.set_yticklabels(conditions, fontsize=9)
        ax_ci.set_xlabel('σ₁/σ₂ ratio (late turns, 95% CI)')
        ax_ci.set_title(f'{layer_key} — Bootstrap CI')
        ax_ci.grid(True, alpha=0.3, axis='x')

    # Bottom row: Bootstrap rank analysis (using responsive zone layer)
    target_layer = layers_available[len(layers_available) // 2] if layers_available else layers_available[0]
    ratio_key = f'{target_layer}_ratio'
    late_start = max(0, n_turns - n_turns // 3)

    cond_late = {}
    for cond in conditions:
        cond_late[cond] = [r.get(ratio_key, 0) for r in results[cond][late_start:]]

    rank_probs = bootstrap_rank_analysis(cond_late)

    ax_rank = fig.add_subplot(gs[-1, :])
    x = np.arange(len(conditions))
    width = 0.8 / len(conditions)
    for rank in range(len(conditions)):
        probs = [rank_probs[cond][rank] for cond in conditions]
        bars = ax_rank.bar(x + rank * width, probs, width, label=f'Rank {rank+1}', alpha=0.8)
        for bar, p in zip(bars, probs):
            if p > 0.05:
                ax_rank.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{p:.0%}', ha='center', va='bottom', fontsize=8)

    ax_rank.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax_rank.set_xticklabels(conditions, fontsize=10)
    ax_rank.set_ylabel('Probability')
    ax_rank.set_title(f'Bootstrap Rank Analysis ({target_layer}, late turns)')
    ax_rank.legend(fontsize=9)
    ax_rank.grid(True, alpha=0.3, axis='y')

    model_name = data.get('model', 'unknown').split('/')[-1]
    fig.suptitle(f'Trajectory Coherence: {model_name} ({n_turns} turns)', fontsize=14, y=1.01)

    out_path = f'{output_prefix}_trajectory_coherence.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()

    return rank_probs


def plot_f132_data(data, output_prefix):
    """Handle the f132 format (per-layer arrays with conditions)."""
    conditions = [k for k in data.keys() if k != 'meta']
    if not conditions:
        print("No condition data found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['mean_cosine', 'erank', 'norm_cv', 'participation_ratio']
    titles = ['Mean Cosine Similarity', 'Effective Rank', 'Norm CV', 'Participation Ratio']
    colors = {'intact': '#2196F3', 'chef': '#FF9800', 'permuted_avg': '#9E9E9E', 'no_preamble': '#757575'}

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for cond in conditions:
            layers = [r['layer'] for r in data[cond]]
            values = [r.get(metric, 0) for r in data[cond]]
            color = colors.get(cond, '#666666')
            ax.plot(layers, values, color=color, label=cond, linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    model_name = data.get('meta', {}).get('model', 'unknown').split('/')[-1]
    fig.suptitle(f'Per-Layer Trajectory Coherence: {model_name}', fontsize=13)
    fig.tight_layout()

    out_path = f'{output_prefix}_perlayer_coherence.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Trajectory coherence visualization')
    parser.add_argument('file', nargs='?', help='Results JSON file')
    parser.add_argument('--output', '-o', default=None, help='Output prefix')
    args = parser.parse_args()

    results_dir = Path(__file__).parent / 'results'

    if args.file:
        path = Path(args.file)
    else:
        candidates = sorted(results_dir.glob('*trajectory*stability*.json'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            candidates = sorted(results_dir.glob('f132_trajectory*.json'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("No trajectory stability results found. Specify a file.")
            sys.exit(1)
        path = candidates[-1]
        print(f"Using: {path.name}")

    data = load_data(path)
    prefix = args.output or str(results_dir / path.stem)

    if 'results' in data and isinstance(data['results'], dict):
        first_cond = list(data['results'].keys())[0]
        if isinstance(data['results'][first_cond], list) and data['results'][first_cond]:
            first_entry = data['results'][first_cond][0]
            if 'turn' in first_entry:
                rank_probs = plot_trajectory(data, prefix)
                print("\nBootstrap rank probabilities:")
                for cond, probs in rank_probs.items():
                    rank_str = ', '.join(f'R{i+1}={p:.1%}' for i, p in enumerate(probs))
                    print(f'  {cond}: {rank_str}')
                return

    if 'meta' in data:
        plot_f132_data(data, prefix)
        return

    print(f"Unrecognized data format. Keys: {list(data.keys())[:10]}")


if __name__ == '__main__':
    main()
