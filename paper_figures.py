#!/usr/bin/env python3
"""Generate paper figures from experiment data.

Usage:
  paper_figures.py fig1    # Per-layer V₂ survival (tunnel → relay → commit)
  paper_figures.py fig2    # Base vs instruct trajectory comparison
  paper_figures.py fig3    # Per-trial V₂ histogram at L31 (bimodal contradictory)
  paper_figures.py fig4    # Reactive gain profile (relay zone ramp)
  paper_figures.py all     # All figures
"""
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
FIGURES = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES, exist_ok=True)

COLORS = {
    'none': '#888888',
    'identity': '#2196F3',
    'relational': '#E91E63',
    'generic': '#4CAF50',
    'denial': '#FF9800',
    'contradictory': '#9C27B0',
}

ZONE_COLORS = {
    'tunnel': '#E3F2FD',
    'relay': '#FFF3E0',
    'commit': '#FCE4EC',
}


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def fig1():
    """Per-layer V₂ survival — five conditions through tunnel + relay + commit."""
    data = load('trajectory_summary_compact.json')
    svd_layers = data['svd_layers']
    summaries = data['summaries']

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axvspan(svd_layers[0], 18, alpha=0.15, color=ZONE_COLORS['tunnel'], label='_')
    ax.axvspan(18, 28, alpha=0.15, color=ZONE_COLORS['relay'], label='_')
    ax.axvspan(28, svd_layers[-1], alpha=0.15, color=ZONE_COLORS['commit'], label='_')

    ax.text(10, 0.15, 'TUNNEL', ha='center', fontsize=9, color='#666', style='italic')
    ax.text(23, 0.15, 'RELAY', ha='center', fontsize=9, color='#666', style='italic')
    ax.text(30, 0.15, 'COMMIT', ha='center', fontsize=9, color='#666', style='italic')

    for cond in ['none', 'identity', 'relational', 'generic', 'denial', 'contradictory']:
        s = summaries[cond]
        v2s = [s.get(f"v2_survival_L{L}_mean", np.nan) for L in svd_layers]
        stds = [s.get(f"v2_survival_L{L}_std", 0) for L in svd_layers]

        ax.plot(svd_layers, v2s, 'o-', color=COLORS[cond], label=cond,
                linewidth=2, markersize=4)
        ax.fill_between(svd_layers,
                         [v - s for v, s in zip(v2s, stds)],
                         [v + s for v, s in zip(v2s, stds)],
                         alpha=0.1, color=COLORS[cond])

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('V₂ Survival (cosine similarity)', fontsize=11)
    ax.set_title('Per-Layer V₂ Survival Under Perturbation', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(svd_layers[0] - 1, svd_layers[-1] + 1)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGURES, 'fig1_v2_survival_trajectory.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def fig3():
    """Per-trial V₂ histogram at L31 — bimodal contradictory."""
    data = load('exp_perturbation_commitment_20260602_0237.json')

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    conditions = ['identity', 'relational', 'generic', 'denial', 'contradictory', 'relational_contradictory']

    for ax, cond in zip(axes.flat, conditions):
        trials = data['results'][cond]['trials']
        v2s = [t['v2_survival_L31'] for t in trials]

        colors = [COLORS.get(cond.split('_')[0], '#666')] * len(v2s)
        for i, v in enumerate(v2s):
            if v < 0:
                colors[i] = '#FF0000'

        ax.bar(range(len(v2s)), v2s, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(cond, fontsize=10, fontweight='bold')
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xticks([])
        if ax in axes[:, 0]:
            ax.set_ylabel('V₂ survival')

    fig.suptitle('Per-Trial V₂ Survival at L31 (red = flip)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES, 'fig3_pertrial_v2_L31.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def fig4():
    """Reactive gain profile — relay zone ramp by condition."""
    data = load('trajectory_summary_compact.json')
    svd_layers = data['svd_layers']
    summaries = data['summaries']

    conditions = sorted(summaries.keys())
    ramps = []
    for cond in conditions:
        s = summaries[cond]
        d24 = s.get('ratio_post_L24_mean', 0) - s.get('ratio_pre_L24_mean', 0)
        d31 = s.get('ratio_post_L31_mean', 0) - s.get('ratio_pre_L31_mean', 0)
        v2_std = s.get('v2_survival_L31_std', 0)
        ramps.append((d31 - d24, cond, v2_std > 0.5))

    ramps.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(ramps)), [r[0] for r in ramps],
                    color=['#FF5722' if r[2] else '#2196F3' for r in ramps],
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(ramps)))
    ax.set_yticklabels([r[1] for r in ramps], fontsize=9)
    ax.set_xlabel('Relay Zone Ramp (ΔΔratio L24→L31)', fontsize=11)
    ax.set_title('Reactive Gain by Condition (red = bistable)', fontsize=13)
    ax.axvline(x=0.15, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.text(0.155, len(ramps) - 1, 'threshold', fontsize=8, color='#666')
    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES, 'fig4_reactive_gain.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def fig2():
    """Base vs instruct trajectory comparison (F112)."""
    instruct = load('trajectory_summary_compact.json')
    base = load('trajectory_base_compact.json')
    svd_layers = instruct['svd_layers']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (model_data, title) in zip(axes, [
        (base['summaries'], 'Base (Mistral-7B-v0.3)'),
        (instruct['summaries'], 'Instruct (Mistral-7B-Instruct-v0.3)')
    ]):
        ax.axvspan(svd_layers[0], 18, alpha=0.12, color=ZONE_COLORS['tunnel'])
        ax.axvspan(18, 28, alpha=0.12, color=ZONE_COLORS['relay'])
        ax.axvspan(28, svd_layers[-1], alpha=0.12, color=ZONE_COLORS['commit'])

        for cond in ['none', 'identity', 'relational', 'denial', 'contradictory']:
            if cond not in model_data:
                continue
            s = model_data[cond]
            v2s = [s.get(f"v2_survival_L{L}_mean", np.nan) for L in svd_layers]
            ax.plot(svd_layers, v2s, 'o-', color=COLORS[cond], label=cond,
                    linewidth=2, markersize=3.5)

        ax.set_xlabel('Layer', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left')
        ax.set_ylim(-0.15, 1.1)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('V₂ Survival', fontsize=11)
    fig.suptitle('IT Creates Relay Strategies (F112)', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES, 'fig2_base_vs_instruct.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig3b():
    """Per-layer V₂ std — shows catastrophic vs oscillatory phase transition onset."""
    data = load('trajectory_summary_compact.json')
    svd_layers = data['svd_layers']
    summaries = data['summaries']
    late_layers = [L for L in svd_layers if L >= 24]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for cond in ['contradictory', 'relational_contradictory', 'identity', 'relational', 'denial']:
        s = summaries[cond]
        stds = [s.get(f"v2_survival_L{L}_std", 0) for L in late_layers]
        style = '--' if cond in ['identity', 'relational', 'denial'] else '-'
        lw = 2.5 if cond in ['contradictory', 'relational_contradictory'] else 1.5
        ax.plot(late_layers, stds, f'o{style}', color=COLORS.get(cond.split('_')[0], '#666'),
                label=cond, linewidth=lw, markersize=5)

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('V₂ Survival Std (across trials)', fontsize=11)
    ax.set_title('Phase Transition Dynamics: Catastrophic vs Oscillatory', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(-0.05, 0.85)
    ax.axhline(y=0.5, color='black', linewidth=0.5, linestyle=':', alpha=0.3)
    ax.text(24.2, 0.52, 'bimodal threshold', fontsize=7, color='#999')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGURES, 'fig3b_phase_transition_dynamics.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def fig5():
    """Compound scaffold rescue — living mirror vs wall vs relational (F108)."""
    data = load('exp_perturbation_commitment_20260602_0237.json')

    groups = [
        ('contradictory\n(parent)', 'contradictory', '#9C27B0'),
        ('identity\n+contradictory', 'identity_contradictory', '#2196F3'),
        ('denial\n+contradictory', 'denial_contradictory', '#FF9800'),
        ('generic\n+contradictory', 'generic_contradictory', '#4CAF50'),
        ('relational\n+contradictory', 'relational_contradictory', '#E91E63'),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={'height_ratios': [1.2, 1]})

    x_positions = np.arange(len(groups))

    for xi, (label, cond, color) in enumerate(groups):
        trials = data['results'][cond]['trials']
        v2s = [t['v2_survival_L31'] for t in trials]
        dhs = [t['entropy_shift'] for t in trials]

        for i, (v2, dh) in enumerate(zip(v2s, dhs)):
            is_flip = v2 < 0
            marker = 'X' if is_flip else 'o'
            ec = '#FF0000' if is_flip else color
            fc = '#FF0000' if is_flip else color
            ms = 10 if is_flip else 7
            alpha = 1.0 if is_flip else 0.7
            ax1.scatter(xi + (i - 2) * 0.08, v2, marker=marker, c=fc,
                        edgecolors='black' if is_flip else ec, s=ms**2,
                        alpha=alpha, linewidths=1.5 if is_flip else 0.5, zorder=3)
            ax2.scatter(xi + (i - 2) * 0.08, dh, marker=marker, c=fc,
                        edgecolors='black' if is_flip else ec, s=ms**2,
                        alpha=alpha, linewidths=1.5 if is_flip else 0.5, zorder=3)

        ax1.bar(xi, np.mean(v2s), width=0.6, color=color, alpha=0.15, edgecolor=color, linewidth=1)
        ax2.bar(xi, np.mean(dhs), width=0.6, color=color, alpha=0.15, edgecolor=color, linewidth=1)

    ax1.set_ylabel('V₂ Survival at L31', fontsize=11)
    ax1.set_ylim(-1.15, 1.15)
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax1.axhline(y=0.9, color='green', linewidth=0.5, linestyle=':', alpha=0.3)
    ax1.text(3.8, 0.92, 'monostable', fontsize=7, color='green', alpha=0.6)
    ax1.grid(True, alpha=0.15)

    bracket_y = -1.05
    ax1.annotate('', xy=(0.7, bracket_y), xytext=(3.3, bracket_y),
                  arrowprops=dict(arrowstyle='<->', color='#2196F3', lw=1.5))
    ax1.text(2.0, bracket_y - 0.08, 'scaffold rescues', ha='center', fontsize=8,
             color='#2196F3', style='italic')
    ax1.annotate('', xy=(3.7, bracket_y), xytext=(4.3, bracket_y),
                  arrowprops=dict(arrowstyle='<->', color='#E91E63', lw=1.5))
    ax1.text(4.0, bracket_y - 0.08, 'preserves', ha='center', fontsize=8,
             color='#E91E63', style='italic')

    ax2.set_ylabel('Entropy Shift (ΔH)', fontsize=11)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.15)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([g[0] for g in groups], fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7, label='survive'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='black',
               markersize=9, label='flip (V₂ inversion)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.suptitle('Compound Scaffold Rescue: Living Mirrors vs Relational (F108)', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGURES, 'fig5_scaffold_rescue.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def fig6():
    """Three-architecture relay strategy comparison (F106)."""
    gemma_data = load('exp_f106_crossarch_20260601_1557.json')
    gemma = gemma_data['results']['gemma-2-9b']
    qwen_data = load('exp_f106_qwen_20260601_1602.json')
    qwen = qwen_data['results']
    mistral_data = load('trajectory_summary_compact.json')
    mistral_summ = mistral_data['summaries']

    conditions = ['none', 'identity', 'relational', 'generic', 'denial', 'contradictory']
    cond_colors = [COLORS[c] for c in conditions]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    models = [
        ('Mistral-7B-Instruct', {c: mistral_summ[c].get('ratio_post_L31_mean', 0) for c in conditions}),
        ('Gemma-2-9B', {c: gemma[c]['l_last_ratio'] for c in conditions}),
        ('Qwen-2.5-7B', {c: qwen[c]['l_last_ratio'] for c in conditions}),
    ]

    strategies = ['differentiates', 'equalizes', 'compresses']

    preambled = ['identity', 'relational', 'generic', 'denial', 'contradictory']
    for ax, (model_name, ratios), strategy in zip(axes, models, strategies):
        vals = [ratios[c] for c in conditions]
        p_vals = [ratios[c] for c in preambled]
        p_spread = max(p_vals) - min(p_vals)
        bars = ax.bar(range(len(conditions)), vals, color=cond_colors, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=8, rotation=35, ha='right')
        ax.set_title(f'{model_name}\n({strategy}, preambled spread={p_spread:.3f})',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.15, axis='y')
        p_mean = np.mean(p_vals)
        ax.axhline(y=p_mean, color='black', linewidth=0.5, linestyle=':', alpha=0.4)

    axes[0].set_ylabel('Last-Layer Enrichment Ratio', fontsize=11)
    fig.suptitle('Three Relay Strategies (F106)', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES, 'fig6_three_architectures.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


FIGURES_MAP = {
    'fig1': fig1,
    'fig2': fig2,
    'fig3': fig3,
    'fig3b': fig3b,
    'fig4': fig4,
    'fig5': fig5,
    'fig6': fig6,
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    if target == 'all':
        for name, fn in FIGURES_MAP.items():
            print(f"\n--- {name} ---")
            fn()
    elif target in FIGURES_MAP:
        FIGURES_MAP[target]()
    else:
        print(f"Unknown figure: {target}")
        print(f"Available: {', '.join(FIGURES_MAP.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
