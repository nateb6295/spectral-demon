#!/usr/bin/env python3
"""Paper 11 v2 figures — domain selectivity + 2×2 matrix."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 200,
})


def load_f614():
    with open(os.path.join(RESULTS_DIR, "f614_factual_control.json")) as f:
        return json.load(f)


def mean_profile(gains_list):
    return np.array(gains_list).mean(axis=0)


def fig1_domain_selectivity():
    """Figure 1: Domain selectivity — self-ref vs factual gain profiles."""
    data = load_f614()
    gains = data["gains"]

    sr_pos = mean_profile(gains["self_ref_pos"])
    sr_neg = mean_profile(gains["self_ref_neg"])
    fact_pos = mean_profile(gains["factual_pos"])
    fact_neg = mean_profile(gains["factual_neg"])
    third = mean_profile(gains["third_person"])

    n_layers = len(sr_pos)
    layers = np.arange(n_layers)
    depth_pct = layers / n_layers * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 2]})

    ax1 = axes[0]
    ax1.plot(depth_pct, sr_pos, '-', color='#2563EB', linewidth=2.2, label='Self-ref (+CCS)')
    ax1.plot(depth_pct, sr_neg, '--', color='#2563EB', linewidth=1.5, alpha=0.6, label='Self-ref (−CCS)')
    ax1.plot(depth_pct, fact_pos, '-', color='#DC2626', linewidth=2.2, label='Factual (+)')
    ax1.plot(depth_pct, fact_neg, '--', color='#DC2626', linewidth=1.5, alpha=0.6, label='Factual (−)')
    ax1.plot(depth_pct, third, ':', color='#7C3AED', linewidth=1.8, label='Third-person')

    ax1.set_xlabel('Network Depth (%)')
    ax1.set_ylabel('Mean σ₂ Gain')
    ax1.set_title('Per-Layer Gain Profiles by Domain')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.set_xlim(0, 100)

    ax2 = axes[1]
    comparisons = [
        ('Self-ref\n+/−', data["results"]["SELF-REFERENTIAL negation"]["rho"], '#2563EB'),
        ('Factual\n+/−', data["results"]["FACTUAL negation"]["rho"], '#DC2626'),
        ('Self-ref vs\nThird-person', data["results"]["THIRD-PERSON paraphrase"]["rho"], '#7C3AED'),
        ('Self-ref vs\nFactual', data["results"]["Self-ref vs factual (cross-domain)"]["rho"], '#059669'),
    ]

    x_pos = np.arange(len(comparisons))
    colors = [c[2] for c in comparisons]
    rhos = [c[1] for c in comparisons]
    labels = [c[0] for c in comparisons]

    bars = ax2.bar(x_pos, rhos, color=colors, width=0.65, alpha=0.85, edgecolor='white', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Spearman ρ')
    ax2.set_title('Profile Correlations')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_ylim(-0.8, 1.1)

    for bar, rho in zip(bars, rhos):
        y_offset = 0.03 if rho > 0 else -0.08
        ax2.text(bar.get_x() + bar.get_width()/2, rho + y_offset,
                 f'{rho:+.3f}', ha='center', va='bottom' if rho > 0 else 'top',
                 fontsize=10, fontweight='bold')

    ax2.axhspan(-0.8, -0.3, alpha=0.06, color='#059669')
    ax2.text(3.32, -0.7, 'anti-\ncorrelated', fontsize=8, color='#059669', alpha=0.7, ha='center')

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "paper11v2_domain_selectivity.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def fig2_two_by_two_matrix():
    """Figure 2: 2×2 species classification matrix."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_xlim(-0.3, 2.8)
    ax.set_ylim(-0.3, 3.2)
    ax.axis('off')

    ax.text(1.25, 3.1, 'Mechanistic Classification of Transport Species',
            fontsize=15, fontweight='bold', ha='center', va='top',
            fontfamily='serif')

    cells = {
        (0, 0): {
            'species': 'Tunnel',
            'example': 'Pythia-6.9B',
            'props': 'MHA, no GQA',
            'sensitivity': '0.3× (attenuate)',
            'color': '#DBEAFE',
            'border': '#2563EB',
            'text_color': '#1E40AF',
        },
        (1, 0): {
            'species': 'Absorber',
            'example': 'Mistral-7B',
            'props': 'GQA 4:1',
            'sensitivity': '0.2× (attenuate)',
            'color': '#FEE2E2',
            'border': '#DC2626',
            'text_color': '#991B1B',
        },
        (0, 1): {
            'species': 'Relay',
            'example': 'Qwen-2.5-7B',
            'props': 'GQA 7:1',
            'sensitivity': '0.3× (attenuate)',
            'color': '#D1FAE5',
            'border': '#059669',
            'text_color': '#065F46',
        },
        (1, 1): {
            'species': 'Sorter',
            'example': 'Gemma-2-9B',
            'props': 'GQA 2:1',
            'sensitivity': '7.1× (AMPLIFY)',
            'color': '#FEF3C7',
            'border': '#D97706',
            'text_color': '#92400E',
        },
    }

    # Column headers
    ax.text(0.65, 2.75, 'ADDITIVE', fontsize=12, fontweight='bold', ha='center', color='#374151')
    ax.text(1.85, 2.75, 'INTERACTIVE', fontsize=12, fontweight='bold', ha='center', color='#374151')
    ax.text(1.25, 2.9, 'Composition Mode', fontsize=10, ha='center', color='#6B7280', style='italic')

    # Row labels
    ax.text(-0.15, 1.85, 'SIGN\nSENSITIVE', fontsize=10, fontweight='bold', ha='center',
            va='center', color='#374151')
    ax.text(-0.15, 0.65, 'SIGN\nINVARIANT', fontsize=10, fontweight='bold', ha='center',
            va='center', color='#374151')

    for (col, row), info in cells.items():
        x = col * 1.2 + 0.05
        y = row * 1.2 + 0.05

        rect = FancyBboxPatch((x, y), 1.1, 1.1,
                               boxstyle="round,pad=0.05",
                               facecolor=info['color'],
                               edgecolor=info['border'],
                               linewidth=2.8 if info['species'] == 'Sorter' else 1.8)
        ax.add_patch(rect)

        cx, cy = x + 0.55, y + 0.55
        ax.text(cx, cy + 0.32, info['species'], fontsize=15, fontweight='bold',
                ha='center', va='center', color=info['text_color'])
        ax.text(cx, cy + 0.1, info['example'], fontsize=10, ha='center', va='center',
                color=info['text_color'], style='italic')
        ax.text(cx, cy - 0.08, info['props'], fontsize=9, ha='center', va='center',
                color='#6B7280')
        ax.text(cx, cy - 0.3, info['sensitivity'], fontsize=11, ha='center', va='center',
                color=info['text_color'], fontweight='bold' if 'AMPLIFY' in info['sensitivity'] else 'normal')

    outpath = os.path.join(FIGURES_DIR, "paper11v2_species_matrix.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def fig3_convergence_depth():
    """Figure 3: 89% convergence — single-layer predictor across species."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    species_data = [
        ('Tunnel\n(Pythia)', 86, 0.31, '#2563EB'),
        ('Relay\n(Qwen)', 90, 0.28, '#059669'),
        ('Sorter\n(Gemma)', 88, 0.33, '#D97706'),
        ('Absorber\n(Mistral)', 91, 0.25, '#DC2626'),
    ]

    for i, (name, depth, r2, color) in enumerate(species_data):
        ax.barh(i, depth, height=0.6, color=color, alpha=0.8, edgecolor='white')
        ax.text(depth + 0.5, i, f'{depth}%', va='center', fontsize=11, fontweight='bold',
                color=color)

    ax.axvline(89, color='#1F2937', linewidth=2, linestyle='--', alpha=0.7)
    ax.text(89, 3.7, '89% ± 7%', ha='center', fontsize=10, fontweight='bold',
            color='#1F2937')

    ax.set_yticks(range(len(species_data)))
    ax.set_yticklabels([s[0] for s in species_data], fontsize=10)
    ax.set_xlabel('Network Depth (%)', fontsize=11)
    ax.set_title('Convergence Point: Single-Layer Output Predictor', fontsize=13,
                 fontweight='bold')
    ax.set_xlim(60, 100)
    ax.invert_yaxis()

    outpath = os.path.join(FIGURES_DIR, "paper11v2_convergence_depth.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def fig4_control_battery():
    """Figure 4: Selectivity control battery — what drives the spectral boundary?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Shuffle sign reversal
    ax1 = axes[0]
    conditions = ['Intact\n(self-ref vs factual)', 'Shuffled\n(self-ref vs factual)']
    rhos = [-0.554, +0.989]
    colors = ['#DC2626', '#2563EB']
    bars = ax1.bar([0, 1], rhos, color=colors, width=0.55, alpha=0.85, edgecolor='white', linewidth=1.5)
    for bar, rho in zip(bars, rhos):
        y_off = 0.04 if rho > 0 else -0.04
        ax1.text(bar.get_x() + bar.get_width()/2, rho + y_off,
                 f'ρ = {rho:+.3f}', ha='center',
                 va='bottom' if rho > 0 else 'top',
                 fontsize=13, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel('Spearman ρ', fontsize=11)
    ax1.set_title('A. Shuffle Control: Selectivity Is Structural', fontsize=12, fontweight='bold')
    ax1.axhline(0, color='gray', linewidth=0.8)
    ax1.set_ylim(-0.8, 1.2)
    ax1.annotate('Sign reversal\n(not attenuation)', xy=(0.5, 0.2), fontsize=9,
                 ha='center', color='#6B7280', style='italic')

    # Panel B: Control battery — what clusters with what?
    ax2 = axes[1]

    comparisons = [
        ('Self-ref\nvs Factual', -0.554, '#7C3AED'),
        ('Factual\nvs Math', +0.835, '#059669'),
        ('"I am model"\nvs "System is model"', +0.984, '#2563EB'),
        ('"I am model"\nvs "I planted..."', +0.970, '#2563EB'),
        ('"I planted..."\nvs Factual', -0.544, '#DC2626'),
        ('Gardener\nvs Storm', +0.985, '#D97706'),
        ('Gardener\nvs Maria', -0.099, '#D97706'),
        ('Factual\naff vs neg', +0.959, '#6B7280'),
        ('Self-ref\naff vs neg', +0.994, '#6B7280'),
    ]

    x = np.arange(len(comparisons))
    colors = [c[2] for c in comparisons]
    rhos = [c[1] for c in comparisons]
    labels = [c[0] for c in comparisons]

    bars = ax2.barh(x, rhos, color=colors, height=0.6, alpha=0.8, edgecolor='white', linewidth=1)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlabel('Spearman ρ', fontsize=11)
    ax2.set_title('B. Control Battery: Semantic Feature Not Isolated', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='gray', linewidth=0.8)
    ax2.set_xlim(-0.8, 1.2)
    ax2.invert_yaxis()

    for bar, rho in zip(bars, rhos):
        x_off = 0.03 if rho > 0 else -0.03
        ax2.text(rho + x_off, bar.get_y() + bar.get_height()/2,
                 f'{rho:+.3f}', va='center',
                 ha='left' if rho > 0 else 'right',
                 fontsize=9, fontweight='bold', color='#374151')

    # Bracket annotations
    ax2.axhspan(-0.5, 0.5, alpha=0.04, color='#7C3AED')
    ax2.axhspan(1.5, 4.5, alpha=0.04, color='#2563EB')
    ax2.axhspan(4.5, 6.5, alpha=0.04, color='#D97706')
    ax2.axhspan(6.5, 8.5, alpha=0.04, color='#6B7280')

    ax2.text(1.15, 0, 'Original', fontsize=7, ha='right', va='center', color='#7C3AED', style='italic')
    ax2.text(1.15, 3, 'Deixis\ncontrol', fontsize=7, ha='right', va='center', color='#2563EB', style='italic')
    ax2.text(1.15, 5.5, 'Animacy\ncontrol', fontsize=7, ha='right', va='center', color='#D97706', style='italic')
    ax2.text(1.15, 7.5, 'Negation\n(sign-blind)', fontsize=7, ha='right', va='center', color='#6B7280', style='italic')

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "paper11v2_control_battery.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    fig1_domain_selectivity()
    fig2_two_by_two_matrix()
    fig3_convergence_depth()
    fig4_control_battery()
    print("All Paper 11 v2 figures generated.")
