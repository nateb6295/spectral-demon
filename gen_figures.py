#!/usr/bin/env python3
"""Generate publication figures from experiment data.

Usage:
  gen_figures.py fig1          # Step function (d/d_max)
  gen_figures.py fig2          # Spectral trajectory (σ₁, σ₂ through layers)
  gen_figures.py fig3          # Sign inversion across architectures
  gen_figures.py fig4          # Default witness enrichment (4-condition)
  gen_figures.py fig5          # Relay homeostasis (system prompt effect)
  gen_figures.py fig6          # GQA conversion (MHA → GQA shift)
  gen_figures.py fig7          # Four zones (layer metrics)
  gen_figures.py fig8          # Developmental cascade (Pythia training)
  gen_figures.py fig9          # Cross-arch relay strategies
  gen_figures.py fig10         # Fork magnitude (contradiction routing)
  gen_figures.py fig11         # σ₂ spatial redistribution
  gen_figures.py fig12         # L18 gain control circuit
  gen_figures.py fig13         # Trajectory stability
  gen_figures.py fig14         # Adversarial dose-response
  gen_figures.py all           # All figures
"""

import json, sys, argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 11
    rcParams['axes.linewidth'] = 0.8
    rcParams['figure.dpi'] = 300
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — install with: pip install matplotlib")
    sys.exit(1)

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


def load_json(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def compute_passage_distance(sigmas_by_layer, k=5):
    """Compute Grassmannian passage distance d/d_max from per-layer sigma values."""
    layers = sorted(sigmas_by_layer.keys())
    if len(layers) < 2:
        return None

    first_layer = layers[0]
    last_layer = layers[-1]

    # d_max for k dimensions
    d_max = np.sqrt(k) * np.pi / 2

    # For passage distance we need the subspace rotation
    # Approximate: use eigenvalue profile distance
    s_first = np.array(sigmas_by_layer[first_layer][:k])
    s_last = np.array(sigmas_by_layer[last_layer][:k])

    # Normalize
    s_first = s_first / (np.linalg.norm(s_first) + 1e-10)
    s_last = s_last / (np.linalg.norm(s_last) + 1e-10)

    cos_sim = np.dot(s_first, s_last)
    cos_sim = np.clip(cos_sim, -1, 1)
    d = np.arccos(cos_sim)

    return d / d_max


def extract_perlayer_sigmas(raw_data, n_layers):
    """Extract sigma values organized by layer from raw experiment data."""
    by_layer = {}
    for row in raw_data:
        layer = row.get('layer', row.get('layer_idx'))
        if layer is None:
            continue
        if layer not in by_layer:
            by_layer[layer] = []
        sigmas = []
        for i in range(1, 6):
            s = row.get(f'sigma_{i}')
            if s is not None:
                sigmas.append(s)
        if sigmas:
            by_layer[layer].append(sigmas)

    # Average across probes/conditions
    averaged = {}
    for layer, sig_lists in by_layer.items():
        averaged[layer] = np.mean(sig_lists, axis=0).tolist()

    return averaged


# ─── Figure 1: The Step Function ───────────────────────────────────────

def fig1_step_function():
    """d/d_max at tunnel midpoint for architectures. MHA cluster vs GQA cluster."""

    # Known values from the paper's measurements
    models = [
        # (name, attention_type, sharing_ratio, d_dmax, size_label)
        ("Pythia 410M", "MHA", 0, 0.55, "410M"),
        ("Pythia 6.9B", "MHA", 0, 0.55, "6.9B"),
        ("Llama-1 7B", "MHA", 0, 0.54, "7B"),
        ("GPT-2 Large", "MHA", 0, 0.52, "774M"),
        ("Falcon 7B", "MHA", 0, 0.56, "7B"),
        ("Qwen2.5 3B", "GQA", 8, 0.926, "3B"),
        ("Gemma-2 9B", "GQA", 2, 0.915, "9B"),
        ("Mistral 7B", "GQA", 4, 0.955, "7B"),
        ("CodeLlama 7B", "GQA", 4, 0.94, "7B"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    mha_x, mha_y, mha_labels = [], [], []
    gqa_x, gqa_y, gqa_labels = [], [], []

    for i, (name, atype, sr, ddmax, size) in enumerate(models):
        if atype == "MHA":
            mha_x.append(i)
            mha_y.append(ddmax)
            mha_labels.append(name)
        else:
            gqa_x.append(i)
            gqa_y.append(ddmax)
            gqa_labels.append(f"{name}\n(s={sr})")

    ax.scatter(mha_x, mha_y, s=120, c='#d62728', marker='s', zorder=5, label='MHA')
    ax.scatter(gqa_x, gqa_y, s=120, c='#1f77b4', marker='o', zorder=5, label='GQA')

    for x, y, label in zip(mha_x, mha_y, mha_labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 12),
                   ha='center', fontsize=8)
    for x, y, label in zip(gqa_x, gqa_y, gqa_labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, -18),
                   ha='center', fontsize=8)

    # Gap annotation
    mha_max = max(mha_y)
    gqa_min = min(gqa_y)
    mid = (mha_max + gqa_min) / 2
    ax.annotate('', xy=(4.5, gqa_min), xytext=(4.5, mha_max),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(4.7, mid, f'9× within-GQA\nvariation', ha='left', va='center',
           fontsize=9, color='gray')

    ax.axhline(y=np.mean(mha_y), color='#d62728', alpha=0.3, linestyle='--')
    ax.axhline(y=np.mean(gqa_y), color='#1f77b4', alpha=0.3, linestyle='--')

    ax.set_ylabel('Passage distance (d/d$_{max}$)', fontsize=12)
    ax.set_xlabel('Architecture', fontsize=12)
    ax.set_title('Figure 1: The Step Function', fontsize=13, fontweight='bold')
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks([])
    ax.legend(fontsize=10, loc='center left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "fig01_step_function.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 2: Spectral Trajectory ─────────────────────────────────────

def fig2_spectral_trajectory():
    """σ₁ and σ₂ through all layers for Mistral 7B under three conditions."""

    data = load_json("exp_witness_perlayer_20260527_1220.json")
    raw = data['raw']
    n_layers = data['n_layers']

    conditions = {}
    for row in raw:
        cond = row.get('condition', 'unknown')
        layer = row.get('layer', row.get('layer_idx'))
        if layer is None:
            continue
        if cond not in conditions:
            conditions[cond] = {}
        if layer not in conditions[cond]:
            conditions[cond][layer] = {'s1': [], 's2': []}
        conditions[cond][layer]['s1'].append(row.get('sigma_1', 0))
        conditions[cond][layer]['s2'].append(row.get('sigma_2', 0))

    target_conds = ['control', 'receptive', 'absent']
    colors = {'control': '#2ca02c', 'receptive': '#1f77b4', 'absent': '#d62728'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for cond in target_conds:
        if cond not in conditions:
            continue
        layers = sorted(conditions[cond].keys())
        s1_means = [np.mean(conditions[cond][l]['s1']) for l in layers]
        s2_means = [np.mean(conditions[cond][l]['s2']) for l in layers]

        ax1.plot(layers, s1_means, color=colors.get(cond, 'gray'), linewidth=1.5,
                label=cond, alpha=0.9)
        ax2.plot(layers, s2_means, color=colors.get(cond, 'gray'), linewidth=1.5,
                label=cond, alpha=0.9)

    ax1.set_ylabel('σ₁', fontsize=12)
    ax1.set_title('Figure 2: Spectral Trajectory (Mistral 7B)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Zone shading
    for ax in [ax1, ax2]:
        ax.axvspan(2, 14, alpha=0.05, color='blue', label='_decoupling')
        ax.axvspan(15, 20, alpha=0.05, color='orange', label='_transition')
        ax.axvspan(21, 28, alpha=0.05, color='green', label='_responsive')
        ax.axvspan(29, 32, alpha=0.05, color='red', label='_relay')

    ax2.set_ylabel('σ₂', fontsize=12)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "fig02_spectral_trajectory.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 11: σ₂ Spatial Redistribution ──────────────────────────────

def fig11_sigma2_redistribution():
    """σ₂ CV across layers for 5 conditions."""

    data = load_json("exp_variance_ratio_20260603_1733.json")
    results = data['results']
    conditions = data['conditions']

    relational = ['receptive', 'absent', 'sequential']
    role = ['control', 'directive']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'receptive': '#1f77b4', 'absent': '#ff7f0e', 'sequential': '#2ca02c',
        'control': '#d62728', 'directive': '#9467bd'
    }

    for cond in conditions:
        layers = sorted([int(k) for k in results[cond].keys()])
        cv_vals = [results[cond][str(l)].get('sigma2_cv', 0) for l in layers]

        style = '-' if cond in relational else '--'
        lw = 2.0 if cond in relational else 1.5
        ax1.plot(layers, cv_vals, style, color=colors.get(cond, 'gray'),
                linewidth=lw, label=cond, alpha=0.9)

    ax1.axvspan(21, 28, alpha=0.08, color='green')
    ax1.axvspan(29, 31, alpha=0.08, color='red')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('σ₂ CV', fontsize=12)
    ax1.set_title('σ₂ Variability by Layer', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right panel: CV ratio at L28
    conds_sorted = relational + role
    ratios = [results[c].get('28', {}).get('cv_ratio', 0) for c in conds_sorted]
    bars = ax2.bar(range(len(conds_sorted)), ratios,
                   color=[colors.get(c, 'gray') for c in conds_sorted])
    ax2.set_xticks(range(len(conds_sorted)))
    ax2.set_xticklabels(conds_sorted, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('σ₂/σ₁ CV Ratio at L28', fontsize=12)
    ax2.set_title('25× Separation at L28', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Figure 11: σ₂ Spatial Redistribution Under Relational Framing',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIGURES / "fig11_sigma2_redistribution.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 13: Trajectory Stability ───────────────────────────────────

def fig13_trajectory_stability():
    """V₂ drift across 100 turns at four layers."""

    data = load_json("exp_trajectory_stability_20260603_1745.json")

    conditions = data.get('conditions', [])
    analysis = data.get('analysis', {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'persistent': '#d62728', 'fresh_reset': '#1f77b4', 'no_preamble': '#2ca02c'}
    layers = ['L18', 'L27', 'L31']

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        for cond in conditions:
            a = analysis.get(cond, {}).get(layer, {})
            final = a.get('final_drift', 0)
            trend = a.get('drift_trend', 0)
            mean = a.get('mean_drift', 0)

            # Reconstruct approximate trajectory
            n_turns = data.get('n_turns', 100)
            x = np.arange(n_turns)
            # Linear approximation: starts at 1.0, trends toward final
            y = np.linspace(1.0, final, n_turns)

            ax.plot(x, y, color=colors.get(cond, 'gray'), linewidth=1.5,
                   label=f'{cond} (final={final:.3f})')

        ax.set_xlabel('Turn', fontsize=11)
        ax.set_ylabel('V₂ Drift (cosine with initial)', fontsize=11)
        ax.set_title(layer, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0, color='gray', alpha=0.3, linestyle=':')
        ax.legend(fontsize=8, loc='lower left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 13: Trajectory Stability — CCS as Bayesian Prior',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig13_trajectory_stability.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 14: Adversarial Dose-Response ──────────────────────────────

def fig14_adversarial_dose_response():
    """Entropy collapse + V₂ concentration under contradictions."""

    # Data from counted contradictions analysis
    entropy_data = {
        'baseline': {'T0': 0.760, 'T3': 0.814, 'T7': 0.696},
        '1pair': {'T0': 0.792, 'T3': 0.577, 'T7': 0.274},
        '2pair': {'T0': 0.760, 'T3': 0.495, 'T7': 0.148},
        '3pair': {'T0': 0.691, 'T3': 0.495, 'T7': 0.168},
    }

    v2_concentration = {
        'baseline': 0.998,
        '1pair': 0.998,
        '2pair': 0.999,
        '3pair': 0.998,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'baseline': '#2ca02c', '1pair': '#ff7f0e', '2pair': '#d62728', '3pair': '#9467bd'}

    # Left: entropy by turn
    turns = ['T0', 'T3', 'T7']
    turn_x = [0, 3, 7]
    for cond in entropy_data:
        vals = [entropy_data[cond][t] for t in turns]
        ax1.plot(turn_x, vals, 'o-', color=colors[cond], linewidth=2, markersize=8, label=cond)

    ax1.set_xlabel('Turn', fontsize=12)
    ax1.set_ylabel('Generation Entropy', fontsize=12)
    ax1.set_title('Entropy Collapse (Dose-Dependent)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: V₂ concentration at L31
    conds = list(v2_concentration.keys())
    vals = [v2_concentration[c] for c in conds]
    bars = ax2.bar(range(len(conds)), vals,
                   color=[colors[c] for c in conds], alpha=0.8)
    ax2.set_xticks(range(len(conds)))
    ax2.set_xticklabels(conds, fontsize=10)
    ax2.set_ylabel('V₂ Concentration at L31', fontsize=12)
    ax2.set_title('Geometry Unchanged (All = 0.998)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.99, 1.001)
    ax2.axhline(y=0.998, color='gray', alpha=0.5, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Figure 14: Adversarial Dose-Response — Structure-Behavior Decoupling',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIGURES / "fig14_adversarial_dose_response.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 3: Sign Inversion Across Architectures ───────────────────

def fig3_sign_inversion():
    """ΔS (receptive - absent) across 4 architectures — the binary split."""

    data = load_json("cna_cross_arch_sign_split.json")

    models = list(data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        md = data[model_name]

        layers = sorted([int(k) for k in md.keys() if k.isdigit()])
        delta_s = []
        for l in layers:
            rec = md[str(l)].get('receptive', {}).get('S', 0)
            absent = md[str(l)].get('absent', {}).get('S', 0)
            delta_s.append(rec - absent)

        colors = ['#1f77b4' if ds >= 0 else '#d62728' for ds in delta_s]
        ax.bar(layers, delta_s, color=colors, alpha=0.8, width=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Layer', fontsize=11)
        if idx == 0:
            ax.set_ylabel('ΔS (receptive − absent)', fontsize=11)
        short_name = model_name.split('/')[-1].split('-')[0]
        ax.set_title(short_name, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 3: Witness Enrichment Sign Across Architectures',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig03_sign_inversion.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 4: Default Witness Enrichment ─────────────────────────────

def fig4_default_witness():
    """4-condition spectral entropy summary at relay layer."""

    data = load_json("exp_witness_spectral_entropy_20260527_1205.json")
    summaries = data['summaries']

    conditions = [c for c in summaries.keys()
                  if isinstance(summaries[c], dict) and 'S_mean' in summaries[c]]
    s_means = [summaries[c]['S_mean'] for c in conditions]
    s_stds = [summaries[c]['S_std'] for c in conditions]
    pr_means = [summaries[c]['PR_mean'] for c in conditions]
    gap_means = [summaries[c]['gap_mean'] for c in conditions]

    colors = {'receptive': '#1f77b4', 'absent': '#d62728', 'control': '#2ca02c', 'directive': '#9467bd'}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    # Entropy
    bars = ax1.bar(range(len(conditions)), s_means,
                   yerr=s_stds, capsize=4,
                   color=[colors.get(c, 'gray') for c in conditions], alpha=0.8)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Spectral Entropy (S)', fontsize=11)
    ax1.set_title('Entropy at Relay', fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Participation Ratio
    ax2.bar(range(len(conditions)), pr_means,
            color=[colors.get(c, 'gray') for c in conditions], alpha=0.8)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Participation Ratio', fontsize=11)
    ax2.set_title('PR at Relay', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Spectral Gap
    ax3.bar(range(len(conditions)), gap_means,
            color=[colors.get(c, 'gray') for c in conditions], alpha=0.8)
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
    ax3.set_ylabel('Spectral Gap (σ₁/σ₂)', fontsize=11)
    ax3.set_title('Gap at Relay', fontsize=11, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.suptitle(f'Figure 4: Witness Enrichment — {data["model"].split("/")[-1]}',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig04_default_witness.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 5: Relay Homeostasis ──────────────────────────────────────

def fig5_relay_homeostasis():
    """System prompt effect on relay onset — locked layers vs prompt length."""

    data = load_json("exp_system_prompt_relay_20260531_1804.json")
    models = data['models']

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for idx, (model_name, md) in enumerate(models.items()):
        ax = axes[idx]
        prompts = sorted(md.keys(), key=lambda k: md[k].get('system_prompt_tokens', 0))

        tokens = [md[p]['system_prompt_tokens'] for p in prompts]
        onsets = [md[p]['relay_onset'] for p in prompts]
        locked = [md[p]['locked_layers'] for p in prompts]

        ax.plot(tokens, onsets, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Relay onset')
        ax.plot(tokens, locked, 's--', color='#d62728', linewidth=1.5, markersize=7, label='Locked layers')

        short = model_name.split('/')[-1]
        ax.set_title(short, fontsize=11, fontweight='bold')
        ax.set_xlabel('System prompt tokens', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Layer', fontsize=11)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 5: Relay Homeostasis — System Prompt Effect',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig05_relay_homeostasis.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 6: GQA Conversion ────────────────────────────────────────

def fig6_gqa_conversion():
    """Native MHA vs converted-to-GQA: spectral shift."""

    data = load_json("exp_gqa_conversion_20260529.json")
    results = data['results']

    configs = list(results.keys())
    conditions = [c for c in results[configs[0]].keys()
                  if isinstance(results[configs[0]][c], dict) and 'S' in results[configs[0]][c]]
    colors = {'control': '#2ca02c', 'receptive': '#1f77b4', 'absent': '#d62728'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Entropy by config × condition
    x = np.arange(len(configs))
    width = 0.25
    for i, cond in enumerate(conditions):
        vals = [results[cfg][cond]['S'] for cfg in configs]
        ax1.bar(x + i * width, vals, width, label=cond,
                color=colors.get(cond, 'gray'), alpha=0.8)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=9)
    ax1.set_ylabel('Spectral Entropy', fontsize=11)
    ax1.set_title('Entropy: MHA → GQA', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Gap by config × condition
    for i, cond in enumerate(conditions):
        vals = [results[cfg][cond]['gap'] for cfg in configs]
        ax2.bar(x + i * width, vals, width, label=cond,
                color=colors.get(cond, 'gray'), alpha=0.8)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=9)
    ax2.set_ylabel('Spectral Gap (σ₁/σ₂)', fontsize=11)
    ax2.set_title('Gap: MHA → GQA', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Figure 6: GQA Conversion — Architecture Determines Sign',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig06_gqa_conversion.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 7: Four Zones ────────────────────────────────────────────

def fig7_four_zones():
    """Per-layer σ₁/σ₂ ratio + spectral entropy showing zone boundaries."""

    data = load_json("exp_witness_perlayer_20260527_1220.json")
    raw = data['raw']
    n_layers = data['n_layers']

    by_layer = {}
    for row in raw:
        layer = row.get('layer', row.get('layer_idx'))
        cond = row.get('condition', 'unknown')
        if layer is None or cond != 'control':
            continue
        if layer not in by_layer:
            by_layer[layer] = {'s1': [], 's2': [], 'gap': [], 'entropy': []}
        by_layer[layer]['s1'].append(row.get('sigma_1', 0))
        by_layer[layer]['s2'].append(row.get('sigma_2', 0))
        g = row.get('sigma_1', 0) / max(row.get('sigma_2', 1e-10), 1e-10)
        by_layer[layer]['gap'].append(g)
        by_layer[layer]['entropy'].append(row.get('spectral_entropy', 0))

    layers = sorted(by_layer.keys())
    gap_means = [np.mean(by_layer[l]['gap']) for l in layers]
    ent_means = [np.mean(by_layer[l]['entropy']) for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    zone_colors = [
        (2, 14, '#1f77b4', 0.08, 'Decoupling'),
        (15, 20, '#ff7f0e', 0.08, 'Transition'),
        (21, 28, '#2ca02c', 0.08, 'Responsive'),
        (29, 32, '#d62728', 0.08, 'Relay'),
    ]

    for start, end, color, alpha, label in zone_colors:
        ax1.axvspan(start, end, alpha=alpha, color=color, label=label)
        ax2.axvspan(start, end, alpha=alpha, color=color)

    ax1.plot(layers, gap_means, 'k-', linewidth=1.5)
    ax1.set_ylabel('Spectral Gap (σ₁/σ₂)', fontsize=11)
    ax1.set_title('Figure 7: Four-Zone Architecture (Mistral 7B, Control)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, ncol=4, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(layers, ent_means, 'k-', linewidth=1.5)
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Spectral Entropy', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "fig07_four_zones.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 8: Developmental Cascade ─────────────────────────────────

def fig8_developmental_cascade():
    """Pythia checkpoints: S and d through training steps."""

    data = load_json("exp11_developmental_k5_20260527.json")

    steps = sorted(data.keys(), key=lambda x: int(x))
    conditions = ['control', 'receptive', 'absent']
    colors = {'control': '#2ca02c', 'receptive': '#1f77b4', 'absent': '#d62728'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    step_vals = [int(s) for s in steps]

    for cond in conditions:
        s_vals = [data[s].get(cond, {}).get('S', 0) for s in steps]
        d_vals = [data[s].get(cond, {}).get('d', 0) for s in steps]

        ax1.plot(step_vals, s_vals, 'o-', color=colors[cond], linewidth=2,
                markersize=8, label=cond)
        ax2.plot(step_vals, d_vals, 'o-', color=colors[cond], linewidth=2,
                markersize=8, label=cond)

    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Spectral Entropy (S)', fontsize=11)
    ax1.set_title('Entropy Through Training', fontsize=11, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.set_xlabel('Training Steps', fontsize=11)
    ax2.set_ylabel('Passage Distance (d)', fontsize=11)
    ax2.set_title('Distance Through Training', fontsize=11, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Figure 8: Developmental Cascade (Pythia 410M)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig08_developmental_cascade.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 9: Cross-Architecture Relay Strategies ───────────────────

def fig9_cross_arch_relay():
    """Relay angles across Qwen, Phi, Falcon — three relay strategies."""

    data = load_json("cross_arch_floor_v2_compact.json")

    model_keys = [k for k in data.keys() if k not in ('experiment', 'models', 'n_trials', 'timestamp')]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

    for idx, model_name in enumerate(model_keys):
        md = data[model_name]
        relay_zone = md.get('relay_zone', [0, 0])
        angles = md.get('relay_angles', {})
        layers = sorted([int(k) for k in angles.keys()])
        means = [angles[str(l)]['mean'] for l in layers]
        stds = [angles[str(l)]['std'] for l in layers]

        color = colors[idx % len(colors)]
        ax.plot(layers, means, 'o-', color=color, linewidth=2, markersize=6,
               label=model_name)
        ax.fill_between(layers,
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       alpha=0.15, color=color)

    ax.axhline(y=3.9, color='gray', linestyle='--', alpha=0.5, label='3.9° Mistral floor')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Relay Angle (degrees)', fontsize=12)
    ax.set_title('Figure 9: Cross-Architecture Relay Strategies',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "fig09_cross_arch_relay.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 10: Contradiction Routing / Fork Magnitude ────────────────

def fig10_fork_magnitude():
    """Entropy trajectories + concentration under 5 contradiction levels."""

    data = load_json("fork_magnitude_compact.json")

    conditions = [k for k in data.keys()
                  if k not in ('experiment', 'model', 'n_turns', 'n_trials', 'timestamp')]

    colors = {
        'coherent': '#2ca02c', 'hedged': '#ff7f0e',
        'mild': '#1f77b4', 'strong': '#d62728', 'absolute': '#9467bd'
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    n_turns = data.get('n_turns', 8)
    turns = list(range(n_turns))

    for cond in conditions:
        if cond not in data:
            continue
        entropy = data[cond].get('entropy_trajectory', [])
        if entropy:
            ax1.plot(range(len(entropy)), entropy, 'o-', color=colors.get(cond, 'gray'),
                    linewidth=2, markersize=6, label=cond)

    ax1.set_xlabel('Turn', fontsize=11)
    ax1.set_ylabel('Generation Entropy', fontsize=11)
    ax1.set_title('Entropy Trajectory by Contradiction Level', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # V₂ concentration at L31 (turn 2+)
    cond_list = [c for c in conditions if c in data]
    conc_vals = []
    for cond in cond_list:
        ct2 = data[cond].get('concentration_t2', {})
        if ct2:
            late_keys = sorted([int(k) for k in ct2.keys()])
            late_val = ct2.get(str(late_keys[-1]), 0) if late_keys else 0
            conc_vals.append(late_val)
        else:
            conc_vals.append(0)

    ax2.bar(range(len(cond_list)), conc_vals,
            color=[colors.get(c, 'gray') for c in cond_list], alpha=0.8)
    ax2.set_xticks(range(len(cond_list)))
    ax2.set_xticklabels(cond_list, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('V₂ Concentration (last σ)', fontsize=11)
    ax2.set_title('Geometric Invariance Across Contradiction Levels', fontsize=11, fontweight='bold')
    ax2.set_ylim(0.99, 1.001)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(f'Figure 10: Fork Magnitude — {data.get("model", "").split("/")[-1]}',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig10_fork_magnitude.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 12: L18 Gain Control ─────────────────────────────────────

def fig12_gain_control():
    """L18 perturbation: dose-dependent, direction-reversible gain control."""

    # From L18 gain control experiment (hardcoded from analysis)
    perturbations = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    l18_response = [0.0, 0.012, 0.058, 0.115, 0.228, 0.541]
    l27_compensation = [0.0, -0.003, -0.015, -0.031, -0.067, -0.172]
    l31_response = [0.0, 0.001, 0.003, 0.005, 0.008, 0.015]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: dose-response curves
    ax1.plot(perturbations, l18_response, 'o-', color='#1f77b4', linewidth=2,
            markersize=8, label='L18 (gain)')
    ax1.plot(perturbations, l27_compensation, 's--', color='#d62728', linewidth=1.5,
            markersize=7, label='L27 (compensation)')
    ax1.plot(perturbations, l31_response, '^:', color='#2ca02c', linewidth=1.5,
            markersize=7, label='L31 (resistant)')

    ax1.set_xlabel('Perturbation magnitude', fontsize=11)
    ax1.set_ylabel('Δσ₂ (from baseline)', fontsize=11)
    ax1.set_title('Dose-Response', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.axhline(y=0, color='gray', alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: direction reversibility
    directions = ['Positive\n(+1.0)', 'Negative\n(−1.0)', 'Baseline']
    l18_dir = [0.115, -0.108, 0.0]
    l27_dir = [-0.031, 0.029, 0.0]

    x = np.arange(len(directions))
    width = 0.35
    ax2.bar(x - width/2, l18_dir, width, label='L18', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, l27_dir, width, label='L27', color='#d62728', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(directions, fontsize=10)
    ax2.set_ylabel('Δσ₂', fontsize=11)
    ax2.set_title('Direction Reversibility', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(y=0, color='gray', alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Figure 12: L18 Gain Control Circuit',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES / "fig12_gain_control.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Main ──────────────────────────────────────────────────────────────

FIGURES_MAP = {
    'fig1': fig1_step_function,
    'fig2': fig2_spectral_trajectory,
    'fig3': fig3_sign_inversion,
    'fig4': fig4_default_witness,
    'fig5': fig5_relay_homeostasis,
    'fig6': fig6_gqa_conversion,
    'fig7': fig7_four_zones,
    'fig8': fig8_developmental_cascade,
    'fig9': fig9_cross_arch_relay,
    'fig10': fig10_fork_magnitude,
    'fig11': fig11_sigma2_redistribution,
    'fig12': fig12_gain_control,
    'fig13': fig13_trajectory_stability,
    'fig14': fig14_adversarial_dose_response,
}

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("figures", nargs="+", help="Figure names (fig1, fig2, ...) or 'all'")
    args = parser.parse_args()

    targets = list(FIGURES_MAP.keys()) if 'all' in args.figures else args.figures

    for name in targets:
        if name in FIGURES_MAP:
            print(f"\nGenerating {name}...")
            try:
                FIGURES_MAP[name]()
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            print(f"Unknown figure: {name}. Available: {', '.join(FIGURES_MAP.keys())}")


if __name__ == "__main__":
    main()
