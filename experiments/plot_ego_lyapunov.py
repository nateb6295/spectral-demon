#!/usr/bin/env python3
"""Generate F499 Ego Lyapunov basin geometry plots."""

import json, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path(__file__).parent.parent / "results" / "ego_lyapunov_sweep.json"
OUT_DIR = Path(__file__).parent.parent / "results" / "figures"

def load_and_group(path=None):
    p = Path(path) if path else RESULTS_PATH
    with open(p) as f:
        data = json.load(f)

    profiles = {}
    if "groups" in data:
        for gid_str, entries in data["groups"].items():
            gid = int(gid_str)
            scales = [e["scale"] for e in entries]
            wc_ratios = [e["wc_ratio"] for e in entries]
            ttr_ratios = [e["ttr_ratio"] for e in entries]
            profiles[gid] = {"scales": scales, "wc": wc_ratios, "ttr": ttr_ratios}
    else:
        results = data["results"] if "results" in data else data
        groups = defaultdict(list)
        for r in results:
            groups[r["group"]].append(r)
        for gid in sorted(groups.keys()):
            by_scale = defaultdict(list)
            for e in groups[gid]:
                by_scale[e["scale"]].append(e)
            base_wc = np.mean([x["wc"] for x in by_scale.get(1.0, [{"wc": 209}])])
            base_ttr = np.mean([x["ttr"] for x in by_scale.get(1.0, [{"ttr": 0.6}])])
            scales, wc_ratios, ttr_ratios = [], [], []
            for s in sorted(by_scale.keys()):
                seeds = by_scale[s]
                scales.append(s)
                wc_ratios.append(np.mean([x["wc"] for x in seeds]) / base_wc)
                ttr_ratios.append(np.mean([x["ttr"] for x in seeds]) / base_ttr if base_ttr > 0 else 0)
            profiles[gid] = {"scales": scales, "wc": wc_ratios, "ttr": ttr_ratios}
    return profiles

def plot_all_groups(profiles, outpath=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey='row')
    fig.suptitle("F499 — Ego Lyapunov: Per-Group KV Perturbation Profiles\n(Llama 3.1 8B, early layers, identity prompt, 3 seeds/condition)",
                 fontsize=14, fontweight='bold')

    colors_wc = '#2196F3'
    colors_ttr = '#FF5722'

    for gid in range(8):
        if gid not in profiles:
            continue
        p = profiles[gid]
        row = gid // 4
        col = gid % 4
        ax = axes[row, col]

        ax.plot(p['scales'], p['wc'], 'o-', color=colors_wc, label='WC ratio', linewidth=2, markersize=4)
        ax.plot(p['scales'], p['ttr'], 's--', color=colors_ttr, label='TTR ratio', linewidth=1.5, markersize=3, alpha=0.7)

        ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.3)
        ax.set_xlabel('Perturbation Scale')
        ax.set_title(f'KV{gid}', fontweight='bold')
        if col == 0:
            ax.set_ylabel('Ratio to baseline')
        ax.set_ylim(-0.05, 1.8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = outpath or (OUT_DIR / "f499_ego_lyapunov_all_groups.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def plot_critical_thresholds(profiles, outpath=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = []
    for gid in sorted(profiles.keys()):
        p = profiles[gid]
        ec = None
        for s, wc in zip(p['scales'], p['wc']):
            if wc < 0.8:
                ec = s
                break
        thresholds.append((gid, ec or 5.5))

    gids = [t[0] for t in thresholds]
    ecs = [t[1] for t in thresholds]

    colors = ['#E53935' if e < 2.5 else '#FF9800' if e < 3.5 else '#4CAF50' if e < 5.0 else '#2196F3'
              for e in ecs]

    bars = ax.bar(gids, ecs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('KV Group', fontsize=12)
    ax.set_ylabel('Critical Threshold (ε_c)', fontsize=12)
    ax.set_title('F499 — Per-Group Ego Lyapunov Critical Thresholds\n(lower = more fragile)', fontsize=14, fontweight='bold')
    ax.set_xticks(gids)
    ax.set_xticklabels([f'KV{g}' for g in gids])
    ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='3x perturbation')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for bar, ec in zip(bars, ecs):
        label = f'{ec:.1f}x' if ec < 5.5 else 'immune'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    out = outpath or (OUT_DIR / "f499_critical_thresholds.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    profiles = load_and_group(path)
    print(f"Loaded {len(profiles)} groups")
    plot_all_groups(profiles)
    plot_critical_thresholds(profiles)
    print("Done.")
