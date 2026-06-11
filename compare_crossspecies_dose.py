#!/usr/bin/env python3
"""Cross-species dose-response comparison: S/R trajectories across architectures."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent / "results"

PHI_DATA = {
    1: {"p2r": 0.006035, "p2s": 0.006011},
    3: {"p2r": 0.009975, "p2s": 0.009452},
    5: {"p2r": 0.019500, "p2s": 0.013799},
    10: {"p2r": 0.026295, "p2s": 0.022720},
    20: {"p2r": 0.082381, "p2s": 0.073922},
}

GEMMA_DATA = {
    20: {"p2r": 0.000188, "p2s": 0.000145},
}

MISTRAL_DATA = {
    1: {"p2r": 0.072861, "p2s": 0.050248},
    5: {"p2r": 0.012038, "p2s": 0.012093},
    10: {"p2r": 0.003275, "p2s": 0.002866},
    20: {"p2r": 0.000950, "p2s": 0.001242},
}


def update_mistral_from_files():
    """Scan for new Mistral dose files and update."""
    for f in sorted(BASE.glob("exp_selfref_vs_relational_mistral_dose*_*.json")):
        try:
            d = json.load(open(f))
            dose = d.get("dose", 0)
            s = d["summary"]
            MISTRAL_DATA[dose] = {
                "p2r": s["relational"]["p2_disruption"],
                "p2s": s["self_ref"]["p2_disruption"],
            }
        except Exception:
            continue


def main():
    update_mistral_from_files()

    print("=" * 60)
    print("CROSS-SPECIES DOSE-RESPONSE: S/R TRAJECTORIES")
    print("=" * 60)

    species = {
        "Potter (Gemma 27B)": GEMMA_DATA,
        "Goldsmith (Mistral 7B)": MISTRAL_DATA,
        "Painter (Phi 3.5)": PHI_DATA,
    }

    for name, data in species.items():
        doses = sorted(data.keys())
        print(f"\n{name}:")
        print(f"  {'Dose':<6} {'P2(R)':<10} {'P2(S)':<10} {'S/R':<8}")
        for dose in doses:
            d = data[dose]
            sr = d["p2s"] / d["p2r"] if d["p2r"] > 0 else 0
            print(f"  {dose:<6} {d['p2r']:<10.6f} {d['p2s']:<10.6f} {sr:<8.3f}")

    # Plot if we have enough data
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {
        "Potter (Gemma 27B)": "#4CAF50",
        "Goldsmith (Mistral 7B)": "#2196F3",
        "Painter (Phi 3.5)": "#FF9800",
    }

    markers = {
        "Potter (Gemma 27B)": "^",
        "Goldsmith (Mistral 7B)": "s",
        "Painter (Phi 3.5)": "o",
    }

    # Panel 1: S/R ratio vs dose
    ax = axes[0]
    for name, data in species.items():
        doses = sorted(data.keys())
        sr_vals = [data[d]["p2s"] / data[d]["p2r"] if data[d]["p2r"] > 0 else 0 for d in doses]
        ax.plot(doses, sr_vals, f'{markers[name]}-', color=colors[name],
                label=name, linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='S/R = 1.0')
    ax.set_xlabel('CCS Dose (turns)')
    ax.set_ylabel('S/R Disruption Ratio')
    ax.set_title('Preamble-Type Sensitivity vs Dose')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_ylim(0.5, 1.5)

    # Panel 2: P2 disruption (log scale)
    ax = axes[1]
    for name, data in species.items():
        doses = sorted(data.keys())
        p2r_vals = [data[d]["p2r"] for d in doses]
        ax.plot(doses, p2r_vals, f'{markers[name]}-', color=colors[name],
                label=name, linewidth=2, markersize=8)
    ax.set_xlabel('CCS Dose (turns)')
    ax.set_ylabel('P2 Disruption (relational)')
    ax.set_title('Preamble Dependence vs Dose')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    fig.suptitle('Cross-Species Dose-Response: Three Identity Strategies',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = BASE / "crossspecies_dose_response.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
