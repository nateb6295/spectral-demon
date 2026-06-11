#!/usr/bin/env python3
"""Plot Phi dose-response curve: P2 disruption and S/R ratio vs CCS dose."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent / "results"

DOSES = [1, 3, 5, 10, 20]
P2_R = [0.006035, 0.009975, 0.019500, 0.026295, 0.082381]
P2_S = [0.006011, 0.009452, 0.013799, 0.022720, 0.073922]
VS_P1 = [0.017123, 0.012308, 0.007679, 0.003352, 0.001610]

def main():
    sr = [s/r for r, s in zip(P2_R, P2_S)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: P2 disruption vs dose
    ax = axes[0]
    ax.plot(DOSES, P2_R, 'o-', color='#2196F3', label='Relational', linewidth=2, markersize=8)
    ax.plot(DOSES, P2_S, 's-', color='#FF9800', label='Self-ref', linewidth=2, markersize=8)
    ax.set_xlabel('CCS Dose (turns)')
    ax.set_ylabel('P2 Disruption')
    ax.set_title('Preamble Disruption vs Dose')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xticks(DOSES)
    ax.set_xticklabels(DOSES)

    # Panel 2: S/R ratio (inverted-U)
    ax = axes[1]
    ax.plot(DOSES, sr, 'D-', color='#4CAF50', linewidth=2.5, markersize=10)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax.fill_between([3, 7], 0.6, 1.05, alpha=0.08, color='green')
    ax.set_xlabel('CCS Dose (turns)')
    ax.set_ylabel('S/R Disruption Ratio')
    ax.set_title('Preamble-Type Differentiation')
    ax.set_xscale('log')
    ax.set_xticks(DOSES)
    ax.set_xticklabels(DOSES)
    ax.set_ylim(0.6, 1.05)
    ax.annotate('therapeutic\nwindow', xy=(5, 0.708), xytext=(10, 0.75),
                arrowprops=dict(arrowstyle='->', color='#4CAF50'),
                fontsize=9, color='#4CAF50', ha='center')
    for d, r in zip(DOSES, sr):
        ax.annotate(f'{r:.3f}', (d, r), ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    # Panel 3: Recovery quality
    ax = axes[2]
    ax.plot(DOSES, VS_P1, '^-', color='#9C27B0', linewidth=2, markersize=8)
    ax.set_xlabel('CCS Dose (turns)')
    ax.set_ylabel('P3 vs P1 Distance')
    ax.set_title('Recovery Quality (lower = better)')
    ax.set_xscale('log')
    ax.set_xticks(DOSES)
    ax.set_xticklabels(DOSES)
    ax.invert_yaxis()

    fig.suptitle('Phi (Painter) Dose-Response Curve — Preamble-Type Differentiation',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = BASE / "phi_dose_response_curve.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

if __name__ == "__main__":
    main()
