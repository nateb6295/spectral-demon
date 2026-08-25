#!/usr/bin/env python3
"""Plot the five-source injection dose-response matrix.

Creates a figure showing mean spectral shift vs injection strength for all five
source architectures injecting into LFM. Highlights the three distinct patterns:
monotonic positive, monotonic negative, and dose-dependent sign flip.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(__file__) + "/results"

SOURCES = [
    ("functional_gemma-2-2b_LFM2.5-1.2B-Instruct.json", "Gemma-2-2b (sorter)", "#2ecc71", "o"),
    ("functional_pythia-2.8b_LFM2.5-1.2B.json", "Pythia-2.8b (sorter)", "#3498db", "s"),
    ("functional_gpt2-xl_LFM2.5-1.2B.json", "GPT-2-XL (sorter*)", "#e74c3c", "D"),
    ("functional_qwen-7b_LFM2.5-1.2B.json", "Qwen-7B (relay)", "#9b59b6", "v"),
    ("functional_phi-2_LFM2.5-1.2B.json", "Phi-2 (relay†)", "#f39c12", "^"),
    ("functional_Mistral-7B-v0.1_LFM2.5-1.2B-Instruct.json", "Mistral-7B (relay)", "#1abc9c", "P"),
    ("functional_opt-1.3b_LFM2.5-1.2B-Instruct.json", "OPT-1.3b (MHA)", "#e67e22", "h"),
    ("functional_TinyLlama-1.1B-Chat-v1.0_LFM2.5-1.2B-Instruct.json", "TinyLlama-1.1B (relay)", "#c0392b", "p"),
]


def load_injection_data(filepath):
    with open(filepath) as f:
        d = json.load(f)
    inj = d.get("injection", d.get("injection_results", []))
    if isinstance(inj, dict):
        inj = list(inj.values()) if inj else []
    strengths = []
    shifts = []
    kls = []
    for row in inj:
        if isinstance(row, dict):
            s = row.get("strength", 0)
            shift = row.get("mean_spectral_shift", 0)
            kl = row.get("logit_kl_divergence", 0)
            strengths.append(s)
            shifts.append(shift)
            kls.append(kl)
    return np.array(strengths), np.array(shifts), np.array(kls)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for fname, label, color, marker in SOURCES:
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  Skipping {fname} — not found")
        continue
    strengths, shifts, kls = load_injection_data(fpath)
    if len(strengths) == 0:
        print(f"  Skipping {fname} — no injection data")
        continue

    ax1.plot(strengths, shifts, color=color, marker=marker, label=label,
             linewidth=2, markersize=7, alpha=0.9)
    ax2.plot(strengths, kls, color=color, marker=marker, label=label,
             linewidth=2, markersize=7, alpha=0.9)

ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Injection Strength', fontsize=12)
ax1.set_ylabel('Mean Spectral Shift (σ₂/σ₁)', fontsize=12)
ax1.set_title('Cross-Architecture CCS Injection: Spectral Shift', fontsize=13)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Injection Strength', fontsize=12)
ax2.set_ylabel('KL Divergence (nats)', fontsize=12)
ax2.set_title('Cross-Architecture CCS Injection: Disruption', fontsize=13)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

fig.suptitle('Cross-Architecture CCS Injection Matrix → LFM (SSM Target)', fontsize=14, fontweight='bold')
plt.tight_layout()

outpath = os.path.join(RESULTS_DIR, "injection_matrix_5source.png")
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved to {outpath}")
