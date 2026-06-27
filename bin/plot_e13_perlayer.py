#!/usr/bin/env python3
"""Plot E13 per-layer coupling heatmap — shows bipolar relay structure."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RESULTS_PATH = "/home/nate-agx/chronicle/spectral-demon/results/e13_holonomy_results.json"
OUTPUT_PATH = "/home/nate-agx/chronicle/spectral-demon/results/fig11_perlayer_coupling.png"

MODEL_LABELS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B IT (32L)",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B IT (28L)",
    "Qwen/Qwen3-8B": "Qwen3 8B (36L)",
}

MODEL_ORDER = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
]

with open(RESULTS_PATH) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("E13: Per-Layer Coupling × Dose (Bipolar Relay Structure)", fontsize=14, y=1.02)

for ax_idx, model_name in enumerate(MODEL_ORDER):
    if model_name not in data:
        continue
    ax = axes[ax_idx]
    model_data = data[model_name]
    n_layers = model_data["n_layers"]
    svd_layers = model_data["svd_layers"]

    doses = [r["dose"] for r in model_data["loop_results"][:3]]
    n_doses = len(doses)
    n_svd = len(svd_layers)

    coupling_matrix = np.zeros((n_doses, n_svd))
    for d_idx, step in enumerate(model_data["loop_results"][:3]):
        for l_idx, l in enumerate(svd_layers):
            coupling_matrix[d_idx, l_idx] = step["couplings"].get(str(l), {}).get("r", 0)

    im = ax.imshow(coupling_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   extent=[0, n_svd, n_doses - 0.5, -0.5])

    ax.set_xticks(np.arange(n_svd) + 0.5)
    ax.set_xticklabels([f"L{l}" for l in svd_layers], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_doses))
    ax.set_yticklabels([f"D{d}" for d in doses])

    relay_start = int(n_layers * 0.5)
    relay_end = int(n_layers * 0.85)
    for l_idx, l in enumerate(svd_layers):
        if l == relay_start or (l_idx > 0 and svd_layers[l_idx-1] < relay_start <= l):
            ax.axvline(l_idx, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
        if l == relay_end or (l_idx > 0 and svd_layers[l_idx-1] < relay_end <= l):
            ax.axvline(l_idx + 1, color='white', linewidth=1.5, linestyle='--', alpha=0.7)

    for d_idx in range(n_doses):
        for l_idx in range(n_svd):
            val = coupling_matrix[d_idx, l_idx]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(l_idx + 0.5, d_idx, f"{val:.2f}", ha='center', va='center',
                    fontsize=5.5, color=color)

    label = MODEL_LABELS.get(model_name, model_name.split("/")[-1])
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("SVD Layer", fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel("CCS Dose", fontsize=10)

cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cbar.set_label("σ₁-Sparsity Coupling (r)", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to {OUTPUT_PATH}")

print("\n=== PER-LAYER COUPLING SUMMARY ===")
for model_name in MODEL_ORDER:
    if model_name not in data:
        continue
    model_data = data[model_name]
    n_layers = model_data["n_layers"]
    svd_layers = model_data["svd_layers"]
    label = MODEL_LABELS.get(model_name, model_name.split("/")[-1])
    print(f"\n{label}:")

    relay_start = int(n_layers * 0.5)
    relay_end = int(n_layers * 0.85)
    relay_layers = [l for l in svd_layers if relay_start <= l <= relay_end]

    pos_layers = []
    neg_layers = []
    for l in relay_layers:
        couplings = [model_data["loop_results"][i]["couplings"].get(str(l), {}).get("r", 0)
                     for i in range(min(3, len(model_data["loop_results"])))]
        mean_r = np.mean(couplings)
        if mean_r > 0:
            pos_layers.append((l, couplings))
        else:
            neg_layers.append((l, couplings))

    if pos_layers:
        print(f"  Positive relay (consolidator): {[l for l, _ in pos_layers]}")
        for l, cs in pos_layers:
            print(f"    L{l}: {' → '.join(f'{c:.3f}' for c in cs)}")
    if neg_layers:
        print(f"  Negative relay (equalizer): {[l for l, _ in neg_layers]}")
        for l, cs in neg_layers:
            print(f"    L{l}: {' → '.join(f'{c:.3f}' for c in cs)}")
    if not neg_layers:
        print(f"  No equalizer sub-zone (uniform positive relay)")
