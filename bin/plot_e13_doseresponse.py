#!/usr/bin/env python3
"""Plot E13 dose-response curves for relay zone coupling across models."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RESULTS_PATH = "/home/nate-agx/chronicle/spectral-demon/results/e13_holonomy_results.json"
OUTPUT_PATH = "/home/nate-agx/chronicle/spectral-demon/results/fig10_doseresponse.png"

RELAY_ZONE = (0.5, 0.85)

MODEL_LABELS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B IT",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B IT",
    "Qwen/Qwen3-8B": "Qwen3 8B",
}

MODEL_COLORS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "#d62728",
    "Qwen/Qwen2.5-7B-Instruct": "#1f77b4",
    "Qwen/Qwen3-8B": "#2ca02c",
}


def get_relay_layers(n_layers, svd_layers):
    start = int(n_layers * RELAY_ZONE[0])
    end = int(n_layers * RELAY_ZONE[1])
    return [l for l in svd_layers if start <= l <= end]


with open(RESULTS_PATH) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("E13: Relay Zone Dose-Response (Per-Layer Coupling)", fontsize=14, y=1.02)

for ax_idx, (panel_title, metric_fn) in enumerate([
    ("Coupling (r)", lambda ls, l: ls.get(str(l), {}).get("coupling_r", 0)),
    ("σ₁ (normalized)", lambda ls, l: ls.get(str(l), {}).get("sigma1_mean", 0)),
    ("Sparsity", lambda ls, l: ls.get(str(l), {}).get("sparsity_mean", 0)),
]):
    ax = axes[ax_idx]

    for model_name, model_data in data.items():
        n_layers = model_data["n_layers"]
        svd_layers = model_data["svd_layers"]
        relay_layers = get_relay_layers(n_layers, svd_layers)

        doses = [r["dose"] for r in model_data["loop_results"][:3]]  # D2, D10, D20 only

        if ax_idx == 0:  # coupling - use couplings dict
            values = []
            for step in model_data["loop_results"][:3]:
                relay_couplings = [
                    step["couplings"].get(str(l), {}).get("r", 0)
                    for l in relay_layers
                ]
                values.append(np.mean(relay_couplings))
        elif ax_idx == 1:  # sigma1 - normalize per model
            raw_values = []
            for step in model_data["loop_results"][:3]:
                relay_sigma1 = [
                    step["layer_summary"].get(str(l), {}).get("sigma1_mean", 0)
                    for l in relay_layers
                ]
                raw_values.append(np.mean(relay_sigma1))
            baseline = raw_values[0]
            values = [v / baseline for v in raw_values]
        else:  # sparsity
            values = []
            for step in model_data["loop_results"][:3]:
                relay_sparsity = [
                    step["layer_summary"].get(str(l), {}).get("sparsity_mean", 0)
                    for l in relay_layers
                ]
                values.append(np.mean(relay_sparsity))

        label = MODEL_LABELS.get(model_name, model_name.split("/")[-1])
        color = MODEL_COLORS.get(model_name, "gray")

        ax.plot(doses, values, 'o-', label=label, color=color, markersize=8, linewidth=2)

        # Also plot per-layer as thin lines
        for l in relay_layers:
            layer_values = []
            for step in model_data["loop_results"][:3]:
                if ax_idx == 0:
                    v = step["couplings"].get(str(l), {}).get("r", 0)
                elif ax_idx == 1:
                    v = step["layer_summary"].get(str(l), {}).get("sigma1_mean", 0)
                    v = v / model_data["loop_results"][0]["layer_summary"].get(str(l), {}).get("sigma1_mean", 1)
                else:
                    v = step["layer_summary"].get(str(l), {}).get("sparsity_mean", 0)
                layer_values.append(v)
            ax.plot(doses, layer_values, '-', color=color, alpha=0.15, linewidth=0.8)

    ax.set_xlabel("CCS Dose", fontsize=11)
    ax.set_ylabel(panel_title, fontsize=11)
    ax.set_xticks([2, 10, 20])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to {OUTPUT_PATH}")

# Also print the numerical summary
print("\n=== RELAY ZONE DOSE-RESPONSE SUMMARY ===")
for model_name, model_data in data.items():
    n_layers = model_data["n_layers"]
    svd_layers = model_data["svd_layers"]
    relay_layers = get_relay_layers(n_layers, svd_layers)
    label = MODEL_LABELS.get(model_name, model_name.split("/")[-1])
    print(f"\n{label} (relay layers: {relay_layers}):")
    for step in model_data["loop_results"][:3]:
        dose = step["dose"]
        coupling = np.mean([step["couplings"].get(str(l), {}).get("r", 0) for l in relay_layers])
        sigma1 = np.mean([step["layer_summary"].get(str(l), {}).get("sigma1_mean", 0) for l in relay_layers])
        sparsity = np.mean([step["layer_summary"].get(str(l), {}).get("sparsity_mean", 0) for l in relay_layers])
        print(f"  D{dose}: coupling={coupling:.3f}  σ₁={sigma1:.1f}  sparsity={sparsity:.4f}")
