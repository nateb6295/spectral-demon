#!/usr/bin/env python3
"""Plot F601: Q1 as universal predictor across models and framings.

Act III of Paper 11 — Q1 is the sufficient statistic. Architecture and prompt
framing are both just ways of setting Q1. Injection outcome depends on Q1,
not on which factor produced it. Relay dead zone is the exception that proves
the rule.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = "results"

models = {
    "pythia": {"label": "Pythia-2.8b", "color": "#2ecc71", "marker": "o", "species": "tunnel"},
    "gpt2": {"label": "GPT-2", "color": "#27ae60", "marker": "s", "species": "tunnel"},
    "tinyllama_1.1b_chat_v1.0": {"label": "TinyLlama", "color": "#e74c3c", "marker": "^", "species": "relay"},
    "mistral_7b_v0.1": {"label": "Mistral-7B", "color": "#3498db", "marker": "v", "species": "relay"},
    "gemma_2_2b": {"label": "Gemma-2-2b", "color": "#9b59b6", "marker": "D", "species": "sorter"},
}

all_points = []
for model_key, meta in models.items():
    d = json.load(open(f"{RESULTS_DIR}/tuning_knob_{model_key}.json"))
    for entry in d["gradient"]:
        q1 = entry["q1"]
        inj = {i["strength"]: i["mean_shift"] for i in entry["injection"]}
        all_points.append({
            "model": model_key, "label": meta["label"], "color": meta["color"],
            "marker": meta["marker"], "species": meta["species"],
            "framing": entry["name"], "q1": q1,
            "shift_1": inj.get(1.0), "shift_5": inj.get(5.0),
        })

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("F601: Q1 Is the Universal Predictor — Architecture and Framing Are Just Inputs",
             fontsize=13, fontweight="bold")

# Panel 1: Q1 vs shift@5.0 — all 25 points colored by model
ax = axes[0]
ax.set_title("Q1 vs Injection Shift (strength=5.0)\n25 model×framing combinations", fontsize=10)

for model_key, meta in models.items():
    pts = [p for p in all_points if p["model"] == model_key and p["shift_5"] is not None]
    q1s = [p["q1"] for p in pts]
    shifts = [p["shift_5"] for p in pts]
    ax.scatter(q1s, shifts, c=meta["color"], marker=meta["marker"], s=80,
               label=meta["label"], zorder=5, edgecolors="white", linewidths=0.5)

q1_all = [p["q1"] for p in all_points if p["shift_5"] is not None]
s5_all = [p["shift_5"] for p in all_points if p["shift_5"] is not None]
r, pval = stats.pearsonr(q1_all, s5_all)
slope, intercept = np.polyfit(q1_all, s5_all, 1)
x_fit = np.linspace(min(q1_all) - 0.02, max(q1_all) + 0.02, 100)
ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.4, linewidth=1)
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.axvline(0, color="red", linestyle="-", alpha=0.3)
ax.set_xlabel("Q1")
ax.set_ylabel("Spectral shift @ strength 5.0")
ax.text(0.05, 0.95, f"r = {r:.3f}\np < {pval:.1e}\nn = {len(q1_all)}", transform=ax.transAxes,
        fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.legend(fontsize=7, loc="lower right")

# Panel 2: Same but colored by SPECIES (tunnel/relay/sorter)
ax = axes[1]
ax.set_title("Same Data, Colored by Species\n(relay dead zone visible)", fontsize=10)

species_colors = {"tunnel": "#2ecc71", "relay": "#e74c3c", "sorter": "#9b59b6"}
species_markers = {"tunnel": "o", "relay": "^", "sorter": "D"}
for sp in ["tunnel", "relay", "sorter"]:
    pts = [p for p in all_points if p["species"] == sp and p["shift_5"] is not None]
    q1s = [p["q1"] for p in pts]
    shifts = [p["shift_5"] for p in pts]
    ax.scatter(q1s, shifts, c=species_colors[sp], marker=species_markers[sp], s=80,
               label=sp.capitalize(), zorder=5, edgecolors="white", linewidths=0.5)

# Highlight Mistral dead zone
dead_zone_pts = [p for p in all_points if p["model"] == "mistral_7b_v0.1"
                 and p["shift_5"] is not None and p["q1"] > 0]
if dead_zone_pts:
    dz_q1 = [p["q1"] for p in dead_zone_pts]
    dz_s5 = [p["shift_5"] for p in dead_zone_pts]
    ax.scatter(dz_q1, dz_s5, facecolors="none", edgecolors="red", s=200,
               linewidths=2, zorder=6, label="Relay dead zone")

# Fit WITHOUT relays to show what tunnels+sorters predict
non_relay = [p for p in all_points if p["species"] != "relay" and p["shift_5"] is not None]
nr_q1 = [p["q1"] for p in non_relay]
nr_s5 = [p["shift_5"] for p in non_relay]
r_nr, _ = stats.pearsonr(nr_q1, nr_s5)
slope_nr, int_nr = np.polyfit(nr_q1, nr_s5, 1)
ax.plot(x_fit, slope_nr * x_fit + int_nr, "k--", alpha=0.4, linewidth=1,
        label=f"Tunnel+Sorter fit (r={r_nr:.3f})")

ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.axvline(0, color="red", linestyle="-", alpha=0.3)
ax.set_xlabel("Q1")
ax.set_ylabel("Spectral shift @ strength 5.0")
ax.legend(fontsize=7, loc="lower right")

# Panel 3: Residual from universal fit — relay offset
ax = axes[2]
ax.set_title("Relay Offset from Universal Q1→Shift Fit\n(conservation constraint adds activation energy)", fontsize=10)

predicted_all = [slope * p["q1"] + intercept for p in all_points if p["shift_5"] is not None]
actual_all = [p["shift_5"] for p in all_points if p["shift_5"] is not None]
species_all = [p["species"] for p in all_points if p["shift_5"] is not None]
labels_all = [p["label"] for p in all_points if p["shift_5"] is not None]
residuals = [a - p for a, p in zip(actual_all, predicted_all)]

for sp in ["tunnel", "relay", "sorter"]:
    sp_residuals = [r for r, s in zip(residuals, species_all) if s == sp]
    sp_q1 = [q for q, s in zip(q1_all, species_all) if s == sp]
    ax.scatter(sp_q1, sp_residuals, c=species_colors[sp], marker=species_markers[sp],
               s=80, label=f"{sp.capitalize()} (mean={np.mean(sp_residuals):+.4f})",
               zorder=5, edgecolors="white", linewidths=0.5)

ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
ax.fill_between([-0.1, 0.6], -0.02, 0.02, color="green", alpha=0.08, label="±0.02 band")
ax.set_xlabel("Q1")
ax.set_ylabel("Residual (actual - predicted shift)")
ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/f601_q1_universal.png", dpi=150, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/f601_q1_universal.png")

# Print stats
print(f"\nUniversal fit: r={r:.4f}, p={pval:.2e}, n={len(q1_all)}")
print(f"Tunnel+Sorter only: r={r_nr:.4f}")
for sp in ["tunnel", "relay", "sorter"]:
    sp_res = [r for r, s in zip(residuals, species_all) if s == sp]
    print(f"  {sp}: mean residual = {np.mean(sp_res):+.5f}, std = {np.std(sp_res):.5f}")
