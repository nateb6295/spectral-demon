#!/usr/bin/env python3
"""Plot Q1 (first-quarter CCS balance) analysis for all source architectures.

Shows that the sum of CCS deltas in the first 25% of source layers perfectly
separates positive from negative injectors. Also shows within-Q1 sign coherence
to explain why Phi-2 flips while Qwen doesn't.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(__file__) + "/results"

SOURCES = [
    ("functional_gemma-2-2b_LFM2.5-1.2B-Instruct.json", "Gemma-2-2b", "#2ecc71", "positive"),
    ("functional_pythia-2.8b_LFM2.5-1.2B.json", "Pythia-2.8b", "#3498db", "positive"),
    ("functional_gpt2-xl_LFM2.5-1.2B.json", "GPT-2-XL", "#e74c3c", "negative"),
    ("functional_qwen-7b_LFM2.5-1.2B.json", "Qwen-7B", "#9b59b6", "negative"),
    ("functional_phi-2_zone_map.json", "Phi-2", "#f39c12", "sign flip"),
]

# Check for new results from latest experiments
EXTRA_SOURCES = [
    ("functional_mistral-7b_LFM2.5-1.2B.json", "Mistral-7B", "#1abc9c", None),
    ("functional_Mistral-7B-v0.1_LFM2.5-1.2B-Instruct.json", "Mistral-7B", "#1abc9c", None),
    ("functional_opt-1.3b_LFM2.5-1.2B.json", "OPT-1.3b", "#e67e22", None),
    ("functional_opt-1.3b_LFM2.5-1.2B-Instruct.json", "OPT-1.3b", "#e67e22", None),
    ("functional_TinyLlama-1.1B-Chat-v1.0_LFM2.5-1.2B-Instruct.json", "TinyLlama-1.1B", "#c0392b", None),
]


def analyze_q1(filepath):
    with open(filepath) as f:
        d = json.load(f)
    zones = d.get("zones_a", d.get("zones", []))
    q1 = [z for z in zones if z["relative_depth"] < 0.25]
    if not q1:
        return None

    deltas = [z["delta"] for z in q1]
    net = sum(deltas)
    abs_sum = sum(abs(d) for d in deltas)
    cancel = 1 - abs(net) / abs_sum if abs_sum > 0 else 0
    net_sign = 1 if net >= 0 else -1
    disagree = sum(1 for d in deltas if (d >= 0) != (net >= 0)) / len(deltas) if deltas else 0

    # Injection data
    injection = d.get("injection", [])
    shifts = {r.get("strength", 0): r.get("mean_spectral_shift", 0) for r in injection if isinstance(r, dict)}

    return {
        "q1_net": net,
        "q1_abs_sum": abs_sum,
        "cancellation": cancel,
        "disagree_frac": disagree,
        "n_layers": len(q1),
        "deltas": deltas,
        "layers": [z["layer"] for z in q1],
        "zones": [z["zone"] for z in q1],
        "shifts": shifts,
    }


all_sources = list(SOURCES)
for fname, label, color, _ in EXTRA_SOURCES:
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        inj = d.get("injection", [])
        if inj:
            shifts = [r.get("mean_spectral_shift", 0) for r in inj if isinstance(r, dict) and r.get("strength") == 1.0]
            sign = "positive" if shifts and shifts[0] > 0.001 else "negative" if shifts and shifts[0] < -0.001 else "near-zero"
        else:
            sign = "unknown"
        all_sources.append((fname, label, color, sign))
        print(f"  Added {label}: {sign}")

results = {}
for fname, label, color, sign in all_sources:
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        continue
    r = analyze_q1(fpath)
    if r:
        r["sign"] = sign
        r["color"] = color
        results[label] = r

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Q1 net balance vs injection sign
ax1 = axes[0]
for label, r in results.items():
    marker = "^" if r["sign"] == "sign flip" else "o"
    edge = "black" if r["sign"] == "sign flip" else None
    ax1.barh(label, r["q1_net"], color=r["color"], alpha=0.8, edgecolor=edge, linewidth=2 if edge else 0)
ax1.axvline(x=0, color="black", linewidth=1.5)
ax1.set_xlabel("Q1 Net Balance (Σ CCS deltas, first 25% of layers)", fontsize=10)
ax1.set_title("Q1 Balance Predicts Injection Sign", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.2, axis="x")

# Add sign labels
for label, r in results.items():
    x = r["q1_net"]
    ha = "left" if x >= 0 else "right"
    offset = 0.01 if x >= 0 else -0.01
    ax1.text(x + offset, label, r["sign"], fontsize=8, ha=ha, va="center", style="italic")

# Panel 2: Cancellation ratio vs Q1 net (Kimi's coherence check)
ax2 = axes[1]
for label, r in results.items():
    marker = "^" if r["sign"] == "sign flip" else "o" if r["sign"] == "positive" else "s"
    ax2.scatter(abs(r["q1_net"]), r["cancellation"] * 100, s=120, c=r["color"],
                marker=marker, edgecolors="black", linewidths=1, zorder=5)
    ax2.annotate(label, (abs(r["q1_net"]), r["cancellation"] * 100),
                 textcoords="offset points", xytext=(8, 4), fontsize=9)
ax2.set_xlabel("|Q1 Net|", fontsize=10)
ax2.set_ylabel("Within-Q1 Cancellation (%)", fontsize=10)
ax2.set_title("Internal Structure: Cancellation vs Balance\n○=positive, □=negative, △=flip",
              fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.3)

# Panel 3: Per-layer Q1 profiles stacked
ax3 = axes[2]
y_offset = 0
y_ticks = []
y_labels = []
for label, r in sorted(results.items(), key=lambda x: -x[1]["q1_net"]):
    layers = r["layers"]
    deltas = r["deltas"]
    zones_list = r["zones"]
    for i, (l, d, z) in enumerate(zip(layers, deltas, zones_list)):
        c = "#2ecc71" if "+" in z else "#e74c3c" if "-" in z else "#95a5a6"
        ax3.barh(y_offset, d, height=0.6, color=c, alpha=0.8)
        y_offset -= 1
    y_ticks.append(y_offset + len(layers) / 2)
    y_labels.append(f"{label}\nQ1={r['q1_net']:+.3f}")
    y_offset -= 1.5

ax3.axvline(x=0, color="black", linewidth=1)
ax3.set_yticks(y_ticks)
ax3.set_yticklabels(y_labels, fontsize=8)
ax3.set_xlabel("CCS Delta per layer", fontsize=10)
ax3.set_title("Q1 Layer Profiles\n(green=R+, red=R-, gray=invariant)", fontsize=11, fontweight="bold")
ax3.grid(True, alpha=0.2, axis="x")

fig.suptitle("F594: Early-Quarter CCS Balance Determines Injection Direction",
             fontsize=14, fontweight="bold")
plt.tight_layout()

outpath = os.path.join(RESULTS_DIR, "q1_analysis.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved to {outpath}")
