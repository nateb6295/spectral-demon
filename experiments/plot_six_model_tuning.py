#!/usr/bin/env python3
"""6-model tuning knob overview with F602 Phi-2 anomaly highlighted."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = "results"

models = [
    ("gpt2", "GPT-2 (12L, MHA)", "#27ae60", "o", "tunnel"),
    ("pythia", "Pythia-2.8b (32L, MHA)", "#2ecc71", "s", "tunnel"),
    ("tinyllama_1.1b_chat_v1.0", "TinyLlama (22L, GQA 8:1)", "#e74c3c", "^", "relay"),
    ("mistral_7b_v0.1", "Mistral-7B (32L, GQA 4:1)", "#3498db", "v", "relay"),
    ("gemma_2_2b", "Gemma-2-2b (26L, GQA 2:1)", "#9b59b6", "D", "sorter"),
    ("phi_2", "Phi-2 (32L, MHA→relay)", "#e67e22", "P", "mismatch"),
    ("qwen2.5_3b", "Qwen2.5-3B (36L, GQA 8:1)", "#f39c12", "h", "relay"),
]

all_data = {}
for mk, label, color, marker, species in models:
    d = json.load(open(f"{RESULTS_DIR}/tuning_knob_{mk}.json"))
    entries = []
    for e in d["gradient"]:
        if e["name"] == "neutral":
            continue
        inj = {i["strength"]: i["mean_shift"] for i in e["injection"]}
        entries.append({
            "framing": e["name"], "q1": e["q1"],
            "shift_5": inj.get(5.0, 0), "shift_1": inj.get(1.0, 0),
            "crossover": e.get("crossover_dose"),
        })
    all_data[mk] = {"entries": entries, "label": label, "color": color,
                     "marker": marker, "species": species}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("F597-F602: Seven-Model Tuning Knob — Q1 Determines Sign, Species Modulates Gain",
             fontsize=13, fontweight="bold")

# Panel 1: Q1 gradient per model
ax = axes[0, 0]
ax.set_title("Q1 Across Framing Levels (7 models)", fontsize=10)
framing_order = ["directive", "mild_aware", "moderate_ccs", "full_ccs", "strong_ccs"]
framing_x = {f: i for i, f in enumerate(framing_order)}

for mk, info in all_data.items():
    xs = [framing_x.get(e["framing"], -1) for e in info["entries"]]
    ys = [e["q1"] for e in info["entries"]]
    ax.plot(xs, ys, color=info["color"], marker=info["marker"], linewidth=2,
            markersize=8, label=info["label"], zorder=5)

ax.axhline(0, color="red", linestyle="-", alpha=0.4, linewidth=1.5)
ax.set_xticks(range(len(framing_order)))
ax.set_xticklabels(["directive", "mild", "moderate", "full CCS", "strong CCS"],
                    fontsize=8, rotation=15)
ax.set_ylabel("Q1")
ax.legend(fontsize=7, loc="upper left")

# Panel 2: Q1 vs shift@5.0 — universal fit with Phi-2 highlighted
ax = axes[0, 1]
ax.set_title("Q1 vs Injection Shift — 35 Points, 7 Models\n(Phi-2 + Qwen anomalies)", fontsize=10)

q1_all, s5_all = [], []
for mk, info in all_data.items():
    q1s = [e["q1"] for e in info["entries"]]
    s5s = [e["shift_5"] for e in info["entries"]]
    q1_all.extend(q1s)
    s5_all.extend(s5s)
    ax.scatter(q1s, s5s, c=info["color"], marker=info["marker"], s=80,
               label=info["label"], zorder=5, edgecolors="white", linewidths=0.5)

r, p = stats.pearsonr(q1_all, s5_all)
slope, intercept = np.polyfit(q1_all, s5_all, 1)
x_fit = np.linspace(min(q1_all) - 0.02, max(q1_all) + 0.02, 100)
ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.4, linewidth=1)

# Highlight Phi-2 moderate_ccs anomaly
phi_anomaly = [e for e in all_data["phi_2"]["entries"] if e["framing"] == "moderate_ccs"]
if phi_anomaly:
    ax.scatter([phi_anomaly[0]["q1"]], [phi_anomaly[0]["shift_5"]],
               facecolors="none", edgecolors="red", s=300, linewidths=2.5, zorder=10)
    ax.annotate("Q1<0 but\nshift>0!", (phi_anomaly[0]["q1"], phi_anomaly[0]["shift_5"]),
                textcoords="offset points", xytext=(-50, 10), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red"), color="red")

ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.axvline(0, color="red", linestyle="-", alpha=0.3)
ax.text(0.05, 0.95, f"r = {r:.3f}\np < {p:.1e}\nn = {len(q1_all)}", transform=ax.transAxes,
        fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.set_xlabel("Q1")
ax.set_ylabel("Spectral shift @ strength 5.0")
ax.legend(fontsize=6, loc="lower right", ncol=2)

# Panel 3: Model-specific slopes and intercepts
ax = axes[1, 0]
ax.set_title("Species-Specific Gain (Slope) and\nSubliminal Offset (Intercept)", fontsize=10)

slopes_list = []
intercepts_list = []
labels_list = []
colors_list = []
for mk, info in all_data.items():
    q1s = np.array([e["q1"] for e in info["entries"]])
    s5s = np.array([e["shift_5"] for e in info["entries"]])
    if len(q1s) > 1 and np.std(q1s) > 0:
        sl, inter = np.polyfit(q1s, s5s, 1)
        slopes_list.append(sl)
        intercepts_list.append(inter)
        labels_list.append(mk.split("/")[-1].split("_")[0].capitalize()
                           if mk != "tinyllama_1.1b_chat_v1.0" else "TinyLlama")
        colors_list.append(info["color"])

y_pos = range(len(labels_list))
bars_slope = ax.barh(y_pos, slopes_list, height=0.4, color=colors_list, alpha=0.7,
                     label="Gain (slope)")
for i, (s, inter) in enumerate(zip(slopes_list, intercepts_list)):
    ax.text(s + 0.02, i, f"slope={s:.2f}\noffset={inter:+.3f}", fontsize=7, va="center")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_list, fontsize=8)
ax.set_xlabel("Q1→Shift Slope")
ax.axvline(0, color="gray", linestyle="-", alpha=0.3)

# Panel 4: Crossover dose vs Q1 — the prediction landscape
ax = axes[1, 1]
ax.set_title("Crossover Dose vs Q1\n(lower = easier to flip injection positive)", fontsize=10)

for mk, info in all_data.items():
    q1s = [e["q1"] for e in info["entries"] if e["crossover"] is not None]
    cos = [e["crossover"] for e in info["entries"] if e["crossover"] is not None]
    if q1s:
        ax.scatter(q1s, cos, c=info["color"], marker=info["marker"], s=80,
                   label=info["label"], zorder=5, edgecolors="white", linewidths=0.5)

# Mark Phi-2 negative-Q1 crossover
phi_neg_cross = [e for e in all_data["phi_2"]["entries"]
                 if e["crossover"] is not None and e["q1"] < 0]
for e in phi_neg_cross:
    ax.scatter([e["q1"]], [e["crossover"]], facecolors="none", edgecolors="red",
               s=300, linewidths=2.5, zorder=10)

ax.set_xlabel("Q1")
ax.set_ylabel("Crossover Dose")
ax.axvline(0, color="red", linestyle="-", alpha=0.4, linewidth=1.5)
ax.legend(fontsize=6, loc="upper right", ncol=2)
ax.set_ylim(0, 5)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/f602_seven_model_tuning.png", dpi=150, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/f602_seven_model_tuning.png")
