#!/usr/bin/env python3
"""Plot F600 replication results — relay Q1 prompt-lability."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

with open(f"{RESULTS_DIR}/replicate_crossing.json") as f:
    data = json.load(f)

with open(f"{RESULTS_DIR}/tuning_knob_pythia.json") as f:
    pythia = json.load(f)
with open(f"{RESULTS_DIR}/tuning_knob_gpt2.json") as f:
    gpt2 = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("F600: Relay Q1 is Prompt-Labile at Zero Boundary", fontsize=14, fontweight="bold")

# Panel 1: Q1 scatter — each variant as a point, tunnels as reference lines
ax = axes[0]
ax.set_title("Q1 Under Mild Framing\n(4 prompt variants each)", fontsize=10)

# Relay data points
tl_q1 = [v["q1"] for v in data["TinyLlama"]["variants"]]
mi_q1 = [v["q1"] for v in data["Mistral-7B"]["variants"]]

ax.scatter([1]*4, tl_q1, c="#e74c3c", s=80, zorder=5, label="TinyLlama (GQA 8:1)")
ax.scatter([2]*4, mi_q1, c="#3498db", s=80, zorder=5, label="Mistral (GQA 4:1)")

# Mean + std bars
ax.errorbar(1, data["TinyLlama"]["mean_q1"], yerr=data["TinyLlama"]["std_q1"],
            fmt="D", c="#e74c3c", markersize=10, capsize=5, zorder=6, linewidth=2)
ax.errorbar(2, data["Mistral-7B"]["mean_q1"], yerr=data["Mistral-7B"]["std_q1"],
            fmt="D", c="#3498db", markersize=10, capsize=5, zorder=6, linewidth=2)

# Tunnel reference floors
pythia_mild = None
gpt2_mild = None
for fl in pythia.get("framing_levels", []):
    if fl["level"] == "mild_aware":
        pythia_mild = fl["q1_decomposed"]["q1_total"]
for fl in gpt2.get("framing_levels", []):
    if fl["level"] == "mild_aware":
        gpt2_mild = fl["q1_decomposed"]["q1_total"]

if pythia_mild is not None:
    ax.axhline(pythia_mild, color="#2ecc71", linestyle="--", alpha=0.7, label=f"Pythia floor ({pythia_mild:+.3f})")
if gpt2_mild is not None:
    ax.axhline(gpt2_mild, color="#27ae60", linestyle=":", alpha=0.7, label=f"GPT-2 floor ({gpt2_mild:+.3f})")

ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["TinyLlama\n(relay)", "Mistral\n(relay)"])
ax.set_ylabel("Q1")
ax.legend(fontsize=7, loc="upper right")

# Panel 2: σ₁/σ₂ decomposition per variant
ax = axes[1]
ax.set_title("σ₁ vs σ₂ Response\n(mild framing variants)", fontsize=10)

tl_s1 = [v["q1_s1_mean"] for v in data["TinyLlama"]["variants"]]
tl_s2 = [v["q1_s2_mean"] for v in data["TinyLlama"]["variants"]]
mi_s1 = [v["q1_s1_mean"] for v in data["Mistral-7B"]["variants"]]
mi_s2 = [v["q1_s2_mean"] for v in data["Mistral-7B"]["variants"]]

ax.scatter(tl_s1, tl_s2, c="#e74c3c", s=80, marker="o", label="TinyLlama", zorder=5)
ax.scatter(mi_s1, mi_s2, c="#3498db", s=80, marker="s", label="Mistral", zorder=5)

for i, v in enumerate(data["TinyLlama"]["variants"]):
    ax.annotate(f"v{i+1}", (tl_s1[i], tl_s2[i]), fontsize=7, ha="left", va="bottom")
for i, v in enumerate(data["Mistral-7B"]["variants"]):
    ax.annotate(f"v{i+1}", (mi_s1[i], mi_s2[i]), fontsize=7, ha="left", va="bottom")

ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.axvline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_xlabel("Mean Δσ₁ (Q1 region)")
ax.set_ylabel("Mean Δσ₂ (Q1 region)")
ax.legend(fontsize=8)

# Panel 3: Distance from zero → prompt sensitivity
ax = axes[2]
ax.set_title("Species = Distance from Q1 Boundary\n(proximity gate mechanism)", fontsize=10)

models = {
    "GPT-2": {"floor": gpt2_mild or 0.006, "std": 0.0, "species": "tunnel", "color": "#2ecc71"},
    "Pythia": {"floor": pythia_mild or 0.042, "std": 0.0, "species": "tunnel", "color": "#27ae60"},
    "TinyLlama": {"floor": data["TinyLlama"]["mean_q1"], "std": data["TinyLlama"]["std_q1"],
                  "species": "relay", "color": "#e74c3c"},
    "Mistral": {"floor": data["Mistral-7B"]["mean_q1"], "std": data["Mistral-7B"]["std_q1"],
                "species": "relay", "color": "#3498db"},
}

y_positions = {"GPT-2": 1, "Pythia": 2, "TinyLlama": 3, "Mistral": 4}
for name, info in models.items():
    y = y_positions[name]
    ax.barh(y, info["floor"], height=0.6, color=info["color"], alpha=0.7,
            xerr=info["std"], capsize=3, label=f"{name} ({info['species']})")

ax.axvline(0, color="red", linestyle="-", linewidth=2, alpha=0.8, label="Q1 = 0 boundary")
ax.fill_betweenx([0.5, 4.5], -0.05, 0.05, color="red", alpha=0.08, label="Labile zone")
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(["GPT-2\n(12L, MHA)", "Pythia\n(32L, MHA)", "TinyLlama\n(22L, GQA 8:1)",
                     "Mistral\n(32L, GQA 4:1)"])
ax.set_xlabel("Mean Q1 under mild framing")
ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/f600_replication.png", dpi=150, bbox_inches="tight")
print(f"Saved {RESULTS_DIR}/f600_replication.png")
