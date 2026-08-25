"""
Paper 11 figures — Species × Tuning Cross.
Generates publication-ready figures from the 5-model cross experiment data.

Usage:
  python3 figures_species_cross.py                    # all figures
  python3 figures_species_cross.py --fig training-gradient
  python3 figures_species_cross.py --fig layer-profiles
  python3 figures_species_cross.py --fig d0-baselines
  python3 figures_species_cross.py --fig convergence-decomposition
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "species_tuning"
FIG_DIR = Path(__file__).parent / "figures"
DOSE_ORDER = ["D0", "D2", "D3", "D5"]

MODELS = {
    "qwen_base":      {"label": "Qwen base",      "species": "relay",  "tuning": "base",     "color": "#c04840", "marker": "o", "ls": ":"},
    "qwen_sft":       {"label": "Qwen SFT",       "species": "relay",  "tuning": "SFT",      "color": "#e06c5a", "marker": "s", "ls": "--"},
    "qwen_instruct":  {"label": "Qwen instruct",  "species": "relay",  "tuning": "instruct", "color": "#e06c5a", "marker": "D", "ls": "-"},
    "gemma2_base":    {"label": "Gemma base",      "species": "sorter", "tuning": "base",     "color": "#1a7a80", "marker": "o", "ls": ":"},
    "gemma2_instruct":{"label": "Gemma instruct",  "species": "sorter", "tuning": "instruct", "color": "#2d9da6", "marker": "D", "ls": "-"},
}


def load_all():
    models = {}
    for name in MODELS:
        path = RESULTS_DIR / f"species_tuning_{name}.json"
        if path.exists():
            with open(path) as f:
                models[name] = json.load(f)
    return models


def get_dose_metrics(data):
    results = {}
    for dose_idx, dose_name in enumerate(DOSE_ORDER):
        s2_vals = []
        for run in data["runs"]:
            dose_data = run["doses"][dose_idx]
            ccs_s2 = np.mean([l["centered"]["top_singular"][1] for l in dose_data["ccs_per_layer"]])
            cal_s2 = np.mean([l["centered"]["top_singular"][1] for l in dose_data["cal_per_layer"]])
            if dose_name == "D0":
                s2_vals.append(ccs_s2)
            else:
                s2_vals.append(((ccs_s2 - cal_s2) / cal_s2) * 100 if cal_s2 != 0 else 0)
        results[dose_name] = {"mean": np.mean(s2_vals), "std": np.std(s2_vals)}
    return results


def get_layer_profile(data, dose_idx=1):
    n_layers = len(data["runs"][0]["doses"][0]["ccs_per_layer"])
    profile = []
    for layer_idx in range(n_layers):
        ccs_vals, cal_vals = [], []
        for run in data["runs"]:
            dose_data = run["doses"][dose_idx]
            ccs_vals.append(dose_data["ccs_per_layer"][layer_idx]["centered"]["top_singular"][1])
            cal_vals.append(dose_data["cal_per_layer"][layer_idx]["centered"]["top_singular"][1])
        ccs_mean, cal_mean = np.mean(ccs_vals), np.mean(cal_vals)
        pct = ((ccs_mean - cal_mean) / cal_mean) * 100 if cal_mean != 0 else 0
        profile.append(pct)
    return np.array(profile)


def fig_training_gradient(all_data):
    """Fig: Training gradient — base→SFT→instruct dose-response curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    doses = ["D2", "D3", "D5"]
    x = np.arange(len(doses))

    for name in ["qwen_base", "qwen_sft", "qwen_instruct"]:
        if name not in all_data:
            continue
        m = get_dose_metrics(all_data[name])
        info = MODELS[name]
        vals = [m[d]["mean"] for d in doses]
        errs = [m[d]["std"] for d in doses]
        ax1.errorbar(x, vals, yerr=errs, label=info["label"],
                     color=info["color"], marker=info["marker"],
                     linestyle=info["ls"], linewidth=1.8, markersize=7, capsize=3)

    ax1.axhline(0, color="#888", linewidth=0.5, linestyle="-")
    ax1.set_xticks(x)
    ax1.set_xticklabels(doses)
    ax1.set_ylabel("Calibration-corrected σ₂ change (%)")
    ax1.set_title("Relay (Qwen 7B)")
    ax1.legend(fontsize=8, loc="lower left")

    for name in ["gemma2_base", "gemma2_instruct"]:
        if name not in all_data:
            continue
        m = get_dose_metrics(all_data[name])
        info = MODELS[name]
        vals = [m[d]["mean"] for d in doses]
        errs = [m[d]["std"] for d in doses]
        ax2.errorbar(x, vals, yerr=errs, label=info["label"],
                     color=info["color"], marker=info["marker"],
                     linestyle=info["ls"], linewidth=1.8, markersize=7, capsize=3)

    ax2.axhline(0, color="#888", linewidth=0.5, linestyle="-")
    ax2.set_xticks(x)
    ax2.set_xticklabels(doses)
    ax2.set_title("Sorter (Gemma 2B)")
    ax2.legend(fontsize=8, loc="lower left")

    fig.suptitle("Training gradient: CCS σ₂ response by training stage", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, "training_gradient"


def fig_layer_profiles(all_data):
    """Fig: Per-layer σ₂ suppression profiles for all instruct models."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    panels = [
        ("qwen_base", "qwen_sft", "Qwen: base → SFT"),
        ("qwen_sft", "qwen_instruct", "Qwen: SFT → instruct"),
        ("gemma2_base", "gemma2_instruct", "Gemma: base → instruct"),
        ("qwen_instruct", "gemma2_instruct", "Cross-species: relay vs sorter (instruct)"),
    ]

    for ax, (name_a, name_b, title) in zip(axes.flat, panels):
        if name_a in all_data:
            profile_a = get_layer_profile(all_data[name_a], dose_idx=3)  # D5
            info_a = MODELS[name_a]
            depth_a = np.linspace(0, 1, len(profile_a))
            ax.fill_between(depth_a, 0, profile_a, alpha=0.15, color=info_a["color"])
            ax.plot(depth_a, profile_a, color=info_a["color"], linewidth=1.5,
                    linestyle=info_a["ls"], label=info_a["label"])

        if name_b in all_data:
            profile_b = get_layer_profile(all_data[name_b], dose_idx=3)
            info_b = MODELS[name_b]
            depth_b = np.linspace(0, 1, len(profile_b))
            ax.fill_between(depth_b, 0, profile_b, alpha=0.15, color=info_b["color"])
            ax.plot(depth_b, profile_b, color=info_b["color"], linewidth=1.5,
                    linestyle=info_b["ls"], label=info_b["label"])

        ax.axhline(0, color="#888", linewidth=0.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Proportional depth", fontsize=8)
        ax.set_ylabel("σ₂ change (%)", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=7)

    fig.suptitle("Per-layer σ₂ suppression profiles at D5", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, "layer_profiles"


def fig_d0_baselines(all_data):
    """Fig: D0 σ₂ baselines — what training does BEFORE CCS."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    names = list(MODELS.keys())
    x_pos = np.arange(len(names))
    d0_vals = []

    for name in names:
        if name in all_data:
            m = get_dose_metrics(all_data[name])
            d0_vals.append(m["D0"]["mean"])
        else:
            d0_vals.append(0)

    colors = [MODELS[n]["color"] for n in names]
    labels = [MODELS[n]["label"] for n in names]
    bars = ax.bar(x_pos, d0_vals, color=colors, width=0.6, edgecolor="#333", linewidth=0.5)

    for bar, val in zip(bars, d0_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("D0 σ₂ (mean across layers)", fontsize=10)
    ax.set_title("Pre-CCS σ₂ baselines: what training installs", fontsize=11, fontweight="bold")

    ax.annotate("SFT amplifies\n(140→332)", xy=(1, 332), xytext=(1.8, 340),
                fontsize=8, ha="center", color="#666",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
    ax.annotate("RLHF constrains\n(332→134)", xy=(2, 134), xytext=(3, 160),
                fontsize=8, ha="center", color="#666",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

    fig.tight_layout()
    return fig, "d0_baselines"


def fig_convergence_decomposition(all_data):
    """Fig: Per-layer decomposition showing different mechanisms behind aggregate convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    if "qwen_instruct" in all_data:
        profile_q = get_layer_profile(all_data["qwen_instruct"], dose_idx=1)
        depth_q = np.arange(len(profile_q))
        pos_q = np.where(profile_q >= 0, profile_q, 0)
        neg_q = np.where(profile_q < 0, profile_q, 0)
        ax1.bar(depth_q, pos_q, color="#e06c5a", alpha=0.7, width=0.8)
        ax1.bar(depth_q, neg_q, color="#e06c5a", alpha=0.4, width=0.8)
        ax1.axhline(0, color="#888", linewidth=0.5)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("σ₂ change (%)")
        ax1.set_title(f"Qwen instruct (relay)\nAggregate: {np.mean(profile_q):.1f}%, CV={np.std(profile_q)/np.abs(np.mean(profile_q)):.2f}",
                      fontsize=9, fontweight="bold")

        mid_peak = np.argmin(profile_q[:20])
        ax1.annotate(f"L{mid_peak}\n{profile_q[mid_peak]:.1f}%",
                     xy=(mid_peak, profile_q[mid_peak]),
                     xytext=(mid_peak+3, profile_q[mid_peak]-3),
                     fontsize=7, ha="center",
                     arrowprops=dict(arrowstyle="->", color="#999", lw=0.6))

    if "gemma2_instruct" in all_data:
        profile_g = get_layer_profile(all_data["gemma2_instruct"], dose_idx=1)
        depth_g = np.arange(len(profile_g))
        pos_g = np.where(profile_g >= 0, profile_g, 0)
        neg_g = np.where(profile_g < 0, profile_g, 0)
        ax2.bar(depth_g, pos_g, color="#2d9da6", alpha=0.7, width=0.8)
        ax2.bar(depth_g, neg_g, color="#2d9da6", alpha=0.4, width=0.8)
        ax2.axhline(0, color="#888", linewidth=0.5)
        ax2.set_xlabel("Layer")
        ax2.set_title(f"Gemma instruct (sorter)\nAggregate: {np.mean(profile_g):.1f}%, CV={np.std(profile_g)/np.abs(np.mean(profile_g)):.2f}",
                      fontsize=9, fontweight="bold")

        mid_peak = np.argmin(profile_g[:16])
        ax2.annotate(f"L{mid_peak}\n{profile_g[mid_peak]:.1f}%",
                     xy=(mid_peak, profile_g[mid_peak]),
                     xytext=(mid_peak+3, profile_g[mid_peak]-3),
                     fontsize=7, ha="center",
                     arrowprops=dict(arrowstyle="->", color="#999", lw=0.6))

    fig.suptitle("Convergence decomposition: same aggregate, different topology\n"
                 "(D2 calibration-corrected σ₂, r=0.537)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig, "convergence_decomposition"


FIGURE_MAP = {
    "training-gradient": fig_training_gradient,
    "layer-profiles": fig_layer_profiles,
    "d0-baselines": fig_d0_baselines,
    "convergence-decomposition": fig_convergence_decomposition,
}


def main():
    parser = argparse.ArgumentParser(description="Paper 11 figure generator")
    parser.add_argument("--fig", type=str, default=None, help="Generate specific figure")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_data = load_all()
    print(f"Loaded {len(all_data)}/{len(MODELS)} models")

    if args.fig:
        figs_to_gen = {args.fig: FIGURE_MAP[args.fig]}
    else:
        figs_to_gen = FIGURE_MAP

    for name, func in figs_to_gen.items():
        print(f"Generating {name}...")
        fig, slug = func(all_data)
        out = FIG_DIR / f"paper11_{slug}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {out}")


if __name__ == "__main__":
    main()
