#!/usr/bin/env python3
"""
Cross-architecture FTLE analysis.
Compares volume dynamics, dose-sensitivity, and architectural strategies
across Mistral, Qwen, and Gemma.
"""

import json
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_results():
    results = {}
    for f in sorted(RESULTS_DIR.glob("ftle_zones_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for model_key in data:
            results[model_key] = data[model_key]
            results[model_key]["_file"] = str(f)
    return results


def analyze_volume_dynamics(data, dose="0"):
    """Analyze volume (det J) trajectory through depth."""
    spec = data["doses"][dose]["ftle_spectrum"]
    layers = sorted(spec.keys(), key=int)

    volume_trajectory = []
    for l in layers:
        entry = spec[l]
        volume_trajectory.append({
            "layer": int(l),
            "mean_ftle": entry["mean_ftle"],
            "max_ftle": entry["max_ftle"],
            "n_expanding": entry["n_expanding"],
            "n_contracting": entry["n_contracting"],
        })
    return volume_trajectory


def find_transition_points(trajectory):
    """Find where volume dynamics change qualitatively."""
    points = {}

    # First expanding layer
    for t in trajectory:
        if t["n_expanding"] > 0:
            points["first_expanding"] = t["layer"]
            break

    # Volume-preserving crossing (mean_ftle crosses zero)
    for i in range(len(trajectory) - 1):
        if trajectory[i]["mean_ftle"] * trajectory[i+1]["mean_ftle"] < 0:
            points.setdefault("zero_crossings", []).append(
                (trajectory[i]["layer"], trajectory[i+1]["layer"]))

    # Max volume expansion
    max_exp = max(trajectory, key=lambda t: t["mean_ftle"])
    points["max_expansion_layer"] = max_exp["layer"]
    points["max_expansion_value"] = max_exp["mean_ftle"]

    # Max volume contraction
    max_con = min(trajectory, key=lambda t: t["mean_ftle"])
    points["max_contraction_layer"] = max_con["layer"]
    points["max_contraction_value"] = max_con["mean_ftle"]

    # Majority expanding (>32/64)
    for t in trajectory:
        if t["n_expanding"] > 32:
            points.setdefault("majority_expanding_start", t["layer"])

    # Total collapse (0/64 expanding)
    for t in trajectory:
        if t["n_expanding"] == 0 and t["layer"] > trajectory[0]["layer"] + 2:
            points.setdefault("total_collapse", []).append(t["layer"])

    return points


def dose_sensitivity(data, base_dose="0", test_dose="1"):
    """Compute dose sensitivity at each layer."""
    d0 = data["doses"][base_dose]["ftle_spectrum"]
    d1 = data["doses"][test_dose]["ftle_spectrum"]
    layers = sorted(d0.keys(), key=int)

    sensitivity = []
    for l in layers:
        if l in d1:
            delta_n = d1[l]["n_expanding"] - d0[l]["n_expanding"]
            delta_mean = d1[l]["mean_ftle"] - d0[l]["mean_ftle"]
            sensitivity.append({
                "layer": int(l),
                "delta_n_expanding": delta_n,
                "delta_mean_ftle": delta_mean,
            })
    return sensitivity


def classify_architecture(trajectory, points):
    """Classify the dynamical architecture type."""
    first_half = trajectory[:len(trajectory)//2]
    second_half = trajectory[len(trajectory)//2:]

    first_mean = np.mean([t["mean_ftle"] for t in first_half])
    second_mean = np.mean([t["mean_ftle"] for t in second_half])

    if first_mean < -0.1 and second_mean > first_mean:
        return "contract-expand (Mistral-type)"
    elif first_mean > 0.1 and second_mean < first_mean:
        return "expand-contract (Qwen-type)"
    elif abs(first_mean) < 0.1 and abs(second_mean) < 0.1:
        return "volume-preserving (symplectic)"
    elif abs(first_mean - second_mean) < 0.1:
        return "equalized (Gemma-type?)"
    else:
        return f"unclassified (first={first_mean:.3f}, second={second_mean:.3f})"


def main():
    results = load_results()

    print(f"{'='*70}")
    print(f"  CROSS-ARCHITECTURE FTLE ANALYSIS")
    print(f"  Models: {', '.join(results.keys())}")
    print(f"{'='*70}")

    for model_key, data in results.items():
        n_layers = data["n_layers"]
        print(f"\n{'#'*60}")
        print(f"  {model_key.upper()} ({data['model']}, {n_layers} layers)")
        print(f"{'#'*60}")

        # Volume dynamics at dose 0
        traj = analyze_volume_dynamics(data, "0")
        points = find_transition_points(traj)
        arch_class = classify_architecture(traj, points)

        print(f"\n  Architecture class: {arch_class}")
        print(f"\n  Transition points (dose 0):")
        for k, v in points.items():
            print(f"    {k}: {v}")

        # Volume trajectory
        print(f"\n  Volume trajectory (dose 0):")
        for t in traj:
            bar_len = int(abs(t["mean_ftle"]) * 20)
            if t["mean_ftle"] >= 0:
                bar = "+" * min(bar_len, 40)
                print(f"    L{t['layer']:3d} [{t['n_expanding']:2d}/64] {t['mean_ftle']:+.3f} |{'':>20s}{bar}")
            else:
                bar = "-" * min(bar_len, 40)
                print(f"    L{t['layer']:3d} [{t['n_expanding']:2d}/64] {t['mean_ftle']:+.3f} |{bar:>20s}")

        # Dose sensitivity
        if "1" in data["doses"]:
            sens = dose_sensitivity(data, "0", "1")
            print(f"\n  Dose sensitivity (dose 1 - dose 0):")
            max_sens = max(sens, key=lambda s: abs(s["delta_n_expanding"]))
            print(f"    Peak sensitivity: L{max_sens['layer']} "
                  f"(Δn={max_sens['delta_n_expanding']:+d}, "
                  f"Δmean={max_sens['delta_mean_ftle']:+.3f})")
            for s in sens:
                if abs(s["delta_n_expanding"]) > 5 or abs(s["delta_mean_ftle"]) > 0.2:
                    print(f"    L{s['layer']:3d}: Δn_expanding={s['delta_n_expanding']:+3d}, "
                          f"Δmean_ftle={s['delta_mean_ftle']:+.4f}")

        # Volume-preserving test (sum of mean_ftle ~ 0?)
        total_volume = sum(t["mean_ftle"] for t in traj)
        print(f"\n  Total volume change (sum of mean_ftle): {total_volume:+.3f}")
        print(f"  Volume-preserving? {'YES' if abs(total_volume) < 1.0 else 'NO'}")

    # Cross-architecture comparison
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print(f"  CROSS-ARCHITECTURE COMPARISON")
        print(f"{'='*70}")

        for model_key, data in results.items():
            traj = analyze_volume_dynamics(data, "0")
            points = find_transition_points(traj)
            arch_class = classify_architecture(traj, points)

            if "1" in data["doses"]:
                sens = dose_sensitivity(data, "0", "1")
                max_sens = max(sens, key=lambda s: abs(s["delta_n_expanding"]))
                peak_layer = max_sens["layer"]
                peak_delta = max_sens["delta_n_expanding"]
            else:
                peak_layer = "N/A"
                peak_delta = "N/A"

            print(f"\n  {model_key.upper()}:")
            print(f"    Class: {arch_class}")
            print(f"    Max expansion: L{points['max_expansion_layer']} ({points['max_expansion_value']:+.3f})")
            print(f"    Max contraction: L{points['max_contraction_layer']} ({points['max_contraction_value']:+.3f})")
            print(f"    Peak dose sensitivity: L{peak_layer} (Δn={peak_delta})")
            total = sum(t["mean_ftle"] for t in traj)
            print(f"    Total volume: {total:+.3f}")


if __name__ == "__main__":
    main()
