#!/usr/bin/env python3
"""Analyze pre-registered CCS dose-trajectory experiment results.

Computes all pre-registered metrics from prereg_dose_trajectory.md:
  H1-C: Frobenius trajectory by species
  H2-C: Zone Selectivity Index (ZSI)
  H3-P: Dose-trajectory shape comparison
  H4-P: Per-Sigma Gradient (PSG) persistence
  H5-P: Mistral interpolation test
  H6-P: Within-context decay

Also runs spec_curve.py (192 specifications) for transparency.

Usage:
  python3 prereg_analyze.py spectral-demon/results/prereg/
  python3 prereg_analyze.py spectral-demon/results/prereg/ --plot
"""

import json, math, sys, os, argparse
import statistics as stats
from pathlib import Path
from collections import defaultdict

def effective_rank(sigmas):
    total = sum(s**2 for s in sigmas)
    if total == 0: return 0.0
    probs = [(s**2 / total) for s in sigmas if s > 0]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return math.exp(entropy)


def spectral_concentration(sigmas):
    total = sum(s**2 for s in sigmas)
    if total == 0: return 0.0
    return sigmas[0]**2 / total


def frobenius_change(d0_layers, dx_layers):
    changes = []
    for l in range(min(len(d0_layers), len(dx_layers))):
        f0 = d0_layers[l]["centered"]["frobenius_sq"]
        fx = dx_layers[l]["centered"]["frobenius_sq"]
        if f0 > 0:
            changes.append((fx - f0) / f0 * 100)
    return changes


def zone_selectivity_index(d0_layers, dx_layers):
    sigma2_changes = []
    for l in range(min(len(d0_layers), len(dx_layers))):
        s0 = d0_layers[l]["centered"]["top_singular"]
        sx = dx_layers[l]["centered"]["top_singular"]
        if len(s0) >= 2 and len(sx) >= 2 and s0[1] > 0:
            sigma2_changes.append((sx[1] - s0[1]) / s0[1] * 100)

    if not sigma2_changes or abs(stats.mean(sigma2_changes)) < 5:
        return float('nan'), sigma2_changes
    zsi = stats.stdev(sigma2_changes) / abs(stats.mean(sigma2_changes))
    return zsi, sigma2_changes


def per_sigma_gradient(d0_layers, dx_layers, n_sigmas=10):
    mean_changes = []
    for i in range(n_sigmas):
        layer_changes = []
        for l in range(min(len(d0_layers), len(dx_layers))):
            s0 = d0_layers[l]["centered"]["top_singular"]
            sx = dx_layers[l]["centered"]["top_singular"]
            if len(s0) > i and len(sx) > i and s0[i] > 0:
                layer_changes.append((sx[i] - s0[i]) / s0[i] * 100)
        if layer_changes:
            mean_changes.append(stats.mean(layer_changes))
        else:
            mean_changes.append(0)

    if len(mean_changes) < 3:
        return 0.0, mean_changes

    ranks = list(range(1, len(mean_changes) + 1))
    n = len(ranks)
    rank_x = [sorted(ranks).index(v) + 1 for v in ranks]
    rank_y = [sorted(mean_changes).index(v) + 1 for v in mean_changes]
    d2 = sum((a-b)**2 for a,b in zip(rank_x, rank_y))
    spearman = 1 - 6*d2 / (n*(n**2-1))
    return spearman, mean_changes


def load_phase1(result_dir, model_name):
    path = result_dir / f"prereg_phase1_{model_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_phase2(result_dir, model_name):
    path = result_dir / f"prereg_phase2_{model_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_h1(results_by_model):
    print("\n" + "="*70)
    print("  H1-C: Frobenius Trajectory by Species (CONFIRMATORY)")
    print("="*70)

    for model_name, data in sorted(results_by_model.items()):
        species = data["species"]
        runs = data.get("runs", [])
        if not runs:
            continue

        print(f"\n  {model_name.upper()} ({species}, GQA {data['gqa']}):")
        print(f"  {'Dose':>6} {'Mean dF/F%':>12} {'Std':>8} {'N_runs':>8}")

        for dose_name in ["D0", "D2", "D3", "D5", "D8"]:
            dose_frobs = []
            for run in runs:
                d0_entry = next((d for d in run["doses"] if d["dose"] == "D0"), None)
                dx_entry = next((d for d in run["doses"] if d["dose"] == dose_name), None)
                if d0_entry and dx_entry and dose_name != "D0":
                    changes = frobenius_change(d0_entry["per_layer"], dx_entry["per_layer"])
                    dose_frobs.append(stats.mean(changes))

            if dose_frobs:
                mean = stats.mean(dose_frobs)
                std = stats.stdev(dose_frobs) if len(dose_frobs) > 1 else 0
                print(f"  {dose_name:>6} {mean:+11.1f}% {std:7.1f}% {len(dose_frobs):>8}")


def analyze_h2(results_by_model):
    print("\n" + "="*70)
    print("  H2-C: Zone Selectivity Index (CONFIRMATORY)")
    print("="*70)

    for model_name, data in sorted(results_by_model.items()):
        species = data["species"]
        runs = data.get("runs", [])
        if not runs:
            continue

        zsis = []
        for run in runs:
            d0_entry = next((d for d in run["doses"] if d["dose"] == "D0"), None)
            d5_entry = next((d for d in run["doses"] if d["dose"] == "D5"), None)
            if not d5_entry:
                d5_entry = next((d for d in run["doses"] if d["dose"] == "D2"), None)
            if d0_entry and d5_entry:
                zsi, _ = zone_selectivity_index(d0_entry["per_layer"], d5_entry["per_layer"])
                if not math.isnan(zsi):
                    zsis.append(zsi)

        if zsis:
            mean_zsi = stats.mean(zsis)
            threshold = 0.4 if "relay" in species else 0.5
            direction = "<" if "relay" in species or species == "tunnel" else ">"
            passes = mean_zsi < threshold if direction == "<" else mean_zsi > threshold
            verdict = "PASS" if passes else "FAIL"
            print(f"  {model_name:>8} ({species:>6}): ZSI = {mean_zsi:.3f} "
                  f"(predicted {direction} {threshold}) [{verdict}]")


def analyze_h4(results_by_model):
    print("\n" + "="*70)
    print("  H4-P: Per-Sigma Gradient Persistence (PROSPECTIVE)")
    print("="*70)

    for model_name, data in sorted(results_by_model.items()):
        species = data["species"]
        runs = data.get("runs", [])
        if not runs:
            continue

        print(f"\n  {model_name.upper()} ({species}):")
        for dose_name in ["D2", "D3", "D5", "D8"]:
            psgs = []
            for run in runs:
                d0_entry = next((d for d in run["doses"] if d["dose"] == "D0"), None)
                dx_entry = next((d for d in run["doses"] if d["dose"] == dose_name), None)
                if d0_entry and dx_entry:
                    psg, _ = per_sigma_gradient(d0_entry["per_layer"], dx_entry["per_layer"])
                    psgs.append(psg)
            if psgs:
                mean_psg = stats.mean(psgs)
                verdict = "PASS" if mean_psg > 0.5 else "FAIL"
                print(f"    {dose_name}: PSG = {mean_psg:.3f} (predicted > 0.5) [{verdict}]")


def analyze_h5(results_by_model):
    print("\n" + "="*70)
    print("  H5-P: Mistral Interpolation (PROSPECTIVE)")
    print("="*70)

    relay_frobs, sorter_frobs, mistral_frob = [], [], None

    for model_name, data in sorted(results_by_model.items()):
        runs = data.get("runs", [])
        if not runs:
            continue

        d5_frobs = []
        for run in runs:
            d0_entry = next((d for d in run["doses"] if d["dose"] == "D0"), None)
            d5_entry = next((d for d in run["doses"] if d["dose"] == "D5"), None)
            if not d5_entry:
                d5_entry = next((d for d in run["doses"] if d["dose"] == "D2"), None)
            if d0_entry and d5_entry:
                changes = frobenius_change(d0_entry["per_layer"], d5_entry["per_layer"])
                d5_frobs.append(stats.mean(changes))

        if not d5_frobs:
            continue

        mean_frob = stats.mean(d5_frobs)
        species = data["species"]
        if "relay" in species and model_name != "mistral":
            relay_frobs.append(mean_frob)
        elif species == "sorter":
            sorter_frobs.append(mean_frob)
        elif model_name == "mistral":
            mistral_frob = mean_frob

    if relay_frobs and sorter_frobs:
        relay_mean = stats.mean(relay_frobs)
        sorter_mean = stats.mean(sorter_frobs)
        print(f"  Relay mean dF/F: {relay_mean:+.1f}%")
        print(f"  Sorter mean dF/F: {sorter_mean:+.1f}%")
        if mistral_frob is not None:
            lo, hi = min(relay_mean, sorter_mean), max(relay_mean, sorter_mean)
            interpolates = lo <= mistral_frob <= hi
            verdict = "PASS" if interpolates else "FAIL"
            print(f"  Mistral dF/F: {mistral_frob:+.1f}% (range: {lo:.1f} to {hi:.1f}) [{verdict}]")
        else:
            print(f"  Mistral: NOT YET RUN")


def analyze_h6(result_dir, model_names):
    print("\n" + "="*70)
    print("  H6-P: Within-Context Decay (PROSPECTIVE)")
    print("="*70)

    for model_name in model_names:
        data = load_phase2(result_dir, model_name)
        if not data:
            continue

        windows = data.get("windows", [])
        if len(windows) < 2:
            continue

        preamble_er = stats.mean([
            l.get("effective_rank", 0) for l in windows[0]["per_layer"]
        ])
        print(f"\n  {model_name.upper()} ({data['species']}):")
        print(f"  {'Position':>12} {'Mean ER':>10} {'vs Preamble':>12}")

        for w in windows:
            pos = w["position"]
            rel = w["relative_to_preamble"]
            mean_er = stats.mean([l.get("effective_rank", 0) for l in w["per_layer"]])
            ratio = mean_er / preamble_er * 100 if preamble_er > 0 else 0
            print(f"  {pos:>12} {mean_er:9.3f} {ratio:10.1f}%")

        last_er = stats.mean([l.get("effective_rank", 0) for l in windows[-1]["per_layer"]])
        decay_pct = (1 - last_er / preamble_er) * 100 if preamble_er > 0 else 0
        verdict = "PASS" if decay_pct > 50 else "FAIL"
        print(f"  Decay at last window: {decay_pct:.1f}% (predicted > 50%) [{verdict}]")


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-registered experiment results")
    parser.add_argument("result_dir", help="Directory containing prereg results")
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Directory not found: {result_dir}")
        return

    model_names = ["qwen", "gemma", "llama", "phi", "gpt2", "mistral"]
    results_by_model = {}

    for model_name in model_names:
        data = load_phase1(result_dir, model_name)
        if data:
            results_by_model[model_name] = data
            print(f"  Loaded {model_name}: {data['species']}, GQA {data['gqa']}, "
                  f"{len(data.get('runs', []))} runs")

    if not results_by_model:
        print("No Phase 1 results found.")
        return

    print(f"\nLoaded {len(results_by_model)} models.")

    analyze_h1(results_by_model)
    analyze_h2(results_by_model)
    analyze_h4(results_by_model)
    analyze_h5(results_by_model)
    analyze_h6(result_dir, model_names)

    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print("  See individual hypothesis sections for PASS/FAIL verdicts.")
    print("  Run spec_curve.py on each model's data for 192-specification transparency.")


if __name__ == "__main__":
    main()
