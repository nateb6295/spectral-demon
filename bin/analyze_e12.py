#!/usr/bin/env python3
"""Analyze E12 covariance disruption results.

Reads e12_disruption_results.json and produces:
1. Per-model, per-dose causal disruption summary (coupling before/after, recovery ratio)
2. E12 vs E13 baseline coupling comparison (7-probe vs 5-probe discrepancy)
3. V₂ survival under shuffle (should be near-zero — no-op confirmation)
4. Per-layer causal disruption heatmap data
"""

import json
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_e12(path=None):
    if path is None:
        candidates = [
            RESULTS_DIR / "e12_disruption_results.json",
            Path("/workspace/e12_disruption_results.json"),
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
    if path is None:
        print("E12 results not found. Pass path as argument.")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


def load_e13():
    p = RESULTS_DIR / "e13_holonomy_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def analyze_causal(data):
    print("\n" + "=" * 70)
    print("CAUSAL DISRUPTION ANALYSIS")
    print("=" * 70)

    for model_name, model_data in data.items():
        relay_layers = model_data["relay_layers"]
        print(f"\n{'─' * 60}")
        print(f"  {model_name}")
        print(f"  Relay layers: {relay_layers}")
        print(f"{'─' * 60}")

        print(f"  {'Dose':>5}  {'Base coupling':>14}  {'Post-disrupt':>13}  {'Recovery':>10}  {'σ₁ Δ%':>7}  {'Sparsity Δ%':>12}")

        for dose_str, dose_data in sorted(model_data["results_by_dose"].items(), key=lambda x: int(x[0])):
            causal = dose_data.get("causal_disruption", {})
            if not causal:
                continue

            baseline_r = causal.get("baseline_coupling", 0)
            disrupted_r = causal.get("disrupted_coupling", 0)
            recovery = causal.get("recovery_ratio", 0)

            base_sigma1 = np.mean([
                dose_data["baseline"]["per_probe"][0].get(str(l), {}).get("sigma1", 0)
                for l in relay_layers
                if str(l) in dose_data["baseline"]["per_probe"][0]
            ])
            disrupt_sigma1 = causal.get("sigma1_disrupted", base_sigma1)

            base_sparsity = np.mean([
                dose_data["baseline"]["per_probe"][0].get(str(l), {}).get("sparsity", 0)
                for l in relay_layers
                if str(l) in dose_data["baseline"]["per_probe"][0]
            ])
            disrupt_sparsity = causal.get("sparsity_disrupted", base_sparsity)

            sigma1_pct = ((disrupt_sigma1 - base_sigma1) / base_sigma1 * 100) if base_sigma1 > 0 else 0
            sparsity_pct = ((disrupt_sparsity - base_sparsity) / base_sparsity * 100) if base_sparsity > 0 else 0

            print(f"  D{dose_str:>4}  {baseline_r:>14.3f}  {disrupted_r:>13.3f}  {recovery:>10.3f}  {sigma1_pct:>6.1f}%  {sparsity_pct:>11.1f}%")


def analyze_shuffle_noop(data):
    print("\n" + "=" * 70)
    print("SHUFFLE NO-OP CONFIRMATION")
    print("=" * 70)

    for model_name, model_data in data.items():
        print(f"\n  {model_name}:")
        relay_layers = model_data["relay_layers"]

        for dose_str, dose_data in sorted(model_data["results_by_dose"].items(), key=lambda x: int(x[0])):
            base_coupling = dose_data["baseline"]["coupling"]
            shuf_coupling = dose_data["shuffle_disruption"]["coupling"]

            base_r_vals = [base_coupling.get(str(l), {}).get("r", 0) for l in relay_layers]
            shuf_r_vals = [shuf_coupling.get(str(l), {}).get("r", 0) for l in relay_layers]

            diff = np.mean(np.abs(np.array(base_r_vals) - np.array(shuf_r_vals)))
            print(f"    D{dose_str}: mean |baseline - shuffle| = {diff:.6f}  {'✓ NO-OP' if diff < 0.001 else '⚠ DIFFERS'}")


def compare_e12_e13(e12_data, e13_data):
    if e13_data is None:
        print("\n  E13 data not found, skipping comparison.")
        return

    print("\n" + "=" * 70)
    print("E12 vs E13 BASELINE COUPLING COMPARISON")
    print("  E12: 7 probes, E13: 5 probes")
    print("=" * 70)

    model_map = {}
    for name in e12_data:
        short = name.split("/")[-1].split("-")[0].lower()
        model_map[short] = name

    for e13_model, e13_results in e13_data.items():
        if not isinstance(e13_results, dict) or "doses" not in e13_results:
            continue

        print(f"\n  {e13_model}:")
        print(f"  {'Dose':>5}  {'E13 coupling':>13}  {'E12 coupling':>13}  {'Diff':>8}")

        e12_model = None
        for short, full in model_map.items():
            if short in e13_model.lower():
                e12_model = full
                break

        if e12_model is None:
            print(f"    No E12 match found")
            continue

        e12_results = e12_data[e12_model]

        for dose_key in sorted(e13_results.get("doses", {}).keys()):
            e13_dose = e13_results["doses"][dose_key]
            e13_coupling = e13_dose.get("relay_coupling_mean", e13_dose.get("coupling_mean", None))

            dose_num = dose_key.replace("D", "").replace("d", "")
            e12_dose = e12_results["results_by_dose"].get(dose_num, {})
            if e12_dose:
                relay_layers = e12_results["relay_layers"]
                e12_coupling_vals = [
                    e12_dose["baseline"]["coupling"].get(str(l), {}).get("r", 0)
                    for l in relay_layers
                ]
                e12_coupling = np.mean(e12_coupling_vals)
            else:
                e12_coupling = None

            if e13_coupling is not None and e12_coupling is not None:
                diff = e12_coupling - e13_coupling
                print(f"  D{dose_num:>4}  {e13_coupling:>13.3f}  {e12_coupling:>13.3f}  {diff:>+8.3f}")
            elif e13_coupling is not None:
                print(f"  D{dose_num:>4}  {e13_coupling:>13.3f}  {'N/A':>13}")


def v2_survival_summary(data):
    print("\n" + "=" * 70)
    print("V₂ SURVIVAL UNDER SHUFFLE")
    print("=" * 70)

    for model_name, model_data in data.items():
        print(f"\n  {model_name}:")
        relay_layers = model_data["relay_layers"]

        for dose_str, dose_data in sorted(model_data["results_by_dose"].items(), key=lambda x: int(x[0])):
            v2_data = dose_data.get("v2_survival", {})
            survivals = [
                v2_data.get(str(l), {}).get("baseline_vs_shuffle", 0)
                for l in relay_layers
                if str(l) in v2_data
            ]
            if survivals:
                print(f"    D{dose_str}: mean V₂ survival = {np.mean(survivals):.3f} (range: {min(survivals):.3f} to {max(survivals):.3f})")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    e12 = load_e12(path)
    e13 = load_e13()

    analyze_causal(e12)
    analyze_shuffle_noop(e12)
    v2_survival_summary(e12)
    compare_e12_e13(e12, e13)

    print("\n" + "=" * 70)
    print("DONE")
