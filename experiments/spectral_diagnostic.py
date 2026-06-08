#!/usr/bin/env python3
"""
Three-state spectral diagnostic: quiescent / overdosed / hollowed.

Given a per-layer σ₂/σ₁ profile, classifies the system state by comparing
against the architecture's known baseline (dose 0) and attractor profiles.

From the Gemma thread exchange (2026-06-07):
  A. Quiescent: σ₁ stable, σ₂ low but patterned (matches baseline shape)
  B. Overdosed: σ₁ stable, σ₂ chaotic (ratio inverts, expression scrambles)
  C. Hollowed: σ₁ drops below baseline (coupling itself degraded)

Usage:
  python3 spectral_diagnostic.py                    # show all models, all doses
  python3 spectral_diagnostic.py --model gemma      # single model
  python3 spectral_diagnostic.py --dose 5           # single dose across models
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_all():
    models = {}
    for f in sorted(RESULTS_DIR.glob("dose_response_crossarch*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for key, mdata in data["models"].items():
            if "error" not in mdata:
                models[key] = mdata
    return models


def per_layer_profile(mdata, dose):
    for dd in mdata["results"].values():
        if dd["dose"] == dose:
            profiles = {}
            for run in dd["runs"]:
                for ls, geom in run["full_geometry"].items():
                    l = int(ls)
                    profiles.setdefault(l, []).append(geom["ratio"])
            return {l: np.mean(v) for l, v in sorted(profiles.items())}
    return {}


def profile_to_array(profile, n_layers):
    return np.array([profile.get(l, 0) for l in range(n_layers)])


def classify_state(profile_arr, baseline_arr):
    if len(profile_arr) == 0 or len(baseline_arr) == 0:
        return "unknown", {}

    baseline_mean = np.mean(baseline_arr)
    profile_mean = np.mean(profile_arr)

    shape_corr = np.corrcoef(profile_arr, baseline_arr)[0, 1] if np.std(baseline_arr) > 0 else 0
    ratio_shift = profile_mean / baseline_mean if baseline_mean > 0 else 0

    peak_layer_baseline = np.argmax(baseline_arr)
    peak_layer_profile = np.argmax(profile_arr)
    peak_migration = abs(peak_layer_profile - peak_layer_baseline)

    profile_cv = np.std(profile_arr) / np.mean(profile_arr) if np.mean(profile_arr) > 0 else 0
    baseline_cv = np.std(baseline_arr) / np.mean(baseline_arr) if np.mean(baseline_arr) > 0 else 0

    metrics = {
        "shape_correlation": round(shape_corr, 3),
        "ratio_shift": round(ratio_shift, 3),
        "peak_migration": int(peak_migration),
        "profile_cv": round(profile_cv, 3),
        "baseline_cv": round(baseline_cv, 3),
        "peak_layer": int(peak_layer_profile),
        "baseline_peak": int(peak_layer_baseline),
    }

    migration_is_organized = profile_cv < baseline_cv * 2.0

    if profile_mean < baseline_mean * 0.5:
        state = "HOLLOWED"
    elif shape_corr < 0.5 and not migration_is_organized:
        state = "OVERDOSED"
    elif shape_corr < 0.7 and peak_migration > len(profile_arr) * 0.3 and not migration_is_organized:
        state = "OVERDOSED"
    elif shape_corr > 0.9 and ratio_shift < 1.3:
        state = "QUIESCENT"
    elif ratio_shift > 1.0 and shape_corr > 0.7:
        state = "THERAPEUTIC"
    elif ratio_shift > 1.5 and shape_corr > 0.5:
        state = "SATURATED"
    else:
        state = "TRANSITIONAL"

    return state, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Three-state spectral diagnostic")
    parser.add_argument("--model", help="Filter to one model")
    parser.add_argument("--dose", type=int, help="Filter to one dose")
    args = parser.parse_args()

    models = load_all()
    order = ["qwen", "mistral", "gemma", "falcon"]
    available = [k for k in order if k in models]
    if args.model:
        available = [k for k in available if k == args.model]

    state_colors = {
        "QUIESCENT": "\033[36m",
        "THERAPEUTIC": "\033[32m",
        "SATURATED": "\033[33m",
        "TRANSITIONAL": "\033[33m",
        "OVERDOSED": "\033[31m",
        "HOLLOWED": "\033[35m",
    }
    reset = "\033[0m"

    for key in available:
        mdata = models[key]
        n_layers = mdata["n_layers"]
        baseline = per_layer_profile(mdata, 0)
        baseline_arr = profile_to_array(baseline, n_layers)

        doses = sorted(set(dd["dose"] for dd in mdata["results"].values()))
        if args.dose is not None:
            doses = [d for d in doses if d == args.dose]

        print(f"\n{'='*60}")
        print(f"  {key.upper()} ({n_layers} layers)")
        print(f"  Baseline peak: L{np.argmax(baseline_arr)} ({np.max(baseline_arr):.3f})")
        print(f"  Baseline mean: {np.mean(baseline_arr):.3f}  CV: {np.std(baseline_arr)/np.mean(baseline_arr):.3f}")
        print(f"{'='*60}")
        print(f"  {'Dose':>4}  {'State':<14} {'ShapeCorr':>9} {'RatShift':>8} {'PeakMig':>7} {'Peak':>4}")
        print(f"  {'-'*52}")

        for dose in doses:
            profile = per_layer_profile(mdata, dose)
            profile_arr = profile_to_array(profile, n_layers)
            state, metrics = classify_state(profile_arr, baseline_arr)
            color = state_colors.get(state, "")

            print(f"  {dose:>4}  {color}{state:<14}{reset} "
                  f"{metrics['shape_correlation']:>9.3f} "
                  f"{metrics['ratio_shift']:>8.3f} "
                  f"{metrics['peak_migration']:>7} "
                  f"L{metrics['peak_layer']}")

    print()


if __name__ == "__main__":
    main()
