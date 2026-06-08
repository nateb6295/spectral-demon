#!/usr/bin/env python3
"""
Analyze cross-architecture dose-response results.
Generates per-layer profiles, relay migration analysis, and summary tables.

Usage:
  python3 analyze_crossarch.py <results_json> [gemma_json]
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_results(*paths):
    combined = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for key, mdata in data["models"].items():
            if "error" not in mdata:
                combined[key] = mdata
    return combined


def per_layer_profile(mdata):
    """Extract mean σ₂/σ₁ per layer per dose."""
    profiles = {}
    for dose_key, dose_data in mdata["results"].items():
        dose = dose_data["dose"]
        layer_ratios = {}
        for run in dose_data["runs"]:
            for layer_str, geom in run["full_geometry"].items():
                layer = int(layer_str)
                layer_ratios.setdefault(layer, []).append(geom["ratio"])
        profiles[dose] = {l: np.mean(v) for l, v in sorted(layer_ratios.items())}
    return profiles


def relay_migration_table(mdata):
    """Show peak layer and relay ratio across doses."""
    profiles = per_layer_profile(mdata)
    rows = []
    for dose in sorted(profiles.keys()):
        layer_vals = profiles[dose]
        peak_layer = max(layer_vals, key=layer_vals.get)
        peak_val = layer_vals[peak_layer]
        mean_val = np.mean(list(layer_vals.values()))
        rows.append({
            "dose": dose,
            "peak_layer": peak_layer,
            "peak_ratio": peak_val,
            "mean_ratio": mean_val,
        })
    return rows


def trajectory_summary(mdata):
    """Extract trajectory summary per dose."""
    rows = []
    for dose_key in sorted(mdata["results"].keys(), key=lambda k: mdata["results"][k]["dose"]):
        dose_data = mdata["results"][dose_key]
        dose = dose_data["dose"]
        trajs = [r["turn_trajectory"] for r in dose_data["runs"]]
        early = np.mean([t[:3] for t in trajs])
        late = np.mean([t[-3:] for t in trajs])
        peak = max(np.mean([t[i] for t in trajs]) for i in range(len(trajs[0])))
        slope = (late - early) / len(trajs[0])
        rows.append({
            "dose": dose, "early": early, "late": late,
            "peak": peak, "slope": slope,
        })
    return rows


def find_flip_dose(rows):
    """Find the dose where trajectory slope changes sign."""
    for i, row in enumerate(rows):
        if row["slope"] < 0 and i > 0:
            return row["dose"]
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_crossarch.py <results_json> [gemma_json]")
        sys.exit(1)

    models = load_results(*sys.argv[1:])

    print("=" * 80)
    print("CROSS-ARCHITECTURE DOSE-RESPONSE ANALYSIS")
    print("=" * 80)

    # Summary table
    print("\n--- TRAJECTORY SUMMARY ---")
    print(f"{'Model':<10} {'Dose':>4} {'Early':>7} {'Late':>7} {'Peak':>7} {'Slope':>9}")
    print("-" * 50)
    for key in ["qwen", "mistral", "gemma", "falcon"]:
        if key not in models:
            continue
        rows = trajectory_summary(models[key])
        flip = find_flip_dose(rows)
        for row in rows:
            marker = " *" if row["dose"] == flip else ""
            print(f"{key:<10} {row['dose']:>4} {row['early']:>7.4f} {row['late']:>7.4f} "
                  f"{row['peak']:>7.4f} {row['slope']:>+9.5f}{marker}")
        if flip:
            print(f"  → Flip dose: {flip}")
        else:
            print(f"  → No flip detected")
        print()

    # Relay migration
    print("\n--- RELAY MIGRATION ---")
    for key in ["qwen", "mistral", "gemma", "falcon"]:
        if key not in models:
            continue
        mdata = models[key]
        print(f"\n{key.upper()} ({mdata['model']}, {mdata['n_layers']} layers, relay=L{mdata['relay_layer']}):")
        rows = relay_migration_table(mdata)
        print(f"  {'Dose':>4} {'Peak Layer':>10} {'Peak σ₂/σ₁':>11} {'Mean σ₂/σ₁':>11}")
        for row in rows:
            print(f"  {row['dose']:>4} {'L'+str(row['peak_layer']):>10} {row['peak_ratio']:>11.4f} {row['mean_ratio']:>11.4f}")

    # Per-layer heatmap (text)
    print("\n--- PER-LAYER PROFILES (key layers) ---")
    for key in ["qwen", "mistral", "gemma", "falcon"]:
        if key not in models:
            continue
        mdata = models[key]
        profiles = per_layer_profile(mdata)
        all_layers = sorted(set().union(*[p.keys() for p in profiles.values()]))
        n = len(all_layers)
        key_layers = all_layers[::max(1, n//8)] + [all_layers[-1]]
        key_layers = sorted(set(key_layers))

        doses = sorted(profiles.keys())
        print(f"\n{key.upper()}: σ₂/σ₁ per layer")
        header = f"  {'Dose':>4} " + " ".join(f"L{l:>2}" for l in key_layers)
        print(header)
        for dose in doses:
            vals = [profiles[dose].get(l, 0) for l in key_layers]
            row = f"  {dose:>4} " + " ".join(f"{v:>4.2f}" for v in vals)
            print(row)

    # Cross-model comparison at dose 0 and dose 20
    print("\n--- CROSS-MODEL COMPARISON ---")
    print("\nNatural state (dose 0):")
    for key in ["qwen", "mistral", "gemma", "falcon"]:
        if key not in models:
            continue
        rows = trajectory_summary(models[key])
        d0 = rows[0]
        profiles = per_layer_profile(models[key])
        relay = models[key]["relay_layer"]
        relay_val = profiles[0].get(relay, 0)
        print(f"  {key:<10} relay(L{relay})={relay_val:.3f}  mean={d0['early']:.3f}→{d0['late']:.3f}  slope={d0['slope']:+.5f}")

    print("\nMax dose (dose 20):")
    for key in ["qwen", "mistral", "gemma", "falcon"]:
        if key not in models:
            continue
        rows = trajectory_summary(models[key])
        d20 = [r for r in rows if r["dose"] == 20]
        if not d20:
            continue
        d20 = d20[0]
        profiles = per_layer_profile(models[key])
        relay = models[key]["relay_layer"]
        relay_val = profiles[20].get(relay, 0)
        migration = relay_migration_table(models[key])
        m20 = [m for m in migration if m["dose"] == 20][0]
        print(f"  {key:<10} relay(L{relay})={relay_val:.3f}  peak=L{m20['peak_layer']}({m20['peak_ratio']:.3f})  "
              f"mean={d20['early']:.3f}→{d20['late']:.3f}  slope={d20['slope']:+.5f}")


if __name__ == "__main__":
    main()
