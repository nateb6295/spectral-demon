#!/usr/bin/env python3
"""
Generate paper-ready markdown tables from cross-arch dose-response results.

Usage:
  python3 results_to_table.py <json_file> [model_key]
  python3 results_to_table.py results/*.json              # merge multiple files
  python3 results_to_table.py results/*.json qwen         # single model from merged
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_and_merge(paths):
    models = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for key, mdata in data["models"].items():
            if "error" not in mdata:
                models[key] = mdata
    return models


def dose_response_table(mdata):
    """Generate the §2 dose-response table for a model."""
    n_layers = mdata["n_layers"]
    relay_layer = mdata["relay_layer"]
    last_layer = n_layers - 1

    rows = []
    for dose_key in sorted(mdata["results"].keys(), key=lambda k: mdata["results"][k]["dose"]):
        dd = mdata["results"][dose_key]
        dose = dd["dose"]

        last_ratios = []
        relay_ratios = []
        for run in dd["runs"]:
            geom = run["full_geometry"]
            last_ratios.append(geom.get(str(last_layer), {}).get("ratio", 0))
            relay_ratios.append(geom.get(str(relay_layer), {}).get("ratio", 0))

        trajs = [r["turn_trajectory"] for r in dd["runs"]]
        early = np.mean([t[:3] for t in trajs])
        late = np.mean([t[-3:] for t in trajs])
        slope = (late - early) / len(trajs[0])

        rows.append({
            "dose": dose,
            "last": np.mean(last_ratios),
            "relay": np.mean(relay_ratios),
            "slope": slope,
            "early": early,
            "late": late,
        })
    return rows, relay_layer, last_layer


def print_table(mdata, model_key):
    rows, relay_layer, last_layer = dose_response_table(mdata)
    name = mdata["model"]
    n = mdata["n_layers"]

    show_both = (relay_layer != last_layer)

    print(f"**{model_key.title()} ({name.split('/')[-1]}, {n} layers):**\n")

    if show_both:
        print(f"| Dose | L{last_layer} σ₂/σ₁ | L{relay_layer} σ₂/σ₁ | Slope | Early | Late |")
        print(f"|------|{'-----------|' * 2}-------|-------|------|")
        for r in rows:
            sign = "+" if r["slope"] >= 0 else "−"
            print(f"| {r['dose']} | {r['last']:.3f} | {r['relay']:.3f} | "
                  f"{sign}{abs(r['slope']):.4f} | {r['early']:.2f} | {r['late']:.2f} |")
    else:
        print(f"| Dose | L{last_layer} σ₂/σ₁ | Slope | Early | Late |")
        print(f"|------|-----------|-------|-------|------|")
        for r in rows:
            sign = "+" if r["slope"] >= 0 else "−"
            print(f"| {r['dose']} | {r['last']:.3f} | "
                  f"{sign}{abs(r['slope']):.4f} | {r['early']:.2f} | {r['late']:.2f} |")
    print()


def print_migration(mdata, model_key):
    """One-line relay migration summary."""
    profiles = {}
    for dose_key, dd in mdata["results"].items():
        dose = dd["dose"]
        layer_ratios = {}
        for run in dd["runs"]:
            for ls, geom in run["full_geometry"].items():
                l = int(ls)
                layer_ratios.setdefault(l, []).append(geom["ratio"])
        profiles[dose] = {l: np.mean(v) for l, v in layer_ratios.items()}

    peaks = []
    for dose in sorted(profiles.keys()):
        vals = profiles[dose]
        peak_l = max(vals, key=vals.get)
        peak_v = vals[peak_l]
        peaks.append((dose, peak_l, peak_v))

    layers_seen = set(p[1] for p in peaks)
    if len(layers_seen) == 1:
        print(f"  {model_key}: No migration (L{peaks[0][1]} stays peak across all doses)")
    else:
        trajectory = "→".join(f"L{p[1]}" for p in peaks)
        print(f"  {model_key}: Migration {trajectory}")
        print(f"    Peak values: {' → '.join(f'{p[2]:.3f}' for p in peaks)}")


def main():
    paths = [p for p in sys.argv[1:] if p.endswith(".json")]
    model_filter = [a for a in sys.argv[1:] if not a.endswith(".json")]

    if not paths:
        print("Usage: results_to_table.py <json_files> [model_key]")
        sys.exit(1)

    models = load_and_merge(paths)

    if model_filter:
        models = {k: v for k, v in models.items() if k in model_filter}

    for key in ["qwen", "mistral", "gemma", "falcon"]:
        if key in models:
            print_table(models[key], key)

    print("**Relay migration:**\n")
    for key in ["qwen", "mistral", "gemma", "falcon"]:
        if key in models:
            print_migration(models[key], key)


if __name__ == "__main__":
    main()
