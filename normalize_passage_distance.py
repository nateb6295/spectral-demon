#!/usr/bin/env python3
"""Add normalized passage distance (d/d_max) to experiment results.

Usage:
    python normalize_passage_distance.py results/some_experiment.json [--k 10]
    python normalize_passage_distance.py results/some_experiment.json --auto

If --auto, infers k from the results file (looks for top_k_subspace k parameter).
If --k N, uses that k value.
Default k=10 if neither specified.

Prints a summary table and optionally updates the JSON with normalized values.
"""

import json
import math
import sys
from pathlib import Path


def d_max(k):
    return math.sqrt(k) * math.pi / 2


def normalize(d, k):
    return d / d_max(k)


def residual_degrees(d_norm):
    return (1 - d_norm) * 90


def process_file(path, k=10):
    with open(path) as f:
        data = json.load(f)

    distances = []

    if "layer_profile" in data:
        for layer, vals in data["layer_profile"].items():
            if "mean_d" in vals:
                d = vals["mean_d"]
                d_norm = normalize(d, k)
                vals["d_normalized"] = round(d_norm, 6)
                vals["residual_degrees"] = round(residual_degrees(d_norm), 2)
                distances.append((int(layer), d, d_norm))

    if "measurements" in data:
        for m in data["measurements"]:
            if "mean_d" in m or "passage_distance" in m:
                d = m.get("mean_d", m.get("passage_distance", 0))
                if d > 0:
                    d_norm = normalize(d, k)
                    m["d_normalized"] = round(d_norm, 6)
                    m["residual_degrees"] = round(residual_degrees(d_norm), 2)
                    layer = m.get("layer", "?")
                    distances.append((layer, d, d_norm))

    data["_normalization"] = {
        "k": k,
        "d_max": round(d_max(k), 4),
        "method": "d / (sqrt(k) * pi/2)"
    }

    return data, distances


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    k = 10

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--k" and i + 1 < len(sys.argv):
            k = int(sys.argv[i + 1])
        elif arg == "--auto":
            with open(path) as f:
                text = f.read()
            if "k=5" in text or '"k": 5' in text:
                k = 5
            elif "k=10" in text or '"k": 10' in text:
                k = 10

    dmax = d_max(k)
    print(f"k={k}, d_max={dmax:.4f}")
    print()

    data, distances = process_file(path, k)

    if distances:
        print(f"{'Layer':>6} {'Raw d':>8} {'d/d_max':>8} {'Residual':>10}")
        print("-" * 36)
        for layer, d, d_norm in distances[-8:]:
            res = residual_degrees(d_norm)
            print(f"{layer:>6} {d:>8.4f} {d_norm:>8.4f} {res:>9.1f}°")

    if "--write" in sys.argv:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nUpdated {path}")
    else:
        print(f"\nDry run. Use --write to update the file.")


if __name__ == "__main__":
    main()
