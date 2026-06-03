#!/usr/bin/env python3
"""Compare eigenvalue profiles across layers to test recovery vs rebuilding.

Usage:
    python layer_correlation.py results/some_experiment.json
    python layer_correlation.py results/some_experiment.json --layers 0,17,30,32

Computes cosine similarity of [S, σ₂] vectors between all layer pairs.
Reports whether the relay resembles the input (recovery) or is novel (rebuilding).
"""

import json
import math
import sys
from pathlib import Path


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def extract_layer_vectors(data):
    """Extract [S, σ₂] vectors per layer from various result formats."""
    vectors = {}

    if "layer_profile" in data:
        for layer_str, vals in data["layer_profile"].items():
            layer = int(layer_str)
            S = vals.get("receptive_S", vals.get("spectral_entropy", 0))
            s2 = vals.get("receptive_sigma2", vals.get("sigma_2", 0))
            if S > 0 or s2 > 0:
                vectors[layer] = {"S": S, "sigma2": s2}

    if "results" in data and isinstance(data["results"], list):
        from collections import defaultdict
        acc = defaultdict(lambda: {"S": [], "sigma2": []})
        for r in data["results"]:
            layer = r.get("layer", r.get("layer_idx"))
            if layer is None:
                continue
            S = r.get("spectral_entropy", r.get("S", 0))
            s2 = r.get("sigma_2", r.get("sigma2", 0))
            if S > 0 or s2 > 0:
                acc[layer]["S"].append(S)
                acc[layer]["sigma2"].append(s2)
        for layer, v in acc.items():
            vectors[layer] = {
                "S": sum(v["S"]) / len(v["S"]) if v["S"] else 0,
                "sigma2": sum(v["sigma2"]) / len(v["sigma2"]) if v["sigma2"] else 0,
            }

    return vectors


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    with open(path) as f:
        data = json.load(f)

    vectors = extract_layer_vectors(data)
    layers = sorted(vectors.keys())

    if not layers:
        print("No layer data found.")
        sys.exit(1)

    target_layers = None
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--layers" and i + 1 < len(sys.argv):
            target_layers = [int(x) for x in sys.argv[i + 1].split(",")]

    if target_layers is None:
        L0 = layers[0]
        L_mid = layers[len(layers) // 2]
        L_end = layers[-1]
        L_pre = layers[-2] if len(layers) > 1 else L_end
        target_layers = sorted(set([L0, L_mid, L_pre, L_end]))

    print(f"File: {path.name}")
    print(f"Layers available: {layers[0]}-{layers[-1]} ({len(layers)} total)")
    print(f"Analyzing: {target_layers}\n")

    print(f"{'Layer':>5}  {'S':>7}  {'σ₂':>9}  {'role':>12}")
    print("-" * 40)
    for l in target_layers:
        if l in vectors:
            v = vectors[l]
            role = "input" if l == layers[0] else "relay" if l == layers[-1] else "tunnel"
            print(f"{l:>5}  {v['S']:>7.3f}  {v['sigma2']:>9.1f}  {role:>12}")

    print(f"\n{'Pair':>14}  {'cos_sim':>8}  {'interpretation'}")
    print("-" * 55)
    for i, la in enumerate(target_layers):
        for lb in target_layers[i + 1:]:
            if la in vectors and lb in vectors:
                va = [vectors[la]["S"], vectors[la]["sigma2"]]
                vb = [vectors[lb]["S"], vectors[lb]["sigma2"]]
                cs = cosine_sim(va, vb)
                if la == layers[0] and lb == layers[-1]:
                    interp = "INPUT↔RELAY"
                elif la == layers[0]:
                    interp = "input↔tunnel"
                elif lb == layers[-1]:
                    interp = "tunnel↔relay"
                else:
                    interp = ""
                print(f"L{la:>2}↔L{lb:<2}       {cs:>8.4f}  {interp}")

    L0 = layers[0]
    L_relay = layers[-1]
    if L0 in vectors and L_relay in vectors:
        s_ratio = vectors[L_relay]["S"] / vectors[L0]["S"] if vectors[L0]["S"] > 0 else 0
        s2_ratio = vectors[L_relay]["sigma2"] / vectors[L0]["sigma2"] if vectors[L0]["sigma2"] > 0 else 0
        print(f"\nS(relay)/S(input) = {s_ratio:.2f}")
        print(f"σ₂(relay)/σ₂(input) = {s2_ratio:.0f}×")
        if s_ratio > 0.5 and s2_ratio > 10:
            print("\nVERDICT: REBUILDING — similar entropy at vastly different eigenvalue scale.")
            print("The relay constructs new spectral structure, not recovering the input.")
        elif s_ratio > 0.5 and s2_ratio < 3:
            print("\nVERDICT: RECOVERY — similar entropy at similar eigenvalue scale.")
        else:
            print("\nVERDICT: AMBIGUOUS — need more dimensions for discrimination.")


if __name__ == "__main__":
    main()
