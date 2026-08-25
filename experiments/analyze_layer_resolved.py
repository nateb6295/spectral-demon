#!/usr/bin/env python3
"""Analyze layer-resolved SR/OR results across species.

Reads per-model JSON results from f614_layer_resolved_sror and produces:
1. Normalized depth comparison (l/L) for cross-species alignment
2. Band-by-band SR/OR separation vs D2 error budget
3. Summary table for paper
"""
import os, json, sys, argparse
import numpy as np

def load_results(path):
    with open(path) as f:
        return json.load(f)

def normalize_depth(gains, n_layers):
    """Return gains with normalized depth axis [0, 1]."""
    depths = np.linspace(0, 1, n_layers, endpoint=False) + 0.5 / n_layers
    return depths, np.array(gains)

def band_stats(gains, n_layers, n_bands=3):
    band_size = n_layers // n_bands
    stats = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n_layers
        band = gains[start:end]
        stats.append({
            "band": ["early", "mid", "late"][i],
            "layers": f"{start}-{end-1}",
            "mean": float(np.mean(band)),
            "std": float(np.std(band)),
            "max_abs": float(np.max(np.abs(band))),
        })
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="JSON result files")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from scipy.stats import spearmanr

    models = []
    for path in args.results:
        r = load_results(path)
        models.append(r)
        print(f"\n{'='*60}")
        print(f"Model: {r['model']} ({r['n_layers']} layers)")
        print(f"{'='*60}")

        sr = np.array(r["per_layer_gains"]["sr"])
        or_ = np.array(r["per_layer_gains"]["or"])
        d2 = np.array(r["per_layer_gains"]["ccs_d2"])
        fct = np.array(r["per_layer_gains"]["factual"])
        n = r["n_layers"]

        # D2 error budget per band
        d2_bands = band_stats(d2, n)
        print("\nD2 positive control (per-band):")
        for b in d2_bands:
            print(f"  {b['band']} ({b['layers']}): mean={b['mean']:+.6f}, std={b['std']:.6f}")

        # D2 deviation from flat
        d2_flat_dev = float(np.std(d2))
        print(f"  Deviation from flat (std): {d2_flat_dev:.6f}")

        # SR/OR difference per band
        diff = sr - or_
        diff_bands = band_stats(diff, n)
        print("\nSR-OR difference (per-band):")
        for b in diff_bands:
            print(f"  {b['band']} ({b['layers']}): mean={b['mean']:+.6f}, max|diff|={b['max_abs']:.6f}")

        # Does mid-band SR/OR separation exceed D2 error budget?
        mid_idx = 1
        mid_diff_max = diff_bands[mid_idx]["max_abs"]
        mid_d2_std = d2_bands[mid_idx]["std"]
        print(f"\nMid-band SR/OR max|diff| vs D2 std: {mid_diff_max:.6f} vs {mid_d2_std:.6f}")
        if mid_diff_max > 2 * mid_d2_std:
            print("  -> SR/OR separation EXCEEDS D2 error budget (2x)")
        elif mid_diff_max > mid_d2_std:
            print("  -> SR/OR separation marginally exceeds D2 error budget")
        else:
            print("  -> SR/OR separation WITHIN D2 error budget — null holds")

        # Band correlations
        for bname, bslice in [("early", slice(0, n//3)), ("mid", slice(n//3, 2*n//3)), ("late", slice(2*n//3, n))]:
            sr_b = sr[bslice]
            or_b = or_[bslice]
            if len(sr_b) >= 3:
                rho, p = spearmanr(sr_b, or_b)
                print(f"  SR vs OR ({bname}): rho={rho:+.4f} (p={p:.2e})")

    # Cross-species comparison at normalized depth
    if len(models) >= 2:
        print(f"\n{'='*60}")
        print("CROSS-SPECIES COMPARISON (normalized depth)")
        print(f"{'='*60}")

        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if j <= i:
                    continue
                n1, n2 = m1["n_layers"], m2["n_layers"]
                n_common = min(n1, n2)

                # Interpolate to common normalized grid
                grid = np.linspace(0, 1, n_common)
                for metric in ["sr", "or", "ccs_d2"]:
                    g1 = np.array(m1["per_layer_gains"][metric])
                    g2 = np.array(m2["per_layer_gains"][metric])
                    d1 = np.linspace(0, 1, n1)
                    d2_axis = np.linspace(0, 1, n2)
                    g1_interp = np.interp(grid, d1, g1)
                    g2_interp = np.interp(grid, d2_axis, g2)
                    rho, p = spearmanr(g1_interp, g2_interp)
                    print(f"  {m1['model']} vs {m2['model']} ({metric}): rho={rho:+.4f} (p={p:.2e})")

                # SR-OR difference alignment
                diff1 = np.array(m1["sr_or_difference"])
                diff2 = np.array(m2["sr_or_difference"])
                d1 = np.linspace(0, 1, n1)
                d2_axis = np.linspace(0, 1, n2)
                diff1_interp = np.interp(grid, d1, diff1)
                diff2_interp = np.interp(grid, d2_axis, diff2)
                rho, p = spearmanr(diff1_interp, diff2_interp)
                print(f"  SR-OR difference alignment: rho={rho:+.4f} (p={p:.2e})")

    if args.output:
        summary = {
            "models": [m["model"] for m in models],
            "n_layers": [m["n_layers"] for m in models],
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
