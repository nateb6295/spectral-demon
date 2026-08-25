#!/usr/bin/env python3
"""Per-layer ρ(g⁺,g⁻) analysis for F614 results.

Kimi probe #2: aggregate ρ≈+1 could mask a polarity-sensitive band.
This script computes sliding-window and zone-specific ρ from saved gain profiles.

Usage: python3 f614_perlayer_rho.py <f614_result.json> [window_size]
"""

import json, sys
import numpy as np
from scipy import stats


def perlayer_rho(gp, gn, window=5):
    n = len(gp)
    results = []
    for i in range(n - window + 1):
        wp = gp[i:i+window]
        wn = gn[i:i+window]
        if np.std(wp) < 1e-12 or np.std(wn) < 1e-12:
            results.append({"center": i + window // 2, "rho": None, "reason": "zero_variance"})
        else:
            r, p = stats.spearmanr(wp, wn)
            results.append({"center": i + window // 2, "rho": float(r), "p": float(p)})
    return results


def zone_rho(gp, gn, n_layers):
    boundary = int(n_layers * 0.7)
    zones = {}
    for name, start, end in [("early", 0, boundary), ("late", boundary, n_layers)]:
        seg_p, seg_n = gp[start:end], gn[start:end]
        if len(seg_p) < 4:
            zones[name] = {"rho": None, "n": len(seg_p)}
        else:
            r, p = stats.spearmanr(seg_p, seg_n)
            zones[name] = {"rho": float(r), "p": float(p), "n": len(seg_p)}
    return zones


def find_polarity_sensitive_bands(perlayer, threshold=0.5):
    bands = []
    current_band = None
    for entry in perlayer:
        if entry["rho"] is not None and entry["rho"] < threshold:
            if current_band is None:
                current_band = {"start": entry["center"], "end": entry["center"], "min_rho": entry["rho"]}
            else:
                current_band["end"] = entry["center"]
                current_band["min_rho"] = min(current_band["min_rho"], entry["rho"])
        else:
            if current_band is not None:
                bands.append(current_band)
                current_band = None
    if current_band is not None:
        bands.append(current_band)
    return bands


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 f614_perlayer_rho.py <f614_result.json> [window_size]")
        sys.exit(1)

    path = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    with open(path) as f:
        d = json.load(f)

    gp = np.array(d["gain_profiles"]["ccs_pos"])
    gn = np.array(d["gain_profiles"]["ccs_neg"])
    n_layers = d["n_layers"]
    model = d.get("short", d.get("model", "unknown"))

    rho_all, p_all = stats.spearmanr(gp, gn)
    perlayer = perlayer_rho(gp, gn, window)
    zones = zone_rho(gp, gn, n_layers)
    bands = find_polarity_sensitive_bands(perlayer)

    rho_vals = [e["rho"] for e in perlayer if e["rho"] is not None]

    print(f"\n{'='*60}")
    print(f"PER-LAYER ρ ANALYSIS: {model}")
    print(f"{'='*60}")
    print(f"Layers: {n_layers}, Window: {window}")
    print(f"\nWhole-model ρ(g⁺,g⁻): {rho_all:+.4f}  (p={p_all:.2e})")
    print(f"Per-layer ρ range: [{min(rho_vals):+.3f}, {max(rho_vals):+.3f}]")
    print(f"Per-layer ρ mean:  {np.mean(rho_vals):+.3f}")
    print(f"Per-layer ρ std:   {np.std(rho_vals):.3f}")

    print(f"\nZone breakdown (70% boundary at L{int(n_layers*0.7)}):")
    for name, z in zones.items():
        if z["rho"] is not None:
            print(f"  {name}: ρ={z['rho']:+.4f} (p={z['p']:.2e}, n={z['n']})")
        else:
            print(f"  {name}: insufficient data (n={z['n']})")

    if bands:
        print(f"\nPOLARITY-SENSITIVE BANDS (ρ < 0.5):")
        for b in bands:
            print(f"  L{b['start']}-L{b['end']}: min ρ={b['min_rho']:+.3f}")
        print("  → Kimi's hypothesis SUPPORTED: aggregate masks local polarity sensitivity")
    else:
        print(f"\nNo polarity-sensitive bands found (all windows ρ ≥ 0.5)")
        print("  → Domain-specificity is uniformly high across all layers")

    # Proportional vs fixed-offset peak test (Kimi correction #9)
    peak_pos = d.get("peak_layers", {}).get("ccs_pos")
    peak_neg = d.get("peak_layers", {}).get("ccs_neg")
    if peak_pos is not None and n_layers >= 40:
        prop = int(0.89 * n_layers)
        fixed = n_layers - 4
        print(f"\nKimi #9 test (80-layer models): proportional=L{prop} vs fixed=L{fixed}")
        print(f"  Peak +CCS: L{peak_pos} (closer to {'proportional' if abs(peak_pos-prop) < abs(peak_pos-fixed) else 'fixed'})")
        if peak_neg is not None:
            print(f"  Peak -CCS: L{peak_neg} (closer to {'proportional' if abs(peak_neg-prop) < abs(peak_neg-fixed) else 'fixed'})")

    out = {
        "model": model, "n_layers": n_layers, "window": window,
        "whole_model_rho": float(rho_all),
        "perlayer_rho": perlayer,
        "zones": zones,
        "polarity_sensitive_bands": bands,
        "perlayer_stats": {
            "min": float(min(rho_vals)), "max": float(max(rho_vals)),
            "mean": float(np.mean(rho_vals)), "std": float(np.std(rho_vals)),
        },
    }
    outpath = path.replace(".json", "_perlayer.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
