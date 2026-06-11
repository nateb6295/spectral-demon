#!/usr/bin/env python3
"""
Three-model FTLE comparison visualization.
Generates terminal-friendly comparison of volume dynamics across Mistral, Qwen, Gemma.
"""

import json
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_all_ftle():
    results = {}
    for f in sorted(RESULTS_DIR.glob("ftle_zones_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for model_key in data:
            results[model_key] = data[model_key]
    return results


def volume_profile(data, dose="0"):
    spec = data["doses"][dose]["ftle_spectrum"]
    layers = sorted(spec.keys(), key=int)
    return [(int(l), spec[l]["mean_ftle"], spec[l]["n_expanding"]) for l in layers]


def print_comparison(results):
    models = list(results.keys())
    print(f"\n{'='*80}")
    print(f"  THREE-MODEL FTLE COMPARISON (dose 0)")
    print(f"{'='*80}")

    # Total volume
    print(f"\n  TOTAL VOLUME:")
    for m in models:
        prof = volume_profile(results[m], "0")
        total = sum(mean for _, mean, _ in prof)
        total_exp = sum(n for _, _, n in prof)
        n_layers = len(prof)
        print(f"    {m:>20s}: total={total:+8.3f}, expanding={total_exp}/{n_layers*64}")

    # Dose 1 response
    print(f"\n  DOSE 1 TOTAL VOLUME:")
    for m in models:
        if "1" in results[m]["doses"]:
            prof0 = volume_profile(results[m], "0")
            prof1 = volume_profile(results[m], "1")
            t0 = sum(mean for _, mean, _ in prof0)
            t1 = sum(mean for _, mean, _ in prof1)
            delta_pct = ((t1 - t0) / abs(t0)) * 100 if t0 != 0 else 0
            print(f"    {m:>20s}: D0={t0:+8.3f} D1={t1:+8.3f} ({delta_pct:+.1f}%)")

    # Architecture classification
    print(f"\n  ARCHITECTURE TYPE:")
    for m in models:
        prof = volume_profile(results[m], "0")
        first_half = prof[:len(prof)//2]
        second_half = prof[len(prof)//2:]
        
        fh_mean = np.mean([mean for _, mean, _ in first_half])
        sh_mean = np.mean([mean for _, mean, _ in second_half])
        
        if fh_mean < -0.1 and sh_mean > fh_mean:
            arch = "CONTRACT→EXPAND (tunnel→relay)"
        elif fh_mean > 0.1 and sh_mean < fh_mean:
            arch = "EXPAND→CONTRACT (expansion→brace)"
        elif abs(fh_mean) < 0.1 and abs(sh_mean) < 0.1:
            arch = "VOLUME-PRESERVING (symplectic)"
        elif abs(fh_mean - sh_mean) < 0.1:
            arch = "EQUALIZED (distributed)"
        else:
            arch = f"MIXED (first={fh_mean:.3f}, second={sh_mean:.3f})"
        
        print(f"    {m:>20s}: {arch}")

    # Side-by-side volume profile
    print(f"\n  VOLUME PROFILES (dose 0):")
    max_layers = max(len(volume_profile(results[m], "0")) for m in models)
    
    # Get profiles
    profiles = {}
    for m in models:
        prof = volume_profile(results[m], "0")
        profiles[m] = {l: (mean, n) for l, mean, n in prof}
    
    all_layers = sorted(set(l for p in profiles.values() for l in p.keys()))
    
    header = f"  {'Layer':>5}"
    for m in models:
        short = m[:12]
        header += f" | {short:>12s} exp  "
    print(header)
    print("  " + "-" * (len(header) - 2))
    
    for l in all_layers:
        row = f"  L{l:3d} "
        for m in models:
            if l in profiles[m]:
                mean, n = profiles[m][l]
                bar_char = "+" if mean >= 0 else "-"
                bar_len = min(int(abs(mean) * 3), 8)
                bar = bar_char * bar_len
                row += f" | {mean:+.3f} {n:2d}/64 {bar:>8s}"
            else:
                row += f" |       ---        "
        print(row)

    # Dose sensitivity peak
    print(f"\n  DOSE SENSITIVITY (D0→D1, peak layer):")
    for m in models:
        if "1" not in results[m]["doses"]:
            print(f"    {m:>20s}: no dose 1 data")
            continue
        
        d0 = results[m]["doses"]["0"]["ftle_spectrum"]
        d1 = results[m]["doses"]["1"]["ftle_spectrum"]
        layers = sorted(d0.keys(), key=int)
        
        max_delta = 0
        max_layer = 0
        for l in layers:
            if l in d1:
                delta = d1[l]["n_expanding"] - d0[l]["n_expanding"]
                if abs(delta) > abs(max_delta):
                    max_delta = delta
                    max_layer = int(l)
        
        print(f"    {m:>20s}: L{max_layer} (delta={max_delta:+d} expanding directions)")

    # Conservation test
    print(f"\n  VOLUME CONSERVATION TEST:")
    totals = []
    for m in models:
        prof = volume_profile(results[m], "0")
        total = sum(mean for _, mean, _ in prof)
        n_layers = results[m]["n_layers"]
        per_layer = total / n_layers
        totals.append(total)
        print(f"    {m:>20s}: total={total:+8.3f}, per_layer_avg={per_layer:+.4f} (n={n_layers})")
    
    if len(totals) >= 2:
        spread = max(totals) - min(totals)
        mean_total = np.mean(totals)
        print(f"    Spread: {spread:.3f}, Mean: {mean_total:.3f}, CV: {spread/abs(mean_total)*100:.1f}%")


if __name__ == "__main__":
    results = load_all_ftle()
    if len(results) < 2:
        print(f"Only {len(results)} model(s) found. Need at least 2 for comparison.")
        sys.exit(1)
    
    print_comparison(results)
