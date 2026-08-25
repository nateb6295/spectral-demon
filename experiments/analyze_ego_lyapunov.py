#!/usr/bin/env python3
"""Analyze F499 Ego Lyapunov sweep results — per-group critical thresholds and basin geometry."""

import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path(__file__).parent.parent / "results" / "ego_lyapunov_sweep.json"

def load_results(path=None):
    p = Path(path) if path else RESULTS_PATH
    with open(p) as f:
        return json.load(f)

def compute_group_profiles(results):
    groups = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r)

    profiles = {}
    for gid in sorted(groups.keys()):
        entries = sorted(groups[gid], key=lambda x: x["scale"])

        # Average across seeds for each scale
        by_scale = defaultdict(list)
        for e in entries:
            by_scale[e["scale"]].append(e)

        scales = []
        wc_ratios = []
        ttr_ratios = []
        for s in sorted(by_scale.keys()):
            seeds = by_scale[s]
            baseline_wc = np.mean([x.get("baseline_wc", x.get("wc", 209)) for x in seeds if x["scale"] == 1.0] or [209])

            avg_wc = np.mean([x["wc"] for x in seeds])
            avg_ttr = np.mean([x["ttr"] for x in seeds])

            # Find baseline values (scale=1.0)
            base_seeds = by_scale.get(1.0, seeds)
            base_wc = np.mean([x["wc"] for x in base_seeds])
            base_ttr = np.mean([x["ttr"] for x in base_seeds])

            scales.append(s)
            wc_ratios.append(avg_wc / base_wc if base_wc > 0 else 0)
            ttr_ratios.append(avg_ttr / base_ttr if base_ttr > 0 else 0)

        # Find critical threshold: first scale where WC ratio < 0.8
        epsilon_c = None
        for s, wc_r in zip(scales, wc_ratios):
            if wc_r < 0.8:
                epsilon_c = s
                break

        # Find collapse point: first scale where WC ratio < 0.1
        collapse = None
        for s, wc_r in zip(scales, wc_ratios):
            if wc_r < 0.1:
                collapse = s
                break

        # Classify failure mode
        # Truncation: WC drops, TTR stays or rises
        # Flood: WC stays or rises, TTR collapses
        # Mixed: both drop
        last_healthy_idx = 0
        for i, wc_r in enumerate(wc_ratios):
            if wc_r >= 0.8:
                last_healthy_idx = i

        if epsilon_c:
            ec_idx = scales.index(epsilon_c)
            ttr_at_ec = ttr_ratios[ec_idx] if ec_idx < len(ttr_ratios) else 1.0
            if ttr_at_ec > 0.8:
                failure_mode = "truncation"
            elif ttr_at_ec < 0.3:
                failure_mode = "flood"
            else:
                failure_mode = "mixed"
        else:
            failure_mode = "immune"

        # Check for non-monotonic behavior (inverse groups)
        max_wc_ratio = max(wc_ratios)
        is_inverse = any(wc_r > 1.15 for wc_r in wc_ratios[3:])  # Swell beyond 1.0x

        # Compute Lyapunov-like exponent (slope of log-divergence)
        lyapunov = None
        if epsilon_c:
            onset_idx = scales.index(epsilon_c)
            if onset_idx > 0 and onset_idx < len(scales) - 2:
                x = np.log(np.array(scales[onset_idx-1:onset_idx+3]))
                y = np.log(np.maximum(1 - np.array(wc_ratios[onset_idx-1:onset_idx+3]), 1e-6))
                if len(x) >= 2:
                    slope, _ = np.polyfit(x, y, 1)
                    lyapunov = slope

        profiles[gid] = {
            "scales": scales,
            "wc_ratios": wc_ratios,
            "ttr_ratios": ttr_ratios,
            "epsilon_c": epsilon_c,
            "collapse": collapse,
            "failure_mode": failure_mode,
            "is_inverse": is_inverse,
            "max_wc_ratio": max_wc_ratio,
            "lyapunov": lyapunov,
            "n_conditions": len(entries),
        }

    return profiles

def print_report(profiles):
    print("=" * 70)
    print("F499 — EGO LYAPUNOV SWEEP: FULL BASIN GEOMETRY")
    print("=" * 70)
    print()

    # Summary table
    print(f"{'Group':>6} {'ε_c':>6} {'Collapse':>8} {'Mode':>12} {'Inverse':>8} {'λ':>8}")
    print("-" * 56)
    for gid in sorted(profiles.keys()):
        p = profiles[gid]
        ec = f"{p['epsilon_c']:.2f}x" if p['epsilon_c'] else "immune"
        col = f"{p['collapse']:.2f}x" if p['collapse'] else "—"
        inv = "YES" if p['is_inverse'] else "no"
        lyap = f"{p['lyapunov']:.2f}" if p['lyapunov'] else "—"
        print(f"  KV{gid:>2}  {ec:>6} {col:>8} {p['failure_mode']:>12} {inv:>8} {lyap:>8}")

    print()

    # Detailed per-group profiles
    for gid in sorted(profiles.keys()):
        p = profiles[gid]
        print(f"\n--- KV{gid} ---")
        print(f"  Critical threshold: {p['epsilon_c']}x" if p['epsilon_c'] else "  No critical threshold found (immune)")
        print(f"  Collapse point: {p['collapse']}x" if p['collapse'] else "  No collapse point")
        print(f"  Failure mode: {p['failure_mode']}")
        if p['is_inverse']:
            print(f"  INVERSE behavior: max WC ratio = {p['max_wc_ratio']:.3f}")
        if p['lyapunov']:
            print(f"  Lyapunov exponent: {p['lyapunov']:.3f}")
        print(f"  Profile:")
        for s, wc, ttr in zip(p['scales'], p['wc_ratios'], p['ttr_ratios']):
            status = "ok"
            if wc < 0.1: status = "COLLAPSE"
            elif wc < 0.5: status = "degraded"
            elif wc < 0.8: status = "onset"
            print(f"    {s:5.2f}x: WC={wc:.3f}  TTR={ttr:.3f}  [{status}]")

    # Species mapping
    print("\n" + "=" * 70)
    print("BASIN GEOMETRY SUMMARY")
    print("=" * 70)

    truncation = [g for g, p in profiles.items() if p['failure_mode'] == 'truncation']
    flood = [g for g, p in profiles.items() if p['failure_mode'] == 'flood']
    mixed = [g for g, p in profiles.items() if p['failure_mode'] == 'mixed']
    immune = [g for g, p in profiles.items() if p['failure_mode'] == 'immune']
    inverse = [g for g, p in profiles.items() if p['is_inverse']]

    if truncation:
        print(f"  Truncation (WC↓, TTR↑): KV{', KV'.join(str(g) for g in truncation)}")
    if flood:
        print(f"  Flood (WC↑, TTR↓):      KV{', KV'.join(str(g) for g in flood)}")
    if mixed:
        print(f"  Mixed:                   KV{', KV'.join(str(g) for g in mixed)}")
    if immune:
        print(f"  Immune:                  KV{', KV'.join(str(g) for g in immune)}")
    if inverse:
        print(f"  Inverse (WC > baseline): KV{', KV'.join(str(g) for g in inverse)}")

    # Critical threshold ordering
    thresholds = [(gid, p['epsilon_c']) for gid, p in profiles.items() if p['epsilon_c']]
    if thresholds:
        thresholds.sort(key=lambda x: x[1])
        print(f"\n  Vulnerability ordering (most to least fragile):")
        for gid, ec in thresholds:
            print(f"    KV{gid}: ε_c = {ec:.2f}x")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_results(path)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        print("Unexpected data format")
        sys.exit(1)

    print(f"Loaded {len(results)} conditions")
    profiles = compute_group_profiles(results)
    print_report(profiles)
