#!/usr/bin/env python3
"""Compute reactive gain profiles from trajectory or perturbation experiment data.

Usage:
  reactive_gain.py trajectory <trajectory_summary.json>    # per-layer Δratio profile
  reactive_gain.py perturbation <perturbation_results.json> # gain_ratio at L18/L31
  reactive_gain.py both <trajectory.json> <perturbation.json> # unified view
"""
import json, sys, statistics


def load(path):
    with open(path) as f:
        return json.load(f)


def trajectory_profile(data):
    svd_layers = data["svd_layers"]
    summaries = data["summaries"]
    relay_layers = [L for L in svd_layers if 24 <= L <= 31]

    results = []
    for cond, s in summaries.items():
        deltas = {}
        for L in svd_layers:
            pre_key = f"ratio_pre_L{L}_mean"
            post_key = f"ratio_post_L{L}_mean"
            if pre_key in s and post_key in s:
                deltas[L] = s[post_key] - s[pre_key]

        d24 = deltas.get(24, 0)
        d31 = deltas.get(31, 0)
        d26 = deltas.get(26, 0)
        d28 = deltas.get(28, 0)
        ramp = d31 - d24
        onset = d28 - d26

        v2_key = "v2_survival_L31_mean"
        v2_std_key = "v2_survival_L31_std"
        v2 = s.get(v2_key, float("nan"))
        v2_std = s.get(v2_std_key, 0)

        results.append({
            "condition": cond,
            "ramp_L24_L31": ramp,
            "onset_L26_L28": onset,
            "delta_L31": d31,
            "v2_mean": v2,
            "v2_std": v2_std,
            "bistable": v2_std > 0.5,
            "per_layer": deltas,
        })

    results.sort(key=lambda r: r["ramp_L24_L31"], reverse=True)

    print(f"\n{'Condition':<28} {'Ramp':>6} {'Onset':>6} {'ΔL31':>5} {'V2':>6} {'V2σ':>5} {'Status':>10}")
    print("-" * 72)
    for r in results:
        status = "BISTABLE" if r["bistable"] else "mono"
        print(f"{r['condition']:<28} {r['ramp_L24_L31']:>6.4f} {r['onset_L26_L28']:>6.4f} "
              f"{r['delta_L31']:>5.2f} {r['v2_mean']:>6.3f} {r['v2_std']:>5.3f} {status:>10}")

    ramps = [r["ramp_L24_L31"] for r in results]
    bistable_ramps = [r["ramp_L24_L31"] for r in results if r["bistable"]]
    mono_ramps = [r["ramp_L24_L31"] for r in results if not r["bistable"]]

    if bistable_ramps and mono_ramps:
        print(f"\nMono ramp range: [{min(mono_ramps):.4f}, {max(mono_ramps):.4f}]")
        print(f"Bistable ramp range: [{min(bistable_ramps):.4f}, {max(bistable_ramps):.4f}]")
        gap = min(mono_ramps) - max(bistable_ramps)
        print(f"Separation gap: {gap:.4f} ({'CLEAN' if gap > 0 else 'OVERLAP'})")

    return results


def perturbation_profile(data):
    results_data = data["results"]

    results = []
    for cond, cdata in results_data.items():
        trials = cdata["trials"]
        pre_gains = [t["ratio_pre_L31"] - t["ratio_pre_L18"] for t in trials]
        post_gains = [t["ratio_post_L31"] - t["ratio_post_L18"] for t in trials]
        d_rat31 = [t["ratio_post_L31"] - t["ratio_pre_L31"] for t in trials]
        v2s = [t["v2_survival_L31"] for t in trials]
        flips = sum(1 for v in v2s if v < 0)

        pg_mean = statistics.mean(pre_gains)
        ppg_mean = statistics.mean(post_gains)
        gain_ratio = ppg_mean / pg_mean if pg_mean > 0.001 else float("inf")
        v2_std = statistics.stdev(v2s) if len(v2s) > 1 else 0

        results.append({
            "condition": cond,
            "pre_gain": pg_mean,
            "post_gain": ppg_mean,
            "gain_ratio": gain_ratio,
            "delta_L31": statistics.mean(d_rat31),
            "v2_std": v2_std,
            "flips": flips,
            "bistable": v2_std > 0.5,
        })

    results.sort(key=lambda r: r["gain_ratio"], reverse=True)

    print(f"\n{'Condition':<28} {'PreGain':>8} {'PostGain':>9} {'Ratio':>6} {'ΔL31':>5} "
          f"{'Flips':>5} {'Status':>10}")
    print("-" * 78)
    for r in results:
        status = "BISTABLE" if r["bistable"] else "mono"
        ratio_str = f"{r['gain_ratio']:>6.3f}" if r['gain_ratio'] < 100 else "  INF"
        print(f"{r['condition']:<28} {r['pre_gain']:>8.4f} {r['post_gain']:>9.4f} {ratio_str} "
              f"{r['delta_L31']:>5.2f} {r['flips']:>5} {status:>10}")

    return results


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "trajectory":
        data = load(sys.argv[2])
        trajectory_profile(data)
    elif mode == "perturbation":
        data = load(sys.argv[2])
        perturbation_profile(data)
    elif mode == "both" and len(sys.argv) >= 4:
        print("=== TRAJECTORY (per-layer reactive gain) ===")
        trajectory_profile(load(sys.argv[2]))
        print("\n=== PERTURBATION (gain ratio L18→L31) ===")
        perturbation_profile(load(sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
