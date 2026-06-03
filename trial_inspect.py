#!/usr/bin/env python3
"""Inspect per-trial data from perturbation/compositionality experiments.
Usage: trial_inspect.py <results.json> [condition]
  No condition: summary table of all conditions
  With condition: per-trial breakdown with flip detection
"""
import json, sys, statistics

def load(path):
    with open(path) as f:
        return json.load(f)

def trial_stats(trials):
    v2s = [t["v2_survival_L31"] for t in trials]
    deltas = [t["ratio_post_L31"] - t["ratio_pre_L31"] for t in trials]
    dhs = [t["entropy_shift"] for t in trials]
    flips = sum(1 for v in v2s if v < 0)
    return {
        "n": len(trials), "flips": flips,
        "v2_mean": statistics.mean(v2s),
        "v2_std": statistics.stdev(v2s) if len(v2s) > 1 else 0,
        "delta_mean": statistics.mean(deltas),
        "dh_mean": statistics.mean(dhs),
    }

def summary(data):
    results = data["results"]
    print(f"{'Condition':<28} {'n':>3} {'flips':>5} {'V2_mean':>8} {'V2_std':>7} {'Δratio':>7} {'ΔH':>7}")
    print("-" * 72)
    for cond in sorted(results.keys()):
        trials = results[cond]["trials"]
        s = trial_stats(trials)
        marker = " *" if s["flips"] > 0 else ""
        print(f"{cond:<28} {s['n']:>3} {s['flips']:>5} {s['v2_mean']:>8.3f} {s['v2_std']:>7.3f} {s['delta_mean']:>7.3f} {s['dh_mean']:>7.3f}{marker}")

def detail(data, cond):
    trials = data["results"][cond]["trials"]
    print(f"\n=== {cond.upper()} ({len(trials)} trials) ===\n")
    print(f"{'Probe':<45} {'V2_L31':>7} {'Δratio':>7} {'ΔH':>7} {'L18→L31':>8}")
    print("-" * 80)
    for t in trials:
        probe = t["probe"][:42]
        v2 = t["v2_survival_L31"]
        delta = t["ratio_post_L31"] - t["ratio_pre_L31"]
        dh = t["entropy_shift"]
        v2_18 = t.get("v2_survival_L18", float("nan"))
        marker = " FLIP" if v2 < 0 else ""
        print(f"{probe:<45} {v2:>7.3f} {delta:>7.3f} {dh:>7.3f} {v2_18:>8.3f}{marker}")

    v2s = [t["v2_survival_L31"] for t in trials]
    deltas = [t["ratio_post_L31"] - t["ratio_pre_L31"] for t in trials]
    survive = [(d, t) for d, t in zip(deltas, trials) if t["v2_survival_L31"] > 0]
    flips = [(d, t) for d, t in zip(deltas, trials) if t["v2_survival_L31"] < 0]

    if flips and survive:
        s_mean = statistics.mean([d for d, _ in survive])
        f_mean = statistics.mean([d for d, _ in flips])
        print(f"\nSurvive Δratio mean: {s_mean:.4f} (n={len(survive)})")
        print(f"Flip Δratio mean:    {f_mean:.4f} (n={len(flips)})")
        print(f"Direction: {'under-enrichment → flip' if f_mean < s_mean else 'over-enrichment → flip'}")

        if len(trials) > 2:
            cov = sum((d - statistics.mean(deltas)) * (v - statistics.mean(v2s))
                      for d, v in zip(deltas, v2s)) / len(trials)
            sd = statistics.stdev(deltas)
            sv = statistics.stdev(v2s)
            if sd > 0 and sv > 0:
                print(f"r(Δratio, V2): {cov / (sd * sv):.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: trial_inspect.py <results.json> [condition]")
        sys.exit(1)
    data = load(sys.argv[1])
    if len(sys.argv) > 2:
        detail(data, sys.argv[2])
    else:
        summary(data)
