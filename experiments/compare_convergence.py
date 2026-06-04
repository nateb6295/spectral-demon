#!/usr/bin/env python3
"""Compare convergence_v2 results across experiment runs.

Usage:
  python3 compare_convergence.py                    # all results
  python3 compare_convergence.py results/exp_*.json # specific files
  python3 compare_convergence.py --sort ratio       # sort by basin ratio
"""

import json, glob, sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_results(paths=None):
    if not paths:
        paths = sorted(RESULTS_DIR.glob("exp_convergence_v2_*.json"))
    rows = []
    for p in paths:
        p = Path(p)
        with open(p) as f:
            data = json.load(f)
        for mname, mdata in data.get("models", {}).items():
            s = mdata.get("summary", {})
            if not s or "phase1_mean_drift" not in s:
                continue
            p1 = s["phase1_mean_drift"]
            p3 = s["phase3_mean_drift"]
            tv = s.get("transition_to_vanilla", 0)
            tr = s.get("transition_to_reapply", 0)
            rows.append({
                "file": p.name,
                "model": mname,
                "phase2_mode": mdata.get("phase2_mode", "vanilla"),
                "p1_drift": p1,
                "p3_drift": p3,
                "ratio": p1 / p3 if p3 > 0 else float("inf"),
                "asym": (tv - tr) / tv * 100 if tv > 0 else 0,
                "tv": tv,
                "tr": tr,
                "steady": s.get("steady_ccs", 0),
            })
    return rows

def print_table(rows, sort_key="ratio"):
    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)
    has_modes = any(r["phase2_mode"] != "vanilla" for r in rows)
    mode_col = f" {'Mode':>10s}" if has_modes else ""
    print(f"{'Model':12s}{mode_col} {'P1 drift':>10s} {'P3 drift':>10s} {'Ratio':>8s} {'Asym%':>7s} {'Diss×':>7s} {'Reapp×':>7s} {'File'}")
    print("-" * (95 + (11 if has_modes else 0)))
    for r in rows:
        steady = r["steady"] if r["steady"] > 0 else r["p1_drift"]
        diss_x = r["tv"] / steady if steady > 0 else 0
        reapp_x = r["tr"] / steady if steady > 0 else 0
        mode_str = f" {r['phase2_mode']:>10s}" if has_modes else ""
        print(f"{r['model']:12s}{mode_str} {r['p1_drift']:10.6f} {r['p3_drift']:10.6f} {r['ratio']:7.1f}× {r['asym']:6.1f}% {diss_x:6.1f}× {reapp_x:6.1f}× {r['file']}")
    print()
    if len(rows) > 1:
        avg_ratio = sum(r["ratio"] for r in rows) / len(rows)
        avg_asym = sum(r["asym"] for r in rows) / len(rows)
        print(f"Mean ratio: {avg_ratio:.1f}×  Mean asymmetry: {avg_asym:.1f}%  N={len(rows)}")

    if has_modes:
        print("\n  --- Weil Attention Comparison ---")
        by_mode = {}
        for r in rows:
            by_mode.setdefault(r["phase2_mode"], []).append(r)
        for mode in ["vanilla", "silent", "structured"]:
            if mode not in by_mode:
                continue
            ratios = [r["ratio"] for r in by_mode[mode]]
            avg = sum(ratios) / len(ratios)
            print(f"  {mode:12s}: mean ratio = {avg:.2f}× (n={len(ratios)})")
        modes_present = [m for m in ["vanilla", "silent", "structured"] if m in by_mode]
        if len(modes_present) >= 2:
            avgs = {m: sum(r["ratio"] for r in by_mode[m])/len(by_mode[m]) for m in modes_present}
            ranked = sorted(avgs, key=avgs.get, reverse=True)
            print(f"  Ranking: {' > '.join(ranked)}")
            if avgs.get("structured", 0) > avgs.get("vanilla", 0):
                print(f"  → Weil prediction SUPPORTED (structured > vanilla)")
            elif abs(avgs.get("structured", 0) - avgs.get("vanilla", 0)) < 0.1:
                print(f"  → Null: duration matters, quality doesn't")
            else:
                print(f"  → Weil prediction FALSIFIED")

if __name__ == "__main__":
    sort_key = "ratio"
    paths = []
    for arg in sys.argv[1:]:
        if arg.startswith("--sort"):
            continue
        elif sys.argv[sys.argv.index(arg) - 1] == "--sort":
            sort_key = arg
        else:
            paths.extend(glob.glob(arg))

    if "--sort" in sys.argv:
        idx = sys.argv.index("--sort")
        if idx + 1 < len(sys.argv):
            sort_key = sys.argv[idx + 1]
        paths = [p for p in paths if p != sort_key]

    rows = load_results(paths if paths else None)
    if not rows:
        print("No convergence_v2 results found.")
        sys.exit(1)
    print_table(rows, sort_key)
