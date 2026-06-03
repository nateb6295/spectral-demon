#!/usr/bin/env python3
"""results_dashboard — Quick summary of all experiment results.

Usage:
  results_dashboard.py           # Print summary table
  results_dashboard.py --json    # JSON output
  results_dashboard.py --detail EXP  # Detailed view of one experiment
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_results():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            results.append({"file": f.name, "data": data})
        except Exception:
            continue
    return results


def extract_summary(entry):
    data = entry["data"]
    name = entry["file"]

    summary = {"file": name}

    if isinstance(data, list):
        summary["experiment"] = name.replace(".json", "")
        summary["n_measurements"] = len(data)
        return summary

    if "experiment" in data:
        summary["experiment"] = data["experiment"]
    elif "exp" in name:
        summary["experiment"] = name.split("_20")[0] if "_20" in name else name.replace(".json", "")

    summary["model"] = data.get("model", data.get("model_name", "?"))

    if "conditions" in data and isinstance(data["conditions"], dict):
        conds = data["conditions"]
        if "receptive" in conds and "absent" in conds:
            r = conds["receptive"]
            a = conds["absent"]
            r_s = r.get("mean_S", r.get("S_mean", r.get("S", None)))
            a_s = a.get("mean_S", a.get("S_mean", a.get("S", None)))
            if r_s is not None and a_s is not None:
                summary["S_receptive"] = r_s
                summary["S_absent"] = a_s
                summary["delta_S"] = round(r_s - a_s, 5)

    if "delta_S" in data:
        summary["delta_S"] = data["delta_S"]
    if "sign" in data:
        summary["sign"] = data["sign"]
    if "n_probes" in data:
        summary["n_probes"] = data["n_probes"]
    if "total_passes" in data:
        summary["total_passes"] = data["total_passes"]
    if "total_measurements" in data:
        summary["total_measurements"] = data["total_measurements"]

    n_results = len(data.get("results", []))
    if n_results:
        summary["n_measurements"] = n_results

    return summary


def print_table(summaries):
    print(f"{'Experiment':<45} {'Model':<25} {'ΔS':>8} {'Sign':>12} {'N':>6}")
    print("-" * 100)
    for s in summaries:
        exp = s.get("experiment", s["file"])[:44]
        model = str(s.get("model", "?"))[:24]
        delta = s.get("delta_S", "")
        if isinstance(delta, (int, float)):
            delta = f"{delta:+.4f}"
        sign = s.get("sign", "")
        n = s.get("n_measurements", s.get("total_measurements", s.get("n_probes", "")))
        print(f"{exp:<45} {model:<25} {delta:>8} {sign:>12} {str(n):>6}")


def main():
    parser = argparse.ArgumentParser(description="Experiment results dashboard")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--detail", type=str, help="Show detail for matching experiment")
    args = parser.parse_args()

    results = load_results()

    if args.detail:
        matches = [r for r in results if args.detail.lower() in r["file"].lower()]
        if not matches:
            print(f"No results matching '{args.detail}'")
            sys.exit(1)
        for m in matches:
            print(json.dumps(m["data"], indent=2)[:2000])
        return

    summaries = [extract_summary(r) for r in results]

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print(f"\nSpectral Demons II — Results Dashboard")
        print(f"{'=' * 100}")
        print(f"{len(results)} result files in {RESULTS_DIR}\n")
        print_table(summaries)
        total_n = sum(s.get("n_measurements", 0) for s in summaries)
        print(f"\nTotal measurements across all experiments: {total_n}")


if __name__ == "__main__":
    main()
