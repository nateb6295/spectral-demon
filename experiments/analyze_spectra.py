#!/usr/bin/env python3
"""Analyze saved spectra from convergence_v2 runs with --save-spectra.

Computes effective rank profiles across the layer stack to test the
Kolmogorov compression prediction: tunnel exit should show low effective
rank (compression), responsive zone should show recovery (reconstruction),
and the tunnel profile should be preamble-invariant.

Usage:
  python3 analyze_spectra.py results/exp_convergence_v2_*.json
  python3 analyze_spectra.py --compare-modes  # compare Weil conditions
"""

import json, glob, sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_spectra(paths):
    runs = []
    for p in paths:
        p = Path(p)
        with open(p) as f:
            data = json.load(f)
        for mname, mdata in data.get("models", {}).items():
            entries = mdata.get("entries", [])
            has_spectra = any(
                any("spectrum" in (e.get(str(l), {}) if isinstance(e.get(str(l)), dict) else {})
                    for l in range(100))
                for e in entries
            )
            for entry in entries:
                eranks_by_layer = {}
                for l_key in sorted(entry.keys()):
                    if l_key.isdigit() or (isinstance(entry.get(l_key), dict) and "effective_rank" in entry.get(l_key, {})):
                        continue
                if entry.get("tunnel_erank") is not None:
                    runs.append({
                        "file": p.name,
                        "model": mname,
                        "phase": entry.get("phase", "?"),
                        "turn": entry.get("turn", 0),
                        "phase2_mode": mdata.get("phase2_mode", "vanilla"),
                        "tunnel_erank": entry.get("tunnel_erank"),
                        "resp_erank": entry.get("resp_erank"),
                        "relay_erank": entry.get("relay_erank"),
                    })
    return runs


def print_erank_profile(runs):
    by_model = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)

    for model, model_runs in sorted(by_model.items()):
        print(f"\n{'='*50}")
        print(f"  {model}")
        print(f"{'='*50}")

        by_phase = {}
        for r in model_runs:
            by_phase.setdefault(r["phase"], []).append(r)

        for phase in ["ccs_first", "vanilla", "ccs_reapply", "novel"]:
            if phase not in by_phase:
                continue
            pr = by_phase[phase]
            t_er = np.mean([r["tunnel_erank"] for r in pr if r["tunnel_erank"]])
            r_er = np.mean([r["resp_erank"] for r in pr if r["resp_erank"]])
            l_er = np.mean([r["relay_erank"] for r in pr if r["relay_erank"]])
            label = {"ccs_first": "CCS", "vanilla": "Phase2", "ccs_reapply": "Reapply", "novel": "Novel"}
            print(f"  {label.get(phase, phase):8s}: tunnel={t_er:5.1f}  resp={r_er:5.1f}  relay={l_er:5.1f}  (n={len(pr)})")

        # Kolmogorov test: tunnel erank should be stable across phases
        phases_present = [p for p in ["ccs_first", "vanilla", "ccs_reapply"] if p in by_phase]
        if len(phases_present) >= 2:
            tunnel_eranks = {p: np.mean([r["tunnel_erank"] for r in by_phase[p] if r["tunnel_erank"]]) for p in phases_present}
            vals = list(tunnel_eranks.values())
            cv = np.std(vals) / np.mean(vals) * 100 if np.mean(vals) > 0 else 0
            print(f"\n  Kolmogorov test: tunnel erank CV across phases = {cv:.1f}%")
            if cv < 5:
                print(f"  → SUPPORTED: tunnel compression preamble-invariant (CV<5%)")
            elif cv < 15:
                print(f"  → WEAK: tunnel mostly stable but some preamble effect (5%<CV<15%)")
            else:
                print(f"  → FALSIFIED: tunnel compression is preamble-dependent (CV>{cv:.0f}%)")

        # Compression gradient: tunnel < resp < relay?
        if "ccs_first" in by_phase:
            pr = by_phase["ccs_first"]
            t = np.mean([r["tunnel_erank"] for r in pr if r["tunnel_erank"]])
            r = np.mean([r["resp_erank"] for r in pr if r["resp_erank"]])
            l = np.mean([r["relay_erank"] for r in pr if r["relay_erank"]])
            if t < r:
                print(f"  Compression gradient: tunnel({t:.1f}) < resp({r:.1f}) → tunnel compresses")
            else:
                print(f"  No compression gradient: tunnel({t:.1f}) >= resp({r:.1f})")


def compare_modes(runs):
    by_mode = {}
    for r in runs:
        by_mode.setdefault(r["phase2_mode"], []).append(r)

    if len(by_mode) < 2:
        print("Only one mode found — need Weil experiment data for comparison.")
        return

    print(f"\n{'='*50}")
    print(f"  Weil Attention × Kolmogorov")
    print(f"{'='*50}")

    for mode in ["vanilla", "silent", "structured"]:
        if mode not in by_mode:
            continue
        mr = by_mode[mode]
        p2_runs = [r for r in mr if r["phase"] == "vanilla"]
        if not p2_runs:
            continue
        t_er = np.mean([r["tunnel_erank"] for r in p2_runs if r["tunnel_erank"]])
        r_er = np.mean([r["resp_erank"] for r in p2_runs if r["resp_erank"]])
        print(f"  {mode:12s}: tunnel_erank={t_er:.1f}  resp_erank={r_er:.1f}  (n={len(p2_runs)})")

    tunnel_by_mode = {}
    for mode in by_mode:
        p2 = [r for r in by_mode[mode] if r["phase"] == "vanilla" and r["tunnel_erank"]]
        if p2:
            tunnel_by_mode[mode] = np.mean([r["tunnel_erank"] for r in p2])

    if len(tunnel_by_mode) >= 2:
        vals = list(tunnel_by_mode.values())
        cv = np.std(vals) / np.mean(vals) * 100 if np.mean(vals) > 0 else 0
        print(f"\n  Cross-mode tunnel CV = {cv:.1f}%")
        if cv < 5:
            print(f"  → Kolmogorov confirmed: attention quality doesn't change compression")
        else:
            print(f"  → Interesting: attention quality DOES affect compression depth")


if __name__ == "__main__":
    do_compare = "--compare-modes" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        paths = sorted(RESULTS_DIR.glob("exp_convergence_v2_*.json"))

    runs = load_spectra(paths if paths else [])
    if not runs:
        print("No effective rank data found. Run convergence_v2 first (erank computed automatically).")
        sys.exit(1)

    print_erank_profile(runs)
    if do_compare:
        compare_modes(runs)
