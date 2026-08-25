#!/usr/bin/env python3
"""Filter shadow probe: attenuation-angular coupling per species and dose.

Tests whether angular drift in σ₂ is predicted by local energy change
(filter shadow) or independent of it (active redirection).

From Aug 4 thread #316 — Kimi's filter shadow critique + crossover finding:
  Sorters at D2: r=-0.75 (filter shadow). D10: decouples.
  Relays at D2: no coupling. D10: r=-0.77 (filter emerges).
  Tunnels: no coupling at any dose.

Usage:
  python3 filter_shadow_probe.py                    # all species, all doses
  python3 filter_shadow_probe.py --species gemma    # single species
  python3 filter_shadow_probe.py --dose D2 D10      # specific doses
  python3 filter_shadow_probe.py --json              # machine-readable output
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("EXP_K_DIR", os.path.expanduser("~/chronicle/data/exp_k")))

SPECIES_FILES = {
    "gemma": ("exp_k_gemma.json", "sorter"),
    "mistral": ("exp_k_mistral.json", "relay"),
    "llama": ("exp_k_llama.json", "relay"),
    "pythia": ("exp_k_pythia.json", "tunnel"),
    "qwen": ("exp_k_qwen.json", "absorber"),
}


def load_species(name):
    fname, stype = SPECIES_FILES[name]
    path = DATA_DIR / fname
    if not path.exists():
        return None, stype
    with open(path) as f:
        return json.load(f), stype


def compute_coupling(dose_data):
    layers = dose_data["per_layer"]
    attenuation = []
    angular = []

    for lyr in layers:
        s2_d0 = lyr["sigma2_d0"]
        s2_dx = lyr["sigma2_dx"]
        if s2_d0 > 0:
            attenuation.append(1 - s2_dx / s2_d0)
            angular.append(lyr["angular_raw"])

    if len(attenuation) < 4:
        return None

    attenuation = np.array(attenuation)
    angular = np.array(angular)
    n = len(attenuation)

    r_global = float(np.corrcoef(attenuation, angular)[0, 1])

    zones = {}
    zone_bounds = {
        "entry": (0, n // 4),
        "mid": (n // 4, 3 * n // 4),
        "exit": (3 * n // 4, n),
    }
    for zone, (s, e) in zone_bounds.items():
        if e - s > 2:
            zr = float(np.corrcoef(attenuation[s:e], angular[s:e])[0, 1])
            zones[zone] = round(zr, 3)

    return {
        "r_global": round(r_global, 3),
        "is_filter": abs(r_global) > 0.5,
        "n_layers": n,
        "mean_attenuation": round(float(np.mean(attenuation)), 3),
        "mean_angular_deg": round(float(np.degrees(np.mean(angular))), 1),
        "zones": zones,
    }


def run_probe(species_list=None, dose_list=None):
    if species_list is None:
        species_list = list(SPECIES_FILES.keys())

    results = {}
    for name in species_list:
        data, stype = load_species(name)
        if data is None:
            continue

        doses = data.get("doses", [])
        species_results = {"type": stype, "doses": {}}

        for dose_entry in doses:
            dlabel = dose_entry["dose"]
            if dose_list and dlabel not in dose_list:
                continue

            coupling = compute_coupling(dose_entry)
            if coupling:
                species_results["doses"][dlabel] = coupling

        if species_results["doses"]:
            results[name] = species_results

    return results


def format_table(results):
    lines = []
    lines.append(f"{'Species':12s} {'Type':8s} {'Dose':4s}  {'r':>7s}  {'Filter?':>7s}  {'Entry':>6s}  {'Mid':>6s}  {'Exit':>6s}  {'Angle°':>6s}")
    lines.append("-" * 80)

    for name, sdata in sorted(results.items()):
        for dlabel, coupling in sorted(sdata["doses"].items()):
            zones = coupling["zones"]
            verdict = "YES" if coupling["is_filter"] else "no"
            entry = f"{zones.get('entry', 0):+.2f}" if "entry" in zones else "  —"
            mid = f"{zones.get('mid', 0):+.2f}" if "mid" in zones else "  —"
            exit_ = f"{zones.get('exit', 0):+.2f}" if "exit" in zones else "  —"
            lines.append(
                f"{name:12s} {sdata['type']:8s} {dlabel:4s}  "
                f"{coupling['r_global']:+.3f}  {verdict:>7s}  "
                f"{entry:>6s}  {mid:>6s}  {exit_:>6s}  "
                f"{coupling['mean_angular_deg']:>6.1f}"
            )

    return "\n".join(lines)


def format_crossover(results):
    lines = ["\nCrossover analysis:"]
    for name, sdata in sorted(results.items()):
        doses = sdata["doses"]
        if "D2" in doses and "D10" in doses:
            r2 = doses["D2"]["r_global"]
            r10 = doses["D10"]["r_global"]
            if (abs(r2) > 0.5) != (abs(r10) > 0.5):
                d2_label = "FILTER" if abs(r2) > 0.5 else "ACTIVE"
                d10_label = "FILTER" if abs(r10) > 0.5 else "ACTIVE"
                lines.append(f"  {name} ({sdata['type']}): D2={d2_label} (r={r2:+.3f}) → D10={d10_label} (r={r10:+.3f})  ** CROSSOVER **")
            else:
                label = "FILTER" if abs(r2) > 0.5 else "ACTIVE"
                lines.append(f"  {name} ({sdata['type']}): D2→D10 both {label} (r={r2:+.3f} → {r10:+.3f})")

    return "\n".join(lines) if len(lines) > 1 else ""


def main():
    parser = argparse.ArgumentParser(description="Filter shadow probe")
    parser.add_argument("--species", nargs="+", choices=list(SPECIES_FILES.keys()))
    parser.add_argument("--dose", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_probe(args.species, args.dose)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_table(results))
        crossover = format_crossover(results)
        if crossover:
            print(crossover)


if __name__ == "__main__":
    main()
