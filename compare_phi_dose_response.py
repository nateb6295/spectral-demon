#!/usr/bin/env python3
"""Phi P15 dose-response curve: P2 disruption vs CCS dose."""

import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent / "results"

DOSE_FILES = {
    1: "exp_selfref_vs_relational_phi_dose1_20260611_1215.json",
    3: "exp_selfref_vs_relational_phi_dose3_20260611_1223.json",
    5: "exp_selfref_vs_relational_phi_dose5_20260611_1233.json",
    10: "exp_selfref_vs_relational_phi_dose10_20260611_1252.json",
    20: "exp_selfref_vs_relational_phi_20260611_1202.json",
}

def find_dose_files():
    """Auto-discover dose sweep files for Phi."""
    results = {}
    for f in sorted(BASE.glob("exp_selfref_vs_relational_*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
            if "Phi" in d.get("model", "") or "phi" in d.get("model", "").lower():
                dose = d.get("dose", 0)
                results[dose] = f.name
        except Exception:
            continue
    return results

def main():
    dose_files = find_dose_files()
    if not dose_files:
        dose_files = {k: v for k, v in DOSE_FILES.items() if v}

    print("=" * 60)
    print("PHI P15 DOSE-RESPONSE CURVE")
    print("=" * 60)

    doses = sorted(dose_files.keys())
    rows = []

    for dose in doses:
        path = BASE / dose_files[dose]
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)

        s = d["summary"]
        r = s["relational"]
        sr = s["self_ref"]

        ratio = sr["p2_disruption"] / r["p2_disruption"] if r["p2_disruption"] > 0 else 0

        rows.append({
            "dose": dose,
            "p2_r": r["p2_disruption"],
            "p2_s": sr["p2_disruption"],
            "ratio": ratio,
            "conv_r": r["convergence_ratio"],
            "vs_p1_r": r["p3_vs_p1"],
            "vs_p1_s": sr["p3_vs_p1"],
            "l27_r": r["l27_spike_ratio"],
        })

    print(f"\n  {'Dose':<6} {'P2(R)':<12} {'P2(S)':<12} {'S/R':<8} {'Conv(R)':<10} {'vsP1(R)':<12} {'L27(R)':<8}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*8} {'─'*10} {'─'*12} {'─'*8}")
    for row in rows:
        print(f"  {row['dose']:<6} {row['p2_r']:<12.6f} {row['p2_s']:<12.6f} {row['ratio']:<8.3f} "
              f"{row['conv_r']:<10.4f} {row['vs_p1_r']:<12.6f} {row['l27_r']:<8.3f}")

    if len(rows) >= 2:
        p2_vals = [r["p2_r"] for r in rows]
        dose_vals = [r["dose"] for r in rows]
        print(f"\n  P2 disruption range: {min(p2_vals):.6f} → {max(p2_vals):.6f} ({max(p2_vals)/min(p2_vals):.1f}×)")

        # Check for inverted-U
        if len(rows) >= 3:
            peak_idx = np.argmax(p2_vals)
            if 0 < peak_idx < len(rows) - 1:
                print(f"  INVERTED-U detected: peak at dose {dose_vals[peak_idx]}")
            elif peak_idx == len(rows) - 1:
                print(f"  MONOTONIC increase: no peak yet (highest at dose {dose_vals[peak_idx]})")
            else:
                print(f"  IMMEDIATE peak at dose {dose_vals[peak_idx]}")

        # S/R ratio trend
        ratios = [r["ratio"] for r in rows]
        print(f"\n  S/R ratio trend: {' → '.join(f'{r:.3f}' for r in ratios)}")
        if ratios[0] > 0.95 and ratios[-1] < 0.95:
            print(f"  Self-ref/relational distinction EMERGES with dose (starts identical, diverges)")
        elif all(r < 0.95 for r in ratios):
            print(f"  Self-ref consistently less disrupted across all doses")

if __name__ == "__main__":
    main()
