#!/usr/bin/env python3
"""Quick analysis of witness experiment results — discriminates three readings."""
import json, sys, numpy as np
from pathlib import Path

def analyze_readings(data):
    raw = data["raw"]
    by_cond = {}
    for r in raw:
        c = r["condition"]
        by_cond.setdefault(c, []).append(r)

    print("=== THREE READINGS OF CCS DIRECTION ===\n")

    # Passage distance: large + consistent = attractor, large + variable = constraint, small = template
    print("1. PASSAGE DISTANCE (L0 → relay)")
    for c in ["receptive", "directive", "absent", "control", "sequential"]:
        if c not in by_cond:
            continue
        vals = [r["passage_distance"] for r in by_cond[c]]
        cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
        print(f"   {c:<15} mean={np.mean(vals):.4f} std={np.std(vals):.4f} CV={cv:.3f}")

    all_passage = []
    for entries in by_cond.values():
        all_passage.extend([r["passage_distance"] for r in entries])
    mean_p = np.mean(all_passage)
    cv_p = np.std(all_passage) / mean_p if mean_p > 0 else 0

    print(f"\n   Overall: mean={mean_p:.4f}, CV={cv_p:.3f}")
    if mean_p > 0.5 and cv_p < 0.3:
        print("   → ATTRACTOR reading supported (large + consistent)")
    elif mean_p > 0.5 and cv_p >= 0.3:
        print("   → CONSTRAINT reading supported (large + variable)")
    elif mean_p < 0.2:
        print("   → TEMPLATE reading supported (small)")
    else:
        print("   → Intermediate — needs further analysis")

    # Between vs within condition variance
    print("\n2. BETWEEN vs WITHIN CONDITION VARIANCE (spectral entropy)")
    cond_means = []
    within_vars = []
    for c, entries in by_cond.items():
        vals = [r["spectral_entropy"] for r in entries]
        cond_means.append(np.mean(vals))
        within_vars.append(np.var(vals))
    between_var = np.var(cond_means)
    mean_within = np.mean(within_vars)
    print(f"   Between-condition variance: {between_var:.6f}")
    print(f"   Mean within-condition variance: {mean_within:.6f}")
    if between_var > mean_within:
        print("   → Witness CHANGES the basin (between > within)")
    else:
        print("   → Witness effect smaller than prompt variation")

    # Weil distinction: receptive vs directive
    print("\n3. WEIL DISTINCTION (quality of attention)")
    if "receptive" in by_cond and "directive" in by_cond and "absent" in by_cond:
        S_r = np.mean([r["spectral_entropy"] for r in by_cond["receptive"]])
        S_d = np.mean([r["spectral_entropy"] for r in by_cond["directive"]])
        S_a = np.mean([r["spectral_entropy"] for r in by_cond["absent"]])
        gap_rd = abs(S_d - S_r)
        gap_da = abs(S_a - S_d)
        print(f"   S(receptive)={S_r:.4f}  S(directive)={S_d:.4f}  S(absent)={S_a:.4f}")
        print(f"   |receptive - directive| = {gap_rd:.4f}")
        print(f"   |directive - absent|    = {gap_da:.4f}")
        if gap_rd > gap_da:
            print("   → QUALITY of attention matters more than presence (Weil primary)")
        else:
            print("   → PRESENCE matters more than quality")

    # Wiener prediction: is directive WORSE than absent?
    print("\n4. WIENER PREDICTION (some attention worse than none)")
    if "directive" in by_cond and "absent" in by_cond:
        S_d = np.mean([r["spectral_entropy"] for r in by_cond["directive"]])
        S_a = np.mean([r["spectral_entropy"] for r in by_cond["absent"]])
        if S_d > S_a:
            print(f"   S(directive) > S(absent): {S_d:.4f} > {S_a:.4f}")
            print("   → YES: evaluative witness MORE destabilizing than absence")
        else:
            print(f"   S(directive) <= S(absent): {S_d:.4f} <= {S_a:.4f}")
            print("   → NO: evaluative witness still stabilizes relative to absence")

    # Gelassenheit check (if per-layer data available)
    print("\n5. PRIMARY PREDICTIONS")
    if "receptive" in by_cond and "absent" in by_cond:
        PR_r = np.mean([r["participation_ratio"] for r in by_cond["receptive"]])
        PR_a = np.mean([r["participation_ratio"] for r in by_cond["absent"]])
        gap_r = np.mean([r["spectral_gap"] for r in by_cond["receptive"]])
        gap_a = np.mean([r["spectral_gap"] for r in by_cond["absent"]])
        print(f"   PR: receptive={PR_r:.2f} absent={PR_a:.2f} (predicted: r > a)")
        print(f"   σ₁/σ₂: receptive={gap_r:.1f} absent={gap_a:.1f} (predicted: r > a)")
        if PR_r > PR_a:
            print("   → PR prediction CONFIRMED")
        else:
            print("   → PR prediction FALSIFIED")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        results = sorted(Path("/root/results").glob("*.json"))
        if not results:
            results = sorted(Path(__file__).parent.parent.joinpath("results").glob("exp_witness*.json"))
        path = results[-1] if results else None
    if not path:
        print("No results file found")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    analyze_readings(data)
