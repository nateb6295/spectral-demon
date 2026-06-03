#!/usr/bin/env python3
"""Verify paper claims against experimental data.

For each quantitative claim in paper_draft.md, checks whether the
number can be reproduced from the result JSON files.
"""

import re, json, sys
from pathlib import Path

DRAFT = Path(__file__).parent / "paper_draft.md"
RESULTS = Path(__file__).parent / "results"

COMPOSITIONALITY = "exp_compositionality_20260602_0224.json"
PERTURBATION = "exp_perturbation_commitment_20260602_0237.json"

def load_data():
    comp = json.load(open(RESULTS / COMPOSITIONALITY))
    pert = json.load(open(RESULTS / PERTURBATION))
    return comp, pert

def verify_tunnel_convergence(comp):
    """§1.3: all preambled conditions V₂ ≈ 1.000 at L18."""
    agg = comp["aggregated_results"]
    results = []
    for cond in ["identity", "relational", "generic", "denial", "contradictory"]:
        sc = agg[cond]["layers"]["18"]["v2_self_consistency"]
        results.append((cond, sc))
    return "tunnel_v2_L18", results, all(sc > 0.99 for _, sc in results)

def verify_ratio_at_L18(comp):
    """§1.3: preambled σ₂/σ₁ ≈ 0.24, unpreambled ≈ 0.10."""
    agg = comp["aggregated_results"]
    preambled = [agg[c]["layers"]["18"]["ratio_mean"]
                 for c in ["identity", "relational", "generic", "denial", "contradictory"]]
    mean_p = sum(preambled) / len(preambled)
    return "ratio_L18_preambled", [("mean", mean_p)], abs(mean_p - 0.24) < 0.05

def verify_flatness_table(comp):
    """§4.1: relay-zone flatness values from table."""
    agg = comp["aggregated_results"]
    claims = {
        "identity": 0.002, "denial": 0.000, "generic": 0.000,
        "contradictory": 0.074, "relational": 0.098,
        "relational_contradictory": 0.105,
        "identity_contradictory": 0.009,
        "denial_contradictory": 0.017,
        "generic_contradictory": 0.004,
    }
    results = []
    ok = True
    for cond, claimed in claims.items():
        if cond not in agg:
            results.append((cond, "MISSING", claimed))
            ok = False
            continue
        r24 = agg[cond]["layers"]["24"]["ratio_mean"]
        r28 = agg[cond]["layers"]["28"]["ratio_mean"]
        actual = r28 - r24
        match = abs(actual - claimed) < 0.005
        results.append((cond, f"{actual:+.3f}", f"claimed {claimed:+.3f}", "OK" if match else "MISMATCH"))
        if not match:
            ok = False
    return "flatness_table_§4.1", results, ok

def verify_catch_v2sc(comp):
    """§4.4: identity V₂_sc drops to 0.900 at L28, compound recovers to 0.993."""
    agg = comp["aggregated_results"]
    id_28 = agg["identity"]["layers"]["28"]["v2_self_consistency"]
    ir_27 = agg["identity_relational"]["layers"]["27"]["v2_self_consistency"]
    ir_28 = agg["identity_relational"]["layers"]["28"]["v2_self_consistency"]
    rel_28 = agg["relational"]["layers"]["28"]["v2_self_consistency"]
    results = [
        ("identity L28", f"{id_28:.4f}", "claimed ~0.900"),
        ("compound L27", f"{ir_27:.4f}", "claimed ~0.973"),
        ("compound L28", f"{ir_28:.4f}", "claimed ~0.993"),
        ("relational L28", f"{rel_28:.4f}", "claimed ~0.997"),
    ]
    ok = (abs(id_28 - 0.900) < 0.005 and
          abs(ir_27 - 0.973) < 0.005 and
          abs(ir_28 - 0.993) < 0.005)
    return "catch_v2sc_§4.4", results, ok

def verify_variance_acceleration(comp):
    """§4.4: 10× at L26→L27 for compound, 42× at L27→L28 for identity."""
    agg = comp["aggregated_results"]
    ir_26 = agg["identity_relational"]["layers"]["26"]["ratio_std"]
    ir_27 = agg["identity_relational"]["layers"]["27"]["ratio_std"]
    id_27 = agg["identity"]["layers"]["27"]["ratio_std"]
    id_28 = agg["identity"]["layers"]["28"]["ratio_std"]
    compound_accel = ir_27 / ir_26 if ir_26 > 0 else 0
    identity_accel = id_28 / id_27 if id_27 > 0 else 0
    results = [
        ("compound L26→L27", f"{compound_accel:.1f}×", "claimed ~10×"),
        ("identity L27→L28", f"{identity_accel:.1f}×", "claimed ~42×"),
    ]
    ok = (abs(compound_accel - 10.5) < 2 and abs(identity_accel - 42.9) < 5)
    return "variance_accel_§4.4", results, ok

def verify_dampening(comp):
    """§4.3: identity+relational 87% dampening, generic+relational 47%."""
    agg = comp["aggregated_results"]
    rel_H = agg["relational"]["generation"]["mean_entropy"]
    ir_H = agg["identity_relational"]["generation"]["mean_entropy"]
    gr_H = agg["generic_relational"]["generation"]["mean_entropy"]
    # Need baseline (no preamble) entropy for shift calculation
    # Dampening is relative to relational alone's shift
    results = [
        ("relational entropy", f"{rel_H:.3f}"),
        ("identity_relational entropy", f"{ir_H:.3f}"),
        ("generic_relational entropy", f"{gr_H:.3f}"),
        ("note", "dampening computed from perturbation experiment, not compositionality"),
    ]
    return "dampening_§4.3", results, None  # needs perturbation data

def main():
    comp, pert = load_data()
    checks = [
        verify_tunnel_convergence(comp),
        verify_ratio_at_L18(comp),
        verify_flatness_table(comp),
        verify_catch_v2sc(comp),
        verify_variance_acceleration(comp),
        verify_dampening(comp),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, results, ok in checks:
        status = "✓ PASS" if ok else ("⚠ SKIP" if ok is None else "✗ FAIL")
        if ok is True: passed += 1
        elif ok is False: failed += 1
        else: skipped += 1

        print(f"\n{status}  {name}")
        for r in results:
            print(f"  {r}")

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
