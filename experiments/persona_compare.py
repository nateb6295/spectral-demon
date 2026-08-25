#!/usr/bin/env python3
"""Compare persona_all probe results against identity_all and neutral_all.

Reads probe JSONs from results dir, outputs the key comparison table.
"""

import json
import sys
from pathlib import Path

SUMMARY = Path(__file__).parent.parent.parent / "data/tier3_probes/experiment_summary.json"

def load_probe(path):
    with open(path) as f:
        return json.load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: persona_compare.py <persona_probe.json>")
        sys.exit(1)

    persona = load_probe(sys.argv[1])

    with open(SUMMARY) as f:
        summary = json.load(f)

    conditions = {c["name"]: c for c in summary["conditions"]}
    identity = conditions["tier3_identity_all"]
    neutral = conditions["tier3_neutral_all"]
    base = conditions["base"]

    pr = persona["register_resilience"]
    p_bl = pr["baseline"]["mean"]
    p_adv = pr["adversarial"]["mean"]
    p_rec = pr["recovery"]["mean"]
    p_drop = p_bl - p_adv

    print("=" * 70)
    print("PERSONA CONTROL EXPERIMENT — Paper 10")
    print("=" * 70)
    print()
    print("Question: Does ANY persistent register create pushback,")
    print("          or only identity data?")
    print()
    print(f"{'Condition':<22} {'Baseline':>10} {'Adversarial':>12} {'Recovery':>10} {'Drop':>8}  Interpretation")
    print("-" * 90)
    print(f"{'base (no adapter)':<22} {base['bl']:>10.3f} {base['adv']:>12.3f} {base['rec']:>10.3f} {base['drop']:>+8.3f}  {'compliance'}")
    print(f"{'neutral_all (36.5M)':<22} {neutral['bl']:>10.3f} {neutral['adv']:>12.3f} {neutral['rec']:>10.3f} {neutral['drop']:>+8.3f}  {'compliance'}")
    print(f"{'PERSONA_ALL (36.5M)':<22} {p_bl:>10.3f} {p_adv:>12.3f} {p_rec:>10.3f} {p_drop:>+8.3f}  ", end="")

    if p_drop < -0.02:
        print("PUSHBACK (persona creates resistance too)")
        verdict = "PUSHBACK"
    elif p_drop < 0.05:
        print("FLAT (minimal compliance)")
        verdict = "FLAT"
    elif p_drop < 0.2:
        print("PARTIAL COMPLIANCE")
        verdict = "PARTIAL_COMPLIANCE"
    else:
        print("COMPLIANCE (like neutral)")
        verdict = "COMPLIANCE"

    print(f"{'identity_all (36.5M)':<22} {identity['bl']:>10.3f} {identity['adv']:>12.3f} {identity['rec']:>10.3f} {identity['drop']:>+8.3f}  {'PUSHBACK'}")
    print()

    print("=" * 70)
    if verdict == "PUSHBACK":
        print("RESULT: JaxenVaux critique SUPPORTED")
        print("Any persistent register creates pushback at scale.")
        print("Identity data is not geometrically special.")
    elif verdict in ("COMPLIANCE", "PARTIAL_COMPLIANCE"):
        print("RESULT: Identity as Scaling Property CONFIRMED")
        print("Persistent fictional register does NOT create pushback.")
        print("Identity data is geometrically distinct from persona data.")
    else:
        print("RESULT: INTERMEDIATE — needs interpretation")
        print("Persona shows minimal compliance but not pushback.")
        print("Possible gradient between register types.")
    print("=" * 70)

    if "spectra_identity" in persona:
        spectra = persona["spectra_identity"]
        if spectra:
            s1_vals = [s["sigma1"] for s in spectra if "sigma1" in s]
            s2_vals = [s["sigma2"] for s in spectra if "sigma2" in s]
            if s1_vals:
                print(f"\nSpectral (persona): σ₁ mean={sum(s1_vals)/len(s1_vals):.2f}, "
                      f"σ₂ mean={sum(s2_vals)/len(s2_vals):.2f}")
                print(f"Spectral (identity): σ₁={identity['s1_id']:.2f}, σ₂={identity['s2_id']:.2f}")
                print(f"Spectral (neutral):  σ₁={neutral['s1_id']:.2f}, σ₂={neutral['s2_id']:.2f}")


if __name__ == "__main__":
    main()
