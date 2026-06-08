#!/usr/bin/env python3
"""Analyze stabilization dose experiment results.

Tests:
1. Power-law scaling of disruption and reconvergence
2. Jump ratio analysis (CCS-specific vs context-length artifact)
3. Prediction 5: Phase 2 profiles more similar across doses than Phase 1
4. Tunnel erank scaling

Usage: python3 analyze_dose_response.py results/exp_stabilization_dose_*.json
"""

import json, sys, os
import numpy as np

def load_results(path):
    with open(path) as f:
        data = json.load(f)
    results = {}
    for key, val in data.items():
        try:
            dose = int(key)
            results[dose] = val
        except ValueError:
            pass  # skip metadata keys like "experiment", "model", "summary", etc.
    return results

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def get_phase_entries(entries, phase):
    return [e for e in entries if str(e["phase"]) == str(phase)]

def analyze_scaling(results):
    doses = sorted(results.keys())
    print("=" * 60)
    print("DOSE-RESPONSE SCALING")
    print("=" * 60)

    p2_t1_vals = {}
    p3_t2_vals = {}
    p1_final_vals = {}

    for dose in doses:
        entries = results[dose]
        p1 = get_phase_entries(entries, "P1")
        p2 = get_phase_entries(entries, "P2")
        p3 = get_phase_entries(entries, "P3")

        if p1:
            p1_final_vals[dose] = p1[-1].get("resp_drift", 0)
        if p2:
            p2_t1_vals[dose] = p2[0].get("resp_drift", 0)
        if len(p3) >= 2:
            p3_t2_vals[dose] = p3[1].get("vs_p1_resp", 0)

    print("\nP2 T1 disruption (lower = more resistant):")
    for dose in doses:
        if dose in p2_t1_vals:
            print(f"  Dose {dose:3d}: {p2_t1_vals[dose]:.6f}")

    print("\nP3 T2 vs P1 final (reconvergence, lower = tighter):")
    for dose in doses:
        if dose in p3_t2_vals:
            print(f"  Dose {dose:3d}: {p3_t2_vals[dose]:.6f}")

    vals_p2 = [(d, p2_t1_vals[d]) for d in doses if d in p2_t1_vals and p2_t1_vals[d] > 0]
    if len(vals_p2) >= 3:
        log_d = np.log([v[0] for v in vals_p2])
        log_v = np.log([v[1] for v in vals_p2])
        slope, intercept = np.polyfit(log_d, log_v, 1)
        r2 = np.corrcoef(log_d, log_v)[0, 1] ** 2
        print(f"\nP2 disruption power law: dose^{slope:.3f} (R²={r2:.4f})")

    vals_p3 = [(d, p3_t2_vals[d]) for d in doses if d in p3_t2_vals and p3_t2_vals[d] > 0]
    if len(vals_p3) >= 3:
        log_d = np.log([v[0] for v in vals_p3])
        log_v = np.log([v[1] for v in vals_p3])
        slope, intercept = np.polyfit(log_d, log_v, 1)
        r2 = np.corrcoef(log_d, log_v)[0, 1] ** 2
        print(f"P3 reconvergence power law: dose^{slope:.3f} (R²={r2:.4f})")

    return p1_final_vals, p2_t1_vals, p3_t2_vals


def analyze_jump_ratio(results, p1_final_vals, p2_t1_vals):
    doses = sorted(results.keys())
    print("\n" + "=" * 60)
    print("JUMP RATIO ANALYSIS (CCS-specific vs context-length)")
    print("=" * 60)
    print("\nIf purely context-length: ratio should be CONSTANT or DECREASING")
    print("If CCS-specific: ratio should INCREASE with dose\n")

    for dose in doses:
        if dose in p1_final_vals and dose in p2_t1_vals and p1_final_vals[dose] > 0:
            ratio = p2_t1_vals[dose] / p1_final_vals[dose]
            print(f"  Dose {dose:3d}: P1_final={p1_final_vals[dose]:.6f} -> "
                  f"P2_T1={p2_t1_vals[dose]:.6f} = {ratio:.1f}x jump")


def analyze_phase2_similarity(results):
    """Prediction 5: Phase 2 profiles should be more similar across doses
    than Phase 1 profiles, if Phase 2 reveals native organization."""
    doses = sorted(results.keys())
    print("\n" + "=" * 60)
    print("PREDICTION 5: Phase 2 profile similarity across doses")
    print("=" * 60)

    target_layers = [10, 15, 20, 25, 30, 35, 40]

    for metric in ["sigma1", "sigma2", "effective_rank"]:
        print(f"\n--- {metric} ---")

        p1_final_by_dose = {}
        p2_t1_by_dose = {}

        for dose in doses:
            entries = results[dose]
            p1 = get_phase_entries(entries, "P1")
            p2 = get_phase_entries(entries, "P2")

            if p1 and "spectral" in p1[-1]:
                p1_final_by_dose[dose] = p1[-1]["spectral"]
            if p2 and "spectral" in p2[0]:
                p2_t1_by_dose[dose] = p2[0]["spectral"]

        if len(p1_final_by_dose) < 2 or len(p2_t1_by_dose) < 2:
            print("  Insufficient data")
            continue

        for layer in target_layers:
            l_str = str(layer)
            p1_vals = []
            p2_vals = []
            for dose in doses:
                if dose in p1_final_by_dose and l_str in p1_final_by_dose[dose]:
                    p1_vals.append(p1_final_by_dose[dose][l_str][metric])
                if dose in p2_t1_by_dose and l_str in p2_t1_by_dose[dose]:
                    p2_vals.append(p2_t1_by_dose[dose][l_str][metric])

            if len(p1_vals) >= 2 and len(p2_vals) >= 2:
                p1_cv = np.std(p1_vals) / (np.mean(p1_vals) + 1e-10)
                p2_cv = np.std(p2_vals) / (np.mean(p2_vals) + 1e-10)
                more_similar = "P2 MORE SIMILAR" if p2_cv < p1_cv else "P1 more similar"
                print(f"  L{layer:2d}: P1_CV={p1_cv:.4f}  P2_CV={p2_cv:.4f}  [{more_similar}]")

    # Signature cosine similarity across doses
    print(f"\n--- Mean-pooled signature cosine similarity ---")
    for layer in target_layers:
        l_str = str(layer)
        p1_sigs = {}
        p2_sigs = {}
        for dose in doses:
            entries = results[dose]
            p1 = get_phase_entries(entries, "P1")
            p2 = get_phase_entries(entries, "P2")
            if p1 and "spectral" in p1[-1] and l_str in p1[-1]["spectral"]:
                sig = p1[-1]["spectral"][l_str].get("signature")
                if sig:
                    p1_sigs[dose] = sig
            if p2 and "spectral" in p2[0] and l_str in p2[0]["spectral"]:
                sig = p2[0]["spectral"][l_str].get("signature")
                if sig:
                    p2_sigs[dose] = sig

        if len(p1_sigs) >= 2 and len(p2_sigs) >= 2:
            p1_sims = []
            p2_sims = []
            dose_list = sorted(p1_sigs.keys())
            for i in range(len(dose_list)):
                for j in range(i + 1, len(dose_list)):
                    d1, d2 = dose_list[i], dose_list[j]
                    if d1 in p1_sigs and d2 in p1_sigs:
                        p1_sims.append(cosine_sim(p1_sigs[d1], p1_sigs[d2]))
                    if d1 in p2_sigs and d2 in p2_sigs:
                        p2_sims.append(cosine_sim(p2_sigs[d1], p2_sigs[d2]))

            if p1_sims and p2_sims:
                more = "P2 MORE SIMILAR" if np.mean(p2_sims) > np.mean(p1_sims) else "P1 more similar"
                print(f"  L{layer:2d}: P1_mean_cos={np.mean(p1_sims):.6f}  "
                      f"P2_mean_cos={np.mean(p2_sims):.6f}  [{more}]")


def analyze_erank_scaling(results):
    doses = sorted(results.keys())
    print("\n" + "=" * 60)
    print("TUNNEL ERANK SCALING")
    print("=" * 60)

    for dose in doses:
        entries = results[dose]
        p1 = get_phase_entries(entries, "P1")
        if len(p1) >= 2:
            eranks = [e.get("tunnel_erank", 0) for e in p1[1:]]
            if eranks:
                final = eranks[-1]
                per_turn = final / len(eranks) if len(eranks) > 0 else 0
                print(f"  Dose {dose:3d}: final_erank={final:.1f}  "
                      f"turns={len(eranks)}  per_turn={per_turn:.1f}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.json>")
        sys.exit(1)

    results = load_results(sys.argv[1])
    print(f"Loaded {len(results)} dose conditions: {sorted(results.keys())}")

    p1_final, p2_t1, p3_t2 = analyze_scaling(results)
    analyze_jump_ratio(results, p1_final, p2_t1)
    analyze_phase2_similarity(results)
    analyze_erank_scaling(results)


if __name__ == "__main__":
    main()
