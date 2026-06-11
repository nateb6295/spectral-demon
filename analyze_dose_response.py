#!/usr/bin/env python3
"""Analyze stabilization dose-response results from JSON."""
import json, sys, argparse
import numpy as np

def load_results(path):
    with open(path) as f:
        data = json.load(f)
    doses = {}
    for k, v in data.items():
        if k.isdigit() and isinstance(v, list):
            doses[int(k)] = v
    return doses

def cfff_pattern(p3_entries):
    vs_p1 = [e.get("vs_p1_resp") for e in p3_entries if e.get("vs_p1_resp") is not None]
    if len(vs_p1) < 2:
        return "?"
    pattern = "C"
    for i in range(1, len(vs_p1)):
        pattern += "C" if vs_p1[i] < vs_p1[i-1] else "F"
    return pattern

def analyze(doses):
    print("=" * 60)
    print("STABILIZATION DOSE-RESPONSE ANALYSIS")
    print("=" * 60)

    summaries = {}
    for dose in sorted(doses.keys()):
        entries = doses[dose]
        p1 = [e for e in entries if e["phase"] == "P1"]
        p2 = [e for e in entries if e["phase"] == "P2"]
        p3 = [e for e in entries if e["phase"] == "P3"]

        p1_final = [e.get("resp_drift", 0) for e in p1[-3:] if e.get("resp_drift") is not None]
        p1_mean = np.mean(p1_final) if p1_final else 0
        p2_t1 = p2[0].get("resp_drift", 0) if p2 else 0
        p3_vs_p1 = [e.get("vs_p1_resp", 0) for e in p3 if e.get("vs_p1_resp") is not None]
        p3_final_drift = p3[-1].get("resp_drift", 0) if p3 else 0
        p1_final_drift = p1[-1].get("resp_drift", 0) if p1 else 0
        dsi = p1_final_drift / p3_final_drift if p3_final_drift > 0 else float("inf")
        jump = p2_t1 / p1_mean if p1_mean > 0 else 0
        pattern = cfff_pattern(p3)

        summaries[dose] = {
            "p1_mean": p1_mean, "p2_t1": p2_t1, "jump": jump,
            "dsi": dsi, "pattern": pattern, "p3_vs_p1": p3_vs_p1,
        }

        print("\nDose %d:" % dose)
        print("  P1 final drift: %.6f  |  P2 T1 disruption: %.6f" % (p1_mean, p2_t1))
        print("  Jump ratio: %.1fx  |  DSI: %.1f" % (jump, dsi))
        print("  P3 vs P1: %s" % ["%.6f" % v for v in p3_vs_p1])
        print("  CFFF pattern: %s" % pattern)

    # Power-law
    valid_doses = sorted(doses.keys())
    if len(valid_doses) >= 3:
        p2_vals = [(d, summaries[d]["p2_t1"]) for d in valid_doses if summaries[d]["p2_t1"] > 0]
        if len(p2_vals) >= 3:
            log_d = np.log([v[0] for v in p2_vals])
            log_v = np.log([v[1] for v in p2_vals])
            slope, intercept = np.polyfit(log_d, log_v, 1)
            r2 = np.corrcoef(log_d, log_v)[0, 1] ** 2
            print("\n" + "=" * 60)
            print("POWER-LAW: P2 disruption ~ dose^%.3f (R²=%.4f)" % (slope, r2))
            for pred_d in [50, 100]:
                pred = np.exp(intercept + slope * np.log(pred_d))
                print("  Predicted dose %d P2 T1: %.6f" % (pred_d, pred))

    # Integration timescale (zone analysis requires spectral data)
    zones = {"tunnel": range(0, 23), "responsive": range(23, 38), "relay": range(38, 46)}
    has_spectral = any("spectral" in e for d in doses.values() for e in d)

    if has_spectral and len(valid_doses) >= 3:
        print("\n" + "=" * 60)
        print("PREDICTION 5: ZONE-SPECIFIC ANALYSIS")
        for zone_name, layer_range in zones.items():
            p2_by_dose = {}
            p1_by_dose = {}
            for dose in valid_doses:
                entries = doses[dose]
                p2_entries = [e for e in entries if e["phase"] == "P2" and "spectral" in e]
                p1_spec = [e for e in entries if e["phase"] == "P1" and "spectral" in e]
                p2_ranks = []
                for p2e in p2_entries:
                    for l in layer_range:
                        if str(l) in p2e["spectral"]:
                            p2_ranks.append(p2e["spectral"][str(l)]["effective_rank"])
                p1_ranks = []
                if p1_spec:
                    last = p1_spec[-1]
                    for l in layer_range:
                        if str(l) in last["spectral"]:
                            p1_ranks.append(last["spectral"][str(l)]["effective_rank"])
                p2_by_dose[dose] = np.mean(p2_ranks) if p2_ranks else 0
                p1_by_dose[dose] = np.mean(p1_ranks) if p1_ranks else 0

            p2_vals = [v for v in p2_by_dose.values() if v > 0]
            p1_vals = [v for v in p1_by_dose.values() if v > 0]
            p2_cv = np.std(p2_vals) / np.mean(p2_vals) * 100 if p2_vals and np.mean(p2_vals) > 0 else 0
            p1_cv = np.std(p1_vals) / np.mean(p1_vals) * 100 if p1_vals and np.mean(p1_vals) > 0 else 0
            result = "CONFIRMED" if p2_cv < p1_cv else "REVERSED"
            print("  %s: P1 CV=%.1f%%  P2 CV=%.1f%%  -> %s" % (zone_name, p1_cv, p2_cv, result))

    # Integration timescale
    if len(valid_doses) >= 3 and has_spectral:
        shifts = []
        for dose in valid_doses:
            entries = doses[dose]
            p1_spec = [e for e in entries if e["phase"] == "P1" and "spectral" in e]
            p2_entries = [e for e in entries if e["phase"] == "P2" and "spectral" in e]
            if not p1_spec or not p2_entries:
                continue
            last_p1 = p1_spec[-1]
            tunnel_range = range(0, 23)
            p1_er = np.mean([last_p1["spectral"][str(l)]["effective_rank"]
                           for l in tunnel_range if str(l) in last_p1["spectral"]])
            p2_er = np.mean([np.mean([p2e["spectral"][str(l)]["effective_rank"]
                           for l in tunnel_range if str(l) in p2e["spectral"]])
                           for p2e in p2_entries])
            shift = 100 * (p2_er - p1_er) / p1_er if p1_er > 0 else 0
            shifts.append((dose, shift))

        if len(shifts) >= 3:
            log_d = np.log([s[0] for s in shifts])
            log_s = np.log([s[1] for s in shifts])
            slope, intercept = np.polyfit(log_d, log_s, 1)
            r2 = np.corrcoef(log_d, log_s)[0, 1] ** 2
            dose_1pct = np.exp((np.log(1.0) - intercept) / slope)
            print("\n" + "=" * 60)
            print("INTEGRATION TIMESCALE (tunnel shift ~ dose^%.2f, R²=%.4f)" % (slope, r2))
            print("  Dose for 1%% shift: %.0f turns" % dose_1pct)
            for s in shifts:
                print("  Dose %d: %.1f%% shift" % s)

    return summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dose-response results")
    parser.add_argument("path", help="Path to results JSON")
    args = parser.parse_args()
    doses = load_results(args.path)
    if not doses:
        print("No dose data found in %s" % args.path)
        sys.exit(1)
    analyze(doses)
