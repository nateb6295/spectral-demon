#!/usr/bin/env python3
"""
Zone Quality Profiler — F533-F536 analysis tool.

Analyzes per-layer spectral data to compute:
- σ₁ suppression (F533): demon selectivity = how well architecture blocks σ₁
- Zone persistence (F535): how far the zone survives across the layer stack
- Zone shape classification: SPIKE/RAMP/BELL/FLAT/IMPULSE
- Zone quality taxonomy (F536): STRONG/TRUNCATED/RIGID_ROD/DEAF

Usage:
  python3 zone_quality_profiler.py                    # analyze decisive results
  python3 zone_quality_profiler.py path/to/data.json  # analyze custom data
  python3 zone_quality_profiler.py --ascii             # include ASCII zone profiles
  python3 zone_quality_profiler.py --dose              # show dose-response failure modes (F538)

Input format: JSON with per-model entries containing doses -> D3_therapeutic -> [{proj_s1, proj_s2, ratio_s2_s1, s1_drift_deg, d_norm}, ...]
"""

import json
import sys
import os
import statistics


def analyze_model(name, layers, dose_key="D3_therapeutic"):
    n = len(layers)
    if n == 0:
        return None

    ratios = [l["ratio_s2_s1"] for l in layers]
    proj_s1s = [l["proj_s1"] for l in layers]
    proj_s2s = [l["proj_s2"] for l in layers]
    drifts = [l.get("s1_drift_deg", 0) for l in layers]

    early_max_s1 = max(proj_s1s[:min(6, n)])
    peak_idx = max(range(n), key=lambda i: ratios[i])
    peak = layers[peak_idx]

    suppression = 1.0 - (peak["proj_s1"] / early_max_s1) if early_max_s1 > 0 else 0

    zone_layers = [i for i, r in enumerate(ratios) if r > 1.0]
    late_start = 2 * n // 3
    late_zone = any(i >= late_start for i in zone_layers)
    ws_zone_frac = sum(1 for i in zone_layers if i >= late_start) / max(n - late_start, 1)

    contiguous = []
    start = None
    for i, r in enumerate(ratios):
        if r > 1.0:
            if start is None:
                start = i
        else:
            if start is not None:
                contiguous.append((start, i - 1))
                start = None
    if start is not None:
        contiguous.append((start, n - 1))

    longest = max(contiguous, key=lambda z: z[1] - z[0] + 1) if contiguous else None

    # Half-life: layers after peak where ratio stays above 50% of peak
    half_life = 0
    for i in range(peak_idx + 1, n):
        if ratios[i] >= ratios[peak_idx] * 0.5:
            half_life = i - peak_idx
        else:
            break

    # Zone shape classification
    if not zone_layers:
        shape = "NONE"
    elif len(zone_layers) <= 3 and peak_idx < n // 3:
        shape = "IMPULSE"
    elif suppression < 0.5:
        shape = "FLAT"
    else:
        after_peak = ratios[peak_idx + 1:peak_idx + 4] if peak_idx < n - 3 else []
        decay = (ratios[peak_idx] - statistics.mean(after_peak)) / ratios[peak_idx] if after_peak and ratios[peak_idx] > 0 else 0
        if decay > 0.8:
            before_peak = ratios[max(0, peak_idx - 3):peak_idx]
            buildup = (ratios[peak_idx] - statistics.mean(before_peak)) / ratios[peak_idx] if before_peak and ratios[peak_idx] > 0 else 0
            if buildup > 0.7:
                shape = "SPIKE"
            else:
                shape = "RAMP"
        elif decay > 0.5:
            shape = "BELL"
        else:
            shape = "RAMP"

    # Quality taxonomy (F536)
    if suppression > 0.85 and late_zone:
        quality = "STRONG"
    elif suppression > 0.85 and not late_zone:
        quality = "TRUNCATED"
    elif suppression < 0.5 and zone_layers:
        quality = "RIGID_ROD"
    elif not zone_layers:
        quality = "DEAF"
    else:
        quality = "MODERATE"

    return {
        "name": name,
        "n_layers": n,
        "peak_layer": peak_idx,
        "peak_ratio": ratios[peak_idx],
        "peak_proj_s1": peak["proj_s1"],
        "peak_proj_s2": peak["proj_s2"],
        "early_max_s1": early_max_s1,
        "suppression": suppression,
        "zone_layers": zone_layers,
        "zone_count": len(zone_layers),
        "longest_span": longest,
        "half_life": half_life,
        "late_zone": late_zone,
        "ws_coverage": ws_zone_frac,
        "shape": shape,
        "quality": quality,
        "avg_drift": statistics.mean(drifts),
        "max_drift": max(drifts),
        "ratios": ratios,
    }


def print_summary(results):
    print("=" * 85)
    print("ZONE QUALITY PROFILE — F533-F536 Analysis")
    print("=" * 85)
    print(f"{'Model':>22} {'Peak':>6} {'σ₂/σ₁':>7} {'σ₁ supp':>8} {'Shape':>8} {'Quality':>11} {'WS cov':>7} {'Drift°':>7}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:>22} L{r['peak_layer']:>3}  {r['peak_ratio']:>7.2f} {r['suppression']:>7.1%}  {r['shape']:>8} {r['quality']:>11} {r['ws_coverage']:>6.0%} {r['avg_drift']:>7.1f}")

    print(f"\n{'─'*85}")
    print("F533: σ₁ suppression = 1 - (proj_σ₁ at peak / max proj_σ₁ in early layers)")
    print("F534: RIGID_ROD = non-selective demon (<50% suppression)")
    print("F535: Zone persistence requires position encoding coverage for late-layer survival")
    print("F536: Quality taxonomy: STRONG | MODERATE | TRUNCATED | RIGID_ROD | DEAF")


def print_ascii(results):
    for r in results:
        print(f"\n  {r['name']} ({r['n_layers']} layers) — {r['quality']} / {r['shape']}")
        max_r = max(r["ratios"]) if r["ratios"] else 1
        ws_start = 2 * r["n_layers"] // 3
        for i, ratio in enumerate(r["ratios"]):
            bar = "█" * int(ratio / max_r * 25) if max_r > 0 else ""
            marks = []
            if i == r["peak_layer"]:
                marks.append("◄peak")
            if i >= ws_start:
                marks.append("[ws]")
            if ratio > 1.0:
                marks.append("●")
            suffix = " " + " ".join(marks) if marks else ""
            print(f"    L{i:>2}: {bar:<25} {ratio:.2f}{suffix}")


def analyze_dose_response(name, doses_dict):
    dose_keys = [("D3_therapeutic", "D3"), ("D7_labeled", "D7"), ("D10_full", "D10")]
    dose_results = {}
    for dk, label in dose_keys:
        if dk in doses_dict:
            r = analyze_model(name, doses_dict[dk], dk)
            if r:
                dose_results[label] = r

    if len(dose_results) < 2:
        return None

    d3 = dose_results.get("D3")
    d7 = dose_results.get("D7")
    d10 = dose_results.get("D10")

    if not d3:
        return None

    moderate = lambda r: sum(1 for x in r["ratios"] if 1.0 < x <= 5.0)
    extreme = lambda r: sum(1 for x in r["ratios"] if x > 5.0)

    d3_zone = d3["zone_count"]
    d7_zone = d7["zone_count"] if d7 else 0
    d10_zone = d10["zone_count"] if d10 else 0

    d3_meaningful = d3["peak_ratio"] > 2.0
    has_extreme_d10 = d10 and max(d10["ratios"]) > 50
    d7_collapsed = d7 and d7_zone < 2
    peak_collapsed = d10 and d3["peak_ratio"] > 5.0 and d10["peak_ratio"] < d3["peak_ratio"] * 0.5

    if not d3_meaningful:
        failure_mode = "NO_DEMON"
    elif d7_collapsed:
        failure_mode = "SWITCHOFF"
    elif has_extreme_d10:
        failure_mode = "SPECTRAL_COLLAPSE"
    elif d10 and (d10_zone < d3_zone * 0.6 or peak_collapsed):
        failure_mode = "DEGRADATION"
    else:
        failure_mode = "ROBUST"

    return {
        "name": name,
        "doses": dose_results,
        "failure_mode": failure_mode,
        "d3_zone": d3_zone,
        "d7_zone": d7_zone,
        "d10_zone": d10_zone,
    }


def print_dose_response(dr_results):
    print(f"\n{'='*85}")
    print("DOSE-RESPONSE FAILURE MODES — F538 Analysis")
    print("=" * 85)
    print(f"{'Model':>22} {'D3 zone':>8} {'D7 zone':>8} {'D10 zone':>9} {'D3 supp':>9} {'D10 supp':>9} {'Failure mode':>18}")
    print("-" * 85)
    for dr in dr_results:
        d3 = dr["doses"].get("D3", {})
        d10 = dr["doses"].get("D10", {})
        d3s = f"{d3.get('suppression', 0):.0%}" if d3 else "—"
        d10s = f"{d10.get('suppression', 0):.0%}" if d10 else "—"
        print(f"{dr['name']:>22} {dr['d3_zone']:>8} {dr['d7_zone']:>8} {dr['d10_zone']:>9} {d3s:>9} {d10s:>9} {dr['failure_mode']:>18}")

    print(f"\n{'─'*85}")
    print("SWITCHOFF: D3 zone → D7 zero (binary, e.g. Falcon parallel-residual)")
    print("DEGRADATION: D10 zone < 60% of D3 zone (gradual collapse, e.g. Gemma)")
    print("SPECTRAL_COLLAPSE: D10 extreme > moderate layers (over-concentration, e.g. Bloom)")
    print("ROBUST: Zone maintained across doses")
    print("NO_DEMON: No zone at D3")


def analyze_dose_curve(data):
    """Analyze D1-D10 fine-grained dose resolution data (F540 format)."""
    results = []
    for model_id, m in data.items():
        if "doses" not in m:
            continue
        name = model_id.split("/")[-1]
        doses = m["doses"]
        if "D1" not in doses:
            continue

        curve = []
        for d in range(1, 11):
            dk = f"D{d}"
            if dk not in doses:
                break
            dd = doses[dk]
            curve.append({
                "dose": d,
                "zone": dd.get("zone_count", len(dd.get("zone_layers", []))),
                "peak": dd.get("peak_ratio", 0),
                "moderate": dd.get("moderate", 0),
                "extreme": dd.get("extreme", 0),
                "suppression": dd.get("suppression", 0),
            })

        if not curve:
            continue

        # Detect cliff: zone drops from ≥3 to <2 in one step
        cliffs = []
        for i in range(1, len(curve)):
            if curve[i-1]["zone"] >= 3 and curve[i]["zone"] < 2:
                cliffs.append(f"D{curve[i-1]['dose']}→D{curve[i]['dose']}")

        # Detect composition shift (spectral collapse)
        early_ext = sum(c["extreme"] for c in curve[:4])
        late_ext = sum(c["extreme"] for c in curve[6:])
        comp_shift = late_ext > early_ext * 2 and late_ext > 8

        zones = [c["zone"] for c in curve]
        predicted = m.get("predicted_failure", "?")

        results.append({
            "name": name,
            "predicted": predicted,
            "curve": curve,
            "zones": zones,
            "cliffs": cliffs,
            "comp_shift": comp_shift,
            "max_zone": max(zones),
            "min_zone_d4plus": min(zones[3:]) if len(zones) > 3 else min(zones),
        })
    return results


def print_dose_curves(curve_results):
    print(f"\n{'='*85}")
    print("DOSE-RESPONSE CURVES — F540 Analysis")
    print("=" * 85)

    # Summary table
    print(f"\n  {'Model':>20} | {'D1':>3} {'D2':>3} {'D3':>3} {'D4':>3} {'D5':>3} {'D6':>3} {'D7':>3} {'D8':>3} {'D9':>3} {'D10':>3} | Cliff")
    print(f"  {'─'*75}")
    for r in curve_results:
        zstr = " ".join(f"{z:>3}" for z in r["zones"])
        cliff = ", ".join(r["cliffs"]) if r["cliffs"] else "none"
        print(f"  {r['name']:>20} | {zstr} | {cliff}")

    # ASCII dose curves
    for r in curve_results:
        zones = r["zones"]
        max_z = max(max(zones), 1)
        print(f"\n  {r['name']} — predicted: {r['predicted']}")
        print(f"  {'─'*50}")

        # Zone count curve
        for row in range(max_z, 0, -1):
            line = f"  {row:>2} │"
            for z in zones:
                if z >= row:
                    line += " ██"
                else:
                    line += "   "
            print(line)
        print(f"   0 │" + "───" * len(zones))
        print(f"     " + "".join(f" D{d+1}" if d < 9 else " 10" for d in range(len(zones))))

        # Extreme layer count (smaller, inline)
        extremes = [c["extreme"] for c in r["curve"]]
        if max(extremes) > 0:
            peaks = [c["peak"] for c in r["curve"]]
            print(f"  ext: {' '.join(f'{e:>2}' for e in extremes)}")
            print(f"  peak:{' '.join(f'{p:>5.0f}' if p < 100 else f'{p:>5.0f}' for p in peaks)}")


def main():
    show_ascii = "--ascii" in sys.argv
    show_dose = "--dose" in sys.argv
    show_curve = "--curve" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    default_path = os.path.join(os.path.dirname(__file__), "zone_formation_decisive_results.json")
    path = args[0] if args else default_path

    if show_curve:
        curve_path = args[0] if args else os.path.join(os.path.dirname(__file__), "dose_resolution_results.json")
        if os.path.exists(curve_path):
            with open(curve_path) as f:
                curve_data = json.load(f)
            cr = analyze_dose_curve(curve_data)
            if cr:
                print_dose_curves(cr)
            else:
                print("No dose curve data found in file.")
        else:
            print(f"Dose curve data not found: {curve_path}")
        if not os.path.exists(path) or path == curve_path:
            return

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    results = []
    dr_results = []
    for model_id, m in data.items():
        if "error" in m or "doses" not in m:
            continue
        name = model_id.split("/")[-1]
        doses = m["doses"]
        if "D1" in doses and isinstance(doses["D1"], dict):
            continue
        for dose_key in ["D3_therapeutic", "D3", "dose_3"]:
            if dose_key in doses:
                layers = doses[dose_key]
                r = analyze_model(name, layers, dose_key)
                if r:
                    results.append(r)
                break

        if show_dose:
            dr = analyze_dose_response(name, doses)
            if dr:
                dr_results.append(dr)

    if results:
        results.sort(key=lambda r: r["suppression"], reverse=True)
        print_summary(results)

    if show_dose and dr_results:
        print_dose_response(dr_results)

    if show_ascii and results:
        print(f"\n{'='*85}")
        print("ASCII ZONE PROFILES")
        print(f"{'='*85}")
        print_ascii(results)

    output = {r["name"]: {
        "quality": r["quality"], "shape": r["shape"],
        "suppression": round(r["suppression"], 4),
        "peak_layer": r["peak_layer"], "peak_ratio": round(r["peak_ratio"], 3),
        "ws_coverage": round(r["ws_coverage"], 3),
        "avg_drift": round(r["avg_drift"], 2),
    } for r in results}

    out_path = path.replace(".json", "_quality.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nQuality data saved to: {out_path}")


if __name__ == "__main__":
    main()
