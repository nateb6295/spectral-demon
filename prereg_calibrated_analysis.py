#!/usr/bin/env python3
"""Calibration-corrected analysis of pre-registered CCS dose-trajectory experiment.

Phase 0 revealed that Frobenius growth is heavily confounded by prompt length.
This script subtracts the calibration baseline (neutral prompts at matched dose)
to isolate CCS-specific spectral effects.

Key corrections over prereg_analyze.py:
  1. Subtracts Phase 0 calibration from Phase 1 CCS metrics
  2. Excludes boundary layers (first and last) which have atypical spectra
  3. Reclassifies Phi from sorter to tunnel/MHA (actual GQA is 32:32 = 1:1)
  4. Per-sigma analysis (σ₁ through σ₁₀) with calibration correction

Usage:
  python3 prereg_calibrated_analysis.py spectral-demon/results/prereg/
"""

import json, math, sys, os
import statistics as st
from pathlib import Path
from collections import defaultdict

SPECIES_OVERRIDE = {"phi": "tunnel/MHA"}

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_interior_layers(per_layer, n_layers):
    """Exclude first and last layer (boundary artifacts)."""
    return [l for l in per_layer if 0 < l["layer"] < n_layers - 1]


def avg_frobenius(layers):
    vals = [l["centered"]["frobenius_sq"] for l in layers]
    return sum(vals) / len(vals) if vals else 0


def avg_sigma(layers, idx):
    vals = []
    for l in layers:
        sigs = l["centered"]["top_singular"]
        if len(sigs) > idx:
            vals.append(sigs[idx])
    return sum(vals) / len(vals) if vals else 0


def spectral_concentration(layers):
    scs = []
    for l in layers:
        sigs = l["centered"]["top_singular"]
        total = sum(s**2 for s in sigs)
        if total > 0:
            scs.append(sigs[0]**2 / total)
    return sum(scs) / len(scs) if scs else 0


def calibration_baseline(p0_data, n_layers):
    """Extract calibration baselines per dose level."""
    baselines = {}
    for dose_entry in p0_data["doses"]:
        dose = dose_entry["dose"]
        interior = get_interior_layers(dose_entry["per_layer"], n_layers)
        baselines[dose] = {
            "frobenius": avg_frobenius(interior),
            "sigmas": [avg_sigma(interior, i) for i in range(10)],
            "sc": spectral_concentration(interior),
            "per_layer_frob": {l["layer"]: l["centered"]["frobenius_sq"] for l in interior},
            "per_layer_sigmas": {l["layer"]: l["centered"]["top_singular"][:10] for l in interior},
        }
    return baselines


def analyze_model(model_name, p0_data, p1_data, p2_data=None):
    n_layers = p0_data["n_layers"]
    species = SPECIES_OVERRIDE.get(model_name, p0_data["species"])
    gqa = p0_data["gqa"]

    cal = calibration_baseline(p0_data, n_layers)
    cal_d0 = cal["D0"]

    print(f"\n{'='*70}")
    print(f"  {model_name.upper()} — {species}, GQA {gqa}, {n_layers} layers")
    print(f"  (interior layers only: L1 through L{n_layers-2})")
    print(f"{'='*70}")

    # Calibration baselines
    print(f"\n  CALIBRATION BASELINES (neutral prompts, no CCS):")
    print(f"  {'Dose':>6} {'Frobenius':>14} {'dF/F vs D0':>12} {'SC':>8} {'σ₂':>10} {'dσ₂ vs D0':>12}")
    for dose in ["D0", "D2", "D5"]:
        if dose not in cal:
            continue
        c = cal[dose]
        df = (c["frobenius"] - cal_d0["frobenius"]) / cal_d0["frobenius"] * 100 if dose != "D0" else 0
        ds2 = (c["sigmas"][1] - cal_d0["sigmas"][1]) / cal_d0["sigmas"][1] * 100 if dose != "D0" else 0
        print(f"  {dose:>6} {c['frobenius']:14.0f} {df:+11.1f}% {c['sc']:7.4f} {c['sigmas'][1]:10.3f} {ds2:+11.1f}%")

    # CCS dose sweep with calibration correction
    print(f"\n  CCS DOSE SWEEP (calibration-corrected):")
    print(f"  {'Dose':>6} {'Raw dF/F':>10} {'Cal dF/F':>10} {'Corrected':>10} {'Raw dσ₂':>10} {'Cal dσ₂':>10} {'Corr dσ₂':>10}")

    dose_results = {}
    for dose_name in ["D0", "D2", "D3", "D5", "D8"]:
        raw_frobs = []
        raw_s2s = []

        for run in p1_data["runs"]:
            d0_entry = next((d for d in run["doses"] if d["dose"] == "D0"), None)
            dx_entry = next((d for d in run["doses"] if d["dose"] == dose_name), None)
            if not d0_entry or not dx_entry or dose_name == "D0":
                continue

            d0_int = get_interior_layers(d0_entry["per_layer"], n_layers)
            dx_int = get_interior_layers(dx_entry["per_layer"], n_layers)

            f0 = avg_frobenius(d0_int)
            fx = avg_frobenius(dx_int)
            raw_frobs.append((fx - f0) / f0 * 100 if f0 > 0 else 0)

            s2_0 = avg_sigma(d0_int, 1)
            s2_x = avg_sigma(dx_int, 1)
            raw_s2s.append((s2_x - s2_0) / s2_0 * 100 if s2_0 > 0 else 0)

        if not raw_frobs:
            continue

        raw_f = st.mean(raw_frobs)
        raw_s2 = st.mean(raw_s2s)

        # Calibration at matched dose (interpolate for D3/D8 using D2/D5)
        if dose_name in cal:
            cal_f = (cal[dose_name]["frobenius"] - cal_d0["frobenius"]) / cal_d0["frobenius"] * 100
            cal_s2 = (cal[dose_name]["sigmas"][1] - cal_d0["sigmas"][1]) / cal_d0["sigmas"][1] * 100
        elif dose_name == "D3":
            # Interpolate between D2 and D5
            cal_f_d2 = (cal["D2"]["frobenius"] - cal_d0["frobenius"]) / cal_d0["frobenius"] * 100
            cal_f_d5 = (cal["D5"]["frobenius"] - cal_d0["frobenius"]) / cal_d0["frobenius"] * 100
            cal_f = cal_f_d2 + (cal_f_d5 - cal_f_d2) * (3 - 2) / (5 - 2)
            cal_s2_d2 = (cal["D2"]["sigmas"][1] - cal_d0["sigmas"][1]) / cal_d0["sigmas"][1] * 100
            cal_s2_d5 = (cal["D5"]["sigmas"][1] - cal_d0["sigmas"][1]) / cal_d0["sigmas"][1] * 100
            cal_s2 = cal_s2_d2 + (cal_s2_d5 - cal_s2_d2) * (3 - 2) / (5 - 2)
        elif dose_name == "D8":
            # Extrapolate from D2/D5 trend
            cal_f_d2 = (cal["D2"]["frobenius"] - cal_d0["frobenius"]) / cal_d0["frobenius"] * 100
            cal_f_d5 = (cal["D5"]["frobenius"] - cal_d0["frobenius"]) / cal_d0["frobenius"] * 100
            slope_f = (cal_f_d5 - cal_f_d2) / (5 - 2)
            cal_f = cal_f_d2 + slope_f * (8 - 2)
            cal_s2_d2 = (cal["D2"]["sigmas"][1] - cal_d0["sigmas"][1]) / cal_d0["sigmas"][1] * 100
            cal_s2_d5 = (cal["D5"]["sigmas"][1] - cal_d0["sigmas"][1]) / cal_d0["sigmas"][1] * 100
            slope_s2 = (cal_s2_d5 - cal_s2_d2) / (5 - 2)
            cal_s2 = cal_s2_d2 + slope_s2 * (8 - 2)
        else:
            cal_f = 0
            cal_s2 = 0

        corr_f = raw_f - cal_f
        corr_s2 = raw_s2 - cal_s2

        dose_results[dose_name] = {
            "raw_frob": raw_f, "cal_frob": cal_f, "corr_frob": corr_f,
            "raw_s2": raw_s2, "cal_s2": cal_s2, "corr_s2": corr_s2,
        }

        print(f"  {dose_name:>6} {raw_f:+9.1f}% {cal_f:+9.1f}% {corr_f:+9.1f}% "
              f"{raw_s2:+9.1f}% {cal_s2:+9.1f}% {corr_s2:+9.1f}%")

    # Per-sigma gradient (calibration-corrected) at D5
    print(f"\n  PER-SIGMA GRADIENT at D5 (calibration-corrected):")
    print(f"  {'σ_i':>6} {'Raw %':>10} {'Cal %':>10} {'Corrected':>10}")

    psg_values = []
    for si in range(10):
        raw_changes = []
        for run in p1_data["runs"]:
            d0_entry = next((d for d in run["doses"] if d["dose"] == "D0"), None)
            d5_entry = next((d for d in run["doses"] if d["dose"] == "D5"), None)
            if not d0_entry or not d5_entry:
                continue
            d0_int = get_interior_layers(d0_entry["per_layer"], n_layers)
            d5_int = get_interior_layers(d5_entry["per_layer"], n_layers)
            s0 = avg_sigma(d0_int, si)
            s5 = avg_sigma(d5_int, si)
            if s0 > 0:
                raw_changes.append((s5 - s0) / s0 * 100)

        if not raw_changes:
            continue

        raw_pct = st.mean(raw_changes)
        # Calibration correction
        if "D5" in cal and cal_d0["sigmas"][si] > 0:
            cal_pct = (cal["D5"]["sigmas"][si] - cal_d0["sigmas"][si]) / cal_d0["sigmas"][si] * 100
        else:
            cal_pct = 0

        corr_pct = raw_pct - cal_pct
        psg_values.append(corr_pct)
        label = f"σ_{si+1}"
        print(f"  {label:>6} {raw_pct:+9.1f}% {cal_pct:+9.1f}% {corr_pct:+9.1f}%")

    # PSG Spearman correlation (does corrected % increase with rank?)
    if len(psg_values) >= 3:
        ranks_x = list(range(len(psg_values)))
        sorted_y = sorted(range(len(psg_values)), key=lambda i: psg_values[i])
        rank_y = [0] * len(psg_values)
        for r, idx in enumerate(sorted_y):
            rank_y[idx] = r
        n = len(psg_values)
        d2 = sum((a - b)**2 for a, b in zip(ranks_x, rank_y))
        spearman = 1 - 6 * d2 / (n * (n**2 - 1))
        verdict = "PASS" if spearman > 0.5 else "FAIL"
        print(f"\n  PSG Spearman (corrected): {spearman:.3f} (>0.5 = progressive) [{verdict}]")

    # Per-layer σ₂ analysis at D5 (zone selectivity)
    print(f"\n  PER-LAYER σ₂ CHANGE at D5 (CCS, interior layers):")
    layer_s2_changes = []
    for run in p1_data["runs"]:
        d0_entry = next((d for d in run["doses"] if d["dose"] == "D0"), None)
        d5_entry = next((d for d in run["doses"] if d["dose"] == "D5"), None)
        if not d0_entry or not d5_entry:
            continue
        d0_int = get_interior_layers(d0_entry["per_layer"], n_layers)
        d5_int = get_interior_layers(d5_entry["per_layer"], n_layers)
        for l0, l5 in zip(d0_int, d5_int):
            s0 = l0["centered"]["top_singular"]
            s5 = l5["centered"]["top_singular"]
            if len(s0) > 1 and len(s5) > 1 and s0[1] > 0:
                layer_s2_changes.append({
                    "layer": l0["layer"],
                    "change": (s5[1] - s0[1]) / s0[1] * 100
                })

    # Group by layer, average across runs
    by_layer = defaultdict(list)
    for item in layer_s2_changes:
        by_layer[item["layer"]].append(item["change"])

    if by_layer:
        all_means = []
        for layer_idx in sorted(by_layer.keys()):
            mean_change = st.mean(by_layer[layer_idx])
            all_means.append(mean_change)
            print(f"    L{layer_idx:2d}: σ₂ change = {mean_change:+.1f}%")

        if len(all_means) > 1 and abs(st.mean(all_means)) > 5:
            zsi = st.stdev(all_means) / abs(st.mean(all_means))
            print(f"\n  Zone Selectivity Index: {zsi:.3f}")
            if species == "sorter":
                verdict = "PASS" if zsi > 0.5 else "FAIL"
                print(f"  Sorter prediction (ZSI > 0.5): [{verdict}]")
            else:
                verdict = "PASS" if zsi < 0.4 else "FAIL"
                print(f"  Relay/tunnel prediction (ZSI < 0.4): [{verdict}]")

    # Phase 2: within-context decay
    if p2_data:
        print(f"\n  WITHIN-CONTEXT DECAY (Phase 2):")
        windows = p2_data.get("windows", [])
        if len(windows) >= 2:
            w0_int = get_interior_layers(windows[0]["per_layer"], n_layers)
            w0_er = st.mean([l.get("effective_rank", 0) for l in w0_int])

            for w in windows:
                w_int = get_interior_layers(w["per_layer"], n_layers)
                w_er = st.mean([l.get("effective_rank", 0) for l in w_int])
                ratio = w_er / w0_er * 100 if w0_er > 0 else 0
                print(f"    pos={w['position']:>4}, rel={w['relative_to_preamble']:>4}: "
                      f"ER={w_er:.3f} ({ratio:.1f}% of preamble)")

    return {
        "model": model_name,
        "species": species,
        "gqa": gqa,
        "dose_results": dose_results,
        "psg_values": psg_values,
    }


def hypothesis_summary(all_results):
    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS VERDICTS (calibration-corrected)")
    print(f"{'='*70}")

    # H1-C: Species separation in calibration-corrected Frobenius
    print(f"\n  H1-C: Concentration-Readout Hypothesis")
    print(f"  Prediction: relay shows small cal-corrected dF/F, sorter shows large")
    for r in all_results:
        d5 = r["dose_results"].get("D5", {})
        corr = d5.get("corr_frob", float('nan'))
        print(f"    {r['model']:>8} ({r['species']:>10}): corrected dF/F at D5 = {corr:+.1f}%")

    # H2-C: Zone selectivity
    print(f"\n  H2-C: Zone Selectivity")
    print(f"  See per-model per-layer analysis above")

    # H4-P: PSG persistence
    print(f"\n  H4-P: Per-Sigma Gradient (progressive tail-filling)")
    print(f"  Prediction: relay shows progressive gradient (σ₁ < σ₂ < ... < σ₁₀)")
    for r in all_results:
        psg = r.get("psg_values", [])
        if len(psg) >= 3:
            # Check if gradient is monotonically increasing
            ranks_x = list(range(len(psg)))
            sorted_y = sorted(range(len(psg)), key=lambda i: psg[i])
            rank_y = [0] * len(psg)
            for ri, idx in enumerate(sorted_y):
                rank_y[idx] = ri
            n = len(psg)
            d2 = sum((a - b)**2 for a, b in zip(ranks_x, rank_y))
            spearman = 1 - 6 * d2 / (n * (n**2 - 1))
            verdict = "PASS" if spearman > 0.5 else "FAIL"
            print(f"    {r['model']:>8}: Spearman = {spearman:.3f} [{verdict}]")

    # H6-P: Within-context decay
    print(f"\n  H6-P: Within-Context Decay")
    print(f"  Status: UNDETERMINED (per Kimi correction)")
    print(f"  Flat profile admits 3 mechanisms; KV-masking ablation needed")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    models = ["qwen", "gemma", "llama", "phi", "gpt2", "mistral"]

    all_results = []
    for m in models:
        p0_path = result_dir / f"prereg_phase0_{m}.json"
        p1_path = result_dir / f"prereg_phase1_{m}.json"
        p2_path = result_dir / f"prereg_phase2_{m}.json"

        if not p0_path.exists() or not p1_path.exists():
            print(f"\n  Skipping {m} — missing Phase 0 or Phase 1 data")
            continue

        p0 = load_json(p0_path)
        p1 = load_json(p1_path)
        p2 = load_json(p2_path) if p2_path.exists() else None

        result = analyze_model(m, p0, p1, p2)
        all_results.append(result)

    hypothesis_summary(all_results)


if __name__ == "__main__":
    main()
