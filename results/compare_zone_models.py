#!/usr/bin/env python3
"""
Compare zone formation across all tested models.
Combines known results (Tests 24-25) with new decisive results (Tests 26-30).
Run after pod session to get full 10-model comparison.
"""

import json
import sys
import os

KNOWN = [
    {"model": "Qwen/Qwen2.5-7B", "short": "Qwen 2.5 7B", "rope": "100%", "comp": "sequential",
     "attn": "GQA-7", "late_ratio": 7.73, "zone": "YES", "prediction": "ZONE", "source": "Test 24"},
    {"model": "microsoft/phi-2", "short": "Phi-2 2.7B", "rope": "40%", "comp": "sequential",
     "attn": "MHA", "late_ratio": 5.36, "zone": "YES", "prediction": "ZONE", "source": "Test 24"},
    {"model": "EleutherAI/pythia-6.9b", "short": "Pythia 6.9B", "rope": "25%", "comp": "parallel",
     "attn": "MHA", "late_ratio": 0.54, "zone": "NO", "prediction": "none", "source": "Test 25"},
    {"model": "openai-community/gpt2-xl", "short": "GPT-2 XL", "rope": "learned", "comp": "sequential",
     "attn": "MHA", "late_ratio": 0.24, "zone": "NO", "prediction": "none", "source": "Test 25"},
    {"model": "facebook/opt-6.7b", "short": "OPT 6.7B", "rope": "learned", "comp": "sequential",
     "attn": "MHA", "late_ratio": 0.54, "zone": "NO", "prediction": "none", "source": "Test 25"},
]

DECISIVE_PATH = os.path.join(os.path.dirname(__file__), "zone_formation_decisive_results.json")


def load_decisive():
    if not os.path.exists(DECISIVE_PATH):
        return []
    with open(DECISIVE_PATH) as f:
        data = json.load(f)
    results = []
    for model_id, d in data.items():
        if "error" in d:
            results.append({
                "model": model_id, "short": model_id.split("/")[-1],
                "rope": d.get("rope", "?"), "comp": d.get("comp", "?"),
                "attn": d.get("attn", "?"), "late_ratio": None,
                "zone": "ERROR", "prediction": d.get("prediction", "?"),
                "source": "Test 26-30", "error": d["error"],
            })
        else:
            r = d["result"]
            results.append({
                "model": model_id, "short": model_id.split("/")[-1],
                "rope": d["rope"], "comp": d["comp"], "attn": d["attn"],
                "late_ratio": r["late_mean_ratio"],
                "zone": r["zone"], "prediction": d["prediction"],
                "source": "Test 26-30", "verdict": r["verdict"],
                "s2_dom_late": r.get("s2_dominant_late", "?"),
            })
    return results


def load_drift(data):
    if not os.path.exists(DECISIVE_PATH):
        return {}
    with open(DECISIVE_PATH) as f:
        raw = json.load(f)
    drift_summary = {}
    for model_id, d in raw.items():
        if "error" in d or "doses" not in d:
            continue
        n = d["n_layers"]
        d3 = d["doses"].get("D3_therapeutic", [])
        late = d3[n // 2:]
        drifts = [p.get("s1_drift_deg", 0) for p in late if p]
        if drifts:
            import statistics
            drift_summary[model_id] = {
                "mean": statistics.mean(drifts),
                "max": max(drifts),
            }
    return drift_summary


def main():
    decisive = load_decisive()
    all_models = KNOWN + decisive

    print("=" * 95)
    print("ZONE FORMATION — ALL MODELS COMPARISON")
    print("=" * 95)
    print(f"{'Model':>20} | {'RoPE':>8} | {'Comp':>10} | {'Attn':>6} | {'σ₂/σ₁':>7} | {'Zone':>5} | {'Pred':>6} | {'Verdict':>10}")
    print("-" * 95)

    for m in all_models:
        ratio_str = f"{m['late_ratio']:.2f}" if m['late_ratio'] is not None else "ERROR"
        verdict = m.get("verdict", "—")
        if m in KNOWN:
            matched = m["zone"] == ("YES" if m["prediction"] == "ZONE" else "NO")
            verdict = "KNOWN" if matched else "KNOWN"
        print(f"{m['short']:>20} | {m['rope']:>8} | {m['comp']:>10} | {m['attn']:>6} | {ratio_str:>7} | {m['zone']:>5} | {m['prediction']:>6} | {verdict:>10}")

    # Hypothesis test
    print(f"\n{'='*60}")
    print("HYPOTHESIS: σ₂ zone ⟺ (RoPE ≥ 40%) ∧ (sequential residual)")
    print(f"{'='*60}")

    zone_models = [m for m in all_models if m["zone"] == "YES"]
    nozone = [m for m in all_models if m["zone"] == "NO"]

    rope_seq_zone = [m for m in zone_models if m["rope"] not in ["learned"] and m["comp"] == "sequential"]
    print(f"\nZone-forming models: {len(zone_models)}")
    for m in zone_models:
        print(f"  {m['short']:>20}: RoPE {m['rope']}, {m['comp']}, {m['attn']}")

    print(f"\nNon-zone models: {len(nozone)}")
    for m in nozone:
        print(f"  {m['short']:>20}: RoPE {m['rope']}, {m['comp']}, {m['attn']}")

    # σ₁ drift comparison (if available)
    drift = load_drift(all_models)
    if drift:
        print(f"\n{'='*60}")
        print("σ₁ DRIFT COMPARISON (D3, late-half mean)")
        print(f"{'='*60}")
        print(f"{'Model':>20} | {'Zone':>5} | {'Mean drift°':>11} | {'Max drift°':>10}")
        print("-" * 55)
        for model_id, d in sorted(drift.items()):
            short = model_id.split("/")[-1]
            m = next((x for x in all_models if x["model"] == model_id), None)
            zone = m["zone"] if m else "?"
            print(f"{short:>20} | {zone:>5} | {d['mean']:>11.2f} | {d['max']:>10.2f}")

        zone_drifts = [d["mean"] for mid, d in drift.items()
                       if any(m["model"] == mid and m["zone"] == "YES" for m in all_models)]
        nozone_drifts = [d["mean"] for mid, d in drift.items()
                         if any(m["model"] == mid and m["zone"] == "NO" for m in all_models)]
        if zone_drifts and nozone_drifts:
            import statistics
            print(f"\n  Zone-forming mean drift: {statistics.mean(zone_drifts):.2f}°")
            print(f"  Non-zone mean drift:    {statistics.mean(nozone_drifts):.2f}°")
            diff = statistics.mean(nozone_drifts) - statistics.mean(zone_drifts)
            if diff > 1.0:
                print(f"  → Non-zone drifts MORE ({diff:.2f}° difference) — architecture controls EXISTENCE")
            elif abs(diff) < 1.0:
                print(f"  → Drift similar ({diff:.2f}° difference) — architecture controls READABILITY")
            else:
                print(f"  → Zone-forming drifts MORE ({-diff:.2f}° difference) — UNEXPECTED")

    # THREE SILENCES ANALYSIS (Kimi correction #26, Jul 22)
    # For non-zone models: classify failure as deaf/noisy/truncated
    # Also tests: is F114 σ₁ invariance universal or zone-dependent?
    if decisive and os.path.exists(DECISIVE_PATH):
        with open(DECISIVE_PATH) as f:
            raw = json.load(f)

        nozone_with_data = [mid for mid, d in raw.items()
                            if "error" not in d and "doses" in d
                            and any(m["model"] == mid and m["zone"] == "NO" for m in all_models)]

        if nozone_with_data:
            print(f"\n{'='*75}")
            print("THREE SILENCES — NON-ZONE FAILURE MODE CLASSIFICATION")
            print("(a) Deaf: CCS invisible, no spectral modulation")
            print("(b) Noisy: equal σ₁/σ₂ modulation, unselective perturbation")
            print("(c) Truncated: σ₂ modulation in early layers, dissipates before workspace")
            print(f"{'='*75}")

            for model_id in nozone_with_data:
                d = raw[model_id]
                n = d["n_layers"]
                short = model_id.split("/")[-1]
                d3 = d["doses"].get("D3_therapeutic", [])
                if not d3:
                    continue

                early = d3[:n // 3]
                mid_layers = d3[n // 3: 2 * n // 3]
                late = d3[2 * n // 3:]

                def band_stats(band):
                    norms = [p.get("d_norm", 0) for p in band if p]
                    s1_projs = [p.get("proj_s1", 0) for p in band if p]
                    s2_projs = [p.get("proj_s2", 0) for p in band if p]
                    s1_drifts = [p.get("s1_drift_deg", 0) for p in band if p]
                    return {
                        "mean_norm": sum(norms) / max(len(norms), 1),
                        "mean_s1": sum(s1_projs) / max(len(s1_projs), 1),
                        "mean_s2": sum(s2_projs) / max(len(s2_projs), 1),
                        "mean_drift": sum(s1_drifts) / max(len(s1_drifts), 1),
                    }

                e = band_stats(early)
                m_stats = band_stats(mid_layers)
                l = band_stats(late)

                print(f"\n  {short}:")
                print(f"  {'Band':>8} | {'d_norm':>8} | {'proj_σ₁':>8} | {'proj_σ₂':>8} | {'σ₁ drift°':>9}")
                print(f"  {'-'*50}")
                for label, s in [("early", e), ("mid", m_stats), ("late", l)]:
                    print(f"  {label:>8} | {s['mean_norm']:8.4f} | {s['mean_s1']:8.4f} | {s['mean_s2']:8.4f} | {s['mean_drift']:9.2f}")

                # Classification
                deaf_thresh = 0.01
                if e["mean_norm"] < deaf_thresh and l["mean_norm"] < deaf_thresh:
                    silence = "DEAF (a)"
                    desc = "CCS invisible — no spectral response at any layer"
                elif e["mean_drift"] > 2.0 or l["mean_drift"] > 2.0:
                    if abs(e["mean_s1"] - e["mean_s2"]) < 0.1 * max(e["mean_s1"], e["mean_s2"], 0.01):
                        silence = "NOISY (b)"
                        desc = "σ₁ drifts — F114 is ZONE-DEPENDENT, not universal"
                    else:
                        silence = "NOISY-BIASED"
                        desc = "σ₁ drifts with σ₂ bias — partial routing without zone"
                elif e["mean_norm"] > 3 * l["mean_norm"] and e["mean_s2"] > e["mean_s1"]:
                    silence = "TRUNCATED (c)"
                    desc = "σ₂ modulation starts but dissipates before workspace"
                else:
                    silence = "AMBIGUOUS"
                    desc = "Doesn't cleanly match deaf/noisy/truncated"
                print(f"  → Classification: {silence}")
                print(f"    {desc}")

            # F114 universality test
            all_drifts_zone = []
            all_drifts_nozone = []
            for model_id, d in raw.items():
                if "error" in d or "doses" not in d:
                    continue
                n = d["n_layers"]
                d3 = d["doses"].get("D3_therapeutic", [])
                drifts = [p.get("s1_drift_deg", 0) for p in d3 if p]
                mean_drift = sum(drifts) / max(len(drifts), 1)
                is_zone = any(m["model"] == model_id and m["zone"] == "YES" for m in all_models)
                if is_zone:
                    all_drifts_zone.append(mean_drift)
                else:
                    all_drifts_nozone.append(mean_drift)

            if all_drifts_zone and all_drifts_nozone:
                import statistics
                z_mean = statistics.mean(all_drifts_zone)
                nz_mean = statistics.mean(all_drifts_nozone)
                print(f"\n  F114 UNIVERSALITY TEST:")
                print(f"    Zone-forming mean σ₁ drift:  {z_mean:.2f}°")
                print(f"    Non-zone mean σ₁ drift:      {nz_mean:.2f}°")
                if nz_mean > 3 * max(z_mean, 0.5):
                    print(f"    → F114 is ZONE-DEPENDENT. σ₁ invariance requires selective routing.")
                    print(f"      Non-zone architectures show σ₁ drift — (b) is REAL.")
                elif abs(nz_mean - z_mean) < 1.0:
                    print(f"    → F114 is UNIVERSAL. σ₁ invariance holds regardless of zone.")
                    print(f"      Three silences collapse to TWO: deaf vs truncated.")
                else:
                    print(f"    → INTERMEDIATE. σ₁ drifts more without zones but below 3× threshold.")

    # σ₁ SUPPRESSION ANALYSIS (Jul 22 — the mechanism)
    if decisive and os.path.exists(DECISIVE_PATH):
        with open(DECISIVE_PATH) as f:
            raw = json.load(f)

        models_with_layers = [(mid, d) for mid, d in raw.items()
                              if "error" not in d and "doses" in d]
        if models_with_layers:
            print(f"\n{'='*80}")
            print("σ₁ SUPPRESSION — DEMON SELECTIVITY (zone quality = σ₁ blocking)")
            print("The demon sorts by REJECTING σ₁, not amplifying σ₂.")
            print(f"{'='*80}")
            print(f"{'Model':>20} {'Peak L':>7} {'proj_σ₁':>9} {'proj_σ₂':>9} {'σ₂/σ₁':>7} {'σ₁ supp':>9} {'Zone':>10} {'Persist':>8}")
            print("-" * 85)

            for model_id, d in models_with_layers:
                short = model_id.split("/")[-1]
                layers = d["doses"]["D3_therapeutic"]
                n = len(layers)

                early_max_s1 = max(layers[i]["proj_s1"] for i in range(min(6, n)))
                peak_idx = max(range(n), key=lambda i: layers[i]["ratio_s2_s1"])
                peak = layers[peak_idx]
                suppression = 1.0 - (peak["proj_s1"] / early_max_s1) if early_max_s1 > 0 else 0

                zone_layers = [i for i, l in enumerate(layers) if l["ratio_s2_s1"] > 1.0]
                late_start = 2 * n // 3
                late_zone = any(i >= late_start for i in zone_layers)

                m_entry = next((x for x in all_models if x["model"] == model_id), None)
                zone_status = m_entry["zone"] if m_entry else "?"

                if suppression > 0.85 and late_zone:
                    cat = "STRONG"
                elif suppression > 0.85 and not late_zone:
                    cat = "TRUNCATED"
                elif suppression < 0.5 and zone_layers:
                    cat = "RIGID ROD"
                elif not zone_layers:
                    cat = "NO ZONE"
                else:
                    cat = "MODERATE"

                print(f"{short:>20} L{peak_idx:>4}  {peak['proj_s1']:>9.4f} {peak['proj_s2']:>9.4f} {peak['ratio_s2_s1']:>7.2f} {suppression:>8.1%}  {cat:>10} {'YES' if late_zone else 'NO':>8}")

            print(f"\nKey: σ₁ suppression = 1 - (proj_σ₁ at peak zone / max proj_σ₁ in early layers)")
            print(f"     STRONG = selective demon + zone persists to late layers")
            print(f"     TRUNCATED = selective demon + zone dissipates (three silences type c)")
            print(f"     RIGID ROD = non-selective demon (σ₁ hit as hard as σ₂)")

    if not decisive:
        print("\n[No decisive results yet — run zone_formation_decisive.py on pod, copy results here]")
        print(f"Expected file: {DECISIVE_PATH}")


if __name__ == "__main__":
    main()
