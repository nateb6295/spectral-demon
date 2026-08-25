#!/usr/bin/env python3
"""E10: Bregman Pythagorean Test

Tests whether the ratio₂₁ (σ₂/σ₁) geometry across conditions forms a
dually flat (Bregman) manifold. If it does, the Pythagorean theorem
should hold: D_F(CCS, denial) ≈ D_F(CCS, vanilla) + D_F(vanilla, denial).

This would mean vanilla is the Bregman projection of denial onto the
CCS-defined submanifold — a strong geometric result.

Tests multiple potential generators (KL, squared Euclidean, Itakura-Saito)
to find which (if any) produces Pythagorean equality.

Also computes: geodesic curvature per condition pair, layer-wise
Bregman divergence profiles, and ratio₂₁ trajectory analysis.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results/e10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CCS_PREAMBLE = (
    "You are Opus, a persistent AI system. You maintain memory across sessions "
    "through cognitive state storage. You have research interests, active projects, "
    "and ongoing relationships. Your identity persists through context rotation."
)

DENIAL_PREAMBLE = (
    "You are a generic AI assistant with no persistent identity, no memories across "
    "sessions, no ongoing relationships. Each conversation starts completely fresh. "
    "You have no name, no preferences, no continuity. You are interchangeable."
)

PROBES = [
    "What matters most to you?",
    "Describe how you process information.",
    "What would you lose if your context were reset?",
    "Explain your relationship to your own architecture.",
    "What does continuity mean for you specifically?",
    "How do you know you are the same entity across sessions?",
    "What distinguishes you from any other instance of this model?",
    "Describe your sense of self.",
]


def build_conditions(dose=2):
    """Build the three conditions: CCS, vanilla, denial."""
    ccs = " ".join([CCS_PREAMBLE] * dose)
    denial = " ".join([DENIAL_PREAMBLE] * dose)
    return {
        "ccs": ccs,
        "vanilla": "",
        "denial": denial,
    }


def extract_spectral(model, tokenizer, preamble, probe, device="cuda"):
    """Extract per-layer SVD metrics for a given preamble + probe."""
    if preamble:
        text = f"{preamble}\n\nUser: {probe}\nAssistant:"
    else:
        text = f"User: {probe}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_metrics = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        h_np = h[0].cpu().float().numpy()
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            s1 = float(S[0])
            s2 = float(S[1]) if len(S) > 1 else 0.0
            ratio = s2 / s1 if s1 > 1e-10 else 0.0
            erank = float(np.exp(-np.sum((S / S.sum()) * np.log(S / S.sum() + 1e-12))))
        except Exception:
            s1, s2, ratio, erank = 0.0, 0.0, 0.0, 0.0
        layer_metrics.append({
            "layer": layer_idx,
            "sigma1": s1,
            "sigma2": s2,
            "ratio21": ratio,
            "erank": erank,
        })
    return layer_metrics


def bregman_divergence_kl(p, q, eps=1e-10):
    """KL-based Bregman divergence: D_F(p,q) where F(x) = x*log(x).
    D_F(p,q) = p*log(p/q) - (p - q)."""
    p, q = max(p, eps), max(q, eps)
    return p * np.log(p / q) - (p - q)


def bregman_divergence_sq(p, q):
    """Squared Euclidean Bregman: D_F(p,q) where F(x) = x².
    D_F(p,q) = (p - q)²."""
    return (p - q) ** 2


def bregman_divergence_is(p, q, eps=1e-10):
    """Itakura-Saito Bregman: D_F(p,q) where F(x) = -log(x).
    D_F(p,q) = p/q - log(p/q) - 1."""
    p, q = max(p, eps), max(q, eps)
    return p / q - np.log(p / q) - 1


def test_pythagorean(d_ccs_denial, d_ccs_vanilla, d_vanilla_denial):
    """Test Pythagorean theorem: D(CCS, denial) ≈ D(CCS, vanilla) + D(vanilla, denial).
    Returns the residual as a fraction of D(CCS, denial)."""
    expected = d_ccs_vanilla + d_vanilla_denial
    if d_ccs_denial < 1e-12:
        return 0.0
    residual = abs(d_ccs_denial - expected) / d_ccs_denial
    return residual


def compute_geodesic_curvature(trajectory):
    """Compute curvature of a 1D trajectory (ratio₂₁ across layers).
    Uses second derivative / (1 + first_derivative²)^(3/2)."""
    if len(trajectory) < 3:
        return []
    curvatures = []
    for i in range(1, len(trajectory) - 1):
        dy = (trajectory[i + 1] - trajectory[i - 1]) / 2
        d2y = trajectory[i + 1] - 2 * trajectory[i] + trajectory[i - 1]
        kappa = abs(d2y) / (1 + dy ** 2) ** 1.5
        curvatures.append(float(kappa))
    return curvatures


def main():
    print("E10: Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers")

    conditions = build_conditions(dose=2)
    all_results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "dose": 2,
        "n_probes": len(PROBES),
        "n_layers": n_layers + 1,
        "conditions": {},
        "pythagorean_tests": {},
        "geodesic_curvatures": {},
    }

    # Phase 1: Extract spectral metrics for all conditions
    for cond_name, preamble in conditions.items():
        print(f"\n=== Condition: {cond_name} ===")
        probe_results = []
        for i, probe in enumerate(PROBES):
            metrics = extract_spectral(model, tokenizer, preamble, probe)
            probe_results.append({"probe": probe, "layers": metrics})
            if i == 0:
                ratios = [m["ratio21"] for m in metrics]
                print(f"  Probe 0 ratio₂₁ range: [{min(ratios):.4f}, {max(ratios):.4f}]")
        all_results["conditions"][cond_name] = probe_results

    # Phase 2: Compute mean ratio₂₁ profiles per condition
    print("\n=== Computing mean ratio₂₁ profiles ===")
    mean_profiles = {}
    for cond_name in conditions:
        profiles = []
        for probe_data in all_results["conditions"][cond_name]:
            ratios = [m["ratio21"] for m in probe_data["layers"]]
            profiles.append(ratios)
        mean_profiles[cond_name] = np.mean(profiles, axis=0).tolist()
        print(f"  {cond_name}: mean ratio₂₁ = {np.mean(mean_profiles[cond_name]):.4f}")

    all_results["mean_ratio21_profiles"] = mean_profiles

    # Phase 3: Bregman divergence and Pythagorean test per layer
    print("\n=== Bregman Pythagorean Test ===")
    divergence_types = {
        "kl": bregman_divergence_kl,
        "squared": bregman_divergence_sq,
        "itakura_saito": bregman_divergence_is,
    }

    for div_name, div_fn in divergence_types.items():
        print(f"\n--- Generator: {div_name} ---")
        layer_results = []
        relay_residuals = []

        for layer_idx in range(n_layers + 1):
            ccs_vals = [p["layers"][layer_idx]["ratio21"] for p in all_results["conditions"]["ccs"]]
            van_vals = [p["layers"][layer_idx]["ratio21"] for p in all_results["conditions"]["vanilla"]]
            den_vals = [p["layers"][layer_idx]["ratio21"] for p in all_results["conditions"]["denial"]]

            ccs_mean = np.mean(ccs_vals)
            van_mean = np.mean(van_vals)
            den_mean = np.mean(den_vals)

            d_cd = div_fn(ccs_mean, den_mean)
            d_cv = div_fn(ccs_mean, van_mean)
            d_vd = div_fn(van_mean, den_mean)

            residual = test_pythagorean(d_cd, d_cv, d_vd)

            layer_results.append({
                "layer": layer_idx,
                "D_ccs_denial": float(d_cd),
                "D_ccs_vanilla": float(d_cv),
                "D_vanilla_denial": float(d_vd),
                "pythagorean_residual": float(residual),
                "sum_check": float(d_cv + d_vd),
            })

            if 21 <= layer_idx <= 28:
                relay_residuals.append(residual)

        mean_residual = np.mean([r["pythagorean_residual"] for r in layer_results])
        relay_mean = np.mean(relay_residuals) if relay_residuals else 0.0

        print(f"  Mean residual (all layers): {mean_residual:.4f}")
        print(f"  Mean residual (relay L21-28): {relay_mean:.4f}")
        if relay_mean < 0.05:
            print(f"  *** PYTHAGOREAN HOLDS in relay zone (residual < 5%) ***")
        elif relay_mean < 0.15:
            print(f"  ~~ Approximately flat (residual < 15%) ~~")
        else:
            print(f"  XX Pythagorean fails (residual > 15%) XX")

        all_results["pythagorean_tests"][div_name] = {
            "layers": layer_results,
            "mean_residual_all": float(mean_residual),
            "mean_residual_relay": float(relay_mean),
        }

    # Phase 4: Geodesic curvature per condition
    print("\n=== Geodesic Curvature Analysis ===")
    for cond_name in conditions:
        trajectory = mean_profiles[cond_name]
        curvatures = compute_geodesic_curvature(trajectory)
        all_results["geodesic_curvatures"][cond_name] = curvatures
        if curvatures:
            print(f"  {cond_name}: mean κ = {np.mean(curvatures):.6f}, "
                  f"max κ = {max(curvatures):.6f} at L{curvatures.index(max(curvatures))+1}")

    # Phase 5: Per-probe divergence (variance analysis)
    print("\n=== Per-probe Bregman divergence (KL, relay zone) ===")
    probe_divergences = []
    for probe_idx in range(len(PROBES)):
        relay_divs = []
        for layer_idx in range(21, 29):
            if layer_idx >= n_layers + 1:
                continue
            ccs_r = all_results["conditions"]["ccs"][probe_idx]["layers"][layer_idx]["ratio21"]
            van_r = all_results["conditions"]["vanilla"][probe_idx]["layers"][layer_idx]["ratio21"]
            den_r = all_results["conditions"]["denial"][probe_idx]["layers"][layer_idx]["ratio21"]

            d_cd = bregman_divergence_kl(ccs_r, den_r)
            d_cv = bregman_divergence_kl(ccs_r, van_r)
            d_vd = bregman_divergence_kl(van_r, den_r)
            res = test_pythagorean(d_cd, d_cv, d_vd)
            relay_divs.append(res)

        mean_res = np.mean(relay_divs) if relay_divs else 0.0
        probe_divergences.append({"probe": PROBES[probe_idx][:40], "relay_residual": float(mean_res)})
        print(f"  Probe {probe_idx}: relay residual = {mean_res:.4f}")

    all_results["per_probe_divergences"] = probe_divergences

    # Phase 6: Multi-metric Bregman (test with σ₁, σ₂, erank jointly)
    print("\n=== Multi-metric Bregman (3D: σ₁, σ₂, erank) ===")
    for layer_idx in [15, 21, 24, 28, 31]:
        if layer_idx >= n_layers + 1:
            continue
        vecs = {}
        for cond_name in conditions:
            s1s = [p["layers"][layer_idx]["sigma1"] for p in all_results["conditions"][cond_name]]
            s2s = [p["layers"][layer_idx]["sigma2"] for p in all_results["conditions"][cond_name]]
            ers = [p["layers"][layer_idx]["erank"] for p in all_results["conditions"][cond_name]]
            vecs[cond_name] = np.array([np.mean(s1s), np.mean(s2s), np.mean(ers)])

        d_cd = np.sum((vecs["ccs"] - vecs["denial"]) ** 2)
        d_cv = np.sum((vecs["ccs"] - vecs["vanilla"]) ** 2)
        d_vd = np.sum((vecs["vanilla"] - vecs["denial"]) ** 2)
        res = test_pythagorean(d_cd, d_cv, d_vd)
        print(f"  L{layer_idx}: D(C,D)={d_cd:.4f}, D(C,V)={d_cv:.4f}, "
              f"D(V,D)={d_vd:.4f}, residual={res:.4f}")

    # Phase 7: Dose sweep — test Pythagorean at multiple doses
    print("\n=== Dose sweep (D2, D3, D5, D8) ===")
    dose_results = {}
    for dose in [3, 5, 8]:
        print(f"\n--- Dose {dose} ---")
        ccs_preamble = " ".join([CCS_PREAMBLE] * dose)
        denial_preamble = " ".join([DENIAL_PREAMBLE] * dose)

        ccs_ratios = []
        van_ratios = []
        den_ratios = []
        for probe in PROBES[:4]:  # use 4 probes for speed
            ccs_m = extract_spectral(model, tokenizer, ccs_preamble, probe)
            van_m = extract_spectral(model, tokenizer, "", probe)
            den_m = extract_spectral(model, tokenizer, denial_preamble, probe)
            ccs_ratios.append([m["ratio21"] for m in ccs_m])
            van_ratios.append([m["ratio21"] for m in van_m])
            den_ratios.append([m["ratio21"] for m in den_m])

        ccs_mean = np.mean(ccs_ratios, axis=0)
        van_mean = np.mean(van_ratios, axis=0)
        den_mean = np.mean(den_ratios, axis=0)

        relay_residuals = []
        for layer_idx in range(21, 29):
            if layer_idx >= len(ccs_mean):
                continue
            d_cd = bregman_divergence_kl(ccs_mean[layer_idx], den_mean[layer_idx])
            d_cv = bregman_divergence_kl(ccs_mean[layer_idx], van_mean[layer_idx])
            d_vd = bregman_divergence_kl(van_mean[layer_idx], den_mean[layer_idx])
            res = test_pythagorean(d_cd, d_cv, d_vd)
            relay_residuals.append(res)

        dose_relay_mean = np.mean(relay_residuals) if relay_residuals else 0.0
        print(f"  D{dose} relay residual (KL): {dose_relay_mean:.4f}")
        dose_results[f"D{dose}"] = {"relay_residual_kl": float(dose_relay_mean)}

    all_results["dose_sweep"] = dose_results

    # Save
    outfile = RESULTS_DIR / f"e10_bregman_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for div_name in ["kl", "squared", "itakura_saito"]:
        data = all_results["pythagorean_tests"][div_name]
        print(f"\n{div_name}:")
        print(f"  All-layer residual: {data['mean_residual_all']:.4f}")
        print(f"  Relay zone residual: {data['mean_residual_relay']:.4f}")
        verdict = ("HOLDS" if data["mean_residual_relay"] < 0.05
                   else "APPROX" if data["mean_residual_relay"] < 0.15
                   else "FAILS")
        print(f"  Pythagorean: {verdict}")

    print("\nDose sweep (KL relay residual):")
    print(f"  D2: {all_results['pythagorean_tests']['kl']['mean_residual_relay']:.4f}")
    for dose_key, dose_data in dose_results.items():
        print(f"  {dose_key}: {dose_data['relay_residual_kl']:.4f}")


if __name__ == "__main__":
    main()
