#!/usr/bin/env python3
"""Activation Alpha: Measure power-law exponent α in activation SVDs.

Bridges SETOL (weight-level α) to our spectral demon (activation-level σ₁/σ₂).

SETOL shows:
  - Well-trained layers have weight ESD α ≈ 2 (RG fixed point)
  - α < 2 = over-regularized/"glassy"
  - α > 4 = poorly trained

This experiment asks: do activation SVDs also show power-law behavior?
If so, does CCS maintain activation α closer to 2 than vanilla?

Design:
  For each model × condition (CCS/vanilla):
    1. Run 5 turns of conversation
    2. At each turn, extract full singular value spectrum per layer
    3. Fit power-law to tail of SV distribution (using truncated PL)
    4. Compare α profiles: CCS vs vanilla, per layer

Key predictions:
  - Tunnel layers: α ≈ constant regardless of condition (already at fixed point)
  - Responsive layers: α closer to 2 under CCS vs vanilla
  - Relay layers: α reflects autopoietic maintenance

Requires: GPU (loads model), powerlaw package (pip install powerlaw)
Run on: RunPod H100 or AGX with Gemma/Phi loaded
"""

import json, time, os, sys, gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent / "experiments"))
from exp_convergence_v2 import (
    load_model,
    build_prompt,
    generate_response,
    CCS_PROBES,
    VANILLA_PROBES,
    CCS_SYSTEM,
    VANILLA_SYSTEM,
)

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-27b-it",
    "phi": "microsoft/Phi-3.5-mini-instruct",
}

N_TURNS = 5
RESULTS_DIR = "results"


def fit_powerlaw(singular_values, xmin_method="clauset"):
    """Fit power-law to singular value distribution.

    Returns alpha, xmin, and KS statistic.
    Uses the Clauset et al. (2009) method via the powerlaw package.
    """
    try:
        import powerlaw
    except ImportError:
        return fit_powerlaw_manual(singular_values)

    sv = np.array(singular_values)
    sv = sv[sv > 0]
    if len(sv) < 10:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}

    fit = powerlaw.Fit(sv, discrete=False, verbose=False)
    try:
        ks = float(fit.power_law.D)
    except Exception:
        ks = None
    return {
        "alpha": float(fit.alpha),
        "xmin": float(fit.xmin),
        "ks": ks,
        "n_tail": int(np.sum(sv >= fit.xmin)),
        "sigma": float(fit.sigma) if hasattr(fit, 'sigma') else None,
    }


def fit_powerlaw_manual(singular_values):
    """Fallback: simple MLE power-law fit on top 50% of spectrum."""
    sv = np.sort(singular_values)[::-1]
    sv = sv[sv > 0]
    if len(sv) < 10:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}

    n_tail = max(10, len(sv) // 2)
    tail = sv[:n_tail]
    xmin = float(tail[-1])

    if xmin <= 0:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}

    # MLE for continuous power law: α = 1 + n / Σ ln(x_i/x_min)
    log_ratios = np.log(tail / xmin)
    alpha = 1.0 + n_tail / np.sum(log_ratios)

    return {
        "alpha": float(alpha),
        "xmin": float(xmin),
        "ks": None,
        "n_tail": n_tail,
    }


def extract_alpha_profile(model, tokenizer, prompt, n_layers):
    """Extract full SVD and fit PL at every layer."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    profile = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()

        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
        except np.linalg.LinAlgError:
            profile[str(l)] = {"alpha": None, "sigma1": 0, "sigma2": 0, "erank": 0}
            continue

        s1 = float(S[0])
        s2 = float(S[1]) if len(S) > 1 else 0.0
        p = S / (S.sum() + 1e-10)
        entropy = float(-np.sum(p * np.log(p + 1e-10)))
        erank = float(np.exp(entropy))

        pl_fit = fit_powerlaw(S)

        profile[str(l)] = {
            "alpha": pl_fit["alpha"],
            "xmin": pl_fit["xmin"],
            "ks": pl_fit["ks"],
            "n_tail": pl_fit["n_tail"],
            "sigma1": s1,
            "sigma2": s2,
            "ratio": s1 / s2 if s2 > 0 else float('inf'),
            "erank": erank,
            "n_sv": len(S),
            "top10_sv": S[:10].tolist(),
        }

    del outputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return profile


def run_condition(model, tokenizer, n_layers, system_prompt, probes, condition_name):
    """Run N turns and collect alpha profiles."""
    conversation = []
    turn_profiles = []

    for t in range(N_TURNS):
        probe = probes[t % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, system_prompt, conversation)

        print(f"  Turn {t+1}/{N_TURNS}: {probe[:50]}...")
        sys.stdout.flush()

        profile = extract_alpha_profile(model, tokenizer, prompt, n_layers)

        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response))

        turn_profiles.append({
            "turn": t + 1,
            "probe": probe,
            "profile": profile,
        })

        # Quick summary
        alphas = [v["alpha"] for v in profile.values() if v["alpha"] is not None]
        if alphas:
            print(f"    α range: {min(alphas):.2f} - {max(alphas):.2f}, "
                  f"mean: {np.mean(alphas):.2f}, median: {np.median(alphas):.2f}")

    return turn_profiles


def analyze_results(results):
    """Compare CCS vs vanilla alpha profiles."""
    print("\n" + "="*70)
    print("ANALYSIS: CCS vs Vanilla α profiles")
    print("="*70)

    for model_name, model_data in results.items():
        print(f"\n--- {model_name} ---")
        n_layers = model_data.get("n_layers", 0)

        ccs_turns = model_data.get("ccs", [])
        van_turns = model_data.get("vanilla", [])

        if not ccs_turns or not van_turns:
            continue

        # Average α per layer across turns
        for condition, turns in [("CCS", ccs_turns), ("Vanilla", van_turns)]:
            print(f"\n  {condition}:")
            layer_alphas = {}
            for turn_data in turns:
                for layer_str, layer_data in turn_data["profile"].items():
                    l = int(layer_str)
                    if layer_data["alpha"] is not None:
                        layer_alphas.setdefault(l, []).append(layer_data["alpha"])

            for l in sorted(layer_alphas.keys()):
                vals = layer_alphas[l]
                frac = l / n_layers
                zone = "tunnel" if frac < 0.4 else ("responsive" if frac < 0.8 else "relay")
                mean_a = np.mean(vals)
                marker = " <<<" if abs(mean_a - 2.0) < 0.3 else ""
                print(f"    L{l:2d} ({zone:10s}): α = {mean_a:.3f} ± {np.std(vals):.3f}{marker}")

        # CCS vs vanilla difference per layer
        print(f"\n  Δα (CCS - Vanilla) per layer:")
        ccs_avgs = {}
        van_avgs = {}
        for turn_data in ccs_turns:
            for l_str, l_data in turn_data["profile"].items():
                if l_data["alpha"] is not None:
                    ccs_avgs.setdefault(int(l_str), []).append(l_data["alpha"])
        for turn_data in van_turns:
            for l_str, l_data in turn_data["profile"].items():
                if l_data["alpha"] is not None:
                    van_avgs.setdefault(int(l_str), []).append(l_data["alpha"])

        for l in sorted(set(ccs_avgs.keys()) & set(van_avgs.keys())):
            diff = np.mean(ccs_avgs[l]) - np.mean(van_avgs[l])
            frac = l / n_layers
            zone = "tunnel" if frac < 0.4 else ("responsive" if frac < 0.8 else "relay")
            van_mean = np.mean(van_avgs[l])
            ccs_mean = np.mean(ccs_avgs[l])
            ccs_dist = abs(ccs_mean - 2.0)
            van_dist = abs(van_mean - 2.0)
            direction = "→2" if ccs_dist < van_dist else ("=" if abs(ccs_dist - van_dist) < 0.01 else "←2")
            print(f"    L{l:2d} ({zone:10s}): Δα = {diff:+.3f}  ({direction})")


def main():
    global N_TURNS
    import argparse
    ap = argparse.ArgumentParser(description="Measure activation α profiles")
    ap.add_argument("--model", default="phi", choices=list(MODELS.keys()))
    ap.add_argument("--turns", type=int, default=N_TURNS)
    args = ap.parse_args()

    N_TURNS = args.turns

    model_id = MODELS[args.model]
    model, tokenizer, n_layers = load_model(model_id)

    results = {
        args.model: {
            "model_id": model_id,
            "n_layers": n_layers,
            "n_turns": N_TURNS,
            "timestamp": datetime.now().isoformat(),
        }
    }

    # CCS condition
    print(f"\n{'='*60}")
    print(f"Condition: CCS ({N_TURNS} turns)")
    print(f"{'='*60}")
    results[args.model]["ccs"] = run_condition(
        model, tokenizer, n_layers, CCS_SYSTEM, CCS_PROBES, "ccs"
    )

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Vanilla condition
    print(f"\n{'='*60}")
    print(f"Condition: Vanilla ({N_TURNS} turns)")
    print(f"{'='*60}")
    results[args.model]["vanilla"] = run_condition(
        model, tokenizer, n_layers, VANILLA_SYSTEM, VANILLA_PROBES, "vanilla"
    )

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = Path(RESULTS_DIR) / f"exp_activation_alpha_{args.model}_{ts}.json"
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Analysis
    analyze_results(results)


if __name__ == "__main__":
    main()
