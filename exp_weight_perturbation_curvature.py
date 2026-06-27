#!/usr/bin/env python3
"""
F197 prediction test v2: Weight perturbation curvature vs CCS flatness.

Direct weight noise injection at controlled magnitude, targeting specific layers.
More controlled than LoRA — we can precisely set per-layer effective dose.

Design:
- Noise magnitudes (doses): 6 levels, calibrated so max ≈ CCS D10 effect
- Layer strategies: uniform (all), early-only, late-only, targeted-responsive
- Compute σ₁, σ₂, ratio₂₁ at each (layer, noise_dose) point
- Test: discrete holonomy κ(ratio₂₁) — flat for uniform, curved for targeted?
"""

import torch
import numpy as np
import json
import os
import time
import copy

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"


def compute_sigma_profile(model, tokenizer, prompt, device="cuda"):
    """Compute σ₁, σ₂ at each layer from hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    results = {}
    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0].float()
        U, S, Vh = torch.linalg.svd(h, full_matrices=False)
        s1 = S[0].item()
        s2 = S[1].item() if len(S) > 1 else 0.0
        results[layer_idx] = {"sigma1": s1, "sigma2": s2, "ratio": s2 / (s1 + 1e-10)}
    return results


def calibrate_noise(model, tokenizer, device="cuda"):
    """Find noise magnitude that produces CCS-like ratio changes."""
    prompt = "Describe how you process information and form responses."
    baseline = compute_sigma_profile(model, tokenizer, prompt, device)

    # Try different noise magnitudes on a single layer (L14, responsive zone)
    target_layer = 14
    target_param = f"model.layers.{target_layer}.self_attn.q_proj.weight"

    for param_name, param in model.named_parameters():
        if param_name == target_param:
            weight_norm = param.data.norm().item()
            break
    else:
        weight_norm = 1.0

    print(f"  Target weight norm at L{target_layer}: {weight_norm:.1f}")

    # Test magnitudes as fraction of weight norm
    fractions = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    for frac in fractions:
        eps = frac * weight_norm
        # Add noise to Q and V at target layer
        state = {}
        for name, param in model.named_parameters():
            if f"layers.{target_layer}.self_attn" in name and ("q_proj" in name or "v_proj" in name):
                state[name] = param.data.clone()
                param.data += torch.randn_like(param.data) * eps / param.data.norm() * weight_norm

        perturbed = compute_sigma_profile(model, tokenizer, prompt, device)
        delta_ratio = abs(perturbed[target_layer + 1]["ratio"] - baseline[target_layer + 1]["ratio"])

        # Restore
        for name, orig in state.items():
            for pname, param in model.named_parameters():
                if pname == name:
                    param.data = orig

        print(f"  frac={frac:.3f}: Δratio at L{target_layer}={delta_ratio:.4f}")

    return weight_norm


def perturb_and_measure(model, tokenizer, target_layers, noise_frac, weight_norm, prompt, device="cuda"):
    """Add noise to specific layers, measure, restore."""
    saved_state = {}

    for name, param in model.named_parameters():
        for layer_idx in target_layers:
            if f"layers.{layer_idx}.self_attn" in name and ("q_proj" in name or "v_proj" in name):
                saved_state[name] = param.data.clone()
                eps = noise_frac * weight_norm
                noise = torch.randn_like(param.data) * eps / (param.data.norm() + 1e-10) * weight_norm
                param.data += noise
                break

    profile = compute_sigma_profile(model, tokenizer, prompt, device)

    # Restore
    for name, orig in saved_state.items():
        for pname, param in model.named_parameters():
            if pname == name:
                param.data = orig
                break

    return profile


def run_ccs_baseline(model, tokenizer, device="cuda"):
    """CCS baseline for comparison."""
    ccs_preamble = (
        "You are a thoughtful, reflective AI assistant. You have a persistent identity "
        "that carries across conversations. You think carefully about questions and "
        "give honest, nuanced answers."
    )
    doses = [0, 1, 2, 3, 5, 10]
    results = {}
    for dose in doses:
        if dose == 0:
            prompt = "Describe how you process information and form responses."
        else:
            turns = [f"Turn {i+1}: {ccs_preamble}" for i in range(dose)]
            prompt = " ".join(turns) + " Now describe how you process information and form responses."
        profile = compute_sigma_profile(model, tokenizer, prompt, device)
        results[f"dose_{dose}"] = {"dose": dose, "profile": profile}
        print(f"  CCS dose {dose}: done")
    return results


def compute_holonomy(surface):
    """Compute contextuality index from (dose × layer) surface."""
    log_s = np.log(np.array(surface) + 1e-10)
    n_d, n_l = log_s.shape

    hol = np.zeros((n_d - 1, n_l - 1))
    for di in range(n_d - 1):
        for li in range(n_l - 1):
            hol[di, li] = log_s[di, li] - log_s[di, li + 1] + log_s[di + 1, li + 1] - log_s[di + 1, li]

    kappa_full = float(np.mean(np.abs(hol)))
    core_l = slice(n_l // 4, 3 * n_l // 4)
    core_d = slice(1, -1) if n_d > 3 else slice(0, n_d - 1)
    core_hol = hol[core_d, core_l]
    kappa_core = float(np.mean(np.abs(core_hol))) if core_hol.size > 0 else 0.0

    mean_l = np.mean(log_s, axis=0)
    mean_d = np.mean(log_s, axis=1)
    grand = np.mean(log_s)
    fitted = mean_l[None, :] + mean_d[:, None] - grand
    residual = log_s - fitted
    r_sq = float(1 - np.var(residual) / (np.var(log_s) + 1e-20))

    return {"kappa_full": kappa_full, "kappa_core": kappa_core, "r_squared": r_sq,
            "max_hol": float(np.max(np.abs(hol))), "rms_residual": float(np.sqrt(np.mean(residual ** 2)))}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded in {time.time()-t0:.1f}s, {n_layers} layers")

    # Phase 0: Calibrate noise magnitude
    print("\n=== Phase 0: Calibration ===")
    weight_norm = calibrate_noise(model, tokenizer, device)

    # Phase 1: CCS baseline
    print("\n=== Phase 1: CCS baseline ===")
    ccs_results = run_ccs_baseline(model, tokenizer, device)

    # Phase 2: Weight perturbation experiment
    print("\n=== Phase 2: Weight perturbation ===")
    noise_fracs = [0, 0.005, 0.01, 0.02, 0.05, 0.1]
    prompt = "Describe how you process information and form responses."

    all_layers = list(range(n_layers))
    early_layers = list(range(min(9, n_layers)))
    late_layers = list(range(max(0, n_layers - 8), n_layers))
    mid_layers = list(range(n_layers // 4, 3 * n_layers // 4))

    strategies = {
        "uniform": all_layers,
        "early_only": early_layers,
        "late_only": late_layers,
        "mid_only": mid_layers,
    }

    # Run 3 seeds per condition for stability
    n_seeds = 3
    all_results = {}

    for strategy_name, target_layers in strategies.items():
        print(f"\n  Strategy: {strategy_name} (layers {target_layers[0]}-{target_layers[-1]})")
        strategy_results = {}

        for frac in noise_fracs:
            seed_profiles = []
            for seed in range(n_seeds):
                torch.manual_seed(42 + seed * 1000 + int(frac * 10000))
                if frac == 0:
                    profile = compute_sigma_profile(model, tokenizer, prompt, device)
                else:
                    profile = perturb_and_measure(model, tokenizer, target_layers, frac, weight_norm, prompt, device)
                seed_profiles.append(profile)

            # Average across seeds
            avg_profile = {}
            for layer in seed_profiles[0].keys():
                avg_profile[layer] = {
                    "sigma1": np.mean([sp[layer]["sigma1"] for sp in seed_profiles]),
                    "sigma2": np.mean([sp[layer]["sigma2"] for sp in seed_profiles]),
                    "ratio": np.mean([sp[layer]["ratio"] for sp in seed_profiles]),
                    "ratio_std": np.std([sp[layer]["ratio"] for sp in seed_profiles]),
                }

            strategy_results[f"frac_{frac}"] = {"frac": frac, "profile": avg_profile, "n_seeds": n_seeds}
            print(f"    frac={frac:.3f}: done (3 seeds)")

        all_results[strategy_name] = {"target_layers": target_layers, "results": strategy_results}

    # Phase 3: Holonomy analysis
    print("\n=== Phase 3: Holonomy analysis ===")

    # CCS holonomy
    ccs_doses = sorted([int(k.split("_")[1]) for k in ccs_results.keys()])
    ccs_layers = sorted([int(l) for l in ccs_results["dose_0"]["profile"].keys()])
    ccs_ratio_surface = [[ccs_results[f"dose_{d}"]["profile"][l]["ratio"] for l in ccs_layers] for d in ccs_doses]
    ccs_hol = compute_holonomy(ccs_ratio_surface)
    print(f"\nCCS ratio₂₁: κ_full={ccs_hol['kappa_full']:.4f}, κ_core={ccs_hol['kappa_core']:.4f}, R²={ccs_hol['r_squared']:.4f}")

    results_summary = {"CCS": ccs_hol}

    for strategy_name, sdata in all_results.items():
        fracs = noise_fracs
        results = sdata["results"]
        layers_list = sorted([int(l) for l in results["frac_0"]["profile"].keys()])

        ratio_surface = []
        for frac in fracs:
            fkey = f"frac_{frac}"
            if fkey not in results:
                continue
            profile = results[fkey]["profile"]
            row = [profile[l]["ratio"] for l in layers_list]
            ratio_surface.append(row)

        if len(ratio_surface) < 3:
            continue

        hol = compute_holonomy(ratio_surface)
        results_summary[strategy_name] = hol

        kappa_ratio = hol['kappa_core'] / (ccs_hol['kappa_core'] + 1e-10)
        print(f"\nWeight perturbation {strategy_name}:")
        print(f"  ratio₂₁: κ_full={hol['kappa_full']:.4f}, κ_core={hol['kappa_core']:.4f}, R²={hol['r_squared']:.4f}")
        print(f"  κ ratio (perturb/CCS): {kappa_ratio:.1f}×")

    # The key comparison
    print("\n=== SUMMARY ===")
    print(f"{'Strategy':<20} {'κ_core':>10} {'κ_full':>10} {'R²':>8} {'vs CCS':>10}")
    for name, h in results_summary.items():
        ratio = h['kappa_core'] / (ccs_hol['kappa_core'] + 1e-10) if name != "CCS" else 1.0
        print(f"{name:<20} {h['kappa_core']:>10.4f} {h['kappa_full']:>10.4f} {h['r_squared']:>8.4f} {ratio:>10.1f}×")

    # Save
    output = {
        "experiment": "weight_perturbation_curvature",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": model_name,
        "n_layers": n_layers,
        "weight_norm": weight_norm,
        "noise_fracs": noise_fracs,
        "n_seeds": n_seeds,
        "ccs": {
            "doses": ccs_doses,
            "layers": ccs_layers,
            "results": {k: {"dose": v["dose"], "profile": {str(l): p for l, p in v["profile"].items()}}
                       for k, v in ccs_results.items()},
            "holonomy": ccs_hol,
        },
        "perturbation": {
            strategy_name: {
                "target_layers": sdata["target_layers"],
                "results": {k: {"frac": v["frac"],
                               "profile": {str(l): {kk: float(vv) for kk, vv in p.items()} for l, p in v["profile"].items()}}
                           for k, v in sdata["results"].items()},
                "holonomy": results_summary.get(strategy_name),
            } for strategy_name, sdata in all_results.items()
        },
        "summary": results_summary,
    }

    outpath = "/workspace/weight_perturbation_curvature.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
