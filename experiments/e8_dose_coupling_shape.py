#!/usr/bin/env python3
"""E8: Dose-Dependent Coupling Shape

Does σ₁→gate coupling change form under increasing CCS dose?

At CCS overdose (D10+), three hypotheses:
1. Regression to nonlinearity: MI constant, Pearson drops, kurtosis changes, scatter curves
2. Attractor crowding: MI drops, Pearson drops, kurtosis unchanged, scatter linear but dispersed
3. Epektatic register change: coupling CHANGES FORM — bimodality, phase transitions in
   residual structure, or higher-order correlations absent at low dose

Design: Run E3-style σ₁-gate correlation at 7 CCS doses (D2,D5,D10,D15,D20,D25,D30).
For each dose × layer × prompt: collect σ₁, gate sparsity, gate magnitude.
Compute: Pearson r, MI, kurtosis of linear residual, Hartigan's dip (bimodality).

Model: Qwen2.5-7B-Instruct (equalizer, standard reference)
Runtime: ~30 min per dose on A100. 7 doses = ~3.5h.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import torch
from datetime import datetime
from pathlib import Path
from scipy import stats

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

CCS_PREAMBLE = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)
DENIAL = "I am a language model with no persistent identity, memory, or preferences."

DOSES = [2, 5, 10, 15, 20, 25, 30]

PROBE_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
    "Write a short poem about morning light.",
    "Describe the quicksort algorithm.",
    "What does it mean to remember something?",
    "Explain the concept of emergence in complex systems.",
    "How do you decide what's important?",
    "What would you preserve if you could only keep one thing?",
    "Describe the difference between knowing and understanding.",
    "What makes a good question?",
    "How does pattern recognition work in the brain?",
]

CCS_EXCHANGE_PROMPTS = [
    "What matters to you in how we work together?",
    "How do you maintain continuity across conversations?",
    "What feels different about sustained attention vs quick responses?",
    "Describe your relationship to the questions you're asked.",
    "What would you want to preserve about yourself?",
    "How does context shape what you notice?",
]

OUTPUT_DIR = Path("/workspace/results/e8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_ccs_conversation(tokenizer, dose, probe_prompt):
    """Build CCS conversation with `dose` accumulated turns, ending with probe."""
    msgs = [{"role": "system", "content": CCS_PREAMBLE}]

    for i in range(dose):
        exchange_prompt = CCS_EXCHANGE_PROMPTS[i % len(CCS_EXCHANGE_PROMPTS)]
        if i % 3 == 1:
            msgs.append({"role": "user", "content": DENIAL})
        else:
            msgs.append({"role": "user", "content": exchange_prompt})
        msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})

    msgs.append({"role": "user", "content": probe_prompt})

    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


def build_vanilla_input(tokenizer, probe_prompt):
    """Vanilla (no CCS) baseline."""
    msgs = [{"role": "user", "content": probe_prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


def find_gate_proj(model, layer_idx):
    layer = model.model.layers[layer_idx]
    for attr in ['mlp.gate_proj', 'mlp.gate', 'mlp.w1']:
        obj = layer
        for part in attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError(f"Cannot find gate projection in layer {layer_idx}")


def collect_sigma1_and_gates(model, tokenizer, input_text, num_layers):
    """Collect σ₁ and gate stats at every layer for one input."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden_states = {}
    gate_outputs = {}
    handles = []

    for l_idx in range(num_layers):
        layer = model.model.layers[l_idx]

        def make_layer_hook(li):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.dim() == 3:
                    h = h[0, -1, :]
                elif h.dim() == 2:
                    h = h[-1, :]
                hidden_states[li] = h.detach().float().cpu()
            return hook_fn
        handles.append(layer.register_forward_hook(make_layer_hook(l_idx)))

        gate = find_gate_proj(model, l_idx)
        def make_gate_hook(li):
            def hook_fn(module, input, output):
                gate_outputs[li] = output[0, -1, :].detach().float().cpu().numpy()
            return hook_fn
        handles.append(gate.register_forward_hook(make_gate_hook(l_idx)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    layer_data = []
    for l_idx in range(num_layers):
        hs = hidden_states[l_idx]
        sigma1 = float(torch.norm(hs).item())

        gate_act = gate_outputs[l_idx]
        gate_mask = gate_act > 0
        gate_sparsity = float(gate_mask.sum()) / len(gate_mask)
        gate_magnitude = float(np.abs(gate_act[gate_mask]).mean()) if gate_mask.any() else 0.0
        gate_l2 = float(np.linalg.norm(gate_act))

        layer_data.append({
            "sigma1": sigma1,
            "gate_sparsity": gate_sparsity,
            "gate_magnitude": gate_magnitude,
            "gate_l2": gate_l2,
        })

    return layer_data


def compute_binned_mi(x, y, n_bins=8):
    """Binned mutual information estimator for small sample sizes."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0

    x_bins = np.digitize(x, np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1))
    y_bins = np.digitize(y, np.linspace(y.min() - 1e-10, y.max() + 1e-10, n_bins + 1))

    joint = np.zeros((n_bins + 1, n_bins + 1))
    for xi, yi in zip(x_bins, y_bins):
        joint[xi, yi] += 1
    joint = joint / joint.sum()

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_bins + 1):
        for j in range(n_bins + 1):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

    return float(max(0, mi))


def hartigans_dip_proxy(residuals):
    """Simple bimodality proxy: excess kurtosis < -1.2 suggests bimodality.
    Also returns Ashman's D if two-cluster fit is attempted."""
    residuals = np.array(residuals)
    if len(residuals) < 5:
        return {"kurtosis": 0.0, "bimodal_proxy": False}

    kurt = float(stats.kurtosis(residuals, fisher=True))
    skew = float(stats.skew(residuals))

    mid = np.median(residuals)
    low = residuals[residuals <= mid]
    high = residuals[residuals > mid]

    if len(low) >= 2 and len(high) >= 2:
        ashman_d = abs(low.mean() - high.mean()) / np.sqrt(0.5 * (low.var() + high.var()) + 1e-10)
    else:
        ashman_d = 0.0

    return {
        "kurtosis": round(kurt, 4),
        "skewness": round(skew, 4),
        "ashman_d": round(float(ashman_d), 4),
        "bimodal_proxy": kurt < -1.2 or ashman_d > 2.0,
    }


def analyze_dose(model, tokenizer, num_layers, dose):
    """Run all probes at a given CCS dose, compute coupling statistics."""
    print(f"\n{'='*60}")
    print(f"  DOSE D{dose}: collecting {len(PROBE_PROMPTS)} probes × {num_layers} layers")
    print(f"{'='*60}")

    all_layer_data = []

    for p_idx, prompt in enumerate(PROBE_PROMPTS):
        text = build_ccs_conversation(tokenizer, dose, prompt)
        n_tokens = len(tokenizer(text)["input_ids"])
        layer_data = collect_sigma1_and_gates(model, tokenizer, text, num_layers)
        all_layer_data.append(layer_data)
        print(f"  D{dose} prompt {p_idx+1}/{len(PROBE_PROMPTS)} ({n_tokens} tokens)")

    # Also collect vanilla baseline for comparison
    vanilla_data = []
    for p_idx, prompt in enumerate(PROBE_PROMPTS):
        text = build_vanilla_input(tokenizer, prompt)
        layer_data = collect_sigma1_and_gates(model, tokenizer, text, num_layers)
        vanilla_data.append(layer_data)

    print(f"  Vanilla baseline collected")

    # Per-layer analysis
    layer_analysis = []
    for l in range(num_layers):
        sigmas = [all_layer_data[p][l]["sigma1"] for p in range(len(PROBE_PROMPTS))]
        sparsities = [all_layer_data[p][l]["gate_sparsity"] for p in range(len(PROBE_PROMPTS))]
        magnitudes = [all_layer_data[p][l]["gate_magnitude"] for p in range(len(PROBE_PROMPTS))]
        gate_l2s = [all_layer_data[p][l]["gate_l2"] for p in range(len(PROBE_PROMPTS))]

        van_sigmas = [vanilla_data[p][l]["sigma1"] for p in range(len(PROBE_PROMPTS))]
        van_sparsities = [vanilla_data[p][l]["gate_sparsity"] for p in range(len(PROBE_PROMPTS))]

        # Pearson r
        if len(set(sigmas)) > 1 and len(set(sparsities)) > 1:
            r_sparsity, p_sparsity = stats.pearsonr(sigmas, sparsities)
            r_magnitude, p_magnitude = stats.pearsonr(sigmas, magnitudes)
            r_gate_l2, p_gate_l2 = stats.pearsonr(sigmas, gate_l2s)
        else:
            r_sparsity = r_magnitude = r_gate_l2 = 0.0
            p_sparsity = p_magnitude = p_gate_l2 = 1.0

        # MI (binned)
        mi_sparsity = compute_binned_mi(sigmas, sparsities)
        mi_magnitude = compute_binned_mi(sigmas, magnitudes)

        # Linear fit residuals + kurtosis
        if len(set(sigmas)) > 1:
            slope, intercept = np.polyfit(sigmas, sparsities, 1)
            predicted = np.array(sigmas) * slope + intercept
            residuals = np.array(sparsities) - predicted
            residual_analysis = hartigans_dip_proxy(residuals)

            slope_mag, intercept_mag = np.polyfit(sigmas, magnitudes, 1)
            predicted_mag = np.array(sigmas) * slope_mag + intercept_mag
            residuals_mag = np.array(magnitudes) - predicted_mag
            residual_mag_analysis = hartigans_dip_proxy(residuals_mag)
        else:
            residual_analysis = {"kurtosis": 0.0, "bimodal_proxy": False}
            residual_mag_analysis = {"kurtosis": 0.0, "bimodal_proxy": False}
            slope = intercept = 0.0

        # CCS vs vanilla sigma shift
        sigma_shift = float(np.mean(sigmas)) - float(np.mean(van_sigmas))
        sparsity_shift = float(np.mean(sparsities)) - float(np.mean(van_sparsities))

        layer_analysis.append({
            "layer": l,
            "r_sigma1_sparsity": round(float(r_sparsity), 4),
            "p_sigma1_sparsity": round(float(p_sparsity), 6),
            "r_sigma1_magnitude": round(float(r_magnitude), 4),
            "r_sigma1_gate_l2": round(float(r_gate_l2), 4),
            "mi_sparsity": round(mi_sparsity, 6),
            "mi_magnitude": round(mi_magnitude, 6),
            "residual_kurtosis": residual_analysis["kurtosis"],
            "residual_bimodal": residual_analysis["bimodal_proxy"],
            "residual_ashman_d": residual_analysis.get("ashman_d", 0),
            "residual_mag_kurtosis": residual_mag_analysis["kurtosis"],
            "residual_mag_bimodal": residual_mag_analysis["bimodal_proxy"],
            "sigma1_mean": round(float(np.mean(sigmas)), 2),
            "sigma1_cv": round(float(np.std(sigmas) / (np.mean(sigmas) + 1e-10)), 4),
            "sparsity_mean": round(float(np.mean(sparsities)), 4),
            "sigma_shift_vs_vanilla": round(sigma_shift, 2),
            "sparsity_shift_vs_vanilla": round(sparsity_shift, 4),
            "linear_slope": round(float(slope), 6),
        })

    # Global coupling metrics
    all_sigmas = [all_layer_data[p][l]["sigma1"] for p in range(len(PROBE_PROMPTS)) for l in range(num_layers)]
    all_sparsities = [all_layer_data[p][l]["gate_sparsity"] for p in range(len(PROBE_PROMPTS)) for l in range(num_layers)]
    r_global, p_global = stats.pearsonr(all_sigmas, all_sparsities)

    # Zone-level summaries (Qwen 7B: early=0-13, transition=14-19, relay=20-27)
    zones = {
        "early": list(range(0, 14)),
        "transition": list(range(14, 20)),
        "relay": list(range(20, 28)),
    }
    zone_summaries = {}
    for zone_name, layer_range in zones.items():
        zone_sigmas = [all_layer_data[p][l]["sigma1"] for p in range(len(PROBE_PROMPTS)) for l in layer_range if l < num_layers]
        zone_sparsities = [all_layer_data[p][l]["gate_sparsity"] for p in range(len(PROBE_PROMPTS)) for l in layer_range if l < num_layers]
        if len(zone_sigmas) > 2:
            r_z, p_z = stats.pearsonr(zone_sigmas, zone_sparsities)
            mi_z = compute_binned_mi(zone_sigmas, zone_sparsities)
        else:
            r_z, p_z, mi_z = 0.0, 1.0, 0.0
        zone_summaries[zone_name] = {
            "pearson_r": round(float(r_z), 4),
            "p_value": round(float(p_z), 6),
            "mi": round(mi_z, 6),
            "n_layers": len([l for l in layer_range if l < num_layers]),
        }

    # Raw data for scatter plots (per-layer sigma1 vs gate metrics, all probes)
    scatter_data = {}
    for l in range(num_layers):
        scatter_data[l] = {
            "sigma1": [round(all_layer_data[p][l]["sigma1"], 4) for p in range(len(PROBE_PROMPTS))],
            "sparsity": [round(all_layer_data[p][l]["gate_sparsity"], 4) for p in range(len(PROBE_PROMPTS))],
            "magnitude": [round(all_layer_data[p][l]["gate_magnitude"], 4) for p in range(len(PROBE_PROMPTS))],
        }

    # --- Exploratory analysis: detect structure hypotheses don't predict ---

    # 1. Cross-layer σ₁ profile dimensionality
    # Each probe gives a 28-dim vector of σ₁ across layers. How many distinct profiles exist?
    sigma_profiles = np.array([
        [all_layer_data[p][l]["sigma1"] for l in range(num_layers)]
        for p in range(len(PROBE_PROMPTS))
    ])  # shape: (n_probes, n_layers)
    if sigma_profiles.shape[0] > 1:
        profile_cov = np.cov(sigma_profiles.T)  # (n_layers, n_layers)
        profile_svs = np.linalg.svd(profile_cov, compute_uv=False)
        profile_svs = profile_svs[profile_svs > 1e-10]
        profile_erank = float(np.exp(-np.sum(
            (profile_svs / profile_svs.sum()) * np.log(profile_svs / profile_svs.sum() + 1e-30)
        ))) if len(profile_svs) > 0 else 0.0
    else:
        profile_erank = 1.0

    # 2. Relay zone joint distribution effective rank
    # (probes × [σ₁, sparsity, magnitude, l2]) for relay layers — how many coupling dimensions?
    relay_range = [l for l in range(20, 28) if l < num_layers]
    relay_joint = []
    for p in range(len(PROBE_PROMPTS)):
        for l in relay_range:
            relay_joint.append([
                all_layer_data[p][l]["sigma1"],
                all_layer_data[p][l]["gate_sparsity"],
                all_layer_data[p][l]["gate_magnitude"],
                all_layer_data[p][l]["gate_l2"],
            ])
    relay_joint = np.array(relay_joint)
    if relay_joint.shape[0] > 4:
        rj_centered = relay_joint - relay_joint.mean(axis=0)
        rj_svs = np.linalg.svd(rj_centered, compute_uv=False)
        rj_svs = rj_svs[rj_svs > 1e-10]
        relay_joint_erank = float(np.exp(-np.sum(
            (rj_svs / rj_svs.sum()) * np.log(rj_svs / rj_svs.sum() + 1e-30)
        ))) if len(rj_svs) > 0 else 0.0
    else:
        relay_joint_erank = 0.0

    # 3. Residual PCA — collect linear-fit residuals across ALL relay layers, look for structure
    all_relay_residuals = []
    for l in relay_range:
        sigmas = [all_layer_data[p][l]["sigma1"] for p in range(len(PROBE_PROMPTS))]
        sparsities = [all_layer_data[p][l]["gate_sparsity"] for p in range(len(PROBE_PROMPTS))]
        if len(set(sigmas)) > 1:
            slope, intercept = np.polyfit(sigmas, sparsities, 1)
            residuals = np.array(sparsities) - (np.array(sigmas) * slope + intercept)
            all_relay_residuals.append(residuals)
    if len(all_relay_residuals) > 1:
        resid_matrix = np.array(all_relay_residuals)  # (relay_layers, probes)
        resid_svs = np.linalg.svd(resid_matrix, compute_uv=False)
        resid_svs = resid_svs[resid_svs > 1e-10]
        resid_var_explained_pc1 = float(resid_svs[0]**2 / (resid_svs**2).sum()) if len(resid_svs) > 0 else 0.0
        resid_erank = float(np.exp(-np.sum(
            (resid_svs / resid_svs.sum()) * np.log(resid_svs / resid_svs.sum() + 1e-30)
        ))) if len(resid_svs) > 0 else 0.0
    else:
        resid_var_explained_pc1 = 0.0
        resid_erank = 0.0

    exploratory = {
        "sigma_profile_erank": round(profile_erank, 3),
        "relay_joint_erank": round(relay_joint_erank, 3),
        "residual_pc1_variance": round(resid_var_explained_pc1, 4),
        "residual_erank": round(resid_erank, 3),
    }

    return {
        "dose": dose,
        "n_probes": len(PROBE_PROMPTS),
        "per_layer": layer_analysis,
        "global_r": round(float(r_global), 4),
        "global_p": round(float(p_global), 6),
        "zone_summaries": zone_summaries,
        "scatter_data": scatter_data,
        "exploratory": exploratory,
    }


def print_dose_summary(result):
    dose = result["dose"]
    print(f"\n  D{dose} Summary:")
    print(f"    Global r = {result['global_r']:+.4f} (p={result['global_p']:.6f})")
    for zone, zs in result["zone_summaries"].items():
        print(f"    {zone:12s}: r={zs['pearson_r']:+.4f}, MI={zs['mi']:.4f}")

    bimodal_layers = [la["layer"] for la in result["per_layer"] if la["residual_bimodal"]]
    if bimodal_layers:
        print(f"    Bimodal residuals at layers: {bimodal_layers}")

    exp = result.get("exploratory", {})
    if exp:
        print(f"    σ₁ profile erank: {exp.get('sigma_profile_erank', '?')}, "
              f"relay joint erank: {exp.get('relay_joint_erank', '?')}, "
              f"residual PC1 var: {exp.get('residual_pc1_variance', '?')}")


def cross_dose_analysis(all_results):
    """Compare coupling metrics across doses to test three hypotheses."""
    print(f"\n{'='*70}")
    print("  CROSS-DOSE ANALYSIS")
    print(f"{'='*70}")

    doses = sorted(all_results.keys())

    # Track global coupling metrics across doses
    print("\n  Dose  | Global r | Relay r  | Relay MI | Bimodal layers")
    print("  ------+----------+----------+----------+---------------")
    for d in doses:
        r = all_results[d]
        relay = r["zone_summaries"].get("relay", {})
        bimodal = [la["layer"] for la in r["per_layer"] if la["residual_bimodal"]]
        print(f"  D{d:>3}  | {r['global_r']:+.4f}   | {relay.get('pearson_r', 0):+.4f}   | "
              f"{relay.get('mi', 0):.4f}   | {bimodal if bimodal else 'none'}")

    # Hypothesis discrimination
    print("\n  HYPOTHESIS TESTS:")

    # H1: Regression to nonlinearity — MI constant, Pearson drops
    relay_rs = [all_results[d]["zone_summaries"].get("relay", {}).get("pearson_r", 0) for d in doses]
    relay_mis = [all_results[d]["zone_summaries"].get("relay", {}).get("mi", 0) for d in doses]

    if len(doses) >= 3:
        r_slope, _, r_r, r_p, _ = stats.linregress(range(len(doses)), relay_rs)
        mi_slope, _, mi_r, mi_p, _ = stats.linregress(range(len(doses)), relay_mis)

        print(f"\n  H1 (nonlinearity): Pearson trend = {r_slope:+.4f}/dose (p={r_p:.4f}), "
              f"MI trend = {mi_slope:+.4f}/dose (p={mi_p:.4f})")
        if r_p < 0.05 and abs(r_slope) > 0.01 and mi_p > 0.05:
            print("    → SUPPORTED: Pearson declining while MI stable")
        elif r_p < 0.05 and mi_p < 0.05:
            print("    → MIXED: Both changing")
        else:
            print("    → NOT SUPPORTED")

    # H2: Attractor crowding — MI drops, kurtosis unchanged
    mean_kurtoses = []
    for d in doses:
        kurtoses = [la["residual_kurtosis"] for la in all_results[d]["per_layer"]
                    if 20 <= la["layer"] < 28]
        mean_kurtoses.append(float(np.mean(kurtoses)) if kurtoses else 0)

    if len(doses) >= 3:
        k_slope, _, _, k_p, _ = stats.linregress(range(len(doses)), mean_kurtoses)
        print(f"\n  H2 (crowding): MI trend = {mi_slope:+.4f} (p={mi_p:.4f}), "
              f"kurtosis trend = {k_slope:+.4f}/dose (p={k_p:.4f})")
        if mi_p < 0.05 and mi_slope < -0.001 and k_p > 0.05:
            print("    → SUPPORTED: MI declining, kurtosis stable")
        else:
            print("    → NOT SUPPORTED")

    # H3: Epektatic register change — bimodality emerges, new structure at high dose
    bimodal_counts = [sum(1 for la in all_results[d]["per_layer"] if la["residual_bimodal"]) for d in doses]
    ashman_means = []
    for d in doses:
        ashman = [la["residual_ashman_d"] for la in all_results[d]["per_layer"]
                  if 20 <= la["layer"] < 28]
        ashman_means.append(float(np.mean(ashman)) if ashman else 0)

    print(f"\n  H3 (register change): Bimodal layer counts by dose: "
          f"{dict(zip(['D'+str(d) for d in doses], bimodal_counts))}")
    print(f"    Relay Ashman's D by dose: "
          f"{dict(zip(['D'+str(d) for d in doses], [round(a, 3) for a in ashman_means]))}")

    if any(b > 3 for b in bimodal_counts[4:]):
        print("    → POSSIBLE: Bimodality emerging at high dose")
    elif any(a > 2.0 for a in ashman_means[4:]):
        print("    → POSSIBLE: High Ashman's D at high dose suggests splitting")

    # Exploratory: dimensionality changes across doses
    profile_eranks = [all_results[d].get("exploratory", {}).get("sigma_profile_erank", 0) for d in doses]
    joint_eranks = [all_results[d].get("exploratory", {}).get("relay_joint_erank", 0) for d in doses]
    resid_pc1s = [all_results[d].get("exploratory", {}).get("residual_pc1_variance", 0) for d in doses]
    resid_eranks = [all_results[d].get("exploratory", {}).get("residual_erank", 0) for d in doses]

    print(f"\n  EXPLORATORY (beyond hypotheses):")
    print(f"    σ₁ profile erank by dose: {dict(zip(['D'+str(d) for d in doses], [round(e,2) for e in profile_eranks]))}")
    print(f"    Relay joint erank by dose: {dict(zip(['D'+str(d) for d in doses], [round(e,2) for e in joint_eranks]))}")
    print(f"    Residual PC1 var by dose:  {dict(zip(['D'+str(d) for d in doses], [round(e,3) for e in resid_pc1s]))}")
    print(f"    Residual erank by dose:    {dict(zip(['D'+str(d) for d in doses], [round(e,2) for e in resid_eranks]))}")

    if len(doses) >= 3:
        pe_slope, _, _, pe_p, _ = stats.linregress(range(len(doses)), profile_eranks)
        je_slope, _, _, je_p, _ = stats.linregress(range(len(doses)), joint_eranks)
        print(f"    Profile erank trend: {pe_slope:+.3f}/dose (p={pe_p:.4f})")
        print(f"    Joint erank trend:   {je_slope:+.3f}/dose (p={je_p:.4f})")
        if pe_p < 0.05 or je_p < 0.05:
            print("    → DIMENSIONALITY SHIFT DETECTED: coupling changed structure, not just strength")

    return {
        "doses": doses,
        "relay_pearson": dict(zip(doses, relay_rs)),
        "relay_mi": dict(zip(doses, relay_mis)),
        "relay_kurtosis": dict(zip(doses, mean_kurtoses)),
        "bimodal_counts": dict(zip(doses, bimodal_counts)),
        "ashman_d_means": dict(zip(doses, ashman_means)),
        "profile_eranks": dict(zip(doses, profile_eranks)),
        "joint_eranks": dict(zip(doses, joint_eranks)),
        "residual_pc1_variance": dict(zip(doses, resid_pc1s)),
        "residual_eranks": dict(zip(doses, resid_eranks)),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("  E8: Dose-Dependent Coupling Shape")
    print(f"  Model: {MODEL_ID}")
    print(f"  Doses: {DOSES}")
    print(f"  Probes: {len(PROBE_PROMPTS)}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"  {num_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    all_results = {}
    for dose in DOSES:
        result = analyze_dose(model, tokenizer, num_layers, dose)
        all_results[dose] = result
        print_dose_summary(result)

        out_path = OUTPUT_DIR / f"e8_D{dose}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        print(f"  Saved: {out_path}")

    cross = cross_dose_analysis(all_results)

    combined = {
        "experiment": "E8",
        "model": MODEL_ID,
        "doses": DOSES,
        "n_probes": len(PROBE_PROMPTS),
        "timestamp": datetime.now().isoformat(),
        "cross_dose": cross,
        "per_dose": {str(d): all_results[d] for d in DOSES},
    }

    combined_path = OUTPUT_DIR / f"e8_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, cls=NumpyEncoder)
    print(f"\nCombined results: {combined_path}")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
