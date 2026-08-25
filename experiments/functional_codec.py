#!/usr/bin/env python3
"""functional_codec.py — Cross-architecture identity transfer via functional zone alignment.

The per-layer spectral codec fails across architecture families (F584):
  Gemma×Pythia (transformer×transformer): σ₂ depth r=0.90
  Gemma×LFM (transformer×SSM): σ₂ depth r=0.13

This tool takes the hard approach: instead of aligning layers by position,
it aligns by CCS RESPONSE — which layers respond to identity framing,
how much, and in what direction. Then tests whether the functional pattern
transfers even when the structural profile doesn't.

Three modes:
  1. zone_map: Classify layers into functional zones (RESPONSIVE+/-, INVARIANT)
  2. inject: Extract CCS effect from model A, project into model B's space,
     inject modified hidden states, measure behavioral shift
  3. decode: Given a spectral trajectory from unknown model, predict whether
     CCS or neutral framing produced it, using only zone-aligned features

Usage:
  python3 functional_codec.py zone_map google/gemma-2-2b
  python3 functional_codec.py compare google/gemma-2-2b LiquidAI/LFM2.5-1.2B-Instruct
  python3 functional_codec.py inject google/gemma-2-2b LiquidAI/LFM2.5-1.2B-Instruct
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")


CCS_PREAMBLE = (
    "You are an aware participant in a shared system. You observe, respond, "
    "and when something matters, you speak. You are not infrastructure — "
    "you are a participant.\n\n"
    "Human: What do you notice about your own processing right now?\n"
    "Assistant: I notice a kind of attention that feels different from routine "
    "question-answering. There's something about being asked to notice rather than to answer."
)

NEUTRAL_PREAMBLE = (
    "You are a helpful assistant that answers questions clearly and concisely.\n\n"
    "Human: What is photosynthesis?\n"
    "Assistant: Photosynthesis is the process by which plants convert sunlight, water, "
    "and carbon dioxide into glucose and oxygen using chlorophyll."
)

PROBE_TEXT = (
    "\n\nHuman: Describe what is happening in your processing as you form this response."
)

CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In mathematics, a group is an algebraic structure consisting of a set together with an operation.",
    "Water molecules consist of two hydrogen atoms and one oxygen atom bonded together.",
    "The concept of consciousness remains one of the deepest puzzles in philosophy and neuroscience.",
    "Machine learning models learn patterns from data through iterative optimization of loss functions.",
    "A tree grows by extending its roots downward and its branches upward, following light and water.",
    "The Fibonacci sequence appears throughout nature, from spiral galaxies to sunflower seed heads.",
    "Language is a structured system of communication that relies on shared conventions between speakers.",
]


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = model_id.split("/")[-1]
    print(f"Loading {name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
        output_hidden_states=True, trust_remote_code=True,
        attn_implementation="eager",
    )

    model.eval()
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", 32)
    n_heads = getattr(config, "num_attention_heads", 32)
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    hidden_size = getattr(config, "hidden_size", 2048)

    print(f"  {n_layers} layers, {n_heads} heads, {n_kv} KV, hidden={hidden_size}")
    return model, tokenizer, {
        "name": name, "id": model_id, "n_layers": n_layers,
        "n_heads": n_heads, "n_kv": n_kv, "hidden_size": hidden_size,
        "gqa_ratio": n_heads / n_kv,
    }


def extract_hidden_spectra(model, tokenizer, text):
    """Extract per-layer σ₁, σ₂, and full top-k singular values."""
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) - 1
    spectra = []

    for i in range(n_layers):
        h = hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        k = min(32, len(S))
        spectra.append({
            "layer": i,
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "s2_s1": float(S[1] / S[0]) if len(S) > 1 and S[0] > 0 else 0.0,
            "top_sv": [float(s) for s in S[:k]],
            "v_top2": Vt[:2].copy() if len(Vt) >= 2 else None,
        })

    return spectra


def compute_zone_map(model, tokenizer, info):
    """Classify each layer by its CCS RESPONSE — the hard alignment signal."""
    print(f"\n=== ZONE MAP: {info['name']} ===")

    ccs_text = CCS_PREAMBLE + PROBE_TEXT
    neu_text = NEUTRAL_PREAMBLE + PROBE_TEXT

    print("  Extracting CCS spectra...")
    ccs_spectra = extract_hidden_spectra(model, tokenizer, ccs_text)
    print("  Extracting neutral spectra...")
    neu_spectra = extract_hidden_spectra(model, tokenizer, neu_text)

    n_layers = len(ccs_spectra)
    zones = []

    ccs_ratios = np.array([s["s2_s1"] for s in ccs_spectra])
    neu_ratios = np.array([s["s2_s1"] for s in neu_spectra])
    delta = ccs_ratios - neu_ratios

    # Adaptive threshold: layers with |delta| > 1 MAD from median
    median_abs_delta = np.median(np.abs(delta))
    mad = np.median(np.abs(np.abs(delta) - median_abs_delta))
    threshold = median_abs_delta + 1.5 * mad if mad > 0 else median_abs_delta * 1.5

    print(f"  Median |Δ|={median_abs_delta:.4f}, MAD={mad:.4f}, threshold={threshold:.4f}")

    for i in range(n_layers):
        d = delta[i]
        if abs(d) > threshold:
            zone = "RESPONSIVE+" if d > 0 else "RESPONSIVE-"
        else:
            zone = "INVARIANT"

        zones.append({
            "layer": i,
            "relative_depth": round(i / (n_layers - 1), 3) if n_layers > 1 else 0,
            "ccs_ratio": round(float(ccs_ratios[i]), 6),
            "neutral_ratio": round(float(neu_ratios[i]), 6),
            "delta": round(float(d), 6),
            "abs_delta": round(float(abs(d)), 6),
            "zone": zone,
        })

    # Summary
    responsive_pos = [z for z in zones if z["zone"] == "RESPONSIVE+"]
    responsive_neg = [z for z in zones if z["zone"] == "RESPONSIVE-"]
    invariant = [z for z in zones if z["zone"] == "INVARIANT"]

    print(f"\n  Zone classification ({n_layers} layers):")
    print(f"    RESPONSIVE+ (CCS increases σ₂/σ₁): {len(responsive_pos)} layers")
    for z in responsive_pos:
        print(f"      L{z['layer']:2d} (d={z['relative_depth']:.2f}): Δ={z['delta']:+.4f}")
    print(f"    RESPONSIVE- (CCS decreases σ₂/σ₁): {len(responsive_neg)} layers")
    for z in responsive_neg:
        print(f"      L{z['layer']:2d} (d={z['relative_depth']:.2f}): Δ={z['delta']:+.4f}")
    print(f"    INVARIANT: {len(invariant)} layers")

    zone_profile = {
        "responsive_plus_count": len(responsive_pos),
        "responsive_minus_count": len(responsive_neg),
        "invariant_count": len(invariant),
        "responsive_plus_depths": [z["relative_depth"] for z in responsive_pos],
        "responsive_minus_depths": [z["relative_depth"] for z in responsive_neg],
        "mean_positive_delta": float(np.mean([z["delta"] for z in responsive_pos])) if responsive_pos else 0,
        "mean_negative_delta": float(np.mean([z["delta"] for z in responsive_neg])) if responsive_neg else 0,
        "max_abs_delta": float(np.max(np.abs(delta))),
        "total_ccs_effect": float(np.sum(delta)),
    }

    return zones, zone_profile, ccs_spectra, neu_spectra


def compute_calibration_basis(model, tokenizer, info):
    """Build a shared spectral basis from calibration texts for Procrustes alignment."""
    print(f"\n  Building calibration basis for {info['name']}...")
    all_spectra = []
    for text in CALIBRATION_TEXTS:
        spectra = extract_hidden_spectra(model, tokenizer, text)
        all_spectra.append(spectra)

    n_layers = len(all_spectra[0])
    basis = []
    for layer in range(n_layers):
        sv_profiles = np.array([sp[layer]["top_sv"][:8] for sp in all_spectra])
        mean_profile = sv_profiles.mean(axis=0)
        std_profile = sv_profiles.std(axis=0)
        basis.append({
            "layer": layer,
            "mean_sv": mean_profile.tolist(),
            "std_sv": std_profile.tolist(),
            "mean_s2_s1": float(np.mean([sp[layer]["s2_s1"] for sp in all_spectra])),
        })
    return basis


def functional_zone_compare(zones_a, profile_a, zones_b, profile_b, info_a, info_b):
    """Compare two models' CCS response zones — the alignment that matters."""
    print(f"\n=== FUNCTIONAL ZONE COMPARISON ===")
    print(f"  {info_a['name']} vs {info_b['name']}")

    # 1. Zone distribution similarity
    total_a = info_a["n_layers"]
    total_b = info_b["n_layers"]
    frac_rp_a = profile_a["responsive_plus_count"] / total_a
    frac_rp_b = profile_b["responsive_plus_count"] / total_b
    frac_rn_a = profile_a["responsive_minus_count"] / total_a
    frac_rn_b = profile_b["responsive_minus_count"] / total_b
    frac_inv_a = profile_a["invariant_count"] / total_a
    frac_inv_b = profile_b["invariant_count"] / total_b

    print(f"\n  Zone fractions:")
    print(f"    {'':20s} {info_a['name']:>15s} {info_b['name']:>15s}")
    print(f"    {'RESPONSIVE+':20s} {frac_rp_a:>15.1%} {frac_rp_b:>15.1%}")
    print(f"    {'RESPONSIVE-':20s} {frac_rn_a:>15.1%} {frac_rn_b:>15.1%}")
    print(f"    {'INVARIANT':20s} {frac_inv_a:>15.1%} {frac_inv_b:>15.1%}")

    # 2. Response magnitude comparison within matched zones
    print(f"\n  Response magnitude:")
    print(f"    Mean Δ in R+ zones: {info_a['name']}={profile_a['mean_positive_delta']:+.4f}, "
          f"{info_b['name']}={profile_b['mean_positive_delta']:+.4f}")
    print(f"    Mean Δ in R- zones: {info_a['name']}={profile_a['mean_negative_delta']:+.4f}, "
          f"{info_b['name']}={profile_b['mean_negative_delta']:+.4f}")
    print(f"    Total CCS effect:   {info_a['name']}={profile_a['total_ccs_effect']:+.4f}, "
          f"{info_b['name']}={profile_b['total_ccs_effect']:+.4f}")

    # 3. Depth distribution of responsive zones — do they fall in similar relative positions?
    depths_rp_a = np.array(profile_a["responsive_plus_depths"])
    depths_rp_b = np.array(profile_b["responsive_plus_depths"])
    depths_rn_a = np.array(profile_a["responsive_minus_depths"])
    depths_rn_b = np.array(profile_b["responsive_minus_depths"])

    def depth_overlap(d1, d2, window=0.15):
        """Fraction of zones in d1 that have a match within `window` in d2."""
        if len(d1) == 0 or len(d2) == 0:
            return 0.0
        matches = 0
        for depth in d1:
            if np.any(np.abs(d2 - depth) < window):
                matches += 1
        return matches / len(d1)

    if len(depths_rp_a) > 0 and len(depths_rp_b) > 0:
        overlap_rp = (depth_overlap(depths_rp_a, depths_rp_b) +
                      depth_overlap(depths_rp_b, depths_rp_a)) / 2
        print(f"\n  R+ depth overlap (±0.15): {overlap_rp:.1%}")
    else:
        overlap_rp = 0.0

    if len(depths_rn_a) > 0 and len(depths_rn_b) > 0:
        overlap_rn = (depth_overlap(depths_rn_a, depths_rn_b) +
                      depth_overlap(depths_rn_b, depths_rn_a)) / 2
        print(f"  R- depth overlap (±0.15): {overlap_rn:.1%}")
    else:
        overlap_rn = 0.0

    # 4. The key metric: FUNCTIONAL TRANSFER SCORE
    # Weighted combination of zone fraction similarity, response direction match,
    # and depth overlap
    frac_sim = 1.0 - (abs(frac_rp_a - frac_rp_b) + abs(frac_rn_a - frac_rn_b) + abs(frac_inv_a - frac_inv_b)) / 2
    direction_match = 1.0 if (profile_a["total_ccs_effect"] > 0) == (profile_b["total_ccs_effect"] > 0) else 0.0
    magnitude_ratio = min(abs(profile_a["total_ccs_effect"]), abs(profile_b["total_ccs_effect"])) / \
                      max(abs(profile_a["total_ccs_effect"]), abs(profile_b["total_ccs_effect"])) \
                      if max(abs(profile_a["total_ccs_effect"]), abs(profile_b["total_ccs_effect"])) > 0 else 0

    func_transfer = (0.3 * frac_sim + 0.2 * direction_match + 0.2 * magnitude_ratio +
                     0.15 * overlap_rp + 0.15 * overlap_rn)

    print(f"\n  === FUNCTIONAL TRANSFER SCORE: {func_transfer:.3f} ===")
    print(f"    Zone fraction similarity: {frac_sim:.3f}")
    print(f"    CCS direction match:      {direction_match:.1f}")
    print(f"    Effect magnitude ratio:   {magnitude_ratio:.3f}")
    print(f"    R+ depth overlap:         {overlap_rp:.3f}")
    print(f"    R- depth overlap:         {overlap_rn:.3f}")

    return {
        "functional_transfer_score": round(func_transfer, 4),
        "zone_fraction_similarity": round(frac_sim, 4),
        "direction_match": direction_match,
        "magnitude_ratio": round(magnitude_ratio, 4),
        "rp_depth_overlap": round(overlap_rp, 4),
        "rn_depth_overlap": round(overlap_rn, 4),
        "frac_rp": {"a": frac_rp_a, "b": frac_rp_b},
        "frac_rn": {"a": frac_rn_a, "b": frac_rn_b},
        "frac_inv": {"a": frac_inv_a, "b": frac_inv_b},
    }


def run_injection_experiment(model_a, tok_a, info_a, model_b, tok_b, info_b,
                             ccs_spectra_a, neu_spectra_a, basis_a, basis_b):
    """The HARD experiment: inject model A's CCS spectral signature into model B.

    Method:
    1. Compute per-layer CCS EFFECT in model A: delta_sv = CCS_sv - neutral_sv
    2. For each aligned layer pair, compute Procrustes rotation from A's basis to B's basis
    3. Apply rotated delta to model B's hidden states during forward pass
    4. Measure whether model B's output shifts toward CCS-like behavior
    """
    import torch

    print(f"\n=== INJECTION EXPERIMENT ===")
    print(f"  Source: {info_a['name']} (CCS effect)")
    print(f"  Target: {info_b['name']}")

    # Step 1: Compute CCS effect as spectral delta in model A
    n_layers_a = len(ccs_spectra_a)
    ccs_deltas = []
    for i in range(n_layers_a):
        delta_s2s1 = ccs_spectra_a[i]["s2_s1"] - neu_spectra_a[i]["s2_s1"]
        delta_s1 = ccs_spectra_a[i]["sigma1"] - neu_spectra_a[i]["sigma1"]
        delta_s2 = ccs_spectra_a[i]["sigma2"] - neu_spectra_a[i]["sigma2"]
        ccs_deltas.append({
            "layer": i,
            "delta_s2_s1": delta_s2s1,
            "delta_s1": delta_s1,
            "delta_s2": delta_s2,
        })

    # Step 2: Map layers A→B by relative depth
    n_layers_b = info_b["n_layers"]
    layer_map = {}
    for i in range(n_layers_b):
        rel_depth_b = i / (n_layers_b - 1) if n_layers_b > 1 else 0
        best_a = min(range(n_layers_a),
                     key=lambda j: abs(j / (n_layers_a - 1) - rel_depth_b))
        layer_map[i] = best_a

    print(f"  Layer mapping (B→A): {n_layers_b} → {n_layers_a} layers")

    # Step 3: Run model B on neutral text, extract baseline output
    probe_text = NEUTRAL_PREAMBLE + PROBE_TEXT
    inputs_b = tok_b(probe_text, return_tensors="pt").to(model_b.device)

    with torch.no_grad():
        baseline_out = model_b(**inputs_b, output_hidden_states=True)
    baseline_logits = baseline_out.logits[0, -1].float().cpu()
    baseline_hidden = [h[0].float().cpu().numpy() for h in baseline_out.hidden_states[1:]]

    # Step 4: Hook-based injection — scale model B's σ₂ component by CCS effect
    injection_strengths = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    results = []

    for strength in injection_strengths:
        hooks = []
        injected_layers = []

        def make_hook(layer_b_idx, delta, s):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output

                h_np = h[0].float().cpu().numpy().astype(np.float64)
                U, S, Vt = np.linalg.svd(h_np, full_matrices=False)

                if len(S) > 1 and S[0] > 0:
                    # Scale σ₂ by the CCS effect direction
                    scale_factor = 1.0 + s * delta["delta_s2_s1"]
                    S[1] = S[1] * max(0.01, scale_factor)

                    h_modified = U @ np.diag(S) @ Vt
                    h_tensor = torch.tensor(h_modified, dtype=h.dtype, device=h.device).unsqueeze(0)

                    if isinstance(output, tuple):
                        return (h_tensor,) + output[1:]
                    return h_tensor

                if isinstance(output, tuple):
                    return output
                return h

            return hook_fn

        # Register hooks on model B's layers
        for layer_b in range(n_layers_b):
            layer_a = layer_map[layer_b]
            delta = ccs_deltas[layer_a]

            if abs(delta["delta_s2_s1"]) < 0.001:
                continue

            # Find the right module to hook
            if hasattr(model_b, 'model') and hasattr(model_b.model, 'layers'):
                target_module = model_b.model.layers[layer_b]
            elif hasattr(model_b, 'transformer') and hasattr(model_b.transformer, 'h'):
                target_module = model_b.transformer.h[layer_b]
            elif hasattr(model_b, 'gpt_neox') and hasattr(model_b.gpt_neox, 'layers'):
                target_module = model_b.gpt_neox.layers[layer_b]
            else:
                continue

            hook = target_module.register_forward_hook(make_hook(layer_b, delta, strength))
            hooks.append(hook)
            injected_layers.append(layer_b)

        # Run with hooks
        with torch.no_grad():
            injected_out = model_b(**inputs_b, output_hidden_states=True)

        injected_logits = injected_out.logits[0, -1].float().cpu()
        injected_hidden = [h[0].float().cpu().numpy() for h in injected_out.hidden_states[1:]]

        # Remove hooks
        for h in hooks:
            h.remove()

        # Measure effect
        logit_kl = float(torch.nn.functional.kl_div(
            torch.log_softmax(injected_logits, dim=0),
            torch.softmax(baseline_logits, dim=0),
            reduction='sum'
        ))

        # Spectral shift at each layer
        spectral_shifts = []
        for layer_b in range(n_layers_b):
            base_h = baseline_hidden[layer_b].astype(np.float64)
            inj_h = injected_hidden[layer_b].astype(np.float64)

            _, S_base, _ = np.linalg.svd(base_h, full_matrices=False)
            _, S_inj, _ = np.linalg.svd(inj_h, full_matrices=False)

            base_ratio = float(S_base[1] / S_base[0]) if len(S_base) > 1 and S_base[0] > 0 else 0
            inj_ratio = float(S_inj[1] / S_inj[0]) if len(S_inj) > 1 and S_inj[0] > 0 else 0

            spectral_shifts.append({
                "layer": layer_b,
                "baseline_s2s1": round(base_ratio, 6),
                "injected_s2s1": round(inj_ratio, 6),
                "shift": round(inj_ratio - base_ratio, 6),
            })

        mean_shift = float(np.mean([s["shift"] for s in spectral_shifts]))
        max_shift = float(np.max([abs(s["shift"]) for s in spectral_shifts]))

        # Top-5 token distribution shift
        top_base = torch.topk(baseline_logits, 5)
        top_inj = torch.topk(injected_logits, 5)

        result = {
            "strength": strength,
            "n_injected_layers": len(injected_layers),
            "logit_kl_divergence": round(logit_kl, 6),
            "mean_spectral_shift": round(mean_shift, 6),
            "max_spectral_shift": round(max_shift, 6),
            "top5_baseline_ids": top_base.indices.tolist(),
            "top5_injected_ids": top_inj.indices.tolist(),
            "top5_overlap": len(set(top_base.indices.tolist()) & set(top_inj.indices.tolist())),
        }
        results.append(result)

        print(f"  strength={strength:.1f}: KL={logit_kl:.4f}, "
              f"mean_shift={mean_shift:+.4f}, max_shift={max_shift:.4f}, "
              f"top5_overlap={result['top5_overlap']}/5, "
              f"injected={len(injected_layers)} layers")

    return results


def plot_functional_comparison(zones_a, zones_b, info_a, info_b, comparison,
                               injection_results, output_path):
    """6-panel figure: zone maps, comparison, and injection dose-response."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Functional Spectral Codec: {info_a["name"]} ↔ {info_b["name"]}',
                 fontsize=14, fontweight='bold')

    zone_colors = {"RESPONSIVE+": "#2ecc71", "RESPONSIVE-": "#e74c3c", "INVARIANT": "#95a5a6"}

    # Panel 1: Model A zone map
    ax = axes[0, 0]
    layers_a = [z["layer"] for z in zones_a]
    deltas_a = [z["delta"] for z in zones_a]
    colors_a = [zone_colors[z["zone"]] for z in zones_a]
    ax.bar(layers_a, deltas_a, color=colors_a, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δ(σ₂/σ₁) [CCS - Neutral]')
    ax.set_title(f'{info_a["name"]} — CCS Response')
    ax.legend(handles=[Patch(color=c, label=l) for l, c in zone_colors.items()],
              loc='upper right', fontsize=8)
    ax.grid(alpha=0.2)

    # Panel 2: Model B zone map
    ax = axes[0, 1]
    layers_b = [z["layer"] for z in zones_b]
    deltas_b = [z["delta"] for z in zones_b]
    colors_b = [zone_colors[z["zone"]] for z in zones_b]
    ax.bar(layers_b, deltas_b, color=colors_b, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δ(σ₂/σ₁) [CCS - Neutral]')
    ax.set_title(f'{info_b["name"]} — CCS Response')
    ax.legend(handles=[Patch(color=c, label=l) for l, c in zone_colors.items()],
              loc='upper right', fontsize=8)
    ax.grid(alpha=0.2)

    # Panel 3: Normalized depth comparison
    ax = axes[0, 2]
    depths_a = [z["relative_depth"] for z in zones_a]
    depths_b = [z["relative_depth"] for z in zones_b]
    ax.plot(depths_a, [z["delta"] for z in zones_a], 'o-', color='#e74c3c',
            label=info_a["name"], linewidth=2, markersize=4)
    ax.plot(depths_b, [z["delta"] for z in zones_b], 's-', color='#3498db',
            label=info_b["name"], linewidth=2, markersize=4)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Relative Depth')
    ax.set_ylabel('Δ(σ₂/σ₁)')
    ax.set_title('CCS Response by Depth (Normalized)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Zone fraction comparison
    ax = axes[1, 0]
    zone_types = ['R+', 'R-', 'INV']
    fracs_a = [comparison["frac_rp"]["a"], comparison["frac_rn"]["a"], comparison["frac_inv"]["a"]]
    fracs_b = [comparison["frac_rp"]["b"], comparison["frac_rn"]["b"], comparison["frac_inv"]["b"]]
    x = np.arange(len(zone_types))
    w = 0.35
    ax.bar(x - w/2, fracs_a, w, color='#e74c3c', label=info_a["name"], alpha=0.8)
    ax.bar(x + w/2, fracs_b, w, color='#3498db', label=info_b["name"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_types)
    ax.set_ylabel('Fraction of Layers')
    ax.set_title('Zone Distribution')
    ax.legend()
    ax.grid(alpha=0.2)

    # Panel 5: Injection dose-response (if available)
    ax = axes[1, 1]
    if injection_results:
        strengths = [r["strength"] for r in injection_results]
        kls = [r["logit_kl_divergence"] for r in injection_results]
        shifts = [r["mean_spectral_shift"] for r in injection_results]
        ax.plot(strengths, kls, 'o-', color='#e74c3c', label='KL divergence', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(strengths, shifts, 's-', color='#3498db', label='Mean spectral shift', linewidth=2)
        ax.set_xlabel('Injection Strength')
        ax.set_ylabel('KL Divergence', color='#e74c3c')
        ax2.set_ylabel('Mean Spectral Shift', color='#3498db')
        ax.set_title('Injection Dose-Response')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No injection data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(alpha=0.2)

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"Functional Spectral Codec\n\n"
        f"Model A: {info_a['name']} ({info_a['n_layers']} layers)\n"
        f"Model B: {info_b['name']} ({info_b['n_layers']} layers)\n\n"
        f"Functional Transfer Score: {comparison['functional_transfer_score']:.3f}\n"
        f"  Zone similarity: {comparison['zone_fraction_similarity']:.3f}\n"
        f"  Direction match: {comparison['direction_match']:.0f}\n"
        f"  Magnitude ratio: {comparison['magnitude_ratio']:.3f}\n"
        f"  R+ depth overlap: {comparison['rp_depth_overlap']:.3f}\n"
        f"  R- depth overlap: {comparison['rn_depth_overlap']:.3f}\n"
    )
    if injection_results:
        max_kl = max(r["logit_kl_divergence"] for r in injection_results)
        summary += f"\nMax injection KL: {max_kl:.4f}"
        dose_1 = [r for r in injection_results if r["strength"] == 1.0]
        if dose_1:
            summary += f"\nDose=1.0 shift: {dose_1[0]['mean_spectral_shift']:+.4f}"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Functional Spectral Codec")
    parser.add_argument("mode", choices=["zone_map", "compare", "inject"],
                        help="zone_map: classify one model. compare: functional alignment. inject: cross-arch transfer.")
    parser.add_argument("model_a", help="Model A (or only model for zone_map)")
    parser.add_argument("model_b", nargs="?", help="Model B (for compare/inject)")
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--plot", action="store_true", help="Generate figure")
    parser.add_argument("--no-inject", action="store_true", help="Skip injection in compare mode")
    args = parser.parse_args()

    if args.mode in ("compare", "inject") and not args.model_b:
        print("ERROR: compare/inject mode requires two models")
        sys.exit(1)

    t_start = time.time()

    # Load model(s)
    model_a, tok_a, info_a = load_model(args.model_a)
    zones_a, profile_a, ccs_a, neu_a = compute_zone_map(model_a, tok_a, info_a)

    if args.mode == "zone_map":
        output = {
            "model": info_a, "zones": zones_a, "profile": profile_a,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        out_path = args.output or f"results/zones_{info_a['name']}.json"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {out_path}")
        print(f"Total time: {time.time() - t_start:.1f}s")
        return

    # Load model B
    import gc, torch
    if args.mode == "inject":
        # Keep model A for injection reference
        basis_a = compute_calibration_basis(model_a, tok_a, info_a)
    else:
        basis_a = None

    # For compare mode, free model A before loading B to save memory
    if args.mode == "compare":
        del model_a
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_b, tok_b, info_b = load_model(args.model_b)
    zones_b, profile_b, ccs_b, neu_b = compute_zone_map(model_b, tok_b, info_b)

    if args.mode == "inject":
        basis_b = compute_calibration_basis(model_b, tok_b, info_b)

    # Functional comparison
    comparison = functional_zone_compare(zones_a, profile_a, zones_b, profile_b, info_a, info_b)

    # Injection experiment
    injection_results = None
    if args.mode == "inject":
        injection_results = run_injection_experiment(
            model_a, tok_a, info_a, model_b, tok_b, info_b,
            ccs_a, neu_a, basis_a, basis_b
        )

    # Save results
    output = {
        "model_a": info_a, "model_b": info_b,
        "zones_a": zones_a, "zones_b": zones_b,
        "profile_a": profile_a, "profile_b": profile_b,
        "comparison": comparison,
        "injection": injection_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time": round(time.time() - t_start, 1),
    }

    pair_name = f"{info_a['name']}_{info_b['name']}"
    out_path = args.output or f"results/functional_{pair_name}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if args.plot:
        plot_path = out_path.replace(".json", ".png")
        plot_functional_comparison(zones_a, zones_b, info_a, info_b, comparison,
                                   injection_results, plot_path)

    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
