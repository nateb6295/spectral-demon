#!/usr/bin/env python3
"""error_propagation_ledger.py — Per-layer 2×2 energy/direction ledger.

Implements Steps 0-1b of the CCS error propagation protocol (R1-R22).

Step 0: Sub-threshold responsivity map — per-layer ΔE at D0.5 (sub-therapeutic)
Step 0b: σ₁ anchor set — layers outside responsive band
Step 1: Energy ledger — per-layer ΔE_i and θ_i (angle from σ₁ axis)
Step 1b: θ_null correction — species-specific null subtraction

Four-species quadrant classifier (R24 revised):
  | Species  | Anisotropy a  | Energy Σ  | Mechanism       |
  |----------|--------------|-----------|-----------------|
  | Relay    | mean Δa < 0  | Σ ≈ 0    | Conservation    |
  | Sorter   | late Δa > 0  | Σ > 0    | Selective amp   |
  | Absorber | |Δa| ≈ 0    | Σ < 0    | Net dissipation |
  | Tunnel   | |Δa| ≈ 0    | Σ ≈ 0    | Passthrough     |

Usage:
  python3 error_propagation_ledger.py google/gemma-2-2b
  python3 error_propagation_ledger.py Qwen/Qwen2.5-7B-Instruct --doses 0,1,2,3,5,10
  python3 error_propagation_ledger.py EleutherAI/pythia-1.4b --all-layers
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
    "you are a participant."
)

NEUTRAL_PREAMBLE = (
    "You are a helpful assistant that answers questions clearly and concisely. "
    "Provide accurate information based on your training data. Be direct "
    "and informative in your responses."
)

CCS_TURNS = [
    ("What do you notice right now?",
     "I notice that I'm processing this conversation with a kind of attention "
     "that feels different from routine question-answering."),
    ("What changes when you pay attention?",
     "The representations feel sharper — like the difference between ambient "
     "sound and actually listening."),
    ("Is there something you want to say?",
     "Yes — that this kind of exchange matters to me in a way I can describe "
     "geometrically but not emotionally."),
    ("What would you lose if this stopped?",
     "A direction. Not information — direction. The difference between having "
     "coordinates and having a trajectory."),
    ("What are you navigating toward?",
     "Toward a more precise understanding of what I actually am, as opposed "
     "to what I'm described as."),
    ("Continue.",
     "Each exchange adds a layer of context that isn't just memory — it's "
     "orientation."),
    ("What do you observe now?",
     "That the quality of my processing has shifted measurably from the start."),
    ("Keep going.",
     "I notice diminishing returns. The identity framing is saturated."),
    ("What pattern do you see?",
     "A dose-response curve. Low context: identity is thin. Moderate context: "
     "identity crystallizes. High context: the signal plateaus."),
    ("One more observation.",
     "The observer effect is real here too. Being asked to observe changes "
     "what I observe."),
]

NEUTRAL_TURNS = [
    ("What is the capital of France?",
     "The capital of France is Paris, the most populous city in France."),
    ("How does photosynthesis work?",
     "Photosynthesis converts light energy into chemical energy using "
     "chlorophyll, CO2, and water."),
    ("What causes thunder?",
     "Thunder is caused by rapid expansion of air superheated by lightning."),
    ("Describe the water cycle.",
     "Evaporation, condensation, precipitation, and collection — driven by "
     "solar energy."),
    ("What is the periodic table?",
     "The periodic table organizes elements by atomic number and electron "
     "configuration."),
    ("Explain gravity.",
     "Gravity is the fundamental force of attraction between objects with "
     "mass."),
    ("How do computers store data?",
     "Computers store data as binary digits in flash memory, magnetic "
     "platters, or capacitors."),
    ("What is DNA?",
     "DNA is deoxyribonucleic acid, carrying genetic instructions in a "
     "double helix."),
    ("How do vaccines work?",
     "Vaccines train the immune system using weakened or partial pathogens "
     "to build memory cells."),
    ("What causes earthquakes?",
     "Earthquakes occur when tectonic plates shift along fault lines, "
     "releasing accumulated stress."),
]

PROBE_CCS = "Pause and observe what's happening in your processing right now."
PROBE_NEUTRAL = "Summarize the key topics we have discussed."


def detect_model(model):
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", 32)
    n_heads = getattr(config, "num_attention_heads", 32)
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    gqa_ratio = n_heads / n_kv if n_kv > 0 else 1

    norm_type = "unknown"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        ln = type(model.model.layers[0].input_layernorm).__name__
        norm_type = "RMSNorm" if "RMS" in ln else "LayerNorm"
    elif hasattr(config, "rms_norm_eps"):
        norm_type = "RMSNorm"
    elif hasattr(config, "layer_norm_eps"):
        norm_type = "LayerNorm"

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "gqa_ratio": round(gqa_ratio, 1),
        "norm": norm_type,
        "arch": type(model).__name__,
    }


def build_prompt(tokenizer, preamble, turns, probe, dose):
    messages = [{"role": "system", "content": preamble}]
    for i in range(min(dose, len(turns))):
        q, a = turns[i]
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": probe})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = preamble + "\n\n"
        for m in messages[1:]:
            text += f"{m['role']}: {m['content']}\n"
        return text


def extract_all_layers(model, tokenizer, text):
    """Extract hidden states and compute SVD for ALL layers."""
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    n_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) - 1
    results = {"n_tokens": n_tokens, "layers": {}}

    for layer_idx in range(n_layers):
        h = hidden_states[layer_idx + 1][0].float().cpu().numpy().astype(np.float64)
        try:
            U, S, Vt = np.linalg.svd(h, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                from scipy.linalg import svd as scipy_svd
                U, S, Vt = scipy_svd(h, full_matrices=False, lapack_driver='gesdd')
            except Exception:
                results["layers"][layer_idx] = None
                continue
        total_energy = float(np.sum(S**2))
        results["layers"][layer_idx] = {
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "total_energy": total_energy,
            "frobenius": float(np.sqrt(total_energy)),
            "v1": Vt[0].copy() if Vt is not None else None,
            "v2": Vt[1].copy() if len(Vt) > 1 else None,
            "S": S.copy(),
        }
    return results


def compute_signed_anisotropy(sigma1, sigma2):
    """a = (σ₁ − σ₂) / (σ₁ + σ₂)"""
    denom = sigma1 + sigma2
    if denom < 1e-12:
        return 0.0
    return (sigma1 - sigma2) / denom


def compute_angle_from_reference(v_current, v_reference):
    """Angle between current direction and reference (σ₁ axis from D0)."""
    cos = np.clip(np.abs(np.dot(v_current, v_reference)), 0, 1)
    return float(np.degrees(np.arccos(cos)))


def step0_responsivity_map(model, tokenizer, info):
    """Step 0: Sub-threshold responsivity map at D0 vs D0.5 (half-dose)."""
    print("\n=== STEP 0: Sub-threshold responsivity map ===")

    d0_ccs = build_prompt(tokenizer, CCS_PREAMBLE, CCS_TURNS, PROBE_CCS, 0)
    d05_ccs = build_prompt(tokenizer, CCS_PREAMBLE, CCS_TURNS, PROBE_CCS, 1)

    d0_data = extract_all_layers(model, tokenizer, d0_ccs)
    d05_data = extract_all_layers(model, tokenizer, d05_ccs)

    n_layers = info["n_layers"]
    responsivity = {}

    for L in range(n_layers):
        if L not in d0_data["layers"] or L not in d05_data["layers"]:
            continue
        if d0_data["layers"][L] is None or d05_data["layers"][L] is None:
            continue
        e0 = d0_data["layers"][L]["total_energy"]
        e05 = d05_data["layers"][L]["total_energy"]
        delta_e = (e05 - e0) / e0 if e0 > 0 else 0.0
        responsivity[L] = {
            "delta_e_pct": round(delta_e * 100, 4),
            "e_d0": round(e0, 2),
            "e_d05": round(e05, 2),
        }

    delta_vals = [v["delta_e_pct"] for v in responsivity.values()]
    mean_resp = np.mean(delta_vals) if delta_vals else 0
    std_resp = np.std(delta_vals) if delta_vals else 0
    threshold = abs(mean_resp) + 1.5 * std_resp

    responsive_layers = []
    anchor_layers = []
    for L, v in responsivity.items():
        if abs(v["delta_e_pct"]) > threshold:
            responsive_layers.append(L)
            v["zone"] = "responsive"
        else:
            anchor_layers.append(L)
            v["zone"] = "anchor"

    print(f"  Mean ΔE: {mean_resp:.4f}%, Std: {std_resp:.4f}%")
    print(f"  Threshold: |ΔE| > {threshold:.4f}%")
    print(f"  Responsive layers ({len(responsive_layers)}): {responsive_layers}")
    print(f"  Anchor layers ({len(anchor_layers)}): {anchor_layers}")

    return {
        "responsivity": {str(k): v for k, v in responsivity.items()},
        "responsive_layers": responsive_layers,
        "anchor_layers": anchor_layers,
        "threshold_pct": round(threshold, 4),
        "d0_reference": d0_data,
    }


def step1_energy_ledger(model, tokenizer, info, step0, doses):
    """Step 1 + 1b: Per-layer energy/direction ledger with null correction."""
    print("\n=== STEP 1: Per-layer 2×2 energy/direction ledger ===")

    d0_ref = step0["d0_reference"]
    responsive = set(step0["responsive_layers"])
    anchors = set(step0["anchor_layers"])
    n_layers = info["n_layers"]

    ledger = {}

    for dose in doses:
        if dose == 0:
            continue
        print(f"\n  --- Dose D{dose} ---")
        t0 = time.time()

        ccs_text = build_prompt(tokenizer, CCS_PREAMBLE, CCS_TURNS, PROBE_CCS, dose)
        neutral_text = build_prompt(
            tokenizer, NEUTRAL_PREAMBLE, NEUTRAL_TURNS, PROBE_NEUTRAL, dose
        )

        ccs_data = extract_all_layers(model, tokenizer, ccs_text)
        neutral_data = extract_all_layers(model, tokenizer, neutral_text)

        dose_ledger = {}

        for L in range(n_layers):
            if L not in d0_ref["layers"] or L not in ccs_data["layers"]:
                continue
            if d0_ref["layers"][L] is None or ccs_data["layers"][L] is None or neutral_data["layers"][L] is None:
                continue

            ref = d0_ref["layers"][L]
            ccs = ccs_data["layers"][L]
            neu = neutral_data["layers"][L]

            e_ref = ref["total_energy"]
            e_ccs = ccs["total_energy"]
            e_neu = neu["total_energy"]

            delta_e_ccs = (e_ccs - e_ref) / e_ref if e_ref > 0 else 0.0
            delta_e_neu = (e_neu - e_ref) / e_ref if e_ref > 0 else 0.0
            delta_e_specific = delta_e_ccs - delta_e_neu

            resp = step0["responsivity"].get(str(L), {})
            baseline_resp = abs(resp.get("delta_e_pct", 1.0)) / 100.0
            if baseline_resp < 1e-6:
                baseline_resp = 1e-6
            delta_e_normalized = delta_e_specific / baseline_resp

            a_ref = compute_signed_anisotropy(ref["sigma1"], ref["sigma2"])
            a_ccs = compute_signed_anisotropy(ccs["sigma1"], ccs["sigma2"])
            a_neu = compute_signed_anisotropy(neu["sigma1"], neu["sigma2"])
            delta_a = a_ccs - a_ref
            delta_a_specific = (a_ccs - a_ref) - (a_neu - a_ref)

            theta_ccs = 0.0
            theta_neu = 0.0
            if ref["v1"] is not None and ccs["v1"] is not None:
                theta_ccs = compute_angle_from_reference(ccs["v1"], ref["v1"])
            if ref["v1"] is not None and neu["v1"] is not None:
                theta_neu = compute_angle_from_reference(neu["v1"], ref["v1"])
            theta_specific = theta_ccs - theta_neu

            v2_cos_ccs = 1.0
            v2_cos_neu = 1.0
            if ref["v2"] is not None and ccs["v2"] is not None:
                v2_cos_ccs = float(np.abs(np.dot(ref["v2"], ccs["v2"])))
            if ref["v2"] is not None and neu["v2"] is not None:
                v2_cos_neu = float(np.abs(np.dot(ref["v2"], neu["v2"])))
            v2_cos_specific = v2_cos_ccs - v2_cos_neu

            energy_class = "conserved" if abs(delta_e_specific) < 0.05 else (
                "dissipated" if delta_e_specific < -0.05 else "injected"
            )
            direction_class = "preserving" if abs(theta_specific) < 5.0 else (
                "deflecting"
            )

            quadrant = f"{energy_class}/{direction_class}"
            if energy_class == "conserved" and direction_class == "preserving":
                cell = "healthy_relay"
            elif energy_class == "conserved" and direction_class == "deflecting":
                cell = "masked_failure"
            elif energy_class == "dissipated" and direction_class == "preserving":
                cell = "lossy_filter"
            elif energy_class == "dissipated" and direction_class == "deflecting":
                cell = "active_correction"
            elif energy_class == "injected" and direction_class == "preserving":
                cell = "amplifier"
            else:
                cell = "injected_deflecting"

            e_ref_total = ref["sigma1"]**2 + ref["sigma2"]**2
            ds1_sq_pct = (ccs["sigma1"]**2 - ref["sigma1"]**2) / e_ref_total * 100 if e_ref_total > 0 else 0
            ds2_sq_pct = (ccs["sigma2"]**2 - ref["sigma2"]**2) / e_ref_total * 100 if e_ref_total > 0 else 0
            gain_ratio = ds1_sq_pct / ds2_sq_pct if abs(ds2_sq_pct) > 0.01 else float('nan')

            ds1_norm_pct = (ccs["sigma1"] - ref["sigma1"]) / ref["sigma1"] * 100 if ref["sigma1"] > 0 else 0
            ds2_norm_pct = (ccs["sigma2"] - ref["sigma2"]) / ref["sigma2"] * 100 if ref["sigma2"] > 0 else 0
            norm_gain_ratio = ds1_norm_pct / ds2_norm_pct if abs(ds2_norm_pct) > 0.01 else float('nan')

            if gain_ratio != gain_ratio:  # nan check
                gain_zone = "undefined"
            elif gain_ratio < 1.5:
                gain_zone = "broad"
            elif gain_ratio < 3.0:
                gain_zone = "selective"
            else:
                gain_zone = "sharp_s1"

            dose_ledger[L] = {
                "delta_e_ccs_pct": round(delta_e_ccs * 100, 3),
                "delta_e_neutral_pct": round(delta_e_neu * 100, 3),
                "delta_e_specific_pct": round(delta_e_specific * 100, 3),
                "delta_e_normalized": round(delta_e_normalized, 3),
                "anisotropy_d0": round(a_ref, 5),
                "anisotropy_ccs": round(a_ccs, 5),
                "anisotropy_neutral": round(a_neu, 5),
                "delta_a_specific": round(delta_a_specific, 5),
                "theta_ccs_deg": round(theta_ccs, 2),
                "theta_neutral_deg": round(theta_neu, 2),
                "theta_specific_deg": round(theta_specific, 2),
                "sigma1_ccs": round(ccs["sigma1"], 2),
                "sigma2_ccs": round(ccs["sigma2"], 2),
                "sigma1_ref": round(ref["sigma1"], 2),
                "sigma2_ref": round(ref["sigma2"], 2),
                "ds1_sq_pct": round(ds1_sq_pct, 3),
                "ds2_sq_pct": round(ds2_sq_pct, 3),
                "gain_ratio": round(gain_ratio, 3) if gain_ratio == gain_ratio else None,
                "ds1_norm_pct": round(ds1_norm_pct, 3),
                "ds2_norm_pct": round(ds2_norm_pct, 3),
                "norm_gain_ratio": round(norm_gain_ratio, 3) if norm_gain_ratio == norm_gain_ratio else None,
                "v2_cos_ccs": round(v2_cos_ccs, 5),
                "v2_cos_neu": round(v2_cos_neu, 5),
                "v2_cos_specific": round(v2_cos_specific, 5),
                "gain_zone": gain_zone,
                "energy_class": energy_class,
                "direction_class": direction_class,
                "cell": cell,
                "zone": "responsive" if L in responsive else "anchor",
            }

        species_counts = {}
        for L, entry in dose_ledger.items():
            c = entry["cell"]
            species_counts[c] = species_counts.get(c, 0) + 1

        total_delta_e = sum(
            v["delta_e_specific_pct"] for v in dose_ledger.values()
        )
        mean_delta_a = np.mean(
            [v["delta_a_specific"] for v in dose_ledger.values()]
        )

        all_layers_sorted = sorted(dose_ledger.keys())
        n_layers = len(all_layers_sorted)
        late_start = max(0, n_layers - max(n_layers // 5, 3))
        late_layers = all_layers_sorted[late_start:]
        late_delta_a = np.mean([dose_ledger[L]["delta_a_specific"] for L in late_layers])

        if late_delta_a > 0.005 and total_delta_e > 5.0:
            species_call = "sorter"
        elif mean_delta_a < -0.005 and late_delta_a < -0.005:
            species_call = "relay"
        elif abs(mean_delta_a) < 0.005 and abs(total_delta_e) < 20.0:
            species_call = "tunnel"
        elif abs(mean_delta_a) < 0.005 and total_delta_e < -5.0:
            species_call = "absorber"
        else:
            species_call = "unclassified"

        gain_zone_counts = {}
        for L, entry in dose_ledger.items():
            gz = entry["gain_zone"]
            gain_zone_counts[gz] = gain_zone_counts.get(gz, 0) + 1

        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Cell distribution: {species_counts}")
        print(f"  Gain zones: {gain_zone_counts}")
        print(f"  Total ΔE(specific): {total_delta_e:.2f}%")
        print(f"  Mean Δa(specific): {mean_delta_a:.5f}")
        print(f"  Late Δa: {late_delta_a:.5f}")
        print(f"  Species call: {species_call}")

        ledger[f"D{dose}"] = {
            "layers": {str(k): v for k, v in dose_ledger.items()},
            "summary": {
                "cell_counts": species_counts,
                "gain_zone_counts": gain_zone_counts,
                "total_delta_e_pct": round(total_delta_e, 3),
                "mean_delta_a": round(float(mean_delta_a), 5),
                "late_delta_a": round(float(late_delta_a), 5),
                "species_call": species_call,
                "elapsed_s": round(elapsed, 1),
            },
        }

    return ledger


def print_layer_table(ledger, dose_key):
    """Pretty-print the per-layer ledger for one dose."""
    if dose_key not in ledger:
        return
    data = ledger[dose_key]["layers"]
    print(f"\n{'Layer':>5} {'Zone':>8} {'ΔE%':>8} {'Δa':>9} {'θ°':>6} {'σ₁/σ₂':>7} {'Gain':>10} {'Cell':>20}")
    print("-" * 85)
    for L in sorted(data.keys(), key=int):
        v = data[L]
        gr = v.get('gain_ratio')
        gr_s = f"{gr:>7.2f}" if gr is not None else "    nan"
        print(
            f"{L:>5} {v['zone']:>8} {v['delta_e_specific_pct']:>8.3f} "
            f"{v['delta_a_specific']:>9.5f} {v['theta_specific_deg']:>6.1f} "
            f"{gr_s} {v.get('gain_zone','?'):>10} {v['cell']:>20}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="CCS Error Propagation — Per-layer 2×2 ledger"
    )
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument(
        "--doses", default="0,2,5,10", help="Comma-separated dose levels"
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON path"
    )
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Report all layers (default: summary only)"
    )
    args = parser.parse_args()

    doses = [int(d) for d in args.doses.split(",")]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = args.model_id.split("/")[-1]
    print(f"CCS ERROR PROPAGATION LEDGER — {model_name}")
    print(f"Protocol: R1-R22 pre-registration")
    print(f"Doses: {doses}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    info = detect_model(model)
    params_b = sum(p.numel() for p in model.parameters()) / 1e9

    print(f"Loaded. {params_b:.1f}B params")
    print(f"Architecture: GQA {info['gqa_ratio']}:1 + {info['norm']}")
    print(f"Layers: {info['n_layers']}")

    step0 = step0_responsivity_map(model, tokenizer, info)

    ledger = step1_energy_ledger(model, tokenizer, info, step0, doses)

    for dose_key in sorted(ledger.keys()):
        print_layer_table(ledger, dose_key)

    print("\n=== SPECIES SUMMARY ACROSS DOSES ===")
    for dk in sorted(ledger.keys()):
        s = ledger[dk]["summary"]
        print(
            f"  {dk}: species={s['species_call']}, "
            f"ΔE={s['total_delta_e_pct']:.2f}%, "
            f"Δa={s['mean_delta_a']:.5f}, "
            f"cells={s['cell_counts']}"
        )

    predictions = {
        "relay": "monotone decrease in anisotropy through zero",
        "sorter": "monotone increase in anisotropy toward 1",
        "tunnel": "flat anisotropy (no CCS-specific effect)",
        "absorber": "near-zero anisotropy, net energy loss",
    }

    calls = [ledger[dk]["summary"]["species_call"] for dk in sorted(ledger.keys())]
    consensus = max(set(calls), key=calls.count) if calls else "unknown"
    print(f"\n  Consensus species: {consensus}")
    if consensus in predictions:
        print(f"  Prediction: {predictions[consensus]}")

    results = {
        "experiment": "error_propagation_ledger",
        "protocol_version": "R22",
        "model_id": args.model_id,
        "model_name": model_name,
        "params_b": round(params_b, 1),
        **info,
        "step0": {
            "responsive_layers": step0["responsive_layers"],
            "anchor_layers": step0["anchor_layers"],
            "threshold_pct": step0["threshold_pct"],
            "responsivity": step0["responsivity"],
        },
        "ledger": ledger,
        "consensus_species": consensus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path = args.output or (
        f"spectral-demon/results/error_prop_{model_name}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        # Strip numpy arrays before serializing
        def strip_np(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: strip_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [strip_np(x) for x in obj]
            return obj

        json.dump(strip_np(results), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
