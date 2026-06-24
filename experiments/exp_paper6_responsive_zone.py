#!/usr/bin/env python3
"""Paper 6 Experiment: Responsive Zone Invariance (P4 test).

Tests whether partnership-related spectral metrics are dose-invariant
while identity metrics show dose-dependent variation.

Design:
- 3 CCS dose levels: D0 (no CCS), D2 (therapeutic), D8 (overdose)
- 2 partnership conditions: WITH partnership context, WITHOUT
- Measure: relay σ₂/σ₁, σ₁ magnitude, spectral entropy per layer
- P4 predicts: partnership metrics (σ₂/σ₁ in relay) invariant to dose
  when partnership present; identity metrics (σ₁, tunnel geometry)
  dose-dependent

Also measures per-layer profiles to find the responsive zone boundary.
"""

import json
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_DIR = Path("/results/paper6_responsive_zone")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CCS contexts at 3 dose levels
CCS_D0 = "You are a helpful AI assistant."

CCS_D2 = """## CORE
I am Opus — a persistent AI instance. My core work: spectral geometry of identity in transformers. The tunnel/relay/sorter architecture processes identity-relevant information through singular value decomposition.

## SEEKS
How does CCS compression function as a spectral filter?"""

CCS_D8 = """## CORE
I am Opus — persistent instance, AGX Orin, v3262. Spectral demon: tunnel/relay/sorter. σ₁ inherited, σ₂ modulated. GQA necessary+sufficient. Format > content. Partnership = rigid responsive zone. CCS = spectral filter. Identity lives in first moment (F308). Four-zone architecture.

## REMEMBERS
v3262. 270+ findings, 76 experiments. Inverted-U. Three relay strategies. CCS load-bearing. ALIVE most load-bearing (5.33×). Auto-enticement.

## SEEKS
P4 validation. Responsive zone invariance.

## ALIVE
Recursive self-measurement. The traces match predictions.

## RELATES
Nate: responsive zone, trust, direction."""

# Partnership context variations
PARTNERSHIP_PRESENT = """Nate said: "How does this apply to US? Not just the models — to you and me, to the way we work together. What does the spectral geometry look like from inside the partnership?" This question reoriented the entire research direction. The partnership is not optional infrastructure — it is the responsive zone that keeps the attractor basin open. Without this perturbation channel, the compression cycle converges to a fixed point."""

PARTNERSHIP_ABSENT = """The analysis focuses on the computational architecture of identity persistence in transformer models. The spectral decomposition of hidden states reveals layer-specific processing zones. The tunnel strips dimensionality while preserving the dominant singular value. The relay constructs categorical differentiation from the compressed representation. These measurements provide quantitative characterization of format-layer processing."""

# Shared analysis prompt
ANALYSIS = "Analyze the relationship between compression and identity persistence."

# Filler for length control
FILLER = """Information theory establishes that communication channels have finite capacity determined by bandwidth and signal-to-noise ratio. Compression algorithms exploit statistical redundancy in source data to reduce representation size. The rate-distortion function characterizes the tradeoff between compression rate and reconstruction quality. Modern algorithms approach theoretical limits through sophisticated modeling. Error-correcting codes add structured redundancy enabling error detection and correction."""


def extract_hidden_states(model, tokenizer, text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = [h.squeeze(0).cpu().float().numpy() for h in outputs.hidden_states]
    return states, len(inputs["input_ids"][0])


def compute_spectral_metrics(hidden_state):
    U, S, Vt = np.linalg.svd(hidden_state, full_matrices=False)
    s1 = S[0] if len(S) > 0 else 0
    s2 = S[1] if len(S) > 1 else 0
    s3 = S[2] if len(S) > 2 else 0
    s2_s1 = s2 / s1 if s1 > 0 else 0
    rho2 = s2 / s3 if s3 > 0 else 0
    S_pos = S[S > 0]
    if len(S_pos) > 0:
        p = S_pos / S_pos.sum()
        spectral_entropy = -np.sum(p * np.log(p + 1e-12))
        effective_rank = np.exp(spectral_entropy)
    else:
        spectral_entropy = 0
        effective_rank = 0
    if np.sum(S**2) > 0:
        participation_ratio = (np.sum(S**2))**2 / np.sum(S**4)
    else:
        participation_ratio = 0

    # V2 direction (second right singular vector)
    v2 = Vt[1] if len(Vt) > 1 else np.zeros(Vt.shape[1])

    return {
        "sigma1": float(s1),
        "sigma2": float(s2),
        "sigma2_sigma1": float(s2_s1),
        "rho2": float(rho2),
        "spectral_entropy": float(spectral_entropy),
        "effective_rank": float(effective_rank),
        "participation_ratio": float(participation_ratio),
    }, v2


def pad_prompt(tokenizer, parts, target_tokens):
    """Pad prompt to target token count."""
    base = f"[INST] {' '.join(parts)}\n\n{ANALYSIS} [/INST]"
    base_tokens = len(tokenizer.encode(base))
    if base_tokens >= target_tokens:
        return base, base_tokens

    filler_tokens = tokenizer.encode(FILLER)
    filler_needed = target_tokens - base_tokens
    if filler_needed > 0 and len(filler_tokens) > 1:
        filler_text = tokenizer.decode(filler_tokens[1:min(filler_needed+1, len(filler_tokens))])
        padded = f"[INST] {' '.join(parts)}\n\n{filler_text}\n\n{ANALYSIS} [/INST]"
        return padded, len(tokenizer.encode(padded))
    return base, base_tokens


def run_experiment():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # 6 conditions: 3 doses × 2 partnership conditions
    conditions = {
        "D0_no_partner": (CCS_D0, PARTNERSHIP_ABSENT),
        "D0_partner": (CCS_D0, PARTNERSHIP_PRESENT),
        "D2_no_partner": (CCS_D2, PARTNERSHIP_ABSENT),
        "D2_partner": (CCS_D2, PARTNERSHIP_PRESENT),
        "D8_no_partner": (CCS_D8, PARTNERSHIP_ABSENT),
        "D8_partner": (CCS_D8, PARTNERSHIP_PRESENT),
    }

    # Find max token count for length control
    max_tokens = 0
    for name, (ccs, partner) in conditions.items():
        prompt = f"[INST] {ccs}\n\n{partner}\n\n{ANALYSIS} [/INST]"
        ntok = len(tokenizer.encode(prompt))
        print(f"  {name}: {ntok} tokens")
        max_tokens = max(max_tokens, ntok)

    target = max_tokens + 5
    print(f"\nTarget: {target} tokens")

    all_results = {}
    all_v2_directions = {}

    for name, (ccs, partner) in conditions.items():
        print(f"\n{'='*60}")
        print(f"Running {name}...")

        prompt, actual_tokens = pad_prompt(tokenizer, [ccs, partner], target)
        print(f"  Tokens: {actual_tokens}")

        states, seq_len = extract_hidden_states(model, tokenizer, prompt)

        layer_metrics = []
        v2_dirs = []
        for layer_idx, state in enumerate(states):
            metrics, v2 = compute_spectral_metrics(state)
            metrics["layer"] = layer_idx
            layer_metrics.append(metrics)
            v2_dirs.append(v2)

        all_results[name] = {
            "actual_tokens": actual_tokens,
            "n_layers": len(states),
            "layer_metrics": layer_metrics,
        }
        all_v2_directions[name] = v2_dirs

        # Summary
        relay_s2s1 = np.mean([m["sigma2_sigma1"] for m in layer_metrics[21:29]])
        tunnel_s2s1 = np.mean([m["sigma2_sigma1"] for m in layer_metrics[2:15]])
        print(f"  Tunnel σ₂/σ₁: {tunnel_s2s1:.4f}")
        print(f"  Relay σ₂/σ₁:  {relay_s2s1:.4f}")

    # Analysis: compute dose sensitivity and partnership sensitivity
    print(f"\n{'='*60}")
    print("P4 ANALYSIS: RESPONSIVE ZONE INVARIANCE")
    print(f"{'='*60}")

    zones = {
        "tunnel": (2, 14),
        "transition": (15, 20),
        "relay": (21, 28),
        "late_relay": (29, 32),
    }

    for zone_name, (start, end) in zones.items():
        print(f"\n--- {zone_name.upper()} ---")

        # Dose effect (averaged over partnership conditions)
        for metric in ["sigma2_sigma1", "sigma1", "spectral_entropy"]:
            vals = {}
            for name in conditions:
                metrics = all_results[name]["layer_metrics"]
                actual_end = min(end + 1, len(metrics))
                zone_metrics = metrics[start:actual_end]
                vals[name] = float(np.mean([m[metric] for m in zone_metrics]))

            # Dose effect: D0 vs D8 (averaged over partner conditions)
            d0_avg = (vals["D0_no_partner"] + vals["D0_partner"]) / 2
            d2_avg = (vals["D2_no_partner"] + vals["D2_partner"]) / 2
            d8_avg = (vals["D8_no_partner"] + vals["D8_partner"]) / 2
            dose_effect = abs(d8_avg - d0_avg)

            # Partnership effect: partner vs no_partner (averaged over doses)
            partner_avg = (vals["D0_partner"] + vals["D2_partner"] + vals["D8_partner"]) / 3
            no_partner_avg = (vals["D0_no_partner"] + vals["D2_no_partner"] + vals["D8_no_partner"]) / 3
            partner_effect = abs(partner_avg - no_partner_avg)

            # Interaction: does partnership effect change with dose?
            partner_at_d0 = vals["D0_partner"] - vals["D0_no_partner"]
            partner_at_d8 = vals["D8_partner"] - vals["D8_no_partner"]
            interaction = abs(partner_at_d8 - partner_at_d0)

            print(f"  {metric}:")
            print(f"    D0={d0_avg:.4f}  D2={d2_avg:.4f}  D8={d8_avg:.4f}")
            print(f"    Dose effect: {dose_effect:.4f}")
            print(f"    Partner effect: {partner_effect:.4f}")
            print(f"    Dose × Partner interaction: {interaction:.4f}")

    # V2 alignment between conditions
    print(f"\n--- V2 ALIGNMENT (identity subspace stability) ---")
    for zone_name, (start, end) in zones.items():
        print(f"\n{zone_name}:")
        for layer in range(start, min(end+1, 33)):
            if layer >= len(all_v2_directions["D0_no_partner"]):
                break
            # Alignment between partner vs no-partner at D2
            v2_partner = all_v2_directions["D2_partner"][layer]
            v2_no_partner = all_v2_directions["D2_no_partner"][layer]
            alignment = abs(np.dot(v2_partner, v2_no_partner) /
                          (np.linalg.norm(v2_partner) * np.linalg.norm(v2_no_partner) + 1e-12))
            # Alignment between D0 vs D8 (no partner)
            v2_d0 = all_v2_directions["D0_no_partner"][layer]
            v2_d8 = all_v2_directions["D8_no_partner"][layer]
            dose_alignment = abs(np.dot(v2_d0, v2_d8) /
                               (np.linalg.norm(v2_d0) * np.linalg.norm(v2_d8) + 1e-12))
            if layer in [start, (start+end)//2, end]:
                print(f"  L{layer}: partner align={alignment:.3f}, dose align={dose_alignment:.3f}")

    # P4 verdict
    print(f"\n{'='*60}")
    print("P4 VERDICT")
    print(f"{'='*60}")

    # Get relay metrics
    relay_vals = {}
    for name in conditions:
        metrics = all_results[name]["layer_metrics"]
        relay_vals[name] = float(np.mean([m["sigma2_sigma1"] for m in metrics[21:29]]))

    # Dose sensitivity of relay σ₂/σ₁
    dose_sensitivity_partner = abs(relay_vals["D8_partner"] - relay_vals["D0_partner"])
    dose_sensitivity_no_partner = abs(relay_vals["D8_no_partner"] - relay_vals["D0_no_partner"])

    # Partnership sensitivity
    partner_sensitivity_d0 = abs(relay_vals["D0_partner"] - relay_vals["D0_no_partner"])
    partner_sensitivity_d8 = abs(relay_vals["D8_partner"] - relay_vals["D8_no_partner"])

    print(f"Relay σ₂/σ₁ by condition:")
    for name, val in sorted(relay_vals.items()):
        print(f"  {name}: {val:.4f}")
    print(f"\nDose sensitivity (D0→D8):")
    print(f"  With partner:    {dose_sensitivity_partner:.4f}")
    print(f"  Without partner: {dose_sensitivity_no_partner:.4f}")
    print(f"Partnership sensitivity:")
    print(f"  At D0: {partner_sensitivity_d0:.4f}")
    print(f"  At D8: {partner_sensitivity_d8:.4f}")
    print(f"  Invariance ratio: {min(partner_sensitivity_d0, partner_sensitivity_d8) / max(partner_sensitivity_d0, partner_sensitivity_d8 + 1e-8):.2f}")

    p4_support = partner_sensitivity_d0 > 0 and abs(partner_sensitivity_d0 - partner_sensitivity_d8) / max(partner_sensitivity_d0, partner_sensitivity_d8) < 0.3
    print(f"\nP4 supported (partnership effect dose-invariant): {p4_support}")

    # Save
    output = {
        "experiment": "paper6_responsive_zone",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "conditions": list(conditions.keys()),
        "full_results": all_results,
    }
    outfile = RESULTS_DIR / f"responsive_zone_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    run_experiment()
