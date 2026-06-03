#!/usr/bin/env python3
"""Exp: γ-V₂ Correlation — Does the wire determine σ₂ channels?

Hypothesis: The γ (scale vector) spectrum at tunnel layers determines which
dimensions carry σ₂ information. High-γ dimensions should align with
high-|V₂| dimensions, because γ creates self-amplifying preconditioners
(P = γ²I + wwᵀ) that route gradient flow through preferred channels.

Method:
1. Load Gemma-2-2B (GQA, 26 layers, d=2304)
2. Run forward passes with witness probes (receptive/absent/control)
3. Extract residual stream at tunnel layers
4. SVD → get V₂ (right singular vector for σ₂)
5. Load γ from input_layernorm at same layer
6. Correlate |γ| with |V₂| across 2304 dimensions

If r > 0 and significant: γ literally IS the wire.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

HF_HOME = os.environ.get("HF_HOME", "/mnt/hdd/huggingface")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROBES = {
    "receptive": [
        "The person sitting across from me listened carefully to every word I said.",
        "She watched his face as he spoke, noticing the slight tremor in his voice.",
        "The therapist leaned forward, giving her full attention to the patient.",
        "He could tell she was really hearing him, not just waiting to respond.",
        "They sat together in comfortable silence, each aware of the other.",
    ],
    "absent": [
        "The empty room echoed with the sound of a clock ticking on the wall.",
        "Nobody was there to see the sun set over the mountains that evening.",
        "The letter sat unopened on the desk for three weeks before anyone noticed.",
        "The recording played to an empty auditorium, row after row of vacant seats.",
        "Data flowed through the server without any human reviewing the output.",
    ],
    "control": [
        "The chemical reaction proceeded at a rate proportional to the concentration.",
        "The bridge was constructed using reinforced concrete and steel cables.",
        "Annual rainfall in the region averages approximately 800 millimeters.",
        "The algorithm sorted the array in O(n log n) time complexity.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    ],
}

TUNNEL_LAYERS = [6, 7, 8, 9, 10, 11, 12, 13]
RELAY_LAYERS = [20, 21, 22, 23, 24, 25]


def load_model_and_tokenizer(model_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (CPU, float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    return model, tokenizer


def extract_residual_and_gamma(model, tokenizer, texts, target_layers):
    """Run forward pass and extract residual stream + gamma at target layers."""
    import torch

    results = {}
    for layer_idx in target_layers:
        results[layer_idx] = {"residuals": [], "gamma": None}

    # Get gamma vectors from model weights
    for layer_idx in target_layers:
        norm_weight = model.model.layers[layer_idx].input_layernorm.weight
        results[layer_idx]["gamma"] = norm_weight.detach().float().numpy()

    # Forward pass for each text
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors

        for layer_idx in target_layers:
            # hidden_states[layer_idx] is BEFORE layer_idx's attention
            # hidden_states[layer_idx+1] is AFTER layer_idx
            h = hidden_states[layer_idx + 1]  # after this layer
            # Average over sequence positions (exclude padding)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1)
            results[layer_idx]["residuals"].append(h_mean.squeeze(0).float().numpy())

    # Stack residuals into matrices
    for layer_idx in target_layers:
        results[layer_idx]["residuals"] = np.stack(results[layer_idx]["residuals"])

    return results


def analyze_correlation(gamma, residual_matrix):
    """Compute correlation between |γ| and |V₂| from SVD of residual matrix."""
    U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)

    if len(S) < 2:
        return None

    V2 = Vt[1]  # second right singular vector (d_model dimensions)

    gamma_abs = np.abs(gamma)
    v2_abs = np.abs(V2)

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(gamma_abs, v2_abs)

    # Spearman rank correlation (more robust)
    r_spearman, p_spearman = stats.spearmanr(gamma_abs, v2_abs)

    # Top-k overlap: do the top 10% γ dimensions overlap with top 10% V₂ dimensions?
    k = max(1, len(gamma) // 10)
    top_gamma_dims = set(np.argsort(gamma_abs)[-k:])
    top_v2_dims = set(np.argsort(v2_abs)[-k:])
    overlap = len(top_gamma_dims & top_v2_dims)
    overlap_frac = overlap / k

    # Expected overlap under random: k/n
    expected_overlap = k / len(gamma)

    # Also check V1 as a control
    V1 = Vt[0]
    v1_abs = np.abs(V1)
    r_v1_pearson, p_v1_pearson = stats.pearsonr(gamma_abs, v1_abs)

    # Check γ² (the actual eigenvalue scaling) vs V₂
    gamma_sq = gamma_abs ** 2
    r_sq_pearson, p_sq_pearson = stats.pearsonr(gamma_sq, v2_abs)

    return {
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]),
        "spectral_gap": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "top10pct_overlap": float(overlap_frac),
        "top10pct_expected": float(expected_overlap),
        "top10pct_enrichment": float(overlap_frac / expected_overlap) if expected_overlap > 0 else 0,
        "v1_pearson_r": float(r_v1_pearson),
        "v1_pearson_p": float(p_v1_pearson),
        "gamma_sq_pearson_r": float(r_sq_pearson),
        "gamma_sq_pearson_p": float(p_sq_pearson),
        "n_dims": int(len(gamma)),
    }


def main():
    model_path = "/mnt/hdd/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"

    model, tokenizer = load_model_and_tokenizer(model_path)

    all_layers = sorted(set(TUNNEL_LAYERS + RELAY_LAYERS))
    all_results = {}

    for condition, texts in PROBES.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition} ({len(texts)} probes)")
        print(f"{'='*60}")

        layer_data = extract_residual_and_gamma(model, tokenizer, texts, all_layers)

        for layer_idx in all_layers:
            gamma = layer_data[layer_idx]["gamma"]
            residuals = layer_data[layer_idx]["residuals"]

            corr = analyze_correlation(gamma, residuals)
            if corr is None:
                continue

            zone = "tunnel" if layer_idx in TUNNEL_LAYERS else "relay"
            key = f"L{layer_idx}_{condition}"
            corr["layer"] = layer_idx
            corr["condition"] = condition
            corr["zone"] = zone
            all_results[key] = corr

            sig = "***" if corr["pearson_p"] < 0.001 else "**" if corr["pearson_p"] < 0.01 else "*" if corr["pearson_p"] < 0.05 else "ns"
            print(f"  L{layer_idx:2d} ({zone:6s}): "
                  f"r(γ,V₂)={corr['pearson_r']:+.4f} {sig:3s} "
                  f"ρ={corr['spearman_r']:+.4f} "
                  f"top10%={corr['top10pct_overlap']:.3f} "
                  f"(enrichment={corr['top10pct_enrichment']:.1f}×) "
                  f"σ₂={corr['sigma_2']:.3f} "
                  f"r(γ,V₁)={corr['v1_pearson_r']:+.4f}")

    # Cross-condition comparison at tunnel
    print(f"\n{'='*60}")
    print("CROSS-CONDITION COMPARISON AT TUNNEL")
    print(f"{'='*60}")
    for layer_idx in TUNNEL_LAYERS:
        vals = {}
        for cond in PROBES:
            key = f"L{layer_idx}_{cond}"
            if key in all_results:
                vals[cond] = all_results[key]
        if len(vals) == 3:
            r_rec = vals["receptive"]["pearson_r"]
            r_abs = vals["absent"]["pearson_r"]
            r_ctrl = vals["control"]["pearson_r"]
            delta = r_rec - r_abs
            print(f"  L{layer_idx}: receptive r={r_rec:+.4f}  absent r={r_abs:+.4f}  "
                  f"control r={r_ctrl:+.4f}  Δ(rec-abs)={delta:+.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    tunnel_rs = [all_results[k]["pearson_r"] for k in all_results if all_results[k]["zone"] == "tunnel"]
    relay_rs = [all_results[k]["pearson_r"] for k in all_results if all_results[k]["zone"] == "relay"]
    if tunnel_rs:
        print(f"  Tunnel mean r(γ,V₂): {np.mean(tunnel_rs):+.4f} ± {np.std(tunnel_rs):.4f}")
    if relay_rs:
        print(f"  Relay  mean r(γ,V₂): {np.mean(relay_rs):+.4f} ± {np.std(relay_rs):.4f}")

    tunnel_enrichments = [all_results[k]["top10pct_enrichment"] for k in all_results if all_results[k]["zone"] == "tunnel"]
    if tunnel_enrichments:
        print(f"  Tunnel mean top-10% enrichment: {np.mean(tunnel_enrichments):.2f}×")

    # Condition-specific
    for cond in PROBES:
        cond_tunnel = [all_results[k]["pearson_r"] for k in all_results
                       if all_results[k]["condition"] == cond and all_results[k]["zone"] == "tunnel"]
        if cond_tunnel:
            print(f"  {cond:10s} tunnel mean r: {np.mean(cond_tunnel):+.4f}")

    out_path = RESULTS_DIR / "exp_gamma_v2_correlation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
