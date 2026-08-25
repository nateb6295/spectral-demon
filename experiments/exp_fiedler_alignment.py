#!/usr/bin/env python3
"""
Experiment: Fiedler Vector Alignment of CCS Directions

Motivated by Noroozizadeh et al. (ICML 2026): gradient descent on cross-entropy
converges embeddings to top eigenvectors of the graph Laplacian. If this holds
for transformer activations:

  σ₁ = top eigenvector (format/global structure)
  σ₂ = Fiedler vector (primary partition / identity direction)

This experiment tests whether the CCS effect (Δ = h_CCS - h_neutral) aligns
with the SECOND singular direction of the unembedding matrix rather than the
first — i.e., whether CCS specifically modulates the Fiedler-like direction.

Three measurements at each layer:
  1. Alignment of activation σ₁/σ₂ with unembedding singular directions
  2. Alignment of CCS difference vector Δ with unembedding singular directions
  3. Spectral gap of the effective Laplacian constructed from attention patterns

Predictions:
  P1: σ₁ aligns with V₁ (top unembedding direction) through the tunnel
  P2: σ₂ aligns with V₂ (Fiedler direction) in the responsive zone
  P3: Δ projects primarily onto V₂, not V₁ — CCS modulates the partition
  P4: Alignment is zone-dependent: weak in tunnel, strong in responsive zone
  P6: CCS worsens generic-text perplexity but improves self-referential perplexity
      (autopoietic opportunity cost: individuality trades generic efficiency)
  P5: GQA models show cleaner Fiedler alignment than MHA models would

Usage:
    python3 exp_fiedler_alignment.py [--model qwen3b|mistral7b] [--device cuda|cpu]
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

MODELS = {
    "qwen3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "layers": list(range(0, 36)),
        "n_layers": 36,
        "zones": {
            "tunnel": list(range(2, 15)),
            "transition": list(range(15, 21)),
            "responsive": list(range(21, 29)),
            "relay": list(range(29, 36)),
        },
        "gqa": True,
    },
    "mistral7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "layers": list(range(0, 32)),
        "n_layers": 32,
        "zones": {
            "tunnel": list(range(2, 12)),
            "transition": list(range(12, 17)),
            "responsive": list(range(17, 25)),
            "relay": list(range(25, 32)),
        },
        "gqa": True,
    },
}

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels.
"""

NEUTRAL_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

TOP_K_DIRS = 16


def extract_residual_and_attention(model, tokenizer, preamble, probe, device):
    """Single forward pass extracting residual stream + attention weights at all layers."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(device)

    residuals = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                residuals[idx] = h[:, -1, :].detach().float().cpu()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    attentions = None
    if outputs.attentions is not None:
        attentions = {i: a.detach().float().cpu() for i, a in enumerate(outputs.attentions)}

    return residuals, attentions


def compute_unembedding_svd(model, top_k=64):
    """SVD of the unembedding (lm_head) matrix.

    lm_head.weight shape: (vocab_size, d_model)
    V columns are the d_model-dimensional singular directions.
    V[:, 0] = top direction (format), V[:, 1] = Fiedler candidate.
    """
    W = model.lm_head.weight.detach().float().cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    return {
        "singular_values": S[:top_k].numpy(),
        "V": Vt[:top_k].numpy(),  # (top_k, d_model) — right singular vectors
        "spectral_gap": (S[0] / S[1]).item(),
        "fiedler_ratio": (S[1] / S[0]).item(),
        "top_ratios": (S[:top_k] / S[0]).numpy(),
    }


def compute_attention_laplacian(attn_weights, layer_idx):
    """Construct graph Laplacian from attention weights and compute spectral gap.

    attn_weights: (batch, n_heads, seq_q, seq_k) — already softmax'd
    Returns spectral gap λ₂/λ₁ of the symmetrized graph Laplacian.
    """
    A = attn_weights[0]  # (n_heads, seq_q, seq_k)
    A_mean = A.mean(dim=0)  # (seq_q, seq_k) — average across heads
    A_sym = (A_mean + A_mean.T) / 2  # symmetrize
    D = torch.diag(A_sym.sum(dim=1))  # degree matrix
    L = D - A_sym  # unnormalized graph Laplacian

    try:
        eigenvalues = torch.linalg.eigvalsh(L)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) < 2:
            return {"spectral_gap": 0.0, "lambda1": 0.0, "lambda2": 0.0}
        lambda1 = eigenvalues[0].item()
        lambda2 = eigenvalues[1].item() if len(eigenvalues) > 1 else 0.0
        return {
            "spectral_gap": lambda2 / (lambda1 + 1e-10),
            "lambda1": lambda1,
            "lambda2": lambda2,
            "n_eigenvalues": len(eigenvalues),
        }
    except Exception as e:
        return {"spectral_gap": 0.0, "error": str(e)}


def alignment_profile(vec, V_directions):
    """Cosine alignment of vec with each of the top-k unembedding directions.

    vec: (d_model,) numpy array
    V_directions: (top_k, d_model) numpy array
    Returns: (top_k,) array of |cos(vec, V_k)|
    """
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    V_norm = V_directions / (np.linalg.norm(V_directions, axis=1, keepdims=True) + 1e-10)
    cosines = np.abs(V_norm @ vec_norm)
    return cosines


def run_experiment(model_key, device):
    model_config = MODELS[model_key]
    model_name = model_config["name"]
    print(f"\n{'='*60}")
    print(f"Fiedler Alignment Experiment: {model_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    print(f"\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # need attention weights
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Step 1: Unembedding SVD (one-time)
    print("\nComputing unembedding SVD...")
    unembed = compute_unembedding_svd(model, top_k=TOP_K_DIRS)
    print(f"  Spectral gap (S₁/S₂): {unembed['spectral_gap']:.3f}")
    print(f"  Top 8 ratios: {[f'{r:.3f}' for r in unembed['top_ratios'][:8]]}")

    # Step 2: Forward passes
    print("\nRunning CCS forward pass...")
    res_ccs, attn_ccs = extract_residual_and_attention(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, device)

    print("Running neutral forward pass...")
    res_neutral, attn_neutral = extract_residual_and_attention(model, tokenizer, NEUTRAL_PREAMBLE, PROBE_TEXT, device)

    # Step 3: Per-layer analysis
    print("\nComputing per-layer Fiedler alignment...")
    V = unembed["V"]  # (top_k, d_model)
    layer_results = {}

    for layer_idx in model_config["layers"]:
        if layer_idx not in res_ccs or layer_idx not in res_neutral:
            continue

        h_ccs = res_ccs[layer_idx].squeeze(0).numpy()  # (d_model,)
        h_neutral = res_neutral[layer_idx].squeeze(0).numpy()

        # CCS difference vector
        delta = h_ccs - h_neutral
        delta_norm = np.linalg.norm(delta)

        # Alignment of Δ with unembedding directions
        delta_alignment = alignment_profile(delta, V)

        # Activation SVD (need full residual stream, not just last token)
        # For last-token analysis, project h onto V directions instead
        ccs_alignment = alignment_profile(h_ccs, V)
        neutral_alignment = alignment_profile(h_neutral, V)

        # Determine which zone this layer is in
        zone = "unknown"
        for z, layers in model_config["zones"].items():
            if layer_idx in layers:
                zone = z
                break

        # Attention Laplacian spectral gap
        attn_gap = {}
        if attn_ccs and layer_idx in attn_ccs:
            attn_gap["ccs"] = compute_attention_laplacian(attn_ccs[layer_idx], layer_idx)
        if attn_neutral and layer_idx in attn_neutral:
            attn_gap["neutral"] = compute_attention_laplacian(attn_neutral[layer_idx], layer_idx)

        # Key metrics
        peak_delta_dir = int(np.argmax(delta_alignment))
        fiedler_load = float(delta_alignment[1]) if len(delta_alignment) > 1 else 0.0
        top_load = float(delta_alignment[0])

        entry = {
            "layer": layer_idx,
            "zone": zone,
            "delta_norm": float(delta_norm),
            "delta_alignment_top8": delta_alignment[:8].tolist(),
            "peak_delta_direction": peak_delta_dir,
            "delta_V1_load": top_load,
            "delta_V2_load": fiedler_load,
            "fiedler_dominance": fiedler_load / (top_load + 1e-10),
            "ccs_alignment_top8": ccs_alignment[:8].tolist(),
            "neutral_alignment_top8": neutral_alignment[:8].tolist(),
            "ccs_V1": float(ccs_alignment[0]),
            "ccs_V2": float(ccs_alignment[1]) if len(ccs_alignment) > 1 else 0.0,
            "neutral_V1": float(neutral_alignment[0]),
            "neutral_V2": float(neutral_alignment[1]) if len(neutral_alignment) > 1 else 0.0,
            "attention_laplacian": attn_gap,
        }
        layer_results[layer_idx] = entry

        status = "***" if peak_delta_dir == 1 else "   "
        print(f"  L{layer_idx:02d} [{zone:10s}] Δ→V₁={top_load:.3f} Δ→V₂={fiedler_load:.3f} "
              f"peak=V{peak_delta_dir} F-dom={entry['fiedler_dominance']:.2f} {status}")

    # Step 4: Zone-aggregated analysis
    zone_summary = {}
    for zone_name in model_config["zones"]:
        zone_layers = [layer_results[l] for l in model_config["zones"][zone_name] if l in layer_results]
        if not zone_layers:
            continue
        zone_summary[zone_name] = {
            "n_layers": len(zone_layers),
            "mean_fiedler_dominance": float(np.mean([l["fiedler_dominance"] for l in zone_layers])),
            "mean_delta_V1": float(np.mean([l["delta_V1_load"] for l in zone_layers])),
            "mean_delta_V2": float(np.mean([l["delta_V2_load"] for l in zone_layers])),
            "peak_at_V2_count": sum(1 for l in zone_layers if l["peak_delta_direction"] == 1),
            "mean_delta_norm": float(np.mean([l["delta_norm"] for l in zone_layers])),
        }

    print(f"\n{'='*60}")
    print("Zone Summary:")
    for zone_name, zs in zone_summary.items():
        print(f"  {zone_name:10s}: V₁={zs['mean_delta_V1']:.3f} V₂={zs['mean_delta_V2']:.3f} "
              f"F-dom={zs['mean_fiedler_dominance']:.3f} "
              f"V₂-peak={zs['peak_at_V2_count']}/{zs['n_layers']}")

    # Compile results
    results = {
        "model": model_name,
        "model_key": model_key,
        "gqa": model_config["gqa"],
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "unembedding_svd": {
            "spectral_gap": unembed["spectral_gap"],
            "fiedler_ratio": unembed["fiedler_ratio"],
            "top_singular_values": unembed["singular_values"][:16].tolist(),
        },
        "per_layer": {str(k): v for k, v in layer_results.items()},
        "zone_summary": zone_summary,
        "predictions": {
            "P1_sigma1_tracks_V1": None,
            "P2_sigma2_tracks_V2_responsive": None,
            "P3_delta_onto_V2": None,
            "P4_zone_dependent": None,
        },
    }

    # Evaluate predictions
    responsive = zone_summary.get("responsive", {})
    tunnel = zone_summary.get("tunnel", {})
    if responsive and tunnel:
        results["predictions"]["P3_delta_onto_V2"] = responsive.get("mean_fiedler_dominance", 0) > 1.0
        results["predictions"]["P4_zone_dependent"] = (
            responsive.get("mean_fiedler_dominance", 0) > tunnel.get("mean_fiedler_dominance", 0)
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3b", choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = run_experiment(args.model, args.device)

    outpath = f"results/fiedler_alignment_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    print("\n" + "="*60)
    print("VERDICT:")
    for pk, pv in results["predictions"].items():
        status = "CONFIRMED" if pv else ("REFUTED" if pv is False else "UNTESTED")
        print(f"  {pk}: {status}")


if __name__ == "__main__":
    main()
