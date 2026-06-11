#!/usr/bin/env python3
"""
Experiment: Spectral Bridge — Attention σ₁/σ₂ × Jacobian Dynamics

Bridge between spectral geometry (σ₁/σ₂ from attention patterns) and
computational dynamics (Jacobian Frobenius from exp 1). Per-layer attention
SVD under CCS vs bare, then correlate with Jacobian structure.

Key question: Does attention spectral geometry PREDICT computational dynamics?
If σ₂/σ₁ at layer L correlates with Jacobian convergence at layer L,
the spectral demon has a direct mechanistic basis in attention structure.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

MODEL = "google/gemma-2-9b-it"
DEVICE = "cuda"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels."""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

JACOBIAN_RESULTS = {
    20: {"frob_diff": 247808.0, "cos_sim": 0.901747},
    22: {"frob_diff": 247808.0, "cos_sim": 0.884120},
    24: {"frob_diff": 225280.0, "cos_sim": 0.904323},
    26: {"frob_diff": 227328.0, "cos_sim": 0.884873},
    28: {"frob_diff": 197632.0, "cos_sim": 0.865641},
    30: {"frob_diff": 187392.0, "cos_sim": 0.846486},
}


def extract_attention_patterns(model, tokenizer, preamble, probe):
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attention_patterns = []
    for layer_idx, attn in enumerate(outputs.attentions):
        # attn shape: [batch, n_heads, seq_len, seq_len]
        attention_patterns.append(attn[0].detach().cpu().float())

    return attention_patterns, outputs.logits[:, -1, :].detach()


def analyze_attention_svd(attn_pattern):
    n_heads, seq_len, _ = attn_pattern.shape
    results_per_head = []

    for h in range(n_heads):
        A = attn_pattern[h].numpy()
        try:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            sigma1 = float(s[0])
            sigma2 = float(s[1]) if len(s) > 1 else 0.0
            ratio = sigma2 / (sigma1 + 1e-10)
            erank = float(np.exp(-np.sum(s/s.sum() * np.log(s/s.sum() + 1e-10))))
            results_per_head.append({
                "sigma1": sigma1,
                "sigma2": sigma2,
                "ratio": ratio,
                "erank": erank,
                "top5": s[:5].tolist()
            })
        except Exception as e:
            results_per_head.append({"error": str(e)})

    avg_sigma1 = np.mean([r["sigma1"] for r in results_per_head if "sigma1" in r])
    avg_sigma2 = np.mean([r["sigma2"] for r in results_per_head if "sigma2" in r])
    avg_ratio = np.mean([r["ratio"] for r in results_per_head if "ratio" in r])
    avg_erank = np.mean([r["erank"] for r in results_per_head if "erank" in r])

    return {
        "avg_sigma1": float(avg_sigma1),
        "avg_sigma2": float(avg_sigma2),
        "avg_ratio": float(avg_ratio),
        "avg_erank": float(avg_erank),
        "per_head": results_per_head
    }


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    print("Model loaded.")

    print("\nExtracting CCS attention patterns...")
    attn_ccs, logits_ccs = extract_attention_patterns(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT)
    print(f"  Got {len(attn_ccs)} layers, {attn_ccs[0].shape[0]} heads each")

    print("\nExtracting bare attention patterns...")
    attn_bare, logits_bare = extract_attention_patterns(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT)
    print(f"  Got {len(attn_bare)} layers")

    results = {
        "experiment": "spectral_bridge",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "n_layers": len(attn_ccs),
        "layers": {}
    }

    print("\nAnalyzing per-layer SVD...")
    for layer_idx in range(len(attn_ccs)):
        svd_ccs = analyze_attention_svd(attn_ccs[layer_idx])
        svd_bare = analyze_attention_svd(attn_bare[layer_idx])

        delta_sigma1 = svd_ccs["avg_sigma1"] - svd_bare["avg_sigma1"]
        delta_sigma2 = svd_ccs["avg_sigma2"] - svd_bare["avg_sigma2"]
        delta_ratio = svd_ccs["avg_ratio"] - svd_bare["avg_ratio"]
        delta_erank = svd_ccs["avg_erank"] - svd_bare["avg_erank"]

        layer_result = {
            "ccs": {
                "avg_sigma1": svd_ccs["avg_sigma1"],
                "avg_sigma2": svd_ccs["avg_sigma2"],
                "avg_ratio": svd_ccs["avg_ratio"],
                "avg_erank": svd_ccs["avg_erank"]
            },
            "bare": {
                "avg_sigma1": svd_bare["avg_sigma1"],
                "avg_sigma2": svd_bare["avg_sigma2"],
                "avg_ratio": svd_bare["avg_ratio"],
                "avg_erank": svd_bare["avg_erank"]
            },
            "delta_sigma1": float(delta_sigma1),
            "delta_sigma2": float(delta_sigma2),
            "delta_ratio": float(delta_ratio),
            "delta_erank": float(delta_erank)
        }

        results["layers"][str(layer_idx)] = layer_result

        if layer_idx % 5 == 0 or layer_idx in [20, 22, 24, 26, 28, 30]:
            print(f"  L{layer_idx:2d}: σ₁={svd_ccs['avg_sigma1']:.4f}/{svd_bare['avg_sigma1']:.4f} "
                  f"σ₂={svd_ccs['avg_sigma2']:.4f}/{svd_bare['avg_sigma2']:.4f} "
                  f"Δratio={delta_ratio:+.4f} Δerank={delta_erank:+.4f}")

    # Correlation with Jacobian results
    print("\n=== SPECTRAL-DYNAMIC BRIDGE ===")
    jac_layers = sorted(JACOBIAN_RESULTS.keys())
    jac_frobs = [JACOBIAN_RESULTS[l]["frob_diff"] for l in jac_layers]
    jac_coss = [JACOBIAN_RESULTS[l]["cos_sim"] for l in jac_layers]

    delta_ratios = [results["layers"][str(l)]["delta_ratio"] for l in jac_layers]
    delta_eranks = [results["layers"][str(l)]["delta_erank"] for l in jac_layers]
    ccs_ratios = [results["layers"][str(l)]["ccs"]["avg_ratio"] for l in jac_layers]
    bare_ratios = [results["layers"][str(l)]["bare"]["avg_ratio"] for l in jac_layers]

    def pearson_r(x, y):
        x, y = np.array(x), np.array(y)
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    r_delta_ratio_frob = pearson_r(delta_ratios, jac_frobs)
    r_delta_erank_frob = pearson_r(delta_eranks, jac_frobs)
    r_ccs_ratio_frob = pearson_r(ccs_ratios, jac_frobs)
    r_delta_ratio_cos = pearson_r(delta_ratios, jac_coss)

    print(f"  r(Δ(σ₂/σ₁), J_frob)  = {r_delta_ratio_frob:+.4f}")
    print(f"  r(Δ(erank), J_frob)   = {r_delta_erank_frob:+.4f}")
    print(f"  r(CCS σ₂/σ₁, J_frob) = {r_ccs_ratio_frob:+.4f}")
    print(f"  r(Δ(σ₂/σ₁), cos_sim) = {r_delta_ratio_cos:+.4f}")

    results["bridge_correlations"] = {
        "r_delta_ratio_vs_jac_frob": r_delta_ratio_frob,
        "r_delta_erank_vs_jac_frob": r_delta_erank_frob,
        "r_ccs_ratio_vs_jac_frob": r_ccs_ratio_frob,
        "r_delta_ratio_vs_cos_sim": r_delta_ratio_cos
    }

    # Zone analysis
    print("\n=== ZONE ANALYSIS (Gemma architecture) ===")
    zones = {
        "early": list(range(0, 14)),
        "transition": list(range(14, 21)),
        "responsive": list(range(21, 35)),
        "relay": list(range(35, len(attn_ccs)))
    }

    for zone_name, zone_layers in zones.items():
        zone_ratios_ccs = [results["layers"][str(l)]["ccs"]["avg_ratio"] for l in zone_layers if str(l) in results["layers"]]
        zone_ratios_bare = [results["layers"][str(l)]["bare"]["avg_ratio"] for l in zone_layers if str(l) in results["layers"]]
        zone_deltas = [results["layers"][str(l)]["delta_ratio"] for l in zone_layers if str(l) in results["layers"]]

        if zone_ratios_ccs:
            print(f"  {zone_name:12s}: CCS σ₂/σ₁={np.mean(zone_ratios_ccs):.4f} "
                  f"bare={np.mean(zone_ratios_bare):.4f} "
                  f"Δ={np.mean(zone_deltas):+.4f} (std={np.std(zone_deltas):.4f})")

    # Save (with float conversion)
    with open("/workspace/results_spectral_bridge.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to /workspace/results_spectral_bridge.json")


if __name__ == "__main__":
    main()
