#!/usr/bin/env python3
"""
Spectral Patch: Fix the rank-1 SVD bug in attention_ablation_bridge.

The original experiment computed SVD on h[:, -1, :] which is (1, d_model) = rank 1.
σ₂ is always 0. Fix: use full sequence activation h[:, :, :] → (seq_len, d_model).

Loads existing ablation results, adds correct spectral measurements, computes correlations.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

DEVICE = "cuda"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels.
"""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

MODELS = {
    "gemma": {
        "name": "google/gemma-2-9b-it",
        "n_layers": 42,
        "layers": list(range(2, 40, 2)),
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "layers": list(range(2, 30, 2)),
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "layers": list(range(2, 26, 2)),
    }
}


def compute_spectral_profiles(model, tokenizer, preamble, probe, layers):
    """Compute σ₁, σ₂, σ₂/σ₁ using FULL sequence activations (not just last token)."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.squeeze(0).detach().float()  # (seq_len, d_model)
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        if i in layers:
            hooks.append(layer.self_attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    profiles = {}
    for layer_idx in layers:
        if layer_idx not in captured:
            continue
        act = captured[layer_idx]  # (seq_len, d_model)
        try:
            svs = torch.linalg.svdvals(act)
            s1 = svs[0].item()
            s2 = svs[1].item() if len(svs) > 1 else 0.0
            erank_raw = svs / (svs.sum() + 1e-10)
            erank = float(np.exp(-float((erank_raw * torch.log(erank_raw + 1e-10)).sum())))
            profiles[layer_idx] = {
                "sigma1": s1,
                "sigma2": s2,
                "ratio": s2 / (s1 + 1e-10),
                "erank": erank,
                "n_svs": min(act.shape),
            }
        except Exception as e:
            print(f"  SVD failed at L{layer_idx}: {e}")
            profiles[layer_idx] = {"sigma1": 0.0, "sigma2": 0.0, "ratio": 0.0}

    return profiles


def main():
    # Load existing ablation results
    try:
        with open("/workspace/results_attention_ablation_bridge.json") as f:
            existing = json.load(f)
        print("Loaded existing ablation results")
    except FileNotFoundError:
        existing = None
        print("No existing results found — computing from scratch")

    results = {
        "experiment": "attention_ablation_bridge_v2",
        "timestamp": datetime.now().isoformat(),
        "description": "Fixed spectral measurement (full sequence SVD) + ablation KL from v1",
        "architectures": {}
    }

    for arch_key, arch_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {arch_key.upper()} — {arch_info['name']}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(arch_info["name"])
        model = AutoModelForCausalLM.from_pretrained(
            arch_info["name"], torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model.eval()

        layers = arch_info["layers"]

        # Correct spectral profiles
        print(f"\n  Computing spectral profiles (full sequence SVD)...")
        ccs_spectral = compute_spectral_profiles(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layers)
        bare_spectral = compute_spectral_profiles(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT, layers)

        for l in layers:
            if l in ccs_spectral and l in bare_spectral:
                enrichment = ccs_spectral[l]["ratio"] - bare_spectral[l]["ratio"]
                print(f"    L{l}: CCS σ₂/σ₁={ccs_spectral[l]['ratio']:.4f}  "
                      f"bare={bare_spectral[l]['ratio']:.4f}  enrichment={enrichment:+.4f}  "
                      f"σ₁={ccs_spectral[l]['sigma1']:.1f}")

        # Merge with existing ablation data
        arch_results = {"model": arch_info["name"], "layers": {}}

        existing_layers = {}
        if existing and arch_key in existing.get("architectures", {}):
            existing_layers = existing["architectures"][arch_key].get("layers", {})

        for l in layers:
            l_str = str(l)
            entry = {}

            if l in ccs_spectral:
                entry["ccs_spectral"] = ccs_spectral[l]
            if l in bare_spectral:
                entry["bare_spectral"] = bare_spectral[l]
            if l in ccs_spectral and l in bare_spectral:
                entry["enrichment"] = ccs_spectral[l]["ratio"] - bare_spectral[l]["ratio"]

            # Pull ablation data from v1
            if l_str in existing_layers:
                for key in ["kl_ccs", "kl_bare", "kl_delta", "j_frob"]:
                    if key in existing_layers[l_str]:
                        entry[key] = existing_layers[l_str][key]

            arch_results["layers"][l_str] = entry

        # Correlations
        print(f"\n  Computing correlations...")
        layer_keys = sorted(arch_results["layers"].keys(), key=int)

        # Filter to layers that have both enrichment and KL
        valid_kl = [k for k in layer_keys
                    if "enrichment" in arch_results["layers"][k]
                    and "kl_ccs" in arch_results["layers"][k]]

        if len(valid_kl) > 3:
            enrichments = [arch_results["layers"][k]["enrichment"] for k in valid_kl]
            kl_ccs_vals = [arch_results["layers"][k]["kl_ccs"] for k in valid_kl]
            kl_deltas = [arch_results["layers"][k].get("kl_delta", 0) for k in valid_kl]
            ratios_ccs = [arch_results["layers"][k]["ccs_spectral"]["ratio"] for k in valid_kl]

            r1, p1 = stats.pearsonr(enrichments, kl_ccs_vals)
            r2, p2 = stats.pearsonr(enrichments, kl_deltas)
            r3, p3 = stats.pearsonr(ratios_ccs, kl_ccs_vals)

            print(f"    r(enrichment, KL_ccs) = {r1:.3f} (p={p1:.4f})")
            print(f"    r(enrichment, ΔKL) = {r2:.3f} (p={p2:.4f})")
            print(f"    r(σ₂/σ₁_ccs, KL_ccs) = {r3:.3f} (p={p3:.4f})")

            arch_results["correlations"] = {
                "enrichment_vs_kl_ccs": {"r": float(r1), "p": float(p1)},
                "enrichment_vs_kl_delta": {"r": float(r2), "p": float(p2)},
                "ratio_ccs_vs_kl_ccs": {"r": float(r3), "p": float(p3)},
            }

            # Delta enrichment for Mistral (absolute predicts nothing, delta predicts)
            if arch_key == "mistral" and len(valid_kl) > 4:
                delta_enrichments = []
                for i, k in enumerate(valid_kl):
                    if i == 0:
                        delta_enrichments.append(0)
                    else:
                        prev = arch_results["layers"][valid_kl[i-1]]["enrichment"]
                        curr = arch_results["layers"][k]["enrichment"]
                        delta_enrichments.append(curr - prev)
                r_delta, p_delta = stats.pearsonr(delta_enrichments[1:], kl_ccs_vals[1:])
                print(f"    r(Δenrichment, KL_ccs) = {r_delta:.3f} (p={p_delta:.4f}) [Mistral delta]")
                arch_results["correlations"]["delta_enrichment_vs_kl_ccs"] = {"r": float(r_delta), "p": float(p_delta)}

        # Bridge: σ₂/σ₁ vs J_frob
        valid_jac = [k for k in layer_keys
                     if "ccs_spectral" in arch_results["layers"][k]
                     and "j_frob" in arch_results["layers"][k]]

        if len(valid_jac) > 3:
            ratios = [arch_results["layers"][k]["ccs_spectral"]["ratio"] for k in valid_jac]
            frobs = [arch_results["layers"][k]["j_frob"] for k in valid_jac]
            r_bridge, p_bridge = stats.pearsonr(ratios, frobs)
            print(f"    r(σ₂/σ₁, J_frob) = {r_bridge:.3f} (p={p_bridge:.4f}) [bridge]")
            arch_results["correlations"]["bridge"] = {"r": float(r_bridge), "p": float(p_bridge)}

            # Enrichment vs J_frob
            jac_enrichments = [arch_results["layers"][k].get("enrichment", 0) for k in valid_jac]
            r_ej, p_ej = stats.pearsonr(jac_enrichments, frobs)
            print(f"    r(enrichment, J_frob) = {r_ej:.3f} (p={p_ej:.4f})")
            arch_results["correlations"]["enrichment_vs_jfrob"] = {"r": float(r_ej), "p": float(p_ej)}

        results["architectures"][arch_key] = arch_results

        del model
        torch.cuda.empty_cache()

    # Save
    out_path = "/workspace/results_attention_ablation_bridge_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    # Summary
    print(f"\n{'='*60}")
    print(f"  ATTENTION ABLATION BRIDGE v2 — SUMMARY")
    print(f"{'='*60}")
    for arch_key, arch_data in results["architectures"].items():
        print(f"\n  {arch_key.upper()}:")
        corrs = arch_data.get("correlations", {})
        for name, vals in corrs.items():
            print(f"    {name}: r={vals['r']:.3f} (p={vals['p']:.4f})")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
