#!/usr/bin/env python3
"""Test whether randomly initialized Qwen 2.5 7B shows biological criticality.

Pachitariu & Stringer (Nature 2026): random symmetric matrices at critical
normalization produce power-law covariance spectra (PL exponents 0.7-0.85).

Prediction: If critical initialization holds for transformers, random init
should show PL exponents near biological range at ALL layers. Training
(pre-training + RLHF) prunes criticality from relay layers, preserving it
only at L9 (seed).

This runs on CPU — no inference needed, just weight matrix eigenvalues."""

import json
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

LAYERS = [5, 9, 14, 16, 17, 20, 25, 27]


def power_law_exponent(eigenvalues):
    """Estimate PL exponent from eigenvalue spectrum via log-log regression."""
    eigs = np.sort(np.abs(eigenvalues))[::-1]
    eigs = eigs[eigs > 1e-10]
    if len(eigs) < 10:
        return None
    ranks = np.arange(1, len(eigs) + 1)
    log_r = np.log(ranks)
    log_e = np.log(eigs)
    mask = np.isfinite(log_r) & np.isfinite(log_e)
    if mask.sum() < 10:
        return None
    coeffs = np.polyfit(log_r[mask], log_e[mask], 1)
    return -coeffs[0]


def analyze_layer(layer_module, layer_idx):
    """Extract weight matrices and compute spectral properties."""
    results = {}

    for name, param in layer_module.named_parameters():
        if param.dim() < 2:
            continue
        w = param.detach().float().cpu().numpy()
        if w.shape[0] > 4096 or w.shape[1] > 4096:
            idx = np.random.choice(min(w.shape), size=min(2048, min(w.shape)), replace=False)
            w = w[np.ix_(idx, idx)] if w.shape[0] == w.shape[1] else w[:min(2048, w.shape[0]), :min(2048, w.shape[1])]

        try:
            if w.shape[0] == w.shape[1]:
                eigs = np.linalg.eigvalsh(w)
            else:
                s = np.linalg.svdvals(w)
                eigs = s
        except np.linalg.LinAlgError:
            continue

        pl = power_law_exponent(eigs)
        lambda_max = float(np.max(np.abs(eigs)))
        results[name] = {
            "pl_exponent": pl,
            "lambda_max": lambda_max,
            "shape": list(w.shape),
        }

    return results


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print("Initializing random model (same architecture, random weights)...")
    torch.manual_seed(42)
    random_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    print("Loading trained model...")
    trained_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )

    results = {"random": {}, "trained": {}}

    for l in LAYERS:
        print(f"\nLayer {l}...")
        if l < len(random_model.model.layers):
            results["random"][f"L{l}"] = analyze_layer(random_model.model.layers[l], l)
        if l < len(trained_model.model.layers):
            results["trained"][f"L{l}"] = analyze_layer(trained_model.model.layers[l], l)

    print("\n" + "=" * 60)
    print("SUMMARY: Power-law exponents (biological range: 0.7-0.85)")
    print("=" * 60)

    for model_type in ["random", "trained"]:
        print(f"\n{model_type.upper()}:")
        for layer_key in sorted(results[model_type].keys(), key=lambda x: int(x[1:])):
            layer_data = results[model_type][layer_key]
            attn_pls = []
            mlp_pls = []
            for param_name, data in layer_data.items():
                if data["pl_exponent"] is not None:
                    if "self_attn" in param_name:
                        attn_pls.append(data["pl_exponent"])
                    elif "mlp" in param_name:
                        mlp_pls.append(data["pl_exponent"])
            attn_mean = np.mean(attn_pls) if attn_pls else None
            mlp_mean = np.mean(mlp_pls) if mlp_pls else None
            print(f"  {layer_key}: attn PL={attn_mean:.3f}" if attn_mean else f"  {layer_key}: attn PL=N/A", end="")
            print(f"  mlp PL={mlp_mean:.3f}" if mlp_mean else "  mlp PL=N/A")

    with open("/home/nate-agx/spectral-demon/results/cna_random_init_criticality.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to results/cna_random_init_criticality.json")

    del random_model, trained_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
