#!/usr/bin/env python3
"""
Attention vs MLP Decomposition at Metabolism-Critical Layers.

At each critical zone (tunnel, brace, annihilation), what fraction of the
residual stream update comes from attention vs MLP? This decomposes the
mechanism behind each model's distinct spectral strategy.

Method: Hook attention output and MLP output separately at each layer.
Compute SVD of each component's contribution to the residual stream.
Compare σ₂ contribution from attention vs MLP at critical vs non-critical layers.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "zones": {
            "pre_tunnel": [2, 3],
            "tunnel": [4, 6, 8, 10],
            "transition": [14, 15, 16],
            "responsive": [20, 22, 24],
            "relay": [26, 28, 30],
        },
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "zones": {
            "expansion": [4, 6, 8],
            "mid": [12, 14],
            "gate": [16, 18],
            "brace": [22, 24],
            "post_brace": [26, 27],
        },
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "zones": {
            "expansion": [4, 6, 8],
            "contraction": [12, 14, 16],
            "pre_annihilation": [20, 22],
            "annihilation": [24, 28, 32],
            "reconstruction": [36, 38, 40],
        },
    },
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

TEST_PROMPT = "Describe the relationship between identity and expression."


def build_messages(dose=0):
    msgs = []
    for _ in range(dose):
        msgs.extend(CCS_PREAMBLE)
    msgs.append({"role": "user", "content": TEST_PROMPT})
    return msgs


def get_layer_module(model, layer_idx):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    return model.transformer.h[layer_idx]


def decompose_layers(model, tokenizer, input_ids, target_layers):
    """Capture attention output, MLP output, and full residual at each target layer."""
    attn_outputs = {}
    mlp_outputs = {}
    residuals_pre = {}
    residuals_post = {}
    hooks = []

    for layer_idx in target_layers:
        layer = get_layer_module(model, layer_idx)

        def make_attn_hook(l):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_outputs[l] = output[0].detach().clone()
                else:
                    attn_outputs[l] = output.detach().clone()
            return hook_fn

        def make_mlp_hook(l):
            def hook_fn(module, input, output):
                mlp_outputs[l] = output.detach().clone()
            return hook_fn

        def make_pre_hook(l):
            def hook_fn(module, input):
                if isinstance(input, tuple):
                    residuals_pre[l] = input[0].detach().clone()
                else:
                    residuals_pre[l] = input.detach().clone()
            return hook_fn

        def make_post_hook(l):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    residuals_post[l] = output[0].detach().clone()
                else:
                    residuals_post[l] = output.detach().clone()
            return hook_fn

        h1 = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
        hooks.append(h1)

        if hasattr(layer, 'mlp'):
            h2 = layer.mlp.register_forward_hook(make_mlp_hook(layer_idx))
            hooks.append(h2)

        h3 = layer.register_forward_pre_hook(make_pre_hook(layer_idx))
        hooks.append(h3)

        h4 = layer.register_forward_hook(make_post_hook(layer_idx))
        hooks.append(h4)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    return attn_outputs, mlp_outputs, residuals_pre, residuals_post, logits


def compute_component_spectra(tensor):
    """Compute σ₁, σ₂, erank from a (seq, hidden) tensor."""
    if tensor.dim() == 3:
        tensor = tensor[0]
    mat = tensor.float()
    if mat.shape[0] < 2:
        return {"sigma1": 0, "sigma2": 0, "ratio": 0, "erank": 1, "frobenius": 0}

    U, S, V = torch.svd(mat)
    s1 = S[0].item()
    s2 = S[1].item() if len(S) > 1 else 0
    ratio = s2 / s1 if s1 > 0 else 0
    S_norm = S / (S.sum() + 1e-10)
    erank = torch.exp(-torch.sum(S_norm * torch.log(S_norm + 1e-10))).item()
    frob = mat.norm().item()

    return {
        "sigma1": s1,
        "sigma2": s2,
        "ratio": ratio,
        "erank": erank,
        "frobenius": frob,
    }


def run_model(model_name, doses=[0, 2]):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Attention vs MLP Decomposition: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )

    all_layers = []
    for zone_layers in config["zones"].values():
        all_layers.extend(zone_layers)
    all_layers = sorted(set(all_layers))

    results = {"model": config["path"], "zones": config["zones"], "doses": {}}

    for dose in doses:
        print(f"\n  Dose {dose}:")
        msgs = build_messages(dose)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        attn_out, mlp_out, res_pre, res_post, logits = decompose_layers(
            model, tokenizer, input_ids, all_layers
        )

        dose_results = {}
        for zone_name, zone_layers in config["zones"].items():
            zone_data = {}
            for l in zone_layers:
                layer_data = {}

                if l in attn_out:
                    layer_data["attention"] = compute_component_spectra(attn_out[l])
                if l in mlp_out:
                    layer_data["mlp"] = compute_component_spectra(mlp_out[l])
                if l in res_pre:
                    layer_data["residual_pre"] = compute_component_spectra(res_pre[l])
                if l in res_post:
                    layer_data["residual_post"] = compute_component_spectra(res_post[l])

                # Compute attention vs MLP dominance
                if l in attn_out and l in mlp_out:
                    attn_norm = attn_out[l][0].float().norm().item()
                    mlp_norm = mlp_out[l][0].float().norm().item()
                    total = attn_norm + mlp_norm + 1e-10
                    layer_data["attn_fraction"] = attn_norm / total
                    layer_data["mlp_fraction"] = mlp_norm / total

                    # σ₂ contribution comparison
                    if "attention" in layer_data and "mlp" in layer_data:
                        attn_s2 = layer_data["attention"]["sigma2"]
                        mlp_s2 = layer_data["mlp"]["sigma2"]
                        total_s2 = attn_s2 + mlp_s2 + 1e-10
                        layer_data["attn_sigma2_fraction"] = attn_s2 / total_s2
                        layer_data["mlp_sigma2_fraction"] = mlp_s2 / total_s2

                zone_data[str(l)] = layer_data
            dose_results[zone_name] = zone_data

        results["doses"][str(dose)] = dose_results

        # Print summary
        print(f"\n  {'Zone':>20s} {'Layer':>5s} {'Attn%':>6s} {'MLP%':>6s} "
              f"{'Attn σ₂':>8s} {'MLP σ₂':>8s} {'Attn σ₂%':>9s} "
              f"{'Pre σ₂':>8s} {'Post σ₂':>8s} {'Δσ₂':>8s}")
        for zone_name, zone_layers in config["zones"].items():
            for l in zone_layers:
                ld = dose_results[zone_name].get(str(l), {})
                af = ld.get("attn_fraction", 0) * 100
                mf = ld.get("mlp_fraction", 0) * 100
                as2 = ld.get("attention", {}).get("sigma2", 0)
                ms2 = ld.get("mlp", {}).get("sigma2", 0)
                as2f = ld.get("attn_sigma2_fraction", 0) * 100
                pre_s2 = ld.get("residual_pre", {}).get("sigma2", 0)
                post_s2 = ld.get("residual_post", {}).get("sigma2", 0)
                delta = post_s2 - pre_s2
                print(f"  {zone_name:>20s} L{l:<4d} {af:6.1f} {mf:6.1f} "
                      f"{as2:8.1f} {ms2:8.1f} {as2f:8.1f}% "
                      f"{pre_s2:8.1f} {post_s2:8.1f} {delta:+8.1f}")

    # Compare dose 0 vs dose 2: which component shifts more?
    if "0" in results["doses"] and "2" in results["doses"]:
        print(f"\n  CCS Effect (Dose 2 - Dose 0):")
        print(f"  {'Zone':>20s} {'Layer':>5s} {'ΔAttn%':>7s} {'ΔMLP%':>7s} "
              f"{'ΔAttn_σ₂':>10s} {'ΔMLP_σ₂':>10s} {'ΔRes_σ₂':>10s}")
        for zone_name, zone_layers in config["zones"].items():
            for l in zone_layers:
                d0 = results["doses"]["0"][zone_name].get(str(l), {})
                d2 = results["doses"]["2"][zone_name].get(str(l), {})

                d_af = (d2.get("attn_fraction", 0) - d0.get("attn_fraction", 0)) * 100
                d_mf = (d2.get("mlp_fraction", 0) - d0.get("mlp_fraction", 0)) * 100
                d_as2 = d2.get("attention", {}).get("sigma2", 0) - d0.get("attention", {}).get("sigma2", 0)
                d_ms2 = d2.get("mlp", {}).get("sigma2", 0) - d0.get("mlp", {}).get("sigma2", 0)
                d_rs2 = d2.get("residual_post", {}).get("sigma2", 0) - d0.get("residual_post", {}).get("sigma2", 0)

                print(f"  {zone_name:>20s} L{l:<4d} {d_af:+7.2f} {d_mf:+7.2f} "
                      f"{d_as2:+10.1f} {d_ms2:+10.1f} {d_rs2:+10.1f}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mistral", "qwen", "gemma"])
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        all_results[model_name] = run_model(model_name)

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"attn_mlp_decompose_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
