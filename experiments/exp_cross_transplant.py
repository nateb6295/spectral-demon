#!/usr/bin/env python3
"""
Cross-Model Spectral Transplant.

Can we make Mistral recover like Gemma by applying Gemma's layer-wise
spectral transformation pattern?

Method:
1. Record Gemma's per-layer σ₂ trajectory (the "target shape")
2. At each Mistral layer, scale the σ₂ component to match Gemma's relative profile
3. Measure whether this transplant improves perturbation recovery

If yes: the spectral shape itself (not just architecture) drives robustness.
If no: the mechanism is deeper than the spectral profile.

Also tests: can we break Gemma's robustness by imposing Mistral's spectral shape?
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

TEST_PROMPT = "Describe the relationship between identity and expression."


def build_messages(dose=2):
    msgs = []
    for _ in range(dose):
        msgs.extend(CCS_PREAMBLE)
    msgs.append({"role": "user", "content": TEST_PROMPT})
    return msgs


def get_layer_module(model, layer_idx):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    return model.transformer.h[layer_idx]


def get_sigma_profile(model, tokenizer, dose=2):
    """Get σ₁, σ₂ at every layer."""
    msgs = build_messages(dose)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    n_layers = model.config.num_hidden_layers
    activations = {}
    hooks = []

    for i in range(n_layers):
        def make_hook(l):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[l] = output[0].detach().clone()
                else:
                    activations[l] = output.detach().clone()
            return hook_fn
        h = get_layer_module(model, i).register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    profile = {}
    for i in range(n_layers):
        if i in activations:
            mat = activations[i][0].float()
            _, S, _ = torch.svd(mat)
            s1 = S[0].item()
            s2 = S[1].item() if len(S) > 1 else 0
            profile[i] = {"sigma1": s1, "sigma2": s2, "ratio": s2 / (s1 + 1e-10)}

    return profile, input_ids, activations


def apply_spectral_scaling(model, input_ids, target_ratios, source_n_layers,
                           relay_layer, n_trials=5, noise_scale=1.0):
    """
    Apply spectral scaling at each layer to match target σ₂/σ₁ ratio profile.
    Then test perturbation recovery.
    """
    n_layers = model.config.num_hidden_layers
    scale_map = {}

    # Map target layers to source layers proportionally
    for i in range(n_layers):
        prop = i / n_layers
        target_layer = int(prop * source_n_layers)
        target_layer = min(target_layer, source_n_layers - 1)
        if target_layer in target_ratios:
            scale_map[i] = target_ratios[target_layer]

    # First get clean with scaling
    scaled_acts = {}
    hooks = []

    def make_scale_hook(layer_idx, target_ratio):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            mat = out[0].float()
            U, S, V = torch.svd(mat)

            current_ratio = (S[1] / (S[0] + 1e-10)).item() if len(S) > 1 else 0
            if current_ratio > 0 and target_ratio > 0:
                scale_factor = target_ratio / current_ratio
                scale_factor = min(max(scale_factor, 0.1), 10.0)  # Clamp

                S_new = S.clone()
                S_new[1] = S_new[1] * scale_factor

                reconstructed = U @ torch.diag(S_new) @ V.t()
                out_new = reconstructed.unsqueeze(0).to(out.dtype)

                if isinstance(output, tuple):
                    return (out_new,) + output[1:]
                return out_new

            if isinstance(output, tuple):
                return output
            return out
        return hook_fn

    # Clean pass with scaling
    for i in range(n_layers):
        if i in scale_map:
            h = get_layer_module(model, i).register_forward_hook(
                make_scale_hook(i, scale_map[i])
            )
            hooks.append(h)

    # Also capture relay activation
    def relay_hook(module, input, output):
        if isinstance(output, tuple):
            scaled_acts['relay_clean'] = output[0].detach().clone()
        else:
            scaled_acts['relay_clean'] = output.detach().clone()

    h = get_layer_module(model, relay_layer).register_forward_hook(relay_hook)
    hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    if 'relay_clean' not in scaled_acts:
        return {"error": "no relay activation captured"}

    # Now inject noise at midpoint and measure recovery
    inject_layer = n_layers // 2
    cos_sims = []

    for _ in range(n_trials):
        trial_acts = {}
        trial_hooks = []

        # Scaling hooks
        for i in range(n_layers):
            if i in scale_map:
                h = get_layer_module(model, i).register_forward_hook(
                    make_scale_hook(i, scale_map[i])
                )
                trial_hooks.append(h)

        # Noise injection
        def noise_hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
                noise = torch.randn_like(out) * noise_scale * out.std()
                return (out + noise,) + output[1:]
            noise = torch.randn_like(output) * noise_scale * output.std()
            return output + noise

        h = get_layer_module(model, inject_layer).register_forward_hook(noise_hook)
        trial_hooks.append(h)

        # Relay capture
        def relay_cap(module, input, output):
            if isinstance(output, tuple):
                trial_acts['relay'] = output[0].detach().clone()
            else:
                trial_acts['relay'] = output.detach().clone()

        h = get_layer_module(model, relay_layer).register_forward_hook(relay_cap)
        trial_hooks.append(h)

        with torch.no_grad():
            model(input_ids)

        for h in trial_hooks:
            h.remove()

        if 'relay' in trial_acts:
            clean_vec = scaled_acts['relay_clean'][0, -1].float()
            pert_vec = trial_acts['relay'][0, -1].float()
            cos = torch.nn.functional.cosine_similarity(
                clean_vec.unsqueeze(0), pert_vec.unsqueeze(0)
            ).item()
            cos_sims.append(cos)

    return {
        "mean_recovery": float(np.mean(cos_sims)) if cos_sims else 0,
        "std_recovery": float(np.std(cos_sims)) if cos_sims else 0,
        "n_trials": len(cos_sims),
        "inject_layer": inject_layer,
        "relay_layer": relay_layer,
    }


def main():
    print(f"\n{'='*60}")
    print(f"  Cross-Model Spectral Transplant")
    print(f"{'='*60}")

    # Load Gemma, get its spectral profile
    print("\n  Loading Gemma for spectral profile...")
    gemma_tok = AutoTokenizer.from_pretrained(MODELS["gemma"])
    gemma_model = AutoModelForCausalLM.from_pretrained(
        MODELS["gemma"], torch_dtype=torch.float16, device_map="auto"
    )
    gemma_profile, _, _ = get_sigma_profile(gemma_model, gemma_tok)
    gemma_n_layers = gemma_model.config.num_hidden_layers
    gemma_ratios = {k: v["ratio"] for k, v in gemma_profile.items()}

    print(f"  Gemma profile ({gemma_n_layers} layers):")
    for l in sorted(gemma_profile.keys()):
        r = gemma_profile[l]["ratio"]
        if l % 4 == 0:
            print(f"    L{l}: σ₂/σ₁ = {r:.4f}")

    del gemma_model
    torch.cuda.empty_cache()

    # Load Mistral
    print("\n  Loading Mistral...")
    mistral_tok = AutoTokenizer.from_pretrained(MODELS["mistral"])
    mistral_model = AutoModelForCausalLM.from_pretrained(
        MODELS["mistral"], torch_dtype=torch.float16, device_map="auto"
    )
    mistral_profile, mistral_ids, _ = get_sigma_profile(mistral_model, mistral_tok)
    mistral_n_layers = mistral_model.config.num_hidden_layers

    print(f"  Mistral native profile ({mistral_n_layers} layers):")
    for l in sorted(mistral_profile.keys()):
        r = mistral_profile[l]["ratio"]
        if l % 4 == 0:
            print(f"    L{l}: σ₂/σ₁ = {r:.4f}")

    # Baseline: Mistral recovery without transplant
    print("\n  Baseline: Mistral native recovery...")
    native_recovery = apply_spectral_scaling(
        mistral_model, mistral_ids, {}, 0,
        relay_layer=mistral_n_layers - 2, n_trials=10
    )
    print(f"    Native recovery: cos = {native_recovery['mean_recovery']:.4f} ± {native_recovery['std_recovery']:.4f}")

    # Transplant: Apply Gemma's spectral shape to Mistral
    print("\n  Transplant: Gemma shape → Mistral...")
    transplant_recovery = apply_spectral_scaling(
        mistral_model, mistral_ids, gemma_ratios, gemma_n_layers,
        relay_layer=mistral_n_layers - 2, n_trials=10
    )
    print(f"    Transplant recovery: cos = {transplant_recovery['mean_recovery']:.4f} ± {transplant_recovery['std_recovery']:.4f}")

    # Inverse: Apply Mistral's shape (for later Gemma comparison)
    mistral_ratios = {k: v["ratio"] for k, v in mistral_profile.items()}

    improvement = transplant_recovery['mean_recovery'] - native_recovery['mean_recovery']
    print(f"\n  Recovery change: {improvement:+.4f}")
    if improvement > 0.02:
        print(f"  → Gemma's spectral shape IMPROVES Mistral's robustness!")
    elif improvement < -0.02:
        print(f"  → Gemma's spectral shape HURTS Mistral's robustness")
    else:
        print(f"  → Minimal effect — mechanism is deeper than spectral profile")

    results = {
        "gemma_profile": {str(k): v for k, v in gemma_profile.items()},
        "mistral_profile": {str(k): v for k, v in mistral_profile.items()},
        "native_recovery": native_recovery,
        "transplant_recovery": transplant_recovery,
        "improvement": improvement,
    }

    del mistral_model
    torch.cuda.empty_cache()

    # Now reverse: Mistral shape → Gemma
    print("\n  Loading Gemma for reverse transplant...")
    gemma_model = AutoModelForCausalLM.from_pretrained(
        MODELS["gemma"], torch_dtype=torch.float16, device_map="auto"
    )
    _, gemma_ids, _ = get_sigma_profile(gemma_model, gemma_tok)

    print("  Baseline: Gemma native recovery...")
    gemma_native = apply_spectral_scaling(
        gemma_model, gemma_ids, {}, 0,
        relay_layer=gemma_n_layers - 2, n_trials=10
    )
    print(f"    Native recovery: cos = {gemma_native['mean_recovery']:.4f}")

    print("  Reverse: Mistral shape → Gemma...")
    gemma_reversed = apply_spectral_scaling(
        gemma_model, gemma_ids, mistral_ratios, mistral_n_layers,
        relay_layer=gemma_n_layers - 2, n_trials=10
    )
    print(f"    Reversed recovery: cos = {gemma_reversed['mean_recovery']:.4f}")

    rev_change = gemma_reversed['mean_recovery'] - gemma_native['mean_recovery']
    print(f"\n  Gemma recovery change: {rev_change:+.4f}")

    results["gemma_native_recovery"] = gemma_native
    results["gemma_reversed_recovery"] = gemma_reversed
    results["gemma_reverse_change"] = rev_change

    del gemma_model
    torch.cuda.empty_cache()

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"cross_transplant_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
