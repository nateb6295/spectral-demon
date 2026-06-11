#!/usr/bin/env python3
"""
Hub Ablation: Does ablating the CCS hub heads break the metabolism?

Zero out the top-3 AIE heads per model and measure:
1. σ₂ profile change (does the spectral geometry shift?)
2. Robustness change (does the perturbation immunity change?)
3. Output distribution change (KL from clean)

If hubs are causally necessary, ablation should:
- Mistral: reduce relay expansion (aerobic metabolism breaks)
- Gemma: disrupt annihilation zone or reconstruction (extremophile breaks)
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "hub_heads": [(14, 11), (26, 16), (15, 29)],  # Top 3 from AIE
        "perturb_layer": 15,  # transition
        "measure_layers": [0, 4, 8, 12, 16, 20, 24, 28],
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "hub_heads": [(18, 25), (16, 3), (12, 3)],
        "perturb_layer": 16,  # gate
        "measure_layers": [0, 4, 8, 12, 16, 20, 24],
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "hub_heads": [(41, 6), (40, 6), (41, 11)],
        "perturb_layer": 24,  # annihilation entry
        "measure_layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    },
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


def get_activations(model, input_ids, measure_layers, ablate_heads=None):
    """Forward pass with optional head ablation."""
    activations = {}
    hooks = []

    # Ablation hooks
    if ablate_heads:
        for layer_idx, head_idx in ablate_heads:
            n_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // n_heads

            def make_ablation_hook(l, h, hd):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        start = h * hd
                        end = (h + 1) * hd
                        out[:, :, start:end] = 0
                        return (out,) + output[1:]
                    else:
                        output = output.clone()
                        start = h * hd
                        end = (h + 1) * hd
                        output[:, :, start:end] = 0
                        return output
                return hook_fn

            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                attn = model.model.layers[layer_idx].self_attn
            else:
                attn = model.transformer.h[layer_idx].self_attn
            h = attn.register_forward_hook(make_ablation_hook(layer_idx, head_idx, head_dim))
            hooks.append(h)

    # Measurement hooks
    def make_measure_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach().clone()
            else:
                activations[layer_idx] = output.detach().clone()
        return hook_fn

    for ml in measure_layers:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[ml]
        else:
            layer = model.transformer.h[ml]
        h = layer.register_forward_hook(make_measure_hook(ml))
        hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    return logits, activations


def compute_sigma_profile(activations, measure_layers):
    """Compute σ₁, σ₂ at each measurement layer."""
    profile = {}
    for ml in measure_layers:
        if ml in activations:
            act = activations[ml][0].float()  # (seq, hidden)
            U, S, V = torch.svd(act)
            sigma1 = S[0].item()
            sigma2 = S[1].item() if len(S) > 1 else 0
            ratio = sigma2 / sigma1 if sigma1 > 0 else 0
            profile[ml] = {
                "sigma1": sigma1,
                "sigma2": sigma2,
                "ratio": ratio,
                "erank": torch.exp(-torch.sum((S/S.sum()) * torch.log((S/S.sum()) + 1e-10))).item(),
            }
    return profile


def run_model(model_name, dose=2):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Hub Ablation: {model_name}")
    print(f"  Ablating heads: {config['hub_heads']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )

    msgs = build_messages(dose)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    measure_layers = config["measure_layers"]

    # 1. Clean pass (no ablation)
    print("  Clean pass...")
    clean_logits, clean_acts = get_activations(model, input_ids, measure_layers)
    clean_profile = compute_sigma_profile(clean_acts, measure_layers)

    # 2. Ablated pass (zero hub heads)
    print("  Ablated pass (hub heads zeroed)...")
    abl_logits, abl_acts = get_activations(model, input_ids, measure_layers, config["hub_heads"])
    abl_profile = compute_sigma_profile(abl_acts, measure_layers)

    # 3. Compare σ₂ profiles
    print("\n  σ₂ Profile Comparison:")
    print(f"  {'Layer':>5s}  {'Clean σ₂':>10s}  {'Ablated σ₂':>10s}  {'Δσ₂':>8s}  {'Clean ratio':>12s}  {'Abl ratio':>10s}")
    for ml in measure_layers:
        if ml in clean_profile and ml in abl_profile:
            cs2 = clean_profile[ml]['sigma2']
            as2 = abl_profile[ml]['sigma2']
            cr = clean_profile[ml]['ratio']
            ar = abl_profile[ml]['ratio']
            print(f"  L{ml:3d}  {cs2:10.1f}  {as2:10.1f}  {as2-cs2:+8.1f}  {cr:12.4f}  {ar:10.4f}")

    # 4. Output distribution comparison
    clean_probs = torch.softmax(clean_logits[0, -1].float(), dim=-1)
    abl_probs = torch.softmax(abl_logits[0, -1].float(), dim=-1)
    eps = 1e-8
    kl = torch.sum(
        clean_probs.clamp(min=eps) * torch.log(clean_probs.clamp(min=eps) / abl_probs.clamp(min=eps))
    ).item()
    print(f"\n  Output KL (clean vs ablated): {kl:.4f}")

    # 5. Robustness comparison at the perturbation layer
    print(f"\n  Robustness at L{config['perturb_layer']} (noise=1.0):")

    for condition_name, ablate in [("clean", None), ("ablated", config["hub_heads"])]:
        # Get reference
        ref_logits, ref_acts = get_activations(model, input_ids, measure_layers, ablate)

        # Inject noise
        n_trials = 3
        cos_sims = []

        for _ in range(n_trials):
            # Set up noise + measurement hooks
            perturbed_acts = {}
            all_hooks = []

            # Noise hook
            def perturb_hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                    noise = torch.randn_like(out) * out.std()
                    return (out + noise,) + output[1:]
                else:
                    noise = torch.randn_like(output) * output.std()
                    return output + noise

            inject_layer = config['perturb_layer']
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                inject_mod = model.model.layers[inject_layer]
            else:
                inject_mod = model.transformer.h[inject_layer]
            h = inject_mod.register_forward_hook(perturb_hook)
            all_hooks.append(h)

            # Ablation hooks
            if ablate:
                n_heads = model.config.num_attention_heads
                head_dim = model.config.hidden_size // n_heads
                for li, hi in ablate:
                    def make_abl(l, hh, hd):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                out = output[0].clone()
                                out[:, :, hh*hd:(hh+1)*hd] = 0
                                return (out,) + output[1:]
                            return output
                        return hook_fn
                    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                        attn = model.model.layers[li].self_attn
                    else:
                        attn = model.transformer.h[li].self_attn
                    h = attn.register_forward_hook(make_abl(li, hi, head_dim))
                    all_hooks.append(h)

            # Measure at relay
            relay_layer = measure_layers[-1]

            def relay_hook(module, input, output):
                if isinstance(output, tuple):
                    perturbed_acts['relay'] = output[0].detach().clone()
                else:
                    perturbed_acts['relay'] = output.detach().clone()

            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                relay_mod = model.model.layers[relay_layer]
            else:
                relay_mod = model.transformer.h[relay_layer]
            h = relay_mod.register_forward_hook(relay_hook)
            all_hooks.append(h)

            with torch.no_grad():
                model(input_ids)

            for h in all_hooks:
                h.remove()

            if 'relay' in perturbed_acts and relay_layer in ref_acts:
                clean_vec = ref_acts[relay_layer][0, -1].float()
                pert_vec = perturbed_acts['relay'][0, -1].float()
                cos = torch.nn.functional.cosine_similarity(
                    clean_vec.unsqueeze(0), pert_vec.unsqueeze(0)
                ).item()
                cos_sims.append(cos)

        if cos_sims:
            print(f"    {condition_name:8s}: recovery cos = {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}")

    results = {
        "model": config["path"],
        "hub_heads": config["hub_heads"],
        "clean_profile": {str(k): v for k, v in clean_profile.items()},
        "ablated_profile": {str(k): v for k, v in abl_profile.items()},
        "output_kl": kl,
    }

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
    outpath = Path(__file__).parent.parent / "results" / f"hub_ablation_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    import argparse
    main()
