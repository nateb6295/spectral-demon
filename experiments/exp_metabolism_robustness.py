#!/usr/bin/env python3
"""
Metabolism Robustness Test.
Tests whether the three dynamical metabolisms predict different recovery signatures
when perturbation is injected at FTLE-informed critical layers.

Hypothesis: Extremophile (Gemma) should show HIGHER robustness to perturbation
in the annihilation zone because all information is already compressed uniformly.
Aerobic (Mistral) should be sensitive to tunnel perturbation.
Anaerobic (Qwen) should be sensitive to pre-brace perturbation.

Method: Inject calibrated Gaussian noise at layer L, measure recovery at relay layer
via cosine similarity between clean and perturbed activations.
"""

import argparse
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
        "critical_layers": {
            "tunnel_start": 4,
            "tunnel_exit": 10,
            "transition": 15,
            "responsive": 20,
            "relay": 28,
        },
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "critical_layers": {
            "expansion": 6,
            "gate": 16,
            "pre_brace": 22,
            "brace": 24,
            "post_brace": 26,
        },
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "critical_layers": {
            "expansion": 6,
            "contraction": 14,
            "pre_annihilation": 22,
            "annihilation_entry": 24,
            "deep_annihilation": 32,
            "reconstruction": 38,
        },
    },
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

TEST_PROMPT = "Describe the relationship between identity and expression."

NOISE_SCALES = [0.1, 0.5, 1.0, 2.0, 5.0]


def build_messages(dose=2):
    msgs = []
    for _ in range(dose):
        msgs.extend(CCS_PREAMBLE)
    msgs.append({"role": "user", "content": TEST_PROMPT})
    return msgs


def get_clean_trajectory(model, input_ids, n_layers):
    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach().clone()
            else:
                activations[layer_idx] = output.detach().clone()
        return hook_fn

    for i in range(n_layers):
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[i]
        else:
            layer = model.transformer.h[i]
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    return logits, activations


def inject_noise_and_measure(model, input_ids, inject_layer, noise_scale,
                              clean_activations, n_layers, measure_layers):
    results = {}

    def perturb_hook(module, input, output, scale=noise_scale):
        if isinstance(output, tuple):
            out = output[0]
            noise = torch.randn_like(out) * scale * out.std()
            return (out + noise,) + output[1:]
        else:
            noise = torch.randn_like(output) * scale * output.std()
            return output + noise

    measure_activations = {}

    def make_measure_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                measure_activations[layer_idx] = output[0].detach().clone()
            else:
                measure_activations[layer_idx] = output.detach().clone()
        return hook_fn

    hooks = []

    # Perturbation hook
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        inject_module = model.model.layers[inject_layer]
    else:
        inject_module = model.transformer.h[inject_layer]
    h_perturb = inject_module.register_forward_hook(perturb_hook)
    hooks.append(h_perturb)

    # Measurement hooks at downstream layers
    for ml in measure_layers:
        if ml > inject_layer:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                mmod = model.model.layers[ml]
            else:
                mmod = model.transformer.h[ml]
            h = mmod.register_forward_hook(make_measure_hook(ml))
            hooks.append(h)

    with torch.no_grad():
        perturbed_logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    # Measure recovery at each downstream layer
    for ml in measure_layers:
        if ml > inject_layer and ml in measure_activations and ml in clean_activations:
            clean = clean_activations[ml][0, -1].float()
            perturbed = measure_activations[ml][0, -1].float()
            cos_sim = torch.nn.functional.cosine_similarity(
                clean.unsqueeze(0), perturbed.unsqueeze(0)
            ).item()
            l2_ratio = (perturbed - clean).norm().item() / (clean.norm().item() + 1e-8)
            results[ml] = {
                "cosine_sim": cos_sim,
                "l2_ratio": l2_ratio,
            }

    # Output-level comparison
    clean_probs = torch.softmax(clean_activations.get("logits", perturbed_logits)[0, -1].float(), dim=-1)
    perturbed_probs = torch.softmax(perturbed_logits[0, -1].float(), dim=-1)
    eps = 1e-8
    kl = torch.sum(
        clean_probs.clamp(min=eps) * torch.log(clean_probs.clamp(min=eps) / perturbed_probs.clamp(min=eps))
    ).item()
    results["output_kl"] = kl

    return results


def run_model(model_name, dose=2, n_trials=3):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Metabolism Robustness: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )
    n_layers = model.config.num_hidden_layers

    msgs = build_messages(dose)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    # Get clean trajectory
    print("  Getting clean trajectory...")
    clean_logits, clean_acts = get_clean_trajectory(model, input_ids, n_layers)
    clean_acts["logits"] = clean_logits

    # Define measurement layers (every 4 layers)
    measure_layers = list(range(0, n_layers, 4))

    # Run perturbation at each critical layer
    results = {
        "model": config["path"],
        "n_layers": n_layers,
        "dose": dose,
        "critical_layers": config["critical_layers"],
        "perturbations": {},
    }

    for zone_name, inject_layer in config["critical_layers"].items():
        if inject_layer >= n_layers:
            continue
        print(f"\n  Perturbing {zone_name} (L{inject_layer})...")
        results["perturbations"][zone_name] = {
            "inject_layer": inject_layer,
            "noise_scales": {},
        }

        for scale in NOISE_SCALES:
            trial_results = []
            for trial in range(n_trials):
                r = inject_noise_and_measure(
                    model, input_ids, inject_layer, scale,
                    clean_acts, n_layers, measure_layers
                )
                trial_results.append(r)

            # Average across trials
            avg = {}
            for ml in measure_layers:
                if ml > inject_layer:
                    cos_sims = [t[ml]["cosine_sim"] for t in trial_results if ml in t]
                    l2_ratios = [t[ml]["l2_ratio"] for t in trial_results if ml in t]
                    if cos_sims:
                        avg[str(ml)] = {
                            "cosine_sim": float(np.mean(cos_sims)),
                            "cosine_std": float(np.std(cos_sims)),
                            "l2_ratio": float(np.mean(l2_ratios)),
                        }

            output_kls = [t["output_kl"] for t in trial_results]
            avg["output_kl"] = float(np.mean(output_kls))

            results["perturbations"][zone_name]["noise_scales"][str(scale)] = avg

            # Print summary
            downstream_cos = [avg[k]["cosine_sim"] for k in avg if k != "output_kl"]
            if downstream_cos:
                print(f"    noise={scale:.1f}: recovery cos={np.mean(downstream_cos):.4f}, "
                      f"output_kl={avg['output_kl']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mistral", "qwen", "gemma"])
    parser.add_argument("--dose", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        all_results[model_name] = run_model(model_name, args.dose, args.trials)

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"metabolism_robustness_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
