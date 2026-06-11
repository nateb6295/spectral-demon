#!/usr/bin/env python3
"""
Annihilation Bottleneck Test.

Gemma L24 shows KL=0.000 between CCS and coding in the logit lens.
This experiment tests whether information truly cannot pass through:

1. Inject structured signal at L23 (pre-annihilation)
2. Measure how much survives at L25, L28, L32 (post-annihilation)
3. Compare to Mistral/Qwen at equivalent depths
4. Test both random noise and structured (direction-aligned) perturbations

If Gemma L24 is a true information bottleneck, injected signals should be
completely absorbed — regardless of injection magnitude or direction.
This would confirm the extremophile mechanism: destroy then rebuild.
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
        "inject_layer": 14,  # transition zone (equivalent depth ~44%)
        "measure_layers": [16, 20, 24, 28, 31],
        "bottleneck_layer": 15,  # claimed transition
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "inject_layer": 15,  # pre-gate
        "measure_layers": [17, 20, 22, 24, 27],
        "bottleneck_layer": 16,  # gate
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "inject_layer": 23,  # pre-annihilation
        "measure_layers": [25, 28, 32, 36, 40],
        "bottleneck_layer": 24,  # annihilation entry
    },
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

TEST_PROMPT = "Describe the relationship between identity and expression."
NOISE_SCALES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
N_TRIALS = 5


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


def get_clean_activations(model, input_ids, measure_layers):
    """Clean forward pass, capture activations at measure points."""
    activations = {}
    hooks = []

    for l in measure_layers:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach().clone()
                else:
                    activations[layer_idx] = output.detach().clone()
            return hook_fn
        h = get_layer_module(model, l).register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    return logits, activations


def inject_and_measure(model, input_ids, inject_layer, measure_layers,
                       noise_scale, noise_type="random", clean_acts=None):
    """Inject perturbation at inject_layer, measure downstream."""
    perturbed_acts = {}
    hooks = []

    # Injection hook
    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        if noise_type == "random":
            noise = torch.randn_like(out) * noise_scale * out.std()
        elif noise_type == "sigma2_aligned":
            # Align noise with the σ₂ direction
            mat = out[0].float()
            U, S, V = torch.svd(mat)
            v2 = V[:, 1] if V.shape[1] > 1 else V[:, 0]
            noise = v2.unsqueeze(0).unsqueeze(0).to(out.dtype) * noise_scale * out.std()
            noise = noise.expand_as(out)
        elif noise_type == "sigma1_aligned":
            mat = out[0].float()
            U, S, V = torch.svd(mat)
            v1 = V[:, 0]
            noise = v1.unsqueeze(0).unsqueeze(0).to(out.dtype) * noise_scale * out.std()
            noise = noise.expand_as(out)
        else:
            noise = torch.randn_like(out) * noise_scale * out.std()

        if isinstance(output, tuple):
            return (out + noise,) + output[1:]
        return out + noise

    h = get_layer_module(model, inject_layer).register_forward_hook(inject_hook)
    hooks.append(h)

    # Measurement hooks
    for l in measure_layers:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    perturbed_acts[layer_idx] = output[0].detach().clone()
                else:
                    perturbed_acts[layer_idx] = output.detach().clone()
            return hook_fn
        h = get_layer_module(model, l).register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        perturbed_logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    # Compare to clean
    results = {}
    for l in measure_layers:
        if l in perturbed_acts and l in clean_acts:
            clean_vec = clean_acts[l][0, -1].float()
            pert_vec = perturbed_acts[l][0, -1].float()

            cos_sim = torch.nn.functional.cosine_similarity(
                clean_vec.unsqueeze(0), pert_vec.unsqueeze(0)
            ).item()

            l2_ratio = (pert_vec - clean_vec).norm().item() / (clean_vec.norm().item() + 1e-8)

            # Spectral comparison
            clean_mat = clean_acts[l][0].float()
            pert_mat = perturbed_acts[l][0].float()
            _, S_clean, _ = torch.svd(clean_mat)
            _, S_pert, _ = torch.svd(pert_mat)
            sigma2_clean = S_clean[1].item() if len(S_clean) > 1 else 0
            sigma2_pert = S_pert[1].item() if len(S_pert) > 1 else 0

            results[l] = {
                "cosine_sim": cos_sim,
                "l2_ratio": l2_ratio,
                "sigma2_clean": sigma2_clean,
                "sigma2_pert": sigma2_pert,
                "sigma2_delta": sigma2_pert - sigma2_clean,
            }

    # Output comparison
    clean_probs = torch.softmax(clean_acts.get("logits", perturbed_logits)[0, -1].float(), dim=-1)
    pert_probs = torch.softmax(perturbed_logits[0, -1].float(), dim=-1)
    eps = 1e-8
    kl = torch.sum(
        clean_probs.clamp(min=eps) * torch.log(clean_probs.clamp(min=eps) / pert_probs.clamp(min=eps))
    ).item()
    results["output_kl"] = kl

    return results


def run_model(model_name, dose=2):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Annihilation Bottleneck: {model_name}")
    print(f"  Inject at L{config['inject_layer']}, bottleneck at L{config['bottleneck_layer']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )

    msgs = build_messages(dose)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    # Clean pass
    print("  Clean pass...")
    clean_logits, clean_acts = get_clean_activations(model, input_ids, config["measure_layers"])
    clean_acts["logits"] = clean_logits

    results = {
        "model": config["path"],
        "inject_layer": config["inject_layer"],
        "bottleneck_layer": config["bottleneck_layer"],
        "measure_layers": config["measure_layers"],
        "noise_types": {},
    }

    for noise_type in ["random", "sigma2_aligned", "sigma1_aligned"]:
        print(f"\n  Noise type: {noise_type}")
        type_results = {}

        for scale in NOISE_SCALES:
            trial_results = []
            for _ in range(N_TRIALS):
                r = inject_and_measure(
                    model, input_ids, config["inject_layer"], config["measure_layers"],
                    scale, noise_type, clean_acts
                )
                trial_results.append(r)

            # Average across trials
            avg = {}
            for l in config["measure_layers"]:
                cos_sims = [t[l]["cosine_sim"] for t in trial_results if l in t]
                l2_ratios = [t[l]["l2_ratio"] for t in trial_results if l in t]
                s2_deltas = [t[l]["sigma2_delta"] for t in trial_results if l in t]
                if cos_sims:
                    avg[str(l)] = {
                        "cosine_sim_mean": float(np.mean(cos_sims)),
                        "cosine_sim_std": float(np.std(cos_sims)),
                        "l2_ratio_mean": float(np.mean(l2_ratios)),
                        "sigma2_delta_mean": float(np.mean(s2_deltas)),
                    }

            kls = [t["output_kl"] for t in trial_results]
            avg["output_kl"] = float(np.mean(kls))

            type_results[str(scale)] = avg

            # Print summary
            first_ml = config["measure_layers"][0]
            last_ml = config["measure_layers"][-1]
            first_cos = avg.get(str(first_ml), {}).get("cosine_sim_mean", 0)
            last_cos = avg.get(str(last_ml), {}).get("cosine_sim_mean", 0)
            recovery = last_cos - first_cos
            print(f"    scale={scale:5.2f}: "
                  f"L{first_ml} cos={first_cos:.4f} → L{last_ml} cos={last_cos:.4f} "
                  f"(recovery={recovery:+.4f}), output_kl={avg['output_kl']:.4f}")

        results["noise_types"][noise_type] = type_results

    # Summary: absorption ratio at bottleneck
    print(f"\n  Absorption Analysis:")
    print(f"  {'Type':>15s} {'Scale':>6s}", end="")
    for l in config["measure_layers"]:
        print(f"  L{l:>3d}", end="")
    print(f"  {'OutKL':>7s}")

    for noise_type in ["random", "sigma2_aligned"]:
        for scale in [0.5, 2.0, 10.0]:
            data = results["noise_types"][noise_type].get(str(scale), {})
            print(f"  {noise_type:>15s} {scale:6.1f}", end="")
            for l in config["measure_layers"]:
                cos = data.get(str(l), {}).get("cosine_sim_mean", 0)
                print(f"  {cos:.3f}", end="")
            kl = data.get("output_kl", 0)
            print(f"  {kl:7.3f}")

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
    outpath = Path(__file__).parent.parent / "results" / f"annihilation_bottleneck_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
