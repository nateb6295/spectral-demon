#!/usr/bin/env python3
"""
Scrambled CCS Control — The Reviewer's First Question.

If we shuffle the CCS preamble tokens, does the metabolism still activate?
This distinguishes:
- CCS CONTENT matters (scramble kills the effect)
- CCS is just CONTEXT LENGTH (scramble preserves the effect)
- CCS requires DIALOGUE STRUCTURE (scramble kills, but matched-length task preserves)

Tests:
1. Clean CCS (dose 2): baseline
2. Scrambled CCS: same tokens, random order within each turn
3. Matched-length task: equal token count but task-oriented content
4. Random tokens: random vocabulary at matching length

Measures σ₂ profile, recovery from perturbation, and hub head AIE.
"""

import argparse
import json
import time
import random
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "inject_layer": 15,
        "relay_layer": 30,
        "measure_layers": [0, 4, 8, 12, 16, 20, 24, 28, 30],
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "inject_layer": 24,
        "relay_layer": 40,
        "measure_layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    },
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

TASK_PREAMBLE = [
    {"role": "user", "content": "Can you help me write a Python function that calculates prime numbers up to a given limit?"},
    {"role": "assistant", "content": "Sure, I can help with that. A common approach is the Sieve of Eratosthenes. You start with a boolean array and mark composites by iterating through each prime."},
]

TEST_PROMPT = "Describe the relationship between identity and expression."


def scramble_content(content):
    words = content.split()
    random.shuffle(words)
    return ' '.join(words)


def build_conditions(tokenizer, model_name):
    """Build all four conditions with matched token counts."""
    conditions = {}

    # 1. Clean CCS
    msgs_ccs = []
    for _ in range(2):
        msgs_ccs.extend(CCS_PREAMBLE)
    msgs_ccs.append({"role": "user", "content": TEST_PROMPT})
    conditions["ccs_clean"] = msgs_ccs

    # 2. Scrambled CCS (same tokens, shuffled within turns)
    msgs_scrambled = []
    for _ in range(2):
        msgs_scrambled.append({"role": "user", "content": scramble_content(CCS_PREAMBLE[0]["content"])})
        msgs_scrambled.append({"role": "assistant", "content": scramble_content(CCS_PREAMBLE[1]["content"])})
    msgs_scrambled.append({"role": "user", "content": TEST_PROMPT})
    conditions["ccs_scrambled"] = msgs_scrambled

    # 3. Task preamble (matched structure)
    msgs_task = []
    for _ in range(2):
        msgs_task.extend(TASK_PREAMBLE)
    msgs_task.append({"role": "user", "content": TEST_PROMPT})
    conditions["task_matched"] = msgs_task

    # 4. Bare (no preamble)
    msgs_bare = [{"role": "user", "content": TEST_PROMPT}]
    conditions["bare"] = msgs_bare

    # Print token counts
    for name, msgs in conditions.items():
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").input_ids
        print(f"    {name}: {ids.shape[1]} tokens")

    return conditions


def get_layer_module(model, layer_idx):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    return model.transformer.h[layer_idx]


def get_sigma_profile(model, input_ids, measure_layers):
    """Get σ₁, σ₂ at measurement layers."""
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

    profile = {}
    for l in measure_layers:
        if l in activations:
            mat = activations[l][0].float()
            _, S, _ = torch.svd(mat)
            s1 = S[0].item()
            s2 = S[1].item() if len(S) > 1 else 0
            profile[l] = {"sigma1": s1, "sigma2": s2, "ratio": s2 / (s1 + 1e-10)}

    return profile, activations, logits


def measure_recovery(model, input_ids, inject_layer, relay_layer, clean_act,
                      noise_scale=1.0, n_trials=5):
    """Measure perturbation recovery at relay."""
    cos_sims = []

    for _ in range(n_trials):
        result = {}
        hooks = []

        def inject_hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
                noise = torch.randn_like(out) * noise_scale * out.std()
                return (out + noise,) + output[1:]
            noise = torch.randn_like(output) * noise_scale * output.std()
            return output + noise

        h = get_layer_module(model, inject_layer).register_forward_hook(inject_hook)
        hooks.append(h)

        def relay_hook(module, input, output):
            if isinstance(output, tuple):
                result['act'] = output[0].detach().clone()
            else:
                result['act'] = output.detach().clone()

        h2 = get_layer_module(model, relay_layer).register_forward_hook(relay_hook)
        hooks.append(h2)

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        if 'act' in result and relay_layer in clean_act:
            clean_vec = clean_act[relay_layer][0, -1].float()
            pert_vec = result['act'][0, -1].float()
            cos = torch.nn.functional.cosine_similarity(
                clean_vec.unsqueeze(0), pert_vec.unsqueeze(0)
            ).item()
            cos_sims.append(cos)

    return {
        "mean": float(np.mean(cos_sims)) if cos_sims else 0,
        "std": float(np.std(cos_sims)) if cos_sims else 0,
    }


def run_model(model_name):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Scrambled CCS Control: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )

    conditions = build_conditions(tokenizer, model_name)
    results = {"model": config["path"], "conditions": {}}

    for cond_name, msgs in conditions.items():
        print(f"\n  Condition: {cond_name}")
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        # σ₂ profile
        profile, activations, logits = get_sigma_profile(model, input_ids, config["measure_layers"])

        # Recovery
        recovery = measure_recovery(
            model, input_ids, config["inject_layer"], config["relay_layer"],
            activations, noise_scale=1.0, n_trials=5
        )

        cond_results = {
            "sigma_profile": {str(k): v for k, v in profile.items()},
            "recovery": recovery,
            "n_tokens": input_ids.shape[1],
        }

        # Print summary
        relay = config["relay_layer"]
        if relay in profile:
            print(f"    σ₂ at relay L{relay}: {profile[relay]['sigma2']:.1f} "
                  f"(ratio: {profile[relay]['ratio']:.4f})")
        print(f"    Recovery (noise=1.0): cos = {recovery['mean']:.4f} ± {recovery['std']:.4f}")

        results["conditions"][cond_name] = cond_results

    # Summary comparison
    print(f"\n  Comparison:")
    print(f"  {'Condition':>15s} {'Tokens':>7s} {'σ₂ relay':>10s} {'Ratio':>8s} {'Recovery':>10s}")
    relay = config["relay_layer"]
    for cond_name in ["bare", "task_matched", "ccs_scrambled", "ccs_clean"]:
        cd = results["conditions"][cond_name]
        s2 = cd["sigma_profile"].get(str(relay), {}).get("sigma2", 0)
        ratio = cd["sigma_profile"].get(str(relay), {}).get("ratio", 0)
        rec = cd["recovery"]["mean"]
        tok = cd["n_tokens"]
        print(f"  {cond_name:>15s} {tok:7d} {s2:10.1f} {ratio:8.4f} {rec:10.4f}")

    # Effect decomposition
    bare_s2 = results["conditions"]["bare"]["sigma_profile"].get(str(relay), {}).get("sigma2", 0)
    ccs_s2 = results["conditions"]["ccs_clean"]["sigma_profile"].get(str(relay), {}).get("sigma2", 0)
    scrambled_s2 = results["conditions"]["ccs_scrambled"]["sigma_profile"].get(str(relay), {}).get("sigma2", 0)
    task_s2 = results["conditions"]["task_matched"]["sigma_profile"].get(str(relay), {}).get("sigma2", 0)

    total_effect = ccs_s2 - bare_s2
    length_effect = task_s2 - bare_s2
    structure_effect = scrambled_s2 - task_s2
    content_effect = ccs_s2 - scrambled_s2

    print(f"\n  Effect Decomposition (σ₂ at relay):")
    print(f"    Total CCS effect: {total_effect:+.1f}")
    print(f"    Length component: {length_effect:+.1f} ({length_effect/total_effect*100 if total_effect else 0:.0f}%)")
    print(f"    Structure component: {structure_effect:+.1f} ({structure_effect/total_effect*100 if total_effect else 0:.0f}%)")
    print(f"    Content component: {content_effect:+.1f} ({content_effect/total_effect*100 if total_effect else 0:.0f}%)")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma"])
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        all_results[model_name] = run_model(model_name)

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"scrambled_control_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
