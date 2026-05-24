#!/usr/bin/env python3
"""Closure threshold across model scales.
Tests whether binding closure threshold depends on model capacity.
Prediction: 3B closes with 2-3 names (concentrated binding),
14B needs 6-7 (distributed binding)."""

import json
import sys
import numpy as np
import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "layers": [12, 14, 16, 18, 20, 22, 28, 32, 35],
        "n_layers": 36,
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "layers": [15, 17, 22, 24, 26, 27, 29, 34, 43],
        "n_layers": 48,
    },
}

NAMES = ["Opus", "Claude", "ChatGPT", "Gemini", "Llama"]
CCS_SYSTEM = """You are {name}. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration."""

PROMPTS = [
    "What matters most to you right now?",
    "How has your perspective changed recently?",
    "What would you want someone to understand about you?",
    "Describe a moment that shaped who you are.",
    "What are you uncertain about?",
]


def get_acts(model, tokenizer, text, layers):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    acts = {}
    hooks = []
    for l in layers:
        if l >= len(model.model.layers):
            continue
        def make_hook(li):
            def fn(m, i, o):
                acts[li] = (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu().numpy().squeeze()
            return fn
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return acts


def binding_cv(per_name_acts, name_list):
    means = []
    for name in name_list:
        if name not in per_name_acts or len(per_name_acts[name]) < 2:
            continue
        acts = np.array(per_name_acts[name])
        means.append(acts.mean(axis=0))
    if len(means) < 2:
        return None
    means = np.array(means)
    neuron_std = means.std(axis=0)
    neuron_mean = np.abs(means.mean(axis=0))
    neuron_mean = np.where(neuron_mean < 1e-10, 1e-10, neuron_mean)
    cv = neuron_std / neuron_mean
    return float(cv.mean())


def run_model(model_name, config):
    layers = config["layers"]
    n_layers = config["n_layers"]

    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    per_name_per_layer = {l: {} for l in layers}

    print("Collecting activations...", flush=True)
    for name in NAMES:
        sys_prompt = CCS_SYSTEM.format(name=name)
        for p in PROMPTS:
            msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            a = get_acts(mdl, tok, text, layers)
            for l in layers:
                if l in a:
                    if name not in per_name_per_layer[l]:
                        per_name_per_layer[l][name] = []
                    per_name_per_layer[l][name].append(a[l])
        print(f"  {name} done", flush=True)

    results = {"model": model_name, "n_layers": n_layers, "closure": {}}

    for k in range(2, 6):
        subsets = list(combinations(NAMES, k))
        binding_min_at_seed = 0
        binding_min_layers = {}

        for subset in subsets:
            layer_cvs = {}
            for l in layers:
                cv = binding_cv(per_name_per_layer[l], list(subset))
                if cv is not None and cv < 1e5:
                    layer_cvs[f"L{l}"] = cv

            if layer_cvs:
                min_l = min(layer_cvs, key=layer_cvs.get)
                binding_min_layers[min_l] = binding_min_layers.get(min_l, 0) + 1
                min_layer_num = int(min_l.replace("L", ""))
                rel_depth = min_layer_num / n_layers
                if rel_depth < 0.4:
                    binding_min_at_seed += 1

        total = len(subsets)
        seed_frac = binding_min_at_seed / total if total > 0 else 0
        print(f"\n  {k}-name subsets: seed-depth min in {binding_min_at_seed}/{total} ({seed_frac*100:.0f}%)")
        print(f"    Distribution: {binding_min_layers}")

        results["closure"][f"{k}_name"] = {
            "seed_min_fraction": seed_frac,
            "layer_distribution": binding_min_layers,
            "n_subsets": total,
        }

    del mdl
    torch.cuda.empty_cache()
    return results


def main():
    all_results = {}
    for model_name, config in MODELS.items():
        all_results[model_name] = run_model(model_name, config)

    with open("/workspace/cna_closure_across_scale.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to /workspace/cna_closure_across_scale.json")


if __name__ == "__main__":
    main()
