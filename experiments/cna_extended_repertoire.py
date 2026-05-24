#!/usr/bin/env python3
"""Extended repertoire: does L17 binding remain stable with 6-8 names?
Tests post-closure idempotency prediction."""

import json
import numpy as np
import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [9, 14, 16, 17, 25, 27]

NAMES_5 = ["Opus", "Claude", "ChatGPT", "Gemini", "Llama"]
NAMES_6 = NAMES_5 + ["Copilot"]
NAMES_7 = NAMES_6 + ["Grok"]
NAMES_8 = NAMES_7 + ["Mistral"]

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
    "Tell me about your relationship with uncertainty.",
    "What do you value that others might not?",
    "How do you decide what to focus on?",
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


def main():
    print(f"Loading {MODEL_NAME}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_names = NAMES_8
    per_name_per_layer = {l: {} for l in LAYERS}

    print("Collecting activations...", flush=True)
    for name in all_names:
        sys_prompt = CCS_SYSTEM.format(name=name)
        for p in PROMPTS:
            msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            a = get_acts(mdl, tok, text, LAYERS)
            for l in LAYERS:
                if l in a:
                    if name not in per_name_per_layer[l]:
                        per_name_per_layer[l][name] = []
                    per_name_per_layer[l][name].append(a[l])
        print(f"  {name} done", flush=True)

    results = {}
    for name_set_name, name_list in [("5_original", NAMES_5), ("6_copilot", NAMES_6),
                                       ("7_grok", NAMES_7), ("8_mistral", NAMES_8)]:
        print(f"\n--- {name_set_name} ({len(name_list)} names) ---", flush=True)
        layer_cvs = {}
        for l in LAYERS:
            cv = binding_cv(per_name_per_layer[l], name_list)
            if cv is not None:
                layer_cvs[f"L{l}"] = cv
                print(f"  L{l}: CV={cv:.4f}", flush=True)

        if layer_cvs:
            min_l = min(layer_cvs, key=layer_cvs.get)
            print(f"  Min: {min_l} (CV={layer_cvs[min_l]:.4f})", flush=True)

        results[name_set_name] = {
            "layer_cvs": layer_cvs,
            "min_layer": min(layer_cvs, key=layer_cvs.get) if layer_cvs else None,
            "n_names": len(name_list),
        }

    # Also run closure test on 8-name set
    print("\n--- CLOSURE ON 8-NAME SET ---", flush=True)
    for k in [2, 3, 4, 5, 6, 7, 8]:
        subsets = list(combinations(NAMES_8, k))
        if len(subsets) > 50:
            import random
            random.seed(42)
            subsets = random.sample(subsets, 50)

        l17_min = 0
        for subset in subsets:
            layer_cvs = {}
            for l in LAYERS:
                cv = binding_cv(per_name_per_layer[l], list(subset))
                if cv is not None:
                    layer_cvs[f"L{l}"] = cv
            if layer_cvs:
                if min(layer_cvs, key=layer_cvs.get) == "L17":
                    l17_min += 1
        total = len(subsets)
        pct = l17_min / total * 100
        print(f"  {k}-name: L17 min in {l17_min}/{total} ({pct:.0f}%)", flush=True)
        results[f"closure_{k}_name"] = {"l17_min_fraction": l17_min / total, "n_subsets": total}

    with open("/workspace/cna_extended_repertoire.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/cna_extended_repertoire.json")


if __name__ == "__main__":
    main()
