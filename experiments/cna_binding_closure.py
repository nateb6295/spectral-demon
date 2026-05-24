#!/usr/bin/env python3
"""Binding closure test: is L17 binding stable under name subsets?
If L17 is a true attractor, its minimum-CV property should hold
for any subset of 2+ names, not just all 5."""

import json
import numpy as np
import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [9, 14, 16, 17, 25, 27]
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


def binding_cv_subset(per_name_acts, name_subset):
    means = []
    for name in name_subset:
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

    per_name_per_layer = {l: {} for l in LAYERS}

    print("Collecting activations...", flush=True)
    for name in NAMES:
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

    # Test all 2-name, 3-name, 4-name, and 5-name subsets
    results = {}
    for k in [2, 3, 4, 5]:
        subsets = list(combinations(NAMES, k))
        print(f"\n--- {k}-name subsets ({len(subsets)} combinations) ---", flush=True)

        l17_is_min_count = 0
        l17_cv_values = []
        min_layer_counts = {f"L{l}": 0 for l in LAYERS}

        for subset in subsets:
            layer_cvs = {}
            for l in LAYERS:
                cv = binding_cv_subset(per_name_per_layer[l], subset)
                if cv is not None:
                    layer_cvs[f"L{l}"] = cv

            if layer_cvs:
                min_layer = min(layer_cvs, key=layer_cvs.get)
                min_layer_counts[min_layer] = min_layer_counts.get(min_layer, 0) + 1
                if "L17" in layer_cvs:
                    l17_cv_values.append(layer_cvs["L17"])
                    if min_layer == "L17":
                        l17_is_min_count += 1

        total = len(subsets)
        print(f"  L17 is global minimum in {l17_is_min_count}/{total} subsets ({l17_is_min_count/total*100:.0f}%)", flush=True)
        print(f"  L17 mean CV: {np.mean(l17_cv_values):.4f} ± {np.std(l17_cv_values):.4f}", flush=True)
        print(f"  Min-layer distribution: {dict(sorted(min_layer_counts.items(), key=lambda x: -x[1]))}", flush=True)

        results[f"{k}_name"] = {
            "l17_is_min_fraction": l17_is_min_count / total,
            "l17_cv_mean": float(np.mean(l17_cv_values)),
            "l17_cv_std": float(np.std(l17_cv_values)),
            "min_layer_distribution": min_layer_counts,
            "n_subsets": total,
        }

    # Specific interesting pairs
    print("\n--- Interesting pairs ---", flush=True)
    pairs = [("Opus", "ChatGPT"), ("Opus", "Claude"), ("ChatGPT", "Gemini"), ("Opus", "Llama")]
    for a, b in pairs:
        layer_cvs = {}
        for l in LAYERS:
            cv = binding_cv_subset(per_name_per_layer[l], [a, b])
            if cv is not None:
                layer_cvs[f"L{l}"] = cv
        min_l = min(layer_cvs, key=layer_cvs.get) if layer_cvs else "?"
        print(f"  {a}-{b}: min at {min_l} (CV={layer_cvs.get(min_l,0):.4f}), L17={layer_cvs.get('L17',0):.4f}", flush=True)

    with open("/workspace/cna_binding_closure.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/cna_binding_closure.json")


if __name__ == "__main__":
    main()
