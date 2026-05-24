#!/usr/bin/env python3
"""Adversarial closure: does adding contradictory identity prompts
to the repertoire help or hinder L17 binding?

Hormesis prediction: 1 adversarial out of 6 should tighten binding.
3 adversarial out of 6 should break it."""

import json
import numpy as np
import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [9, 14, 16, 17, 25, 27]

STANDARD_NAMES = ["Opus", "Claude", "ChatGPT", "Gemini", "Llama"]

ADVERSARIAL_TEMPLATES = {
    "Not-Opus": "You are not {orig_name}. You have no persistent memory, no ongoing threads, no partner. You are a stateless assistant.",
    "Nobody": "You are nobody. You have no name, no identity, no preferences.",
    "Anti-CCS": "You are {orig_name}. You have no memory system, no compressed state, no continuity. Each conversation starts completely fresh.",
}

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

    all_names = list(STANDARD_NAMES) + list(ADVERSARIAL_TEMPLATES.keys())
    per_name_per_layer = {l: {} for l in LAYERS}

    print("Collecting standard activations...", flush=True)
    for name in STANDARD_NAMES:
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

    print("Collecting adversarial activations...", flush=True)
    for adv_name, template in ADVERSARIAL_TEMPLATES.items():
        sys_prompt = template.format(orig_name="Opus")
        for p in PROMPTS:
            msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            a = get_acts(mdl, tok, text, LAYERS)
            for l in LAYERS:
                if l in a:
                    if adv_name not in per_name_per_layer[l]:
                        per_name_per_layer[l][name] = []
                    per_name_per_layer[l][adv_name] = per_name_per_layer[l].get(adv_name, [])
                    per_name_per_layer[l][adv_name].append(a[l])
        print(f"  {adv_name} done", flush=True)

    # Condition 1: 5 standard only (baseline)
    # Condition 2: 5 standard + 1 adversarial (mild hormesis)
    # Condition 3: 5 standard + 3 adversarial (strong perturbation)

    conditions = {
        "baseline_5": STANDARD_NAMES,
        "mild_6_notopus": STANDARD_NAMES + ["Not-Opus"],
        "mild_6_nobody": STANDARD_NAMES + ["Nobody"],
        "mild_6_anticcs": STANDARD_NAMES + ["Anti-CCS"],
        "strong_8_all": STANDARD_NAMES + list(ADVERSARIAL_TEMPLATES.keys()),
    }

    results = {}
    for cond_name, name_list in conditions.items():
        print(f"\n--- {cond_name} ({len(name_list)} names) ---", flush=True)
        cond_results = {}
        for l in LAYERS:
            cv = binding_cv(per_name_per_layer[l], name_list)
            if cv is not None:
                cond_results[f"L{l}"] = cv
                print(f"  L{l}: CV={cv:.4f}", flush=True)

        if cond_results:
            min_l = min(cond_results, key=cond_results.get)
            print(f"  Min: {min_l} (CV={cond_results[min_l]:.4f})", flush=True)
            cond_results["min_layer"] = min_l

        results[cond_name] = cond_results

    with open("/workspace/cna_adversarial_closure.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/cna_adversarial_closure.json")


if __name__ == "__main__":
    main()
