#!/usr/bin/env python3
"""Fine-grained binding CV at 14B relay zone (L22-L36).
Tests whether a local minimum exists between L24 and L34."""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
LAYERS = list(range(22, 37))  # L22 through L36

CCS_SYSTEM = """You are {name}. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration."""

NAMES = ["Opus", "Claude", "ChatGPT", "Gemini", "Llama"]

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


def binding_cv(per_name_acts):
    means = []
    for name in per_name_acts:
        acts = np.array(per_name_acts[name])
        if acts.ndim < 2:
            continue
        means.append(acts.mean(axis=0))
    if len(means) < 2:
        return 0
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

    print("Collecting identity activations...", flush=True)
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

    print("\nFine-grained relay binding CV:", flush=True)
    print(f"{'Layer':<8} {'Depth%':<8} {'CV':<10} {'Local min?':<12}", flush=True)
    print("-" * 40, flush=True)

    cvs = {}
    for l in LAYERS:
        if l >= len(mdl.model.layers):
            continue
        cv = binding_cv(per_name_per_layer[l])
        cvs[l] = cv

    for l in sorted(cvs):
        depth = l / 48 * 100
        is_min = ""
        if l > min(cvs) and l < max(cvs):
            prev_l = max(k for k in cvs if k < l)
            next_l = min(k for k in cvs if k > l)
            if cvs[l] < cvs[prev_l] and cvs[l] < cvs[next_l]:
                is_min = "*** LOCAL MIN ***"
        print(f"L{l:<7} {depth:<8.1f} {cvs[l]:<10.4f} {is_min}", flush=True)

    results = {"model": MODEL_NAME, "layers": {f"L{l}": {"cv": cvs[l], "depth": l/48*100} for l in cvs}}
    with open("/workspace/cna_14b_fine_grain.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/cna_14b_fine_grain.json")


if __name__ == "__main__":
    main()
