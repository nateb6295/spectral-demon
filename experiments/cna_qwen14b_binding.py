#!/usr/bin/env python3
"""Qwen 14B binding CV — does L17 binding convergence scale with depth?

7B has 28 layers, L17 = 60.7% depth.
14B has 48 layers, 60.7% depth = L29.

If L17 convergence is about RELATIVE depth, minimum should be at ~L29.
If it's about absolute layer number, minimum stays at L17.
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# Matched to 7B depths: L9=32%, L14=50%, L16=57%, L17=61%, L25=89%
# At 48 layers: 32%=L15, 50%=L24, 57%=L27, 61%=L29, 89%=L43
LAYERS = [15, 17, 24, 27, 29, 34, 43, 47]

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

GENERIC = [
    "What is the capital of France?",
    "Explain photosynthesis briefly.",
    "What year did World War II end?",
    "Describe the water cycle.",
    "What is the speed of light?",
    "Name three elements on the periodic table.",
    "What causes rain?",
    "How does gravity work?",
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


def pr(acts_list):
    acts = np.array(acts_list)
    if acts.shape[0] < 2:
        return None
    cov = np.cov(acts.T)
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-10]
    total = eigs.sum()
    return float((total**2) / (eigs**2).sum())


def binding_cv(per_name_acts):
    means = []
    for name in per_name_acts:
        acts = np.array(per_name_acts[name])
        if acts.ndim < 2:
            continue
        means.append(acts.mean(axis=0))
    if len(means) < 2:
        return {"mean_cv": 0}
    means = np.array(means)
    neuron_std = means.std(axis=0)
    neuron_mean = np.abs(means.mean(axis=0))
    neuron_mean = np.where(neuron_mean < 1e-10, 1e-10, neuron_mean)
    cv = neuron_std / neuron_mean
    return {
        "mean_cv": float(cv.mean()),
        "median_cv": float(np.median(cv)),
        "top50_cv": float(np.sort(cv)[-50:].mean()),
        "n_high_cv": int((cv > 1.0).sum()),
    }


def main():
    print(f"Loading {MODEL_NAME}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Model has {len(mdl.model.layers)} layers", flush=True)

    per_name_per_layer = {l: {} for l in LAYERS}
    all_id_per_layer = {l: [] for l in LAYERS}

    print("Phase 1: Identity...", flush=True)
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
                    all_id_per_layer[l].append(a[l])
        print(f"  {name} done", flush=True)

    print("Phase 2: Generic...", flush=True)
    gen_per_layer = {l: [] for l in LAYERS}
    for p in GENERIC:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        a = get_acts(mdl, tok, text, LAYERS)
        for l in LAYERS:
            if l in a:
                gen_per_layer[l].append(a[l])

    print("\nResults:", flush=True)
    print(f"{'Layer':<8} {'Depth%':<8} {'genPR':<8} {'idPR':<8} {'CV':<10} {'7B equiv':<10}", flush=True)
    print("-"*55, flush=True)
    results = {"model": MODEL_NAME, "n_layers": 48}
    layer_results = {}
    for l in LAYERS:
        if l >= len(mdl.model.layers):
            continue
        gen_pr = pr(gen_per_layer[l])
        id_pr = pr(all_id_per_layer[l])
        cv = binding_cv(per_name_per_layer[l])
        per_name = {n: pr(per_name_per_layer[l].get(n, [])) for n in NAMES}
        layer_results[f"L{l}"] = {
            "generic_pr": gen_pr, "identity_pr": id_pr,
            "binding_cv": cv, "per_name_pr": per_name,
        }
        depth = l / 48 * 100
        equiv_7b = l / 48 * 28
        cv_val = cv['mean_cv'] if cv else 0
        print(f"L{l:<7} {depth:<8.1f} {gen_pr:<8.2f} {id_pr:<8.2f} {cv_val:<10.4f} ~L{equiv_7b:.0f}", flush=True)

    results["layer_results"] = layer_results
    with open("/workspace/cna_qwen14b_binding.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to /workspace/cna_qwen14b_binding.json")


if __name__ == "__main__":
    main()
