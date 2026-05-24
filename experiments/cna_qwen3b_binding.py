#!/usr/bin/env python3
"""Qwen 3B binding CV — third scale point (3B/7B/14B)."""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# 36 layers. Matched depths: seed=L12(33%), presort=L18(50%), relay=L20(56%), apex=L22(61%), expression=L32(89%), final=L35(97%)
# Also fine-grain the relay zone
LAYERS = [12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 28, 32, 35]

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
    return {"mean_cv": float(cv.mean()), "n_high_cv": int((cv > 1.0).sum())}


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
    gen_per_layer = {l: [] for l in LAYERS}

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
    generic_prompts = [
        "What is the capital of France?", "Explain photosynthesis briefly.",
        "What year did World War II end?", "Describe the water cycle.",
        "What is the speed of light?", "Name three elements.",
        "What causes rain?", "How does gravity work?",
    ]
    for p in generic_prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        a = get_acts(mdl, tok, text, LAYERS)
        for l in LAYERS:
            if l in a:
                gen_per_layer[l].append(a[l])

    print(f"\n{'Layer':<8} {'Depth%':<8} {'genPR':<8} {'idPR':<8} {'CV':<10} {'Local min?':<12}", flush=True)
    print("-" * 58, flush=True)
    results = {"model": MODEL_NAME, "n_layers": 36}
    layer_results = {}
    cvs = {}
    for l in LAYERS:
        if l >= len(mdl.model.layers):
            continue
        gen_pr = pr(gen_per_layer[l])
        id_pr = pr(all_id_per_layer[l])
        cv = binding_cv(per_name_per_layer[l])
        per_name = {n: pr(per_name_per_layer[l].get(n, [])) for n in NAMES}
        cvs[l] = cv['mean_cv']
        layer_results[f"L{l}"] = {
            "generic_pr": gen_pr, "identity_pr": id_pr,
            "binding_cv": cv, "per_name_pr": per_name,
        }

    for l in sorted(cvs):
        depth = l / 36 * 100
        is_min = ""
        keys = sorted(cvs.keys())
        idx = keys.index(l)
        if idx > 0 and idx < len(keys) - 1:
            if cvs[l] < cvs[keys[idx-1]] and cvs[l] < cvs[keys[idx+1]]:
                is_min = "*** LOCAL MIN ***"
        lr = layer_results[f"L{l}"]
        gpr = lr['generic_pr'] or 0
        ipr = lr['identity_pr'] or 0
        print(f"L{l:<7} {depth:<8.1f} {gpr:<8.2f} {ipr:<8.2f} {cvs[l]:<10.4f} {is_min}", flush=True)

    results["layer_results"] = layer_results
    with open("/workspace/cna_qwen3b_binding.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to /workspace/cna_qwen3b_binding.json")


if __name__ == "__main__":
    main()
