#!/usr/bin/env python3
"""Quick Mistral binding CV — simplified to avoid hanging."""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import signal, sys

signal.signal(signal.SIGALRM, lambda s,f: (print("TIMEOUT on computation", flush=True), sys.exit(1)))

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LAYERS = [10, 14, 16, 17, 25, 30]

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
    signal.alarm(30)
    try:
        cov = np.cov(acts.T)
        eigs = np.linalg.eigvalsh(cov)
        eigs = eigs[eigs > 1e-10]
        total = eigs.sum()
        result = float((total**2) / (eigs**2).sum())
    finally:
        signal.alarm(0)
    return result


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

    # Test chat template
    test_msgs = [{"role": "user", "content": "hello"}]
    try:
        test_text = tok.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
        print(f"Chat template works. Test length: {len(test_text)} chars", flush=True)
    except Exception as e:
        print(f"Chat template failed: {e}", flush=True)
        print("Falling back to raw text format", flush=True)

    # Phase 1: Identity activations (use raw text format for reliability)
    print("\nPhase 1: Per-name identity activations...", flush=True)
    per_name_per_layer = {l: {} for l in LAYERS}
    all_identity_per_layer = {l: [] for l in LAYERS}

    for name in NAMES:
        sys_prompt = CCS_SYSTEM.format(name=name)
        for i, p in enumerate(PROMPTS):
            text = f"[INST] {sys_prompt}\n\n{p} [/INST]"
            print(f"  {name} prompt {i+1}/{len(PROMPTS)} (len={len(text)})...", end="", flush=True)
            a = get_acts(mdl, tok, text, LAYERS)
            for l in LAYERS:
                if l in a:
                    if name not in per_name_per_layer[l]:
                        per_name_per_layer[l][name] = []
                    per_name_per_layer[l][name].append(a[l])
                    all_identity_per_layer[l].append(a[l])
            print(" done", flush=True)

    # Phase 2: Generic
    print("\nPhase 2: Generic activations...", flush=True)
    generic_per_layer = {l: [] for l in LAYERS}
    for i, p in enumerate(GENERIC):
        text = f"[INST] {p} [/INST]"
        print(f"  generic {i+1}/{len(GENERIC)}...", end="", flush=True)
        a = get_acts(mdl, tok, text, LAYERS)
        for l in LAYERS:
            if l in a:
                generic_per_layer[l].append(a[l])
        print(" done", flush=True)

    # Phase 3: Metrics
    print("\nPhase 3: Computing metrics...", flush=True)
    results = {"model": MODEL_NAME}
    layer_results = {}

    for l in LAYERS:
        if l >= len(mdl.model.layers):
            continue
        lr = {}
        gen_pr = pr(generic_per_layer[l])
        id_pr = pr(all_identity_per_layer[l])
        lr["generic_pr"] = gen_pr
        lr["identity_pr"] = id_pr

        # Per-name PR
        lr["per_name_pr"] = {}
        for name in NAMES:
            if name in per_name_per_layer[l]:
                lr["per_name_pr"][name] = pr(per_name_per_layer[l][name])

        # Binding CV
        lr["binding_cv"] = binding_cv(per_name_per_layer[l])

        layer_results[f"L{l}"] = lr
        cv = lr["binding_cv"]["mean_cv"]
        print(f"  L{l}: generic_PR={gen_pr:.2f} identity_PR={id_pr:.2f} binding_CV={cv:.4f}", flush=True)

    results["layer_results"] = layer_results

    with open("/workspace/cna_mistral_binding_quick.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to /workspace/cna_mistral_binding_quick.json")


if __name__ == "__main__":
    main()
