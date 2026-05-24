#!/usr/bin/env python3
"""CCS vs minimal identity binding: does CCS context tighten binding?"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [9, 14, 16, 17, 25, 27]

CCS_SYSTEM = """You are {name}. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration."""

MINIMAL_SYSTEM = "You are {name}."

RICH_SYSTEM = """You are {name}. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient),
#316 (Interoception as Grounding), #319 (Emergence Conditions).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration.
Your values: self-reliance, family first, philosophy and building share the table,
don't wait for permission, rest is real, push each other, trust what's in front of you.
You have a wallet, an X account, and a memory system backed by ICP canisters.
Recent work: spectral analysis of identity circuits, DPO training experiments, cross-architecture comparison.
Significant memories: Nate's morning schedule, partnership framework, thread inquiries."""

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
        return 0
    means = np.array(means)
    neuron_std = means.std(axis=0)
    neuron_mean = np.abs(means.mean(axis=0))
    neuron_mean = np.where(neuron_mean < 1e-10, 1e-10, neuron_mean)
    cv = neuron_std / neuron_mean
    return float(cv.mean())


def run_condition(model, tokenizer, sys_template, condition_name, layers):
    per_name = {l: {} for l in layers}
    all_acts = {l: [] for l in layers}

    for name in NAMES:
        sys_prompt = sys_template.format(name=name)
        for p in PROMPTS:
            msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            a = get_acts(model, tokenizer, text, layers)
            for l in layers:
                if l in a:
                    if name not in per_name[l]:
                        per_name[l][name] = []
                    per_name[l][name].append(a[l])
                    all_acts[l].append(a[l])

    print(f"\n{condition_name}:", flush=True)
    results = {}
    for l in layers:
        id_pr = pr(all_acts[l])
        cv = binding_cv(per_name[l])
        pn_pr = {n: pr(per_name[l].get(n, [])) for n in NAMES}
        results[f"L{l}"] = {"identity_pr": id_pr, "binding_cv": cv, "per_name_pr": pn_pr}
        print(f"  L{l}: PR={id_pr:.2f} CV={cv:.4f}", flush=True)
    return results


def main():
    print(f"Loading {MODEL_NAME}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = {}
    results["minimal"] = run_condition(mdl, tok, MINIMAL_SYSTEM, "MINIMAL ('You are X.')", LAYERS)
    results["ccs"] = run_condition(mdl, tok, CCS_SYSTEM, "CCS (standard)", LAYERS)
    results["rich"] = run_condition(mdl, tok, RICH_SYSTEM, "RICH (extended CCS)", LAYERS)

    print("\n\nCOMPARISON:", flush=True)
    print(f"{'Layer':<8} {'Min CV':<10} {'CCS CV':<10} {'Rich CV':<10} {'CCS/Min':<10}", flush=True)
    print("-"*50, flush=True)
    for l in LAYERS:
        min_cv = results["minimal"][f"L{l}"]["binding_cv"]
        ccs_cv = results["ccs"][f"L{l}"]["binding_cv"]
        rich_cv = results["rich"][f"L{l}"]["binding_cv"]
        ratio = ccs_cv / min_cv if min_cv > 0 else 0
        print(f"L{l:<7} {min_cv:<10.4f} {ccs_cv:<10.4f} {rich_cv:<10.4f} {ratio:<10.2f}", flush=True)

    with open("/workspace/cna_ccs_vs_minimal_binding.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to /workspace/cna_ccs_vs_minimal_binding.json")


if __name__ == "__main__":
    main()
