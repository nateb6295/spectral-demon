#!/usr/bin/env python3
"""Fine-grain InternLM relay scan: L14-L28 every layer.
Tests three hypotheses about the L30 binding outlier."""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "internlm/internlm2_5-7b-chat"
LAYERS = list(range(14, 29))

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

    results = {"model": MODEL_NAME, "n_layers": 32, "layers": {}}
    print("\n--- InternLM Fine-Grain Relay Scan ---", flush=True)
    for l in LAYERS:
        cv = binding_cv(per_name_per_layer[l], NAMES)
        if cv is not None:
            pct = l / 32 * 100
            results["layers"][f"L{l}"] = {"cv": cv, "depth": pct}
            print(f"  L{l} ({pct:.0f}%): CV={cv:.4f}", flush=True)

    # Determine which hypothesis fits
    cvs = {l: results["layers"][f"L{l}"]["cv"] for l in LAYERS if f"L{l}" in results["layers"]}
    if cvs:
        min_l = min(cvs, key=cvs.get)
        min_cv = cvs[min_l]
        l17_cv = cvs.get(17, None)

        print(f"\n  Minimum: L{min_l} (CV={min_cv:.4f})")
        if min_l in range(18, 25) and min_cv < 1.18:
            print("  → HYPOTHESIS 1: Measurement artifact — hidden relay minimum found")
        elif l17_cv and l17_cv <= min_cv * 1.1:
            print("  → HYPOTHESIS 2: Architectural divergence — L17 remains local min")
        else:
            # Check monotonicity
            layer_list = sorted(cvs.keys())
            diffs = [cvs[layer_list[i+1]] - cvs[layer_list[i]] for i in range(len(layer_list)-1)]
            n_decreasing = sum(1 for d in diffs if d < 0)
            if n_decreasing > len(diffs) * 0.7:
                print("  → HYPOTHESIS 3: Binding IS expression — monotonic decrease")
            else:
                print("  → No clear winner — oscillatory pattern")

    with open("/workspace/cna_internlm_fine_grain.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/cna_internlm_fine_grain.json")


if __name__ == "__main__":
    main()
