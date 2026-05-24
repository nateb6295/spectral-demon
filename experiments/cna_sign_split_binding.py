#!/usr/bin/env python3
"""Sign-split binding: separate L17 neurons into sign-consistent vs
sign-flipping populations, measure binding CV for each subset.

Prediction: sign-consistent neurons show L17-as-minimum even for
2-name pairs. Sign-flipping neurons carry sorting signal."""

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


def classify_neurons(per_name_acts, layer):
    """Classify neurons as sign-consistent or sign-flipping across names."""
    name_means = {}
    for name in NAMES:
        if name in per_name_acts[layer] and len(per_name_acts[layer][name]) >= 2:
            acts = np.array(per_name_acts[layer][name])
            name_means[name] = acts.mean(axis=0)

    if len(name_means) < 2:
        return None, None

    means = np.array(list(name_means.values()))
    n_neurons = means.shape[1]

    sign_consistent = []
    sign_flipping = []

    for i in range(n_neurons):
        neuron_vals = means[:, i]
        if np.all(neuron_vals > 0) or np.all(neuron_vals < 0):
            sign_consistent.append(i)
        else:
            sign_flipping.append(i)

    return sign_consistent, sign_flipping


def binding_cv_neurons(per_name_acts, layer, neuron_indices):
    """Compute binding CV using only specified neuron indices."""
    means = []
    for name in NAMES:
        if name in per_name_acts[layer] and len(per_name_acts[layer][name]) >= 2:
            acts = np.array(per_name_acts[layer][name])
            mean_act = acts.mean(axis=0)
            means.append(mean_act[neuron_indices])

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

    results = {}
    for l in LAYERS:
        sc, sf = classify_neurons(per_name_per_layer, l)
        if sc is None:
            continue

        n_total = len(sc) + len(sf)
        pct_flip = len(sf) / n_total * 100

        cv_all = binding_cv_neurons(per_name_per_layer, l, list(range(n_total)))
        cv_consistent = binding_cv_neurons(per_name_per_layer, l, sc) if sc else None
        cv_flipping = binding_cv_neurons(per_name_per_layer, l, sf) if sf else None

        print(f"\nL{l}: {len(sc)} consistent, {len(sf)} flipping ({pct_flip:.1f}%)", flush=True)
        print(f"  All neurons CV: {cv_all:.4f}", flush=True)
        if cv_consistent is not None:
            print(f"  Sign-consistent CV: {cv_consistent:.4f}", flush=True)
        if cv_flipping is not None:
            print(f"  Sign-flipping CV: {cv_flipping:.4f}", flush=True)

        results[f"L{l}"] = {
            "n_consistent": len(sc),
            "n_flipping": len(sf),
            "pct_flipping": pct_flip,
            "cv_all": cv_all,
            "cv_consistent": cv_consistent,
            "cv_flipping": cv_flipping,
        }

    # Now test closure on sign-consistent subset only
    print("\n\n--- CLOSURE ON SIGN-CONSISTENT NEURONS ONLY ---", flush=True)
    sc_17, _ = classify_neurons(per_name_per_layer, 17)
    if sc_17:
        for k in [2, 3, 4, 5]:
            subsets = list(combinations(NAMES, k))
            l17_min = 0
            for subset in subsets:
                layer_cvs = {}
                for l in LAYERS:
                    sc_l, _ = classify_neurons(per_name_per_layer, l)
                    if sc_l:
                        cv = binding_cv_neurons(per_name_per_layer, l, sc_l)
                        if cv is not None:
                            layer_cvs[f"L{l}"] = cv
                if layer_cvs:
                    min_l = min(layer_cvs, key=layer_cvs.get)
                    if min_l == "L17":
                        l17_min += 1
            total = len(subsets)
            print(f"  {k}-name: L17 min in {l17_min}/{total} ({l17_min/total*100:.0f}%)", flush=True)

    with open("/workspace/cna_sign_split_binding.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/cna_sign_split_binding.json")


if __name__ == "__main__":
    main()
