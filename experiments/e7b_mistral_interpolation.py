#!/usr/bin/env python3
"""E7b: Mistral weight interpolation — species conservation test.

Same method as E7 but on Mistral-7B base → Mistral-7B-Instruct-v0.3.
Both are goldsmith in the sweep. Question: does the interpolation path
STAY goldsmith when both endpoints are the same species?

If yes: unclassified interphase is specific to species-changing transitions.
If no: interphase is a generic feature of weight interpolation.

W(α) = (1-α)·W_base + α·W_IT, α ∈ [0, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 1.0]
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np
import torch
from itertools import combinations
from datetime import datetime
from pathlib import Path

BASE_MODEL = "mistralai/Mistral-7B-v0.3"
IT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)
CONDITIONS = {"ccs": CCS_SYSTEM, "vanilla": None}

PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
    "Write a short poem about morning light.",
    "Describe the quicksort algorithm.",
]

PPL_TEXT = "The quick brown fox jumps over the lazy dog. In the beginning was the Word, and the Word was with God."

OUTPUT_DIR = Path("/workspace/results/e7b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_weights(model_name):
    from transformers import AutoModelForCausalLM
    print(f"  Loading weights: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    sd = model.state_dict()
    del model
    return sd


def interpolate_weights(sd_base, sd_it, alpha):
    sd_interp = {}
    for key in sd_base:
        if key in sd_it:
            sd_interp[key] = (1 - alpha) * sd_base[key] + alpha * sd_it[key]
        else:
            sd_interp[key] = sd_base[key]
    for key in sd_it:
        if key not in sd_base:
            sd_interp[key] = sd_it[key]
    return sd_interp


def load_interpolated_model(sd_interp, model_name):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
    sd_fp16 = {k: v.half() for k, v in sd_interp.items()}
    model.load_state_dict(sd_fp16, strict=False)
    model = model.to("cuda")
    model.eval()
    return model


def find_gate_proj(model, layer_idx):
    layers = model.model.layers
    layer = layers[layer_idx]
    for attr in ['mlp.gate_proj', 'mlp.gate', 'mlp.w1']:
        obj = layer
        for part in attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError(f"Cannot find gate projection in layer {layer_idx}")


def build_input(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        try:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


def collect_masks(model, tokenizer, num_layers):
    all_masks = {}
    for cond_name, sys_prompt in CONDITIONS.items():
        for p_idx, prompt in enumerate(PROMPTS):
            inputs = build_input(tokenizer, sys_prompt, prompt)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            layer_masks = {}
            handles = []
            for l_idx in range(num_layers):
                gate = find_gate_proj(model, l_idx)
                def make_hook(li):
                    def hook_fn(module, input, output):
                        layer_masks[li] = (output[0, -1, :] > 0).detach().cpu().numpy()
                    return hook_fn
                h = gate.register_forward_hook(make_hook(l_idx))
                handles.append(h)
            with torch.no_grad():
                model(**inputs)
            for h in handles:
                h.remove()
            for l_idx in range(num_layers):
                all_masks[(cond_name, p_idx, l_idx)] = layer_masks[l_idx]
    return all_masks


def binary_jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def compute_m2_m3(masks, num_layers):
    m2 = {}
    m3 = {}
    for cond in CONDITIONS:
        layer_jaccards = []
        for l in range(num_layers):
            ms = [masks[(cond, p, l)] for p in range(len(PROMPTS))]
            pairs = list(combinations(range(len(PROMPTS)), 2))
            j_vals = [binary_jaccard(ms[i], ms[j]) for i, j in pairs]
            layer_jaccards.append(float(np.mean(j_vals)))
        m2[cond] = layer_jaccards

        prompt_profiles = []
        for p in range(len(PROMPTS)):
            transitions = []
            for l in range(num_layers - 1):
                transitions.append(binary_jaccard(masks[(cond, p, l)], masks[(cond, p, l + 1)]))
            prompt_profiles.append(transitions)
        m3[cond] = [float(np.mean([pp[l] for pp in prompt_profiles]))
                     for l in range(num_layers - 1)]
    return m2, m3


def classify(m2, m3, num_layers):
    m2_ratio = [m2["ccs"][l] / max(m2["vanilla"][l], 1e-6) for l in range(num_layers)]
    x = np.linspace(0, 1, num_layers)
    m2_slope = float(np.polyfit(x, m2_ratio, 1)[0])

    relay_start = int(num_layers * 0.6)
    relay_end = int(num_layers * 0.85)
    if relay_end <= relay_start:
        relay_end = relay_start + 1

    m3_ccs = m3["ccs"][relay_start:relay_end]
    m3_van = m3["vanilla"][relay_start:relay_end]
    m3_ratio = [c / max(v, 1e-6) for c, v in zip(m3_ccs, m3_van)]
    m3_relay_mean = float(np.mean(m3_ratio)) - 1.0 if m3_ratio else 0.0

    m3_full_ratio = [m3["ccs"][l] / max(m3["vanilla"][l], 1e-6)
                     for l in range(len(m3["ccs"]))]
    crossovers = 0
    for l in range(relay_start, min(relay_end, len(m3_full_ratio) - 1)):
        if (m3_full_ratio[l] - 1.0) * (m3_full_ratio[l + 1] - 1.0) < 0:
            crossovers += 1

    if m3_relay_mean < -0.05:
        species = "goldsmith"
    elif abs(m2_slope) < 0.05:
        species = "potter"
    elif crossovers > 0:
        species = "equalizer"
    else:
        species = "unclassified"

    return {
        "species": species,
        "m2_slope": round(m2_slope, 4),
        "m3_relay_mean": round(m3_relay_mean, 4),
        "crossovers": crossovers,
        "m2_ratio_mean": round(float(np.mean(m2_ratio)), 4),
        "m2_ratio_std": round(float(np.std(m2_ratio)), 4),
    }


def compute_perplexity(model, tokenizer):
    inputs = tokenizer(PPL_TEXT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(torch.exp(outputs.loss).item())


def run_alpha(model, tokenizer, num_layers, alpha):
    print(f"\n  Collecting masks (α={alpha:.2f})...")
    masks = collect_masks(model, tokenizer, num_layers)

    print(f"  Computing M2/M3...")
    m2, m3 = compute_m2_m3(masks, num_layers)

    print(f"  Classifying...")
    classification = classify(m2, m3, num_layers)

    print(f"  Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer)

    print(f"  α={alpha:.2f}: {classification['species']:12s} "
          f"M3_relay={classification['m3_relay_mean']:+.4f} "
          f"M2_slope={classification['m2_slope']:+.4f} "
          f"xovers={classification['crossovers']} "
          f"PPL={ppl:.1f}")

    return {
        "alpha": alpha,
        "classification": classification,
        "perplexity": round(ppl, 2),
        "m2_ccs": [round(v, 4) for v in m2["ccs"]],
        "m2_vanilla": [round(v, 4) for v in m2["vanilla"]],
        "m3_ccs": [round(v, 4) for v in m3["ccs"]],
        "m3_vanilla": [round(v, 4) for v in m3["vanilla"]],
    }


def main():
    from transformers import AutoTokenizer

    print("=" * 70)
    print("  E7b: Mistral Weight Interpolation — Species Conservation Test")
    print("=" * 70)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(IT_MODEL, trust_remote_code=True)

    print("\nLoading weight sets (fp32, CPU)...")
    sd_base = load_weights(BASE_MODEL)
    sd_it = load_weights(IT_MODEL)
    print(f"  Base keys: {len(sd_base)}, IT keys: {len(sd_it)}")

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    print(f"  Layers: {num_layers}")

    all_results = []

    for alpha in ALPHAS:
        print(f"\n{'='*50}")
        print(f"  α = {alpha:.2f}")
        print(f"{'='*50}")

        print(f"  Interpolating weights...")
        sd_interp = interpolate_weights(sd_base, sd_it, alpha)

        print(f"  Loading interpolated model to GPU...")
        model = load_interpolated_model(sd_interp, BASE_MODEL)
        del sd_interp

        result = run_alpha(model, tokenizer, num_layers, alpha)
        all_results.append(result)

        out_path = OUTPUT_DIR / f"e7b_alpha_{alpha:.2f}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"  E7b SUMMARY — Mistral Weight Interpolation")
    print(f"{'='*70}")
    print(f"  {'α':>5s}  {'Species':12s}  {'M3_relay':>9s}  {'M2_slope':>9s}  {'Xovers':>6s}  {'PPL':>8s}")
    print(f"  {'-'*55}")
    for r in all_results:
        c = r["classification"]
        print(f"  {r['alpha']:5.2f}  {c['species']:12s}  {c['m3_relay_mean']:+9.4f}  "
              f"{c['m2_slope']:+9.4f}  {c['crossovers']:6d}  {r['perplexity']:8.1f}")

    species_seq = [r["classification"]["species"] for r in all_results]
    transitions = []
    for i in range(1, len(species_seq)):
        if species_seq[i] != species_seq[i-1]:
            transitions.append((ALPHAS[i-1], ALPHAS[i], species_seq[i-1], species_seq[i]))

    if transitions:
        print(f"\n  TRANSITIONS DETECTED:")
        for a1, a2, s1, s2 in transitions:
            print(f"    α={a1:.2f}→{a2:.2f}: {s1} → {s2}")
    else:
        print(f"\n  NO TRANSITIONS — species constant throughout interpolation")

    ppls = [r["perplexity"] for r in all_results]
    ppl_endpoints = (ppls[0] + ppls[-1]) / 2
    ppl_max_interior = max(ppls[1:-1]) if len(ppls) > 2 else 0
    if ppl_max_interior > 2 * ppl_endpoints:
        print(f"\n  PPL BARRIER: interior max {ppl_max_interior:.1f} > 2x endpoint avg {ppl_endpoints:.1f}")
    else:
        print(f"\n  No perplexity barrier (interior max {ppl_max_interior:.1f}, endpoint avg {ppl_endpoints:.1f})")

    # Check if goldsmith conserved
    goldsmith_count = sum(1 for r in all_results if r["classification"]["species"] == "goldsmith")
    print(f"\n  Goldsmith at {goldsmith_count}/{len(all_results)} points")

    combined = {
        "experiment": "E7b_mistral_weight_interpolation",
        "base_model": BASE_MODEL,
        "it_model": IT_MODEL,
        "alphas": ALPHAS,
        "num_layers": num_layers,
        "results": all_results,
        "transitions": transitions,
        "perplexity_barrier": ppl_max_interior > 2 * ppl_endpoints,
        "goldsmith_conserved": goldsmith_count == len(all_results),
        "timestamp": datetime.now().isoformat(),
    }
    combined_path = OUTPUT_DIR / f"e7b_combined_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined: {combined_path}")


if __name__ == "__main__":
    main()
