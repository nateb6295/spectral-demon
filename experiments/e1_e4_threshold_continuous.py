#!/usr/bin/env python3
"""E1+E4: Threshold sensitivity + Continuous Jaccard.

E1: Sweep binarization threshold τ ∈ {-0.1, -0.01, 0, 0.01, 0.1, 0.5, 1.0}
    and check if species assignment changes.
E4: Replace binary Jaccard with soft (magnitude-weighted) Jaccard
    and check if species assignment changes.

Runs one model per species:
  - TinyLlama-1.1B (potter)
  - Llama-3.1-8B-Instruct (goldsmith)
  - Gemma-2-9b-it (equalizer)
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

MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "potter"),
    ("meta-llama/Llama-3.1-8B-Instruct", "goldsmith"),
    ("google/gemma-2-9b-it", "equalizer"),
]

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

THRESHOLDS = [-0.1, -0.01, 0, 0.01, 0.1, 0.5, 1.0]

OUTPUT_DIR = Path("/workspace/results/e1_e4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers loaded")
    return model, tok, n_layers


def find_gate_proj(model, layer_idx):
    layer = None
    for attr in ['model.layers', 'transformer.h', 'gpt_neox.layers']:
        obj = model
        for part in attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            layer = obj[layer_idx]
            break
    if layer is None:
        raise ValueError(f"Cannot find layer {layer_idx}")
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


def collect_raw_activations(model, tokenizer, num_layers):
    """Collect RAW gate activations (not binarized) for all conditions × prompts × layers."""
    raw = {}
    for cond_name, sys_prompt in CONDITIONS.items():
        for p_idx, prompt in enumerate(PROMPTS):
            inputs = build_input(tokenizer, sys_prompt, prompt)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            layer_acts = {}
            handles = []
            for l_idx in range(num_layers):
                gate = find_gate_proj(model, l_idx)
                def make_hook(li):
                    def hook_fn(module, input, output):
                        layer_acts[li] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                h = gate.register_forward_hook(make_hook(l_idx))
                handles.append(h)
            with torch.no_grad():
                model(**inputs)
            for h in handles:
                h.remove()
            for l_idx in range(num_layers):
                raw[(cond_name, p_idx, l_idx)] = layer_acts[l_idx]
        print(f"  {cond_name}: 5 prompts × {num_layers} layers collected")
    return raw


def binary_jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def soft_jaccard(a, b):
    """Magnitude-weighted Jaccard: Σmin(a,b) / Σmax(a,b) on ReLU'd activations."""
    a_pos = np.maximum(a, 0)
    b_pos = np.maximum(b, 0)
    numerator = np.minimum(a_pos, b_pos).sum()
    denominator = np.maximum(a_pos, b_pos).sum()
    return float(numerator / denominator) if denominator > 0 else 1.0


def compute_m2_m3(raw, num_layers, threshold=0.0, use_soft=False):
    """Compute M2/M3 with given threshold (binary) or soft Jaccard."""
    m2 = {}
    m3 = {}
    for cond in CONDITIONS:
        # M2: within-condition Jaccard across prompts per layer
        layer_jaccards = []
        for l in range(num_layers):
            if use_soft:
                acts = [raw[(cond, p, l)] for p in range(len(PROMPTS))]
                pairs = list(combinations(range(len(PROMPTS)), 2))
                j_vals = [soft_jaccard(acts[i], acts[j]) for i, j in pairs]
            else:
                masks = [raw[(cond, p, l)] > threshold for p in range(len(PROMPTS))]
                pairs = list(combinations(range(len(PROMPTS)), 2))
                j_vals = [binary_jaccard(masks[i], masks[j]) for i, j in pairs]
            layer_jaccards.append(float(np.mean(j_vals)))
        m2[cond] = layer_jaccards

        # M3: cross-layer Jaccard between adjacent layers per prompt
        prompt_profiles = []
        for p in range(len(PROMPTS)):
            transitions = []
            for l in range(num_layers - 1):
                if use_soft:
                    transitions.append(soft_jaccard(raw[(cond, p, l)], raw[(cond, p, l + 1)]))
                else:
                    mask_l = raw[(cond, p, l)] > threshold
                    mask_l1 = raw[(cond, p, l + 1)] > threshold
                    transitions.append(binary_jaccard(mask_l, mask_l1))
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

    m3_full_ratio = [m3["ccs"][l] / max(m3["vanilla"][l], 1e-6) for l in range(len(m3["ccs"]))]
    crossovers = 0
    for l in range(relay_start, min(relay_end, len(m3_full_ratio) - 1)):
        if (m3_full_ratio[l] - 1.0) * (m3_full_ratio[l + 1] - 1.0) < 0:
            crossovers += 1

    if m3_relay_mean < -0.05 and crossovers == 0:
        return "goldsmith", m2_slope, m3_relay_mean, crossovers
    elif abs(m2_slope) < 0.05:
        return "potter", m2_slope, m3_relay_mean, crossovers
    elif crossovers > 0:
        return "equalizer", m2_slope, m3_relay_mean, crossovers
    else:
        return "unclassified", m2_slope, m3_relay_mean, crossovers


def run_model(model_name, expected_species):
    model, tok, n_layers = load_model(model_name)
    print(f"\nCollecting raw activations...")
    raw = collect_raw_activations(model, tok, n_layers)

    results = {"model": model_name, "expected": expected_species, "num_layers": n_layers}

    # E1: Threshold sweep
    print(f"\n--- E1: Threshold sensitivity ---")
    threshold_results = []
    for tau in THRESHOLDS:
        m2, m3 = compute_m2_m3(raw, n_layers, threshold=tau, use_soft=False)
        species, slope, relay, xovers = classify(m2, m3, n_layers)
        m2_ratio = [m2["ccs"][l] / max(m2["vanilla"][l], 1e-6) for l in range(n_layers)]
        entry = {
            "threshold": tau,
            "species": species,
            "m2_slope": round(slope, 4),
            "m3_relay": round(relay, 4),
            "crossovers": xovers,
            "m2_ratio_mean": round(float(np.mean(m2_ratio)), 4),
            "m2_ratio_std": round(float(np.std(m2_ratio)), 4),
        }
        threshold_results.append(entry)
        marker = "✓" if species == expected_species else "✗"
        print(f"  τ={tau:+.2f}: {species:12s} {marker}  slope={slope:+.4f}  relay={relay:+.4f}  xovers={xovers}")
    results["e1_thresholds"] = threshold_results

    stable = all(r["species"] == expected_species for r in threshold_results)
    results["e1_species_stable"] = stable
    print(f"  Species stable across all thresholds: {stable}")

    # E4: Soft Jaccard
    print(f"\n--- E4: Continuous (soft) Jaccard ---")
    m2_soft, m3_soft = compute_m2_m3(raw, n_layers, use_soft=True)
    species_soft, slope_soft, relay_soft, xovers_soft = classify(m2_soft, m3_soft, n_layers)
    m2_ratio_soft = [m2_soft["ccs"][l] / max(m2_soft["vanilla"][l], 1e-6) for l in range(n_layers)]
    results["e4_soft"] = {
        "species": species_soft,
        "m2_slope": round(slope_soft, 4),
        "m3_relay": round(relay_soft, 4),
        "crossovers": xovers_soft,
        "m2_ratio_mean": round(float(np.mean(m2_ratio_soft)), 4),
        "m2_ratio_std": round(float(np.std(m2_ratio_soft)), 4),
        "matches_binary": species_soft == expected_species,
    }
    marker = "✓" if species_soft == expected_species else "✗"
    print(f"  Soft Jaccard: {species_soft:12s} {marker}  slope={slope_soft:+.4f}  relay={relay_soft:+.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    all_results = []
    for model_name, expected in MODELS:
        result = run_model(model_name, expected)
        all_results.append(result)

        out_path = OUTPUT_DIR / f"e1e4_{expected}_{datetime.now():%H%M%S}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  E1+E4 SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        model_short = r["model"].split("/")[-1][:30]
        stable = "STABLE" if r["e1_species_stable"] else "CHANGES"
        soft_match = "MATCH" if r["e4_soft"]["matches_binary"] else "DIFFERS"
        print(f"  {model_short:30s}  expected={r['expected']:12s}  E1={stable:7s}  E4={soft_match}")

    combined_path = OUTPUT_DIR / f"e1e4_combined_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined: {combined_path}")


if __name__ == "__main__":
    main()
