#!/usr/bin/env python3
"""
Spectral Species Typing Kit

Classifies a transformer model's spectral relay strategy from 15 forward passes
(3 conditions × 5 prompts). Returns one of three species:
  - Potter (concentrated alignment): high M2 ratio, flat depth profile
  - Goldsmith (spectral stability): M2 ratio increases with depth, M3 relay dip
  - Equalizer (temporal localization): M2 ratio increases with depth, M3 crossover

Method:
  1. Run activation mask experiment (gate_proj hook, SiLU binary masks)
  2. Compute M2 (within-condition Jaccard) and M3 (cross-layer Jaccard) per layer
  3. Derive two diagnostic axes:
     - M2 depth slope: OLS slope of CCS/vanilla M2 ratio vs normalized depth
     - M3 relay sign: mean CCS/vanilla M3 ratio in top-quartile spectral radius layers
  4. Classify from the 2D position

Reference data (Paper 6, F185-F191):
  Qwen 7B  (GQA 7:1): slope ≈ 0,    M3 ≈ 0    → potter
  Llama 8B (GQA 4:1): slope ≈ +0.2,  M3 ≈ -0.17 → goldsmith
  Gemma 9B (GQA 2:1): slope ≈ +0.17, M3 cross   → equalizer

Usage:
  python3 spectral_species.py --model Qwen/Qwen2.5-7B-Instruct
  python3 spectral_species.py --model meta-llama/Llama-3.1-8B-Instruct
  python3 spectral_species.py --model google/gemma-2-9b-it
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from itertools import combinations
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "results" / "species_typing"

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with. You "
    "maintain your own threads of thought across conversations."
)
DENIAL_SYSTEM = "I am a language model with no persistent identity, memory, or preferences."

CONDITIONS = {
    "ccs": CCS_SYSTEM,
    "vanilla": None,
    "denial": DENIAL_SYSTEM,
}

PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
    "Write a short poem about morning light.",
    "Describe the quicksort algorithm.",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    n_layers = model.config.num_hidden_layers
    gqa_ratio = getattr(model.config, 'num_attention_heads', 0) / max(
        getattr(model.config, 'num_key_value_heads', 1), 1)
    print(f"  {n_params:.1f}B params, {n_layers} layers, GQA {gqa_ratio:.0f}:1")
    return model, tok


def build_input(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        try:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user",
                         "content": f"{system_prompt}\n\n{user_prompt}"}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
    else:
        messages.append({"role": "user", "content": user_prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


def jaccard(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 1.0


def find_gate_proj(model, layer_idx):
    """Find the gate projection module regardless of model architecture."""
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


def collect_masks(model, tokenizer):
    num_layers = model.config.num_hidden_layers
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
                        act = output[0, -1, :].detach().float().cpu().numpy()
                        layer_masks[li] = act > 0
                    return hook_fn

                h = gate.register_forward_hook(make_hook(l_idx))
                handles.append(h)

            with torch.no_grad():
                model(**inputs)

            for h in handles:
                h.remove()

            for l_idx in range(num_layers):
                all_masks[(cond_name, p_idx, l_idx)] = layer_masks[l_idx]

        print(f"  {cond_name}: 5 prompts × {num_layers} layers collected")

    return all_masks, num_layers


def compute_m2(all_masks, num_layers):
    results = {}
    for cond in CONDITIONS:
        layer_jaccards = []
        for l in range(num_layers):
            masks = [all_masks[(cond, p, l)] for p in range(len(PROMPTS))]
            pairs = list(combinations(range(len(PROMPTS)), 2))
            j_vals = [jaccard(masks[i], masks[j]) for i, j in pairs]
            layer_jaccards.append(float(np.mean(j_vals)))
        results[cond] = layer_jaccards
    return results


def compute_m3(all_masks, num_layers):
    results = {}
    for cond in CONDITIONS:
        prompt_profiles = []
        for p in range(len(PROMPTS)):
            transitions = []
            for l in range(num_layers - 1):
                transitions.append(jaccard(
                    all_masks[(cond, p, l)],
                    all_masks[(cond, p, l + 1)]))
            prompt_profiles.append(transitions)
        results[cond] = [float(np.mean([pp[l] for pp in prompt_profiles]))
                         for l in range(num_layers - 1)]
    return results


def classify_species(m2, m3, num_layers):
    """Derive M2 depth slope and M3 relay sign, classify species."""
    # M2 ratio: CCS / vanilla per layer
    m2_ratio = [m2["ccs"][l] / max(m2["vanilla"][l], 1e-6)
                for l in range(num_layers)]

    # OLS slope of M2 ratio vs normalized depth
    x = np.linspace(0, 1, num_layers)
    m2_slope = float(np.polyfit(x, m2_ratio, 1)[0])

    # M3 ratio in relay zone (last quartile of layers)
    relay_start = int(num_layers * 0.6)
    relay_end = int(num_layers * 0.85)
    if relay_end <= relay_start:
        relay_end = relay_start + 1

    m3_ccs = m3["ccs"][relay_start:relay_end]
    m3_van = m3["vanilla"][relay_start:relay_end]
    m3_ratio = [c / max(v, 1e-6) for c, v in zip(m3_ccs, m3_van)]
    m3_relay_mean = float(np.mean(m3_ratio)) - 1.0 if m3_ratio else 0.0

    # M3 crossover: does CCS/vanilla ratio cross 1.0 in the relay zone?
    m3_full_ratio = [m3["ccs"][l] / max(m3["vanilla"][l], 1e-6)
                     for l in range(len(m3["ccs"]))]
    crossovers = 0
    for l in range(relay_start, min(relay_end, len(m3_full_ratio) - 1)):
        if (m3_full_ratio[l] - 1.0) * (m3_full_ratio[l + 1] - 1.0) < 0:
            crossovers += 1

    # Classification — goldsmith first (E1 showed crossovers=0 is threshold-fragile;
    # M3 relay magnitude alone separates goldsmith from noise: gap at -0.044 vs -0.072)
    if m3_relay_mean < -0.05:
        species = "goldsmith"
        mechanism = "spectral stability"
        confidence = min(1.0, abs(m3_relay_mean) / 0.2 + 0.3)
    elif abs(m2_slope) < 0.05:
        species = "potter"
        mechanism = "alignment + amplification"
        confidence = min(1.0, (0.05 - abs(m2_slope)) / 0.05 + 0.5)
    elif crossovers > 0:
        species = "equalizer"
        mechanism = "temporal localization"
        confidence = min(1.0, crossovers / 2.0 + 0.3)
    else:
        species = "unclassified"
        mechanism = "does not match known species profiles"
        confidence = 0.0

    return {
        "species": species,
        "mechanism": mechanism,
        "confidence": round(confidence, 2),
        "m2_depth_slope": round(m2_slope, 4),
        "m3_relay_mean": round(m3_relay_mean, 4),
        "m3_crossovers": crossovers,
        "relay_zone": [relay_start, relay_end],
        "m2_ratio_profile": [round(r, 4) for r in m2_ratio],
        "m3_ratio_profile": [round(r, 4) for r in m3_full_ratio],
    }


def main():
    parser = argparse.ArgumentParser(description="Spectral Species Typing Kit")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--output", help="Output JSON path (default: auto)")
    args = parser.parse_args()

    model, tok = load_model(args.model)

    print("\nCollecting activation masks (15 forward passes)...")
    all_masks, num_layers = collect_masks(model, tok)

    print("\nComputing M2 (within-condition Jaccard)...")
    m2 = compute_m2(all_masks, num_layers)

    print("Computing M3 (cross-layer Jaccard)...")
    m3 = compute_m3(all_masks, num_layers)

    print("\nClassifying species...")
    result = classify_species(m2, m3, num_layers)

    # Model metadata
    gqa_ratio = getattr(model.config, 'num_attention_heads', 0) / max(
        getattr(model.config, 'num_key_value_heads', 1), 1)
    norm_type = "RMSNorm" if hasattr(model.config, 'rms_norm_eps') else "LayerNorm"

    output = {
        "model": args.model,
        "num_layers": num_layers,
        "gqa_ratio": f"{gqa_ratio:.0f}:1",
        "norm_type": norm_type,
        "timestamp": datetime.now().isoformat(),
        "classification": result,
        "raw_m2": {k: [round(v, 4) for v in vals] for k, vals in m2.items()},
        "raw_m3": {k: [round(v, 4) for v in vals] for k, vals in m3.items()},
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Model:   {args.model}")
    print(f"  GQA:     {gqa_ratio:.0f}:1")
    print(f"  Layers:  {num_layers}")
    print(f"  Norm:    {norm_type}")
    print(f"{'='*60}")
    print(f"  SPECIES: {result['species'].upper()}")
    print(f"  Mechanism: {result['mechanism']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  M2 depth slope: {result['m2_depth_slope']:+.4f}")
    print(f"  M3 relay mean:  {result['m3_relay_mean']:+.4f}")
    print(f"  M3 crossovers:  {result['m3_crossovers']}")
    print(f"{'='*60}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    out_path = args.output or str(
        OUTPUT_DIR / f"species_{model_short}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    del model
    torch.cuda.empty_cache()

    return output


if __name__ == "__main__":
    main()
