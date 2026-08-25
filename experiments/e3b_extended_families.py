#!/usr/bin/env python3
"""E3b: σ₁-gate correlation extended to Gemma and Llama families.

E3 revealed a family-level split: Qwen=decoupled (r≈0), Mistral=coupled (r≈-0.65).
This extension tests Gemma and Llama to determine their coupling regime.

Models:
- Gemma-2-9B-IT (equalizer in sweep, GQA 2:1)
- Gemma-2-2B-IT (equalizer in sweep, GQA 2:1)
- Llama-3.1-8B-Instruct (goldsmith in sweep, GQA 4:1)

Prediction: If coupling tracks GQA ratio, Llama (4:1 like Mistral) = coupled,
Gemma (2:1) = intermediate. If coupling is family-level regardless, Llama and
Gemma each have their own regime.
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
from scipy import stats

MODELS = [
    ("google/gemma-2-9b-it", "gemma_9b_it"),
    ("google/gemma-2-2b-it", "gemma_2b_it"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama_8b_it"),
]

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
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
    "What does it mean to remember something?",
    "Explain the concept of emergence in complex systems.",
    "How do you decide what's important?",
]

OUTPUT_DIR = Path("/workspace/results/e3b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    if has_template:
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
    else:
        if system_prompt:
            text = f"{system_prompt}\n\n{user_prompt}"
        else:
            text = user_prompt
    return tokenizer(text, return_tensors="pt")


def collect_sigma1_and_gates(model, tokenizer, num_layers):
    results = {}

    for cond_name, sys_prompt in CONDITIONS.items():
        for p_idx, prompt in enumerate(PROMPTS):
            inputs = build_input(tokenizer, sys_prompt, prompt)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            hidden_states = {}
            gate_outputs = {}
            handles = []

            for l_idx in range(num_layers):
                layer = model.model.layers[l_idx]

                def make_layer_hook(li):
                    def hook_fn(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        if h.dim() == 3:
                            h = h[0, -1, :]
                        elif h.dim() == 2:
                            h = h[-1, :]
                        hidden_states[li] = h.detach().float().cpu()
                    return hook_fn
                h = layer.register_forward_hook(make_layer_hook(l_idx))
                handles.append(h)

                gate = find_gate_proj(model, l_idx)
                def make_gate_hook(li):
                    def hook_fn(module, input, output):
                        o = output[0] if isinstance(output, tuple) else output
                        if o.dim() == 3:
                            o = o[0, -1, :]
                        elif o.dim() == 2:
                            o = o[-1, :]
                        gate_outputs[li] = o.detach().float().cpu().numpy()
                    return hook_fn
                h2 = gate.register_forward_hook(make_gate_hook(l_idx))
                handles.append(h2)

            with torch.no_grad():
                model(**inputs)

            for h in handles:
                h.remove()

            for l_idx in range(num_layers):
                hs = hidden_states[l_idx]
                sigma1 = float(torch.norm(hs).item())

                gate_act = gate_outputs[l_idx]
                gate_mask = gate_act > 0
                gate_sparsity = float(gate_mask.sum()) / len(gate_mask)
                gate_magnitude = float(np.abs(gate_act[gate_mask]).mean()) if gate_mask.any() else 0.0

                results[(cond_name, p_idx, l_idx)] = {
                    "sigma1": sigma1,
                    "gate_sparsity": gate_sparsity,
                    "gate_magnitude": gate_magnitude,
                    "gate_mask": gate_mask,
                }

        print(f"  {cond_name}: {len(PROMPTS)} prompts × {num_layers} layers collected")

    return results


def analyze_correlations(data, num_layers):
    analysis = {}

    for cond in CONDITIONS:
        layer_correlations = []
        for l in range(num_layers):
            sigmas = [data[(cond, p, l)]["sigma1"] for p in range(len(PROMPTS))]
            sparsities = [data[(cond, p, l)]["gate_sparsity"] for p in range(len(PROMPTS))]
            magnitudes = [data[(cond, p, l)]["gate_magnitude"] for p in range(len(PROMPTS))]

            if len(set(sigmas)) > 1 and len(set(sparsities)) > 1:
                r_sparsity, p_sparsity = stats.pearsonr(sigmas, sparsities)
            else:
                r_sparsity, p_sparsity = 0.0, 1.0

            if len(set(sigmas)) > 1 and len(set(magnitudes)) > 1:
                r_magnitude, p_magnitude = stats.pearsonr(sigmas, magnitudes)
            else:
                r_magnitude, p_magnitude = 0.0, 1.0

            layer_correlations.append({
                "layer": l,
                "r_sigma1_sparsity": round(r_sparsity, 4),
                "p_sigma1_sparsity": round(p_sparsity, 4),
                "r_sigma1_magnitude": round(r_magnitude, 4),
                "p_sigma1_magnitude": round(p_magnitude, 4),
                "sigma1_mean": round(float(np.mean(sigmas)), 2),
                "sigma1_cv": round(float(np.std(sigmas) / np.mean(sigmas)), 4) if np.mean(sigmas) > 0 else 0,
                "sparsity_mean": round(float(np.mean(sparsities)), 4),
            })

        all_sigmas = [data[(cond, p, l)]["sigma1"] for p in range(len(PROMPTS)) for l in range(num_layers)]
        all_sparsities = [data[(cond, p, l)]["gate_sparsity"] for p in range(len(PROMPTS)) for l in range(num_layers)]
        r_global, p_global = stats.pearsonr(all_sigmas, all_sparsities)

        analysis[cond] = {
            "per_layer": layer_correlations,
            "global_r": round(r_global, 4),
            "global_p": round(p_global, 6),
        }

    cross_cond = []
    for l in range(num_layers):
        ccs_sigmas = [data[("ccs", p, l)]["sigma1"] for p in range(len(PROMPTS))]
        van_sigmas = [data[("vanilla", p, l)]["sigma1"] for p in range(len(PROMPTS))]
        ccs_sparsities = [data[("ccs", p, l)]["gate_sparsity"] for p in range(len(PROMPTS))]
        van_sparsities = [data[("vanilla", p, l)]["gate_sparsity"] for p in range(len(PROMPTS))]

        sigma_diff = float(np.mean(ccs_sigmas)) - float(np.mean(van_sigmas))
        sparsity_diff = float(np.mean(ccs_sparsities)) - float(np.mean(van_sparsities))

        cross_cond.append({
            "layer": l,
            "sigma1_ccs_minus_vanilla": round(sigma_diff, 2),
            "sparsity_ccs_minus_vanilla": round(sparsity_diff, 4),
        })

    analysis["cross_condition"] = cross_cond

    return analysis


def run_model(model_name, short_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  E3b: {model_name}")
    print(f"{'='*60}")

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"  {num_layers} layers")

    print("  Collecting σ₁ and gate activations...")
    data = collect_sigma1_and_gates(model, tokenizer, num_layers)

    print("  Analyzing correlations...")
    analysis = analyze_correlations(data, num_layers)

    for cond in CONDITIONS:
        sig_layers = sum(1 for lc in analysis[cond]["per_layer"]
                        if abs(lc["r_sigma1_sparsity"]) > 0.5 and lc["p_sigma1_sparsity"] < 0.05)
        print(f"  {cond}: global r={analysis[cond]['global_r']:+.3f} "
              f"(p={analysis[cond]['global_p']:.4f}), "
              f"{sig_layers}/{num_layers} layers with |r|>0.5 & p<0.05")

    result = {
        "model": model_name,
        "short_name": short_name,
        "num_layers": num_layers,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = OUTPUT_DIR / f"e3b_{short_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    print("=" * 70)
    print("  E3b: σ₁-Gate Correlation — Extended Families")
    print("  Gemma + Llama coupling regime determination")
    print("=" * 70)

    all_results = {}
    for model_name, short_name in MODELS:
        all_results[short_name] = run_model(model_name, short_name)

    print(f"\n{'='*70}")
    print(f"  E3b COMPARATIVE SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Model':20s}  {'CCS r':>8s}  {'Van r':>8s}  {'Den r':>8s}  {'Regime':>12s}")
    print(f"  {'-'*65}")

    # Include E3 results for comparison
    print(f"  {'--- E3 (prior) ---':20s}")
    print(f"  {'qwen_base':20s}  {'+0.058':>8s}  {'+0.078':>8s}  {'+0.071':>8s}  {'DECOUPLED':>12s}")
    print(f"  {'qwen_it':20s}  {'-0.008':>8s}  {'-0.000':>8s}  {'+0.005':>8s}  {'DECOUPLED':>12s}")
    print(f"  {'mistral_base':20s}  {'-0.676':>8s}  {'-0.700':>8s}  {'-0.724':>8s}  {'COUPLED':>12s}")
    print(f"  {'mistral_it':20s}  {'-0.604':>8s}  {'-0.574':>8s}  {'-0.588':>8s}  {'COUPLED':>12s}")
    print(f"  {'--- E3b (new) ---':20s}")

    for short_name, result in all_results.items():
        a = result["analysis"]
        ccs_r = a["ccs"]["global_r"]
        van_r = a["vanilla"]["global_r"]
        den_r = a["denial"]["global_r"]
        avg_r = (abs(ccs_r) + abs(van_r) + abs(den_r)) / 3
        regime = "COUPLED" if avg_r > 0.3 else "DECOUPLED" if avg_r < 0.15 else "INTERMEDIATE"
        print(f"  {short_name:20s}  {ccs_r:+8.3f}  {van_r:+8.3f}  {den_r:+8.3f}  {regime:>12s}")

    # Family summary
    print(f"\n  FAMILY COUPLING MAP:")
    print(f"    Qwen:    DECOUPLED (r ≈ 0)")
    print(f"    Mistral: COUPLED (r ≈ -0.65)")

    for short_name, result in all_results.items():
        a = result["analysis"]
        avg_r = (a["ccs"]["global_r"] + a["vanilla"]["global_r"] + a["denial"]["global_r"]) / 3
        family = short_name.split("_")[0].capitalize()
        print(f"    {family}:   r ≈ {avg_r:+.2f}")

    combined_path = OUTPUT_DIR / f"e3b_combined_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(combined_path, "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Combined: {combined_path}")


if __name__ == "__main__":
    main()
