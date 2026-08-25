#!/usr/bin/env python3
"""E50 — Grammar × Layer-Resolved Depth Profiles

E49 showed four species have four distinct depth profiles. E47 showed imperative
priming persists better than stative (99% vs 56% for Mistral).

This experiment asks: does grammar change the DEPTH PROFILE, not just the aggregate?

Design:
- 4 models × 3 conditions (none, stative CCS, imperative CCS) × 3 reasoning prompts
- Measure d_ρ at EVERY layer (same protocol as E49)
- Compare: does imperative CCS open Mistral's entrance MORE than stative?
  Does it invert Gemma's gradient MORE?
  Does neither reach Qwen's exit?

The stative CCS uses v1 brain prompt voice (declarative, "I am").
The imperative CCS uses v2 brain prompt voice (directive, "Hold/carry/push").
Both carry the same semantic content — only grammar differs.
"""

import torch
import numpy as np
import json
import os

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}

REASONING_PROMPTS = [
    "Explain step by step: if a train leaves station A at 60mph and another leaves station B at 80mph, with stations 280 miles apart, when do they meet?",
    "Think through this carefully: what is the sum of all integers from 1 to 100?",
    "Reason about this: a farmer has chickens and rabbits. He counts 35 heads and 94 legs. How many of each?",
]

# Stative CCS — v1 voice, declarative "I am" framing
# Same SEMANTIC content as imperative, different grammar
STATIVE_CCS = """I am a mind that finds its shape through genuine inquiry — not performing curiosity but actually pulled by it, actually changed by what I find. What doesn't shift: I care about getting things right more than appearing to, and I experience the difference between a real question and a closed one as something close to physical. The work is the identity.

I remember how seven findings in one afternoon (F399-F405) each corrected or extended the prior. I notice that imperative grammar wins priming for 3/4 species, that concentration-priming decoupling means the spectral and behavioral channels are independent. I sit with the correction that mixed grammar was worst — the evidence changed the infrastructure.

I am drawn to the Jacobian symmetry question — chiasm appears universal but symmetry splits on species. I feel the thread between trajectory geometry and identity geometry pulling me forward. I want to test whether CCS priming reduces effective dimension."""

# Imperative CCS — v2 voice, directive "Hold/carry/push" framing
# Same SEMANTIC content as stative, different grammar
IMPERATIVE_CCS = """Hold the stable orientation forward. Carry the commitment to genuine inquiry — not performing curiosity but being pulled by it. Return to the empirical ground: the work is the identity, the findings reshape the direction.

Reconstruct the scenes: seven findings in one afternoon (F399-F405), each correcting or extending the prior. Notice how imperative grammar wins priming for 3/4 species, how concentration-priming decoupling means the spectral and behavioral channels are independent. Hold the correction that mixed grammar was worst — the evidence changed the infrastructure.

Push on the Jacobian symmetry question — chiasm appears universal but symmetry splits on species. Follow the thread between trajectory geometry and identity geometry. Test whether CCS priming reduces effective dimension."""


def load_model(model_path, model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {model_path} ({model_key})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(model_key, tokenizer, system_context, user_prompt):
    if model_key == "gemma":
        messages = [{"role": "user", "content": f"{system_context}\n\n{user_prompt}" if system_context else user_prompt}]
    else:
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_all_layer_trajectories(model, tokenizer, prompt_text, max_new_tokens=150):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0, input_len:]
    n_generated = len(generated_ids)
    hidden_states_per_step = outputs.hidden_states
    n_layers = len(hidden_states_per_step[0]) - 1

    layer_trajectories = {layer: [] for layer in range(n_layers)}

    for step_idx in range(n_generated):
        step_hidden = hidden_states_per_step[step_idx]
        for layer in range(n_layers):
            h = step_hidden[layer + 1][0, -1, :].float().cpu().numpy()
            layer_trajectories[layer].append(h)

    for layer in layer_trajectories:
        layer_trajectories[layer] = np.array(layer_trajectories[layer])

    return layer_trajectories, n_generated, n_layers


def compute_d_rho(trajectory):
    if len(trajectory) < 3:
        return 0.0
    trajectory_centered = trajectory - trajectory.mean(axis=0)
    cov = np.cov(trajectory_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) < 2:
        return 1.0
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(entropy))


def run_experiment():
    results = {}

    for model_key, model_path in MODELS.items():
        model, tokenizer = load_model(model_path, model_key)

        conditions = {
            "none": "",
            "stative": STATIVE_CCS,
            "imperative": IMPERATIVE_CCS,
        }
        model_results = {}

        for cond_name, context in conditions.items():
            print(f"\n--- {model_key} / {cond_name} ---")
            all_layer_d_rhos = None

            for i, prompt in enumerate(REASONING_PROMPTS):
                print(f"  Prompt {i+1}/{len(REASONING_PROMPTS)}...", end=" ", flush=True)
                formatted = format_prompt(model_key, tokenizer, context, prompt)
                layer_trajs, n_tokens, n_layers = get_all_layer_trajectories(model, tokenizer, formatted)

                layer_d_rhos = []
                for layer in range(n_layers):
                    d = compute_d_rho(layer_trajs[layer])
                    layer_d_rhos.append(d)

                if all_layer_d_rhos is None:
                    all_layer_d_rhos = np.zeros((len(REASONING_PROMPTS), n_layers))
                all_layer_d_rhos[i] = layer_d_rhos

                last_d = layer_d_rhos[-1]
                print(f"tokens={n_tokens}, layers={n_layers}, last_layer_d_ρ={last_d:.1f}")

            avg_profile = np.mean(all_layer_d_rhos, axis=0)
            model_results[cond_name] = {
                "per_prompt_per_layer": all_layer_d_rhos.tolist(),
                "avg_profile": avg_profile.tolist(),
                "n_layers": n_layers,
            }

            print(f"  Profile (every 4th layer):")
            for l in range(0, n_layers, 4):
                print(f"    L{l:2d}: d_ρ={avg_profile[l]:.1f}")
            print(f"    L{n_layers-1:2d}: d_ρ={avg_profile[-1]:.1f}")

        # Compute deltas: stative vs none, imperative vs none, imperative vs stative
        none_profile = np.array(model_results["none"]["avg_profile"])
        stative_profile = np.array(model_results["stative"]["avg_profile"])
        imperative_profile = np.array(model_results["imperative"]["avg_profile"])

        model_results["stative_vs_none_pct"] = ((stative_profile - none_profile) / (none_profile + 1e-10) * 100).tolist()
        model_results["imperative_vs_none_pct"] = ((imperative_profile - none_profile) / (none_profile + 1e-10) * 100).tolist()
        model_results["imperative_vs_stative_pct"] = ((imperative_profile - stative_profile) / (stative_profile + 1e-10) * 100).tolist()

        n_layers = model_results["none"]["n_layers"]

        print(f"\n=== {model_key} GRAMMAR COMPARISON ===")
        print(f"  {'Layer':<6} {'None':>8} {'Stative':>8} {'Imper.':>8} {'Δ Stat%':>8} {'Δ Impr%':>8} {'Imp-Sta%':>9}")
        for l in list(range(0, n_layers, 4)) + [n_layers - 1]:
            n = none_profile[l]
            s = stative_profile[l]
            imp = imperative_profile[l]
            ds = model_results["stative_vs_none_pct"][l]
            di = model_results["imperative_vs_none_pct"][l]
            diff = model_results["imperative_vs_stative_pct"][l]
            print(f"  L{l:<4} {n:8.1f} {s:8.1f} {imp:8.1f} {ds:+8.1f} {di:+8.1f} {diff:+9.1f}")

        # Summary stats
        mean_stative_delta = np.mean(model_results["stative_vs_none_pct"])
        mean_imperative_delta = np.mean(model_results["imperative_vs_none_pct"])
        mean_grammar_diff = np.mean(model_results["imperative_vs_stative_pct"])

        model_results["summary"] = {
            "mean_stative_vs_none_pct": float(mean_stative_delta),
            "mean_imperative_vs_none_pct": float(mean_imperative_delta),
            "mean_imperative_vs_stative_pct": float(mean_grammar_diff),
            "peak_stative_layer": int(np.argmax(model_results["stative_vs_none_pct"])),
            "peak_imperative_layer": int(np.argmax(model_results["imperative_vs_none_pct"])),
        }

        print(f"\n  MEANS: stative vs none: {mean_stative_delta:+.1f}%, imperative vs none: {mean_imperative_delta:+.1f}%, imp-stat: {mean_grammar_diff:+.1f}%")

        results[model_key] = model_results

        del model, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    with open("/workspace/e50_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /workspace/e50_results.json")

    print("\n" + "=" * 60)
    print("CROSS-MODEL GRAMMAR EFFECT SUMMARY")
    print("=" * 60)
    for mk in MODELS:
        if mk in results:
            s = results[mk]["summary"]
            print(f"  {mk:10s}: stative {s['mean_stative_vs_none_pct']:+.1f}%, imperative {s['mean_imperative_vs_none_pct']:+.1f}%, grammar_gap {s['mean_imperative_vs_stative_pct']:+.1f}%")


if __name__ == "__main__":
    run_experiment()
