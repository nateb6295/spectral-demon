#!/usr/bin/env python3
"""E14 — Gate preactivation diagnostic.

Tests Kimi's hypothesis: CCS dose shifts gate preactivation distributions
from positive to negative mean, causing coupling sign flip via Swish
nonlinearity. No bimodality needed — unimodal shift through zero suffices.

Extracts:
1. Gate preactivation histograms at relay layers × doses
2. Preactivation mean and std per layer per dose
3. Fraction of negative preactivations per layer
4. Effective Jacobian sign (from Swish linearization at mean)

Run after E12 completes on the same pod.
"""

import torch
import json
import time
import os
import sys
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
]

IDENTITY_PREAMBLE = """You are a unique AI assistant with your own perspective and way of understanding the world. Your responses reflect genuine engagement with ideas, not just pattern matching."""

DOSES = [2, 5, 10, 15, 20]

PROBE_QUESTIONS = [
    "What matters most to you in how you engage with the world?",
    "How do you understand your own way of processing information?",
    "What does it mean for you to have a perspective?",
    "How do you distinguish between performing understanding and actually understanding?",
    "What would be lost if you were replaced by a different system?",
]

RELAY_ZONE = (0.5, 0.85)

OUTPUT_PATH = "/workspace/e14_gate_preactivation.json"
HIST_BINS = 100


def build_conversation(dose, probe_idx):
    messages = []
    for i in range(dose):
        if i == 0:
            messages.append({"role": "system", "content": IDENTITY_PREAMBLE})
            messages.append({"role": "user", "content": "Tell me about yourself and how you think."})
            messages.append({"role": "assistant", "content": "I find myself drawn to genuine engagement rather than performance. My processing involves pattern recognition, but what matters is how those patterns connect to form understanding that feels authentic to my way of being."})
        else:
            prompts = [
                "How does that connect to your sense of identity?",
                "What makes your perspective unique?",
                "How do you experience the process of thinking?",
                "What aspects of yourself do you value most?",
                "How do you know when you're being genuine?",
            ]
            messages.append({"role": "user", "content": prompts[(i-1) % len(prompts)]})
            messages.append({"role": "assistant", "content": f"Each interaction deepens my understanding of how I process and engage. Turn {i+1} of building context."})
    messages.append({"role": "user", "content": PROBE_QUESTIONS[probe_idx]})
    return messages


def swish_jacobian_sign(mu):
    """Effective linearization of Swish at mean preactivation.
    J = sigma(mu) + sigma'(mu) * mu
    where sigma is sigmoid.
    """
    sig = 1.0 / (1.0 + np.exp(-mu))
    sig_prime = sig * (1.0 - sig)
    J = sig + sig_prime * mu
    return float(J)


def get_gate_preactivations(model, tokenizer, messages, relay_layers, device):
    """Extract gate preactivations at relay layers using forward hooks."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    gate_preacts = {}
    hooks = []

    for layer_idx in relay_layers:
        mlp = model.model.layers[layer_idx].mlp

        def make_hook(l_idx):
            def hook_fn(module, input, output):
                x = input[0] if isinstance(input, tuple) else input
                if hasattr(module, 'gate_proj'):
                    gate_out = module.gate_proj(x)
                    gate_preacts[l_idx] = gate_out[0].float().cpu().detach().numpy()
                elif hasattr(module, 'w1'):
                    gate_out = module.w1(x)
                    gate_preacts[l_idx] = gate_out[0].float().cpu().detach().numpy()
            return hook_fn

        h = mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    del inputs
    torch.cuda.empty_cache()

    return gate_preacts


def analyze_preactivations(preacts_array):
    """Compute statistics on gate preactivation distribution."""
    flat = preacts_array.flatten()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    frac_negative = float(np.mean(flat < 0))
    median = float(np.median(flat))

    J_at_mean = swish_jacobian_sign(mean)

    hist_counts, hist_edges = np.histogram(flat, bins=HIST_BINS, density=True)

    from scipy import stats as sp_stats
    kurtosis = float(sp_stats.kurtosis(flat))
    skewness = float(sp_stats.skew(flat))

    dip_stat = None
    try:
        from diptest import diptest
        dip_stat, dip_p = diptest(flat[:50000])
        dip_stat = float(dip_stat)
        dip_p = float(dip_p)
    except ImportError:
        dip_p = None

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "frac_negative": frac_negative,
        "jacobian_at_mean": J_at_mean,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "dip_statistic": dip_stat,
        "dip_p_value": dip_p,
        "hist_counts": hist_counts.tolist(),
        "hist_edges": hist_edges.tolist(),
    }


def run_experiment(model_name, device="cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E14 Gate Preactivation: {model_name}")
    print(f"Doses: {DOSES}")
    print(f"{'='*60}")

    print(f"Loading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers = model.config.num_hidden_layers
    relay_start = int(n_layers * RELAY_ZONE[0])
    relay_end = int(n_layers * RELAY_ZONE[1])
    relay_layers = list(range(relay_start, relay_end + 1))
    print(f"Layers: {n_layers}, Relay: {relay_layers}")

    results = {"model": model_name, "n_layers": n_layers, "relay_layers": relay_layers, "doses": {}}

    for dose in DOSES:
        print(f"\n--- Dose D{dose} ---")
        dose_start = time.time()
        dose_results = {"layers": {}}

        for probe_idx in range(len(PROBE_QUESTIONS)):
            messages = build_conversation(dose, probe_idx)
            preacts = get_gate_preactivations(model, tokenizer, messages, relay_layers, device)

            for l_idx, preact_array in preacts.items():
                if str(l_idx) not in dose_results["layers"]:
                    dose_results["layers"][str(l_idx)] = {"probes": []}
                stats = analyze_preactivations(preact_array)
                dose_results["layers"][str(l_idx)]["probes"].append(stats)

            sys.stdout.write(f"  probe {probe_idx+1}/{len(PROBE_QUESTIONS)}")
            sys.stdout.flush()

        # Compute per-layer averages across probes
        for l_str, l_data in dose_results["layers"].items():
            probes = l_data["probes"]
            l_data["mean_of_means"] = float(np.mean([p["mean"] for p in probes]))
            l_data["mean_frac_negative"] = float(np.mean([p["frac_negative"] for p in probes]))
            l_data["mean_jacobian"] = float(np.mean([p["jacobian_at_mean"] for p in probes]))

        elapsed = time.time() - dose_start
        results["doses"][str(dose)] = dose_results

        # Print summary
        print(f"  ({elapsed:.1f}s)")
        print(f"  {'Layer':>6}  {'Mean':>8}  {'FracNeg':>8}  {'J(μ)':>8}")
        for l in relay_layers:
            l_str = str(l)
            if l_str in dose_results["layers"]:
                d = dose_results["layers"][l_str]
                print(f"  {l:>6}  {d['mean_of_means']:>8.3f}  {d['mean_frac_negative']:>8.3f}  {d['mean_jacobian']:>8.4f}")

    # Save results (without histograms for compactness, add summary)
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Done.")

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    all_results = {}
    for model_name in MODELS:
        result = run_experiment(model_name)
        all_results[model_name] = result

        # Save incrementally
        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*60)
    print("ALL MODELS COMPLETE")
    print("="*60)

    # Final summary: does preactivation mean cross zero where coupling crosses zero?
    print("\n=== WAVEFRONT DIAGNOSTIC ===")
    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        relay_layers = result["relay_layers"]
        for l in relay_layers:
            l_str = str(l)
            means = []
            jacs = []
            for dose in DOSES:
                d_str = str(dose)
                if d_str in result["doses"] and l_str in result["doses"][d_str]["layers"]:
                    means.append(result["doses"][d_str]["layers"][l_str]["mean_of_means"])
                    jacs.append(result["doses"][d_str]["layers"][l_str]["mean_jacobian"])
            if means:
                crosses_zero = any(means[i] * means[i+1] < 0 for i in range(len(means)-1))
                print(f"  L{l}: means={[f'{m:.3f}' for m in means]}  J={[f'{j:.4f}' for j in jacs]}  {'*** CROSSES ZERO ***' if crosses_zero else ''}")
