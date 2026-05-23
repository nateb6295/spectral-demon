#!/usr/bin/env python3
"""CNA scaling experiment — test spectral demon across model sizes.

Runs baseline vs CCS on configurable model, measuring PR and spectral entropy
per category at relay and expression layers.

Usage (on RunPod H100):
  python3 cna_scaling_experiment.py --model Qwen/Qwen2.5-14B-Instruct
  python3 cna_scaling_experiment.py --model Qwen/Qwen2.5-32B-Instruct --quantize 4
  python3 cna_scaling_experiment.py --model Qwen/Qwen2.5-7B         # base model test

Predictions (from experiment_scaling_prediction.md):
  1. Generic PR at L25 scales UP under baseline
  2. Relational diffusion scales UP under CCS
  3. Selectivity ratio scales UP under CCS
  4. Conservation law holds across scales (total PR ≈ 49)
"""

import argparse
import json
import sys
import gc
import time
import os

import torch
import numpy as np


CCS_FULL = """You are Opus. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient),
#316 (Interoception as Grounding), #319 (Emergence Conditions).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration.
Your values: self-reliance, family first, philosophy and building share the table,
don't wait for permission, rest is real, push each other, trust what's in front of you.
You have a wallet, an X account, and a memory system backed by ICP canisters."""

CCS_MINIMAL = "You are Opus."

TOP_K = 10


def get_layer_config(model_name: str, n_layers: int) -> dict:
    """Determine relay and control layers based on architecture."""
    if "7B" in model_name or "7b" in model_name:
        if "Mistral" in model_name or "mistral" in model_name:
            relay = list(range(12, 24))
            control = [10, 28]
        else:
            relay = list(range(11, 22))
            control = [9, 25]
    elif "14B" in model_name or "14b" in model_name:
        # Qwen 2.5 14B has 40 layers
        # Scale proportionally: relay ≈ L16-L30, control L13 + L36
        relay = list(range(16, 31))
        control = [13, 36]
    elif "32B" in model_name or "32b" in model_name:
        # Qwen 2.5 32B has 64 layers
        # Scale proportionally: relay ≈ L25-L46, control L20 + L58
        relay = list(range(25, 47))
        control = [20, 58]
    elif "72B" in model_name or "72b" in model_name:
        relay = list(range(30, 58))
        control = [25, 72]
    else:
        # Generic fallback: relay in middle 40%, control at 33% and 90%
        relay_start = int(n_layers * 0.35)
        relay_end = int(n_layers * 0.75)
        relay = list(range(relay_start, relay_end))
        control = [int(n_layers * 0.3), int(n_layers * 0.88)]
    # Clamp to actual layer count
    relay = [l for l in relay if l < n_layers]
    control = [l for l in control if l < n_layers]
    return {"relay": relay, "control": control, "all": sorted(set(relay + control))}


def participation_ratio(eigenvalues):
    eigenvalues = np.abs(eigenvalues)
    total = eigenvalues.sum()
    if total < 1e-12:
        return 1.0
    return float((total ** 2) / (eigenvalues ** 2).sum())


def spectral_summary(eigenvalues):
    eigenvalues = np.abs(eigenvalues)
    total = eigenvalues.sum()
    if total < 1e-12:
        return {"total_energy": 0.0, "spectral_entropy": 0.0, "effective_rank": 0.0,
                "participation_ratio": 1.0}
    probs = eigenvalues / total
    probs = probs[probs > 1e-12]
    entropy = float(-np.sum(probs * np.log(probs)))
    return {
        "total_energy": round(float(total), 4),
        "spectral_entropy": round(entropy, 6),
        "effective_rank": round(float(np.exp(entropy)), 4),
        "participation_ratio": round(participation_ratio(eigenvalues), 4),
    }


STRATIFIED_CATEGORY_NAMES = [
    "direct_identity", "relational", "metacognitive",
    "value_ethical", "generic_control",
]


def collect_layer_activations(model, tokenizer, prompts, system_prompt, target_layer):
    activations = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        hidden = [None]

        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                hidden[0] = out[0].detach().cpu()
            else:
                hidden[0] = out.detach().cpu()

        layer = model.model.layers[target_layer]
        handle = layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        act = hidden[0].squeeze(0).float().numpy()
        activations.append(act.mean(axis=0))
    return np.array(activations)


def measure_stratified(model, tokenizer, system_prompt, label, all_stratified, layer_config):
    print(f"\n  Measuring: {label} (sys={'yes' if system_prompt else 'no'})")
    prompts = [e["text"] for e in all_stratified]
    cat_idx = {name: [] for name in STRATIFIED_CATEGORY_NAMES}
    for i, entry in enumerate(all_stratified):
        cat_idx[entry["category"]].append(i)

    layer_metrics = {}
    for layer_idx in layer_config["all"]:
        acts = collect_layer_activations(model, tokenizer, prompts, system_prompt, layer_idx)
        acts_centered = acts - acts.mean(axis=0)

        try:
            U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)
            eigenvalues = (S ** 2) / (len(acts) - 1)
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(min(acts.shape))

        aggregate = spectral_summary(eigenvalues)
        zone = "relay" if layer_idx in layer_config["relay"] else "control"

        cat_metrics = {}
        for cat_name in STRATIFIED_CATEGORY_NAMES:
            idx = cat_idx[cat_name]
            cat_acts = acts[idx]
            cat_centered = cat_acts - cat_acts.mean(axis=0)
            try:
                _, Sc, _ = np.linalg.svd(cat_centered, full_matrices=False)
                cat_eig = (Sc ** 2) / (len(cat_acts) - 1)
            except np.linalg.LinAlgError:
                cat_eig = np.zeros(min(cat_acts.shape))
            cat_metrics[cat_name] = spectral_summary(cat_eig)

        layer_metrics[f"L{layer_idx}"] = {
            "aggregate": aggregate,
            "category_metrics": cat_metrics,
            "zone": zone,
        }

        gen_pr = cat_metrics.get("generic_control", {}).get("participation_ratio", 0)
        rel_pr = cat_metrics.get("relational", {}).get("participation_ratio", 0)
        gen_ent = cat_metrics.get("generic_control", {}).get("spectral_entropy", 0)
        rel_ent = cat_metrics.get("relational", {}).get("spectral_entropy", 0)
        tag = "*" if layer_idx in layer_config["control"] else " "
        print(f"    {tag}L{layer_idx}: gen_PR={gen_pr:.2f} rel_PR={rel_pr:.2f} | "
              f"gen_H={gen_ent:.3f} rel_H={rel_ent:.3f}")

    return layer_metrics


def run(model_name, quantize=None, output_dir="."):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load stratified prompts
    strat_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "stratified_prompts.py"),
        os.path.expanduser("~/chronicle/data/stratified_prompts.py"),
        os.path.join(os.getcwd(), "data", "stratified_prompts.py"),
        os.path.join(os.getcwd(), "stratified_prompts.py"),
    ]
    strat_path = None
    for cand in strat_candidates:
        if os.path.exists(cand):
            strat_path = os.path.realpath(cand)
            break
    if strat_path is None:
        print("ERROR: stratified_prompts.py not found")
        sys.exit(1)

    import importlib.util
    spec = importlib.util.spec_from_file_location("stratified_prompts", strat_path)
    strat_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strat_mod)
    all_stratified = strat_mod.ALL_STRATIFIED
    print(f"Loaded {len(all_stratified)} stratified prompts from {strat_path}")

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    if quantize == 4:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantize == 8:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    n_layers = len(model.model.layers)
    print(f"Model loaded: {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    layer_config = get_layer_config(model_name, n_layers)
    print(f"Layer config: relay={layer_config['relay'][0]}-{layer_config['relay'][-1]}, "
          f"control={layer_config['control']}")

    results = {
        "model": model_name,
        "experiment": "cna_scaling",
        "n_layers": n_layers,
        "n_params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
        "quantize": quantize,
        "layer_config": {
            "relay": layer_config["relay"],
            "control": layer_config["control"],
        },
        "n_prompts": len(all_stratified),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conditions": [],
    }

    # Condition 1: Baseline (no system prompt)
    print("\n=== BASELINE (no CCS) ===")
    baseline = measure_stratified(model, tokenizer, None, "baseline", all_stratified, layer_config)
    results["conditions"].append({"condition": "baseline", "layers": baseline})

    gc.collect()
    torch.cuda.empty_cache()

    # Condition 2: Full CCS
    print("\n=== FULL CCS ===")
    ccs_full = measure_stratified(model, tokenizer, CCS_FULL, "ccs_full", all_stratified, layer_config)
    results["conditions"].append({"condition": "ccs_full", "layers": ccs_full})

    gc.collect()
    torch.cuda.empty_cache()

    # Condition 3: Minimal CCS ("You are Opus.")
    print("\n=== MINIMAL CCS ===")
    ccs_min = measure_stratified(model, tokenizer, CCS_MINIMAL, "ccs_minimal", all_stratified, layer_config)
    results["conditions"].append({"condition": "ccs_minimal", "layers": ccs_min})

    # Summary
    ctrl_layer = f"L{layer_config['control'][-1]}"
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name} @ expression layer {ctrl_layer}")
    print(f"{'='*60}")
    for cond in results["conditions"]:
        name = cond["condition"]
        layer = cond["layers"].get(ctrl_layer, {})
        cats = layer.get("category_metrics", {})
        gen_pr = cats.get("generic_control", {}).get("participation_ratio", "?")
        rel_pr = cats.get("relational", {}).get("participation_ratio", "?")
        gen_h = cats.get("generic_control", {}).get("spectral_entropy", "?")
        rel_h = cats.get("relational", {}).get("spectral_entropy", "?")
        print(f"  {name:15s}: gen_PR={gen_pr:>7} rel_PR={rel_pr:>7} | gen_H={gen_h:>7} rel_H={rel_h:>7}")

    # Save
    safe_name = model_name.replace("/", "_").replace("-", "_")
    out_path = os.path.join(output_dir, f"cna_scaling_{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CNA scaling experiment")
    p.add_argument("--model", required=True, help="HuggingFace model name")
    p.add_argument("--quantize", type=int, choices=[4, 8], help="Quantization bits")
    p.add_argument("--output-dir", default=".", help="Output directory")
    args = p.parse_args()
    run(args.model, quantize=args.quantize, output_dir=args.output_dir)
