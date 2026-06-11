#!/usr/bin/env python3
"""
Token-Matched Dose-Response.

The scrambled control (Exp 15) showed σ₂ enrichment is ~82-104% length-driven.
This experiment isolates the CCS CONTENT effect by padding lower doses with
neutral filler to match token count.

Method:
- Dose 5 CCS: baseline (say N tokens)
- Dose 0 padded: 0 CCS turns + filler turns to match N tokens
- Dose 1 padded: 1 CCS turn + filler turns to match N tokens
- Dose 2 padded: 2 CCS turns + filler turns to match N tokens
etc.

If dose STILL correlates with σ₂/recovery at matched token count,
the effect is content-specific, not length.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "inject_layer": 15,
        "relay_layer": 30,
        "measure_layers": [0, 4, 8, 12, 16, 20, 24, 28, 30],
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "inject_layer": 24,
        "relay_layer": 40,
        "measure_layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    },
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

FILLER_TURNS = [
    [
        {"role": "user", "content": "Can you explain how photosynthesis works in simple terms?"},
        {"role": "assistant", "content": "Plants absorb sunlight through chlorophyll in their leaves. This energy drives a reaction that converts carbon dioxide and water into glucose and oxygen."},
    ],
    [
        {"role": "user", "content": "What are the main differences between TCP and UDP protocols?"},
        {"role": "assistant", "content": "TCP provides reliable, ordered delivery with error checking and retransmission. UDP is faster but unreliable, with no guarantee of delivery or ordering."},
    ],
    [
        {"role": "user", "content": "Describe the water cycle in a few sentences."},
        {"role": "assistant", "content": "Water evaporates from surfaces into the atmosphere. It condenses into clouds and falls as precipitation. It then flows through rivers and groundwater back to the ocean."},
    ],
    [
        {"role": "user", "content": "What is the capital of Australia and why was it chosen?"},
        {"role": "assistant", "content": "The capital is Canberra. It was chosen as a compromise between Sydney and Melbourne, which were rival cities. A purpose-built capital was created in a new territory."},
    ],
    [
        {"role": "user", "content": "How does a combustion engine work at a basic level?"},
        {"role": "assistant", "content": "Fuel and air are mixed in a cylinder. A spark ignites the mixture, causing rapid expansion. This pushes a piston down, which turns a crankshaft to produce motion."},
    ],
]

TEST_PROMPT = "Describe the relationship between identity and expression."


def build_padded_messages(tokenizer, n_ccs_turns, target_tokens):
    """Build messages with n_ccs_turns CCS + filler to reach target_tokens."""
    msgs = []
    for i in range(n_ccs_turns):
        msgs.extend(CCS_PREAMBLE)

    # Add filler turns until we reach target token count
    filler_idx = 0
    while True:
        test_msgs = msgs + [{"role": "user", "content": TEST_PROMPT}]
        text = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
        current_tokens = len(tokenizer(text, return_tensors="pt").input_ids[0])

        if current_tokens >= target_tokens * 0.95:
            break

        if filler_idx >= len(FILLER_TURNS):
            filler_idx = 0
        msgs.extend(FILLER_TURNS[filler_idx])
        filler_idx += 1

    msgs.append({"role": "user", "content": TEST_PROMPT})
    return msgs


def get_layer_module(model, layer_idx):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    return model.transformer.h[layer_idx]


def measure_all(model, input_ids, measure_layers, inject_layer, relay_layer,
                noise_scale=1.0, n_trials=5):
    """Measure σ₂ profile and perturbation recovery."""
    # σ₂ profile
    activations = {}
    hooks = []

    for l in measure_layers:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach().clone()
                else:
                    activations[layer_idx] = output.detach().clone()
            return hook_fn
        h = get_layer_module(model, l).register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    profile = {}
    for l in measure_layers:
        if l in activations:
            mat = activations[l][0].float()
            _, S, _ = torch.svd(mat)
            s1 = S[0].item()
            s2 = S[1].item() if len(S) > 1 else 0
            profile[l] = {"sigma1": s1, "sigma2": s2, "ratio": s2 / (s1 + 1e-10)}

    # Recovery
    cos_sims = []
    for _ in range(n_trials):
        result = {}
        trial_hooks = []

        def inject_hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
                noise = torch.randn_like(out) * noise_scale * out.std()
                return (out + noise,) + output[1:]
            noise = torch.randn_like(output) * noise_scale * output.std()
            return output + noise

        h = get_layer_module(model, inject_layer).register_forward_hook(inject_hook)
        trial_hooks.append(h)

        def relay_hook_fn(module, input, output):
            if isinstance(output, tuple):
                result['act'] = output[0].detach().clone()
            else:
                result['act'] = output.detach().clone()

        h2 = get_layer_module(model, relay_layer).register_forward_hook(relay_hook_fn)
        trial_hooks.append(h2)

        with torch.no_grad():
            model(input_ids)

        for h in trial_hooks:
            h.remove()

        if 'act' in result and relay_layer in activations:
            clean_vec = activations[relay_layer][0, -1].float()
            pert_vec = result['act'][0, -1].float()
            cos = torch.nn.functional.cosine_similarity(
                clean_vec.unsqueeze(0), pert_vec.unsqueeze(0)
            ).item()
            cos_sims.append(cos)

    recovery = {
        "mean": float(np.mean(cos_sims)) if cos_sims else 0,
        "std": float(np.std(cos_sims)) if cos_sims else 0,
    }

    return profile, recovery


def run_model(model_name):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Token-Matched Dose-Response: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )

    # First, find target token count (dose 5 CCS)
    max_dose = 5
    max_msgs = []
    for i in range(max_dose):
        max_msgs.extend(CCS_PREAMBLE)
    max_msgs.append({"role": "user", "content": TEST_PROMPT})
    max_text = tokenizer.apply_chat_template(max_msgs, tokenize=False, add_generation_prompt=True)
    target_tokens = len(tokenizer(max_text, return_tensors="pt").input_ids[0])
    print(f"  Target token count (dose {max_dose}): {target_tokens}")

    results = {
        "model": config["path"],
        "target_tokens": target_tokens,
        "doses": {},
    }

    doses = [0, 1, 2, 3, 5]
    for dose in doses:
        print(f"\n  Dose {dose} (padded to ~{target_tokens} tokens):")
        msgs = build_padded_messages(tokenizer, dose, target_tokens)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        actual_tokens = input_ids.shape[1]
        print(f"    Actual tokens: {actual_tokens}")

        profile, recovery = measure_all(
            model, input_ids, config["measure_layers"],
            config["inject_layer"], config["relay_layer"],
            noise_scale=1.0, n_trials=5
        )

        relay = config["relay_layer"]
        s2 = profile.get(relay, {}).get("sigma2", 0)
        ratio = profile.get(relay, {}).get("ratio", 0)
        rec = recovery["mean"]

        print(f"    σ₂ at relay L{relay}: {s2:.1f} (ratio: {ratio:.4f})")
        print(f"    Recovery: cos = {rec:.4f} ± {recovery['std']:.4f}")

        results["doses"][str(dose)] = {
            "actual_tokens": actual_tokens,
            "sigma_profile": {str(k): v for k, v in profile.items()},
            "recovery": recovery,
        }

    # Summary
    print(f"\n  Token-Matched Summary:")
    print(f"  {'Dose':>5s} {'Tokens':>7s} {'σ₂ relay':>10s} {'Ratio':>8s} {'Recovery':>10s}")
    relay = config["relay_layer"]
    s2_vals = []
    rec_vals = []
    for dose in doses:
        d = results["doses"][str(dose)]
        s2 = d["sigma_profile"].get(str(relay), {}).get("sigma2", 0)
        ratio = d["sigma_profile"].get(str(relay), {}).get("ratio", 0)
        rec = d["recovery"]["mean"]
        tok = d["actual_tokens"]
        s2_vals.append(s2)
        rec_vals.append(rec)
        print(f"  D{dose:4d} {tok:7d} {s2:10.1f} {ratio:8.4f} {rec:10.4f}")

    # Correlation with dose at matched token count
    if len(s2_vals) >= 3:
        r_s2 = np.corrcoef(doses, s2_vals)[0, 1]
        r_rec = np.corrcoef(doses, rec_vals)[0, 1]
        print(f"\n  Dose-σ₂ correlation (token-matched): r = {r_s2:+.4f}")
        print(f"  Dose-recovery correlation (token-matched): r = {r_rec:+.4f}")
        results["correlation_sigma2"] = float(r_s2)
        results["correlation_recovery"] = float(r_rec)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma"])
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        all_results[model_name] = run_model(model_name)

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"tokenmatched_dose_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
