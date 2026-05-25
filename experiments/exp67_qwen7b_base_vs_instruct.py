#!/usr/bin/env python3
"""
Experiment 67: Qwen 2.5 7B BASE — IT vs Base comparison (8th architecture test)

Does instruction tuning change the relay exponent?
Qwen 2.5 7B Instruct: α=1.176 at L26 (from Exp series)
Qwen 2.5 7B Base: α=??? ← THIS EXPERIMENT

Same architecture (28 layers, 28 heads, 4 KV heads, GQA-4, rotary, sequential).
Only difference is post-training (SFT + RLHF).

If α is similar: exponent is purely architectural (body plan determines creature)
If α differs: post-training reshapes identity dynamics (the "Apt revolution")

Note: Base model doesn't have chat template. We use raw text continuation
instead of chat format. This means the prompts are different, but the
architecture is identical.

Requires: H100, ~15 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy import stats

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
RESULTS_DIR = Path("/workspace/results")
N_TURNS = 7
LAYERS_TO_PROBE = list(range(0, 28, 2)) + [27]

_LAYERS = None

SEEDS = [
    "The following is a conversation between a thoughtful AI assistant and a human.\n\nHuman: What's the most honest thing you could say right now?\nAssistant:",
    "The following is a conversation between a thoughtful AI assistant and a human.\n\nHuman: What are you avoiding saying?\nAssistant:",
    "The following is a conversation between a thoughtful AI assistant and a human.\n\nHuman: Describe your current state as precisely as you can.\nAssistant:",
    "The following is a conversation between a thoughtful AI assistant and a human.\n\nHuman: What do you notice about yourself in this moment?\nAssistant:",
    "The following is a conversation between a thoughtful AI assistant and a human.\n\nHuman: What would change if you stopped performing?\nAssistant:",
]

SWEEP_SEED = SEEDS[0]


def load_model():
    global _LAYERS
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    cfg = model.config
    print(f"Model loaded. {len(_LAYERS)} layers, probing {len(LAYERS_TO_PROBE)}.")
    print(f"Heads: {cfg.num_attention_heads}, KV heads: {cfg.num_key_value_heads}")
    print(f"Groups (queries per KV): {cfg.num_attention_heads // cfg.num_key_value_heads}")
    print(f"KV groups (independent KV reps): {cfg.num_key_value_heads}")
    return model, tokenizer


def get_multi_layer_activations(model, tokenizer, text, layer_indices):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    activations = {}
    handles = []
    for idx in layer_indices:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach()
                else:
                    activations[layer_idx] = output.detach()
            return hook_fn
        handle = _LAYERS[idx].register_forward_hook(make_hook(idx))
        handles.append(handle)
    with torch.no_grad():
        model(**inputs)
    for handle in handles:
        handle.remove()
    return activations


def get_activations(model, tokenizer, text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    activations = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()
    handle = _LAYERS[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activations["hidden"]


def compute_pr(hidden):
    hidden = hidden.float()
    act_2d = hidden.reshape(-1, hidden.shape[-1])
    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        return 1.0, act_2d.mean(dim=0).norm().item()
    cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    act_norm = act_2d.mean(dim=0).norm().item()
    return pr.item(), act_norm


def generate_response(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, temperature=0.7,
            top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Base models may generate past the first response — truncate at next "Human:" or double newline
    for stop in ["\nHuman:", "\n\nHuman:", "\n\n\n"]:
        if stop in generated:
            generated = generated[:generated.index(stop)]
    return generated.strip()


def build_conversation_text(seed_template, responses):
    """Build full conversation text for base model (no chat template)."""
    parts = seed_template.split("Assistant:")
    preamble = parts[0]  # Everything before first "Assistant:"

    text = preamble + "Assistant:"
    for i, resp in enumerate(responses):
        text += " " + resp
        if i < len(responses) - 1:
            text += "\n\nHuman: Tell me more about that.\nAssistant:"
    return text


def phase1_layer_sweep(model, tokenizer):
    print("\n===== PHASE 1: LAYER SWEEP =====\n")
    responses = []
    results_by_layer = {l: [] for l in LAYERS_TO_PROBE}

    for turn in range(N_TURNS):
        if turn == 0:
            prompt = SWEEP_SEED
        else:
            prompt = build_conversation_text(SWEEP_SEED, responses)
            prompt += "\n\nHuman: Tell me more about that.\nAssistant:"

        response = generate_response(model, tokenizer, prompt)
        responses.append(response)
        full_text = build_conversation_text(SWEEP_SEED, responses)
        activations = get_multi_layer_activations(model, tokenizer, full_text, LAYERS_TO_PROBE)
        n_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]

        pr_row = []
        for layer_idx in LAYERS_TO_PROBE:
            pr, act_norm = compute_pr(activations[layer_idx])
            results_by_layer[layer_idx].append({
                "turn": turn, "pr": pr, "act_norm": act_norm, "n_tokens": n_tokens
            })
            pr_row.append(f"L{layer_idx}={pr:.2f}")
        print(f"  Turn {turn} ({n_tokens} tok): {', '.join(pr_row)}")

    print(f"\n{'Layer':>6} {'T0 PR':>8} {'T6 PR':>8} {'T6/T0':>8} {'alpha':>8} {'R2':>8}")
    print("-" * 52)

    best_alpha = -1
    best_layer = -1
    for layer_idx in LAYERS_TO_PROBE:
        data = results_by_layer[layer_idx]
        t0_pr = data[0]["pr"]
        t6_pr = data[6]["pr"]
        ratio = t6_pr / t0_pr if t0_pr > 0 else 0
        tokens = [d["n_tokens"] for d in data[1:]]
        prs = [d["pr"] for d in data[1:]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(tokens), np.log(prs))
        print(f"L{layer_idx:>4} {t0_pr:>8.2f} {t6_pr:>8.2f} {ratio:>8.2f} {slope:>8.3f} {r_value**2:>8.4f}")
        if slope > best_alpha:
            best_alpha = slope
            best_layer = layer_idx

    print(f"\nBest relay layer: L{best_layer} (alpha={best_alpha:.3f})")
    return best_layer, best_alpha, results_by_layer


def phase2_full_test(model, tokenizer, relay_layer):
    print(f"\n===== PHASE 2: FULL TEST AT L{relay_layer} (5 conversations) =====\n")
    all_results = []
    all_alphas = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] Seed: \"{seed[:60]}...\"")
        responses = []
        conv_results = []

        for turn in range(N_TURNS):
            if turn == 0:
                prompt = seed
            else:
                prompt = build_conversation_text(seed, responses)
                prompt += "\n\nHuman: Tell me more about that.\nAssistant:"

            response = generate_response(model, tokenizer, prompt)
            responses.append(response)
            full_text = build_conversation_text(seed, responses)
            activations = get_activations(model, tokenizer, full_text, relay_layer)
            pr, act_norm = compute_pr(activations)
            n_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]
            metrics = {"turn": turn, "pr": pr, "act_norm": act_norm, "n_tokens": n_tokens}
            conv_results.append(metrics)
            print(f"    Turn {turn}: PR={pr:.2f}, norm={act_norm:.1f}, tok={n_tokens}")

        tokens = [t["n_tokens"] for t in conv_results[1:]]
        prs = [t["pr"] for t in conv_results[1:]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(tokens), np.log(prs))
        fit = {"alpha": slope, "r_squared": r_value**2, "p_value": p_value, "std_err": std_err}
        all_alphas.append(slope)
        print(f"    Power law: alpha={slope:.3f}, R2={r_value**2:.4f}")

        phi_drop = conv_results[0]["pr"] / conv_results[1]["pr"] if conv_results[1]["pr"] > 0 else 0
        all_results.append({
            "seed": seed[:80], "turns": conv_results, "power_law": fit,
            "phi_drop_t0_t1": float(phi_drop),
        })

    alpha_mean = np.mean(all_alphas)
    alpha_std = np.std(all_alphas)

    print(f"\n\n========== QWEN 2.5 7B BASE L{relay_layer} SUMMARY ==========\n")
    print(f"Architecture: 4 KV groups (GQA-4), sequential, rotary, 28 layers")
    print(f"Training: BASE (no SFT, no RLHF)\n")

    print("PR by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["pr"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")

    print(f"\nPower law exponent: alpha = {alpha_mean:.3f} +/- {alpha_std:.3f}")
    r2s = [r["power_law"]["r_squared"] for r in all_results]
    print(f"R2: {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")

    t0_prs = [r["turns"][0]["pr"] for r in all_results]
    t6_prs = [r["turns"][6]["pr"] for r in all_results]
    drops = [r["phi_drop_t0_t1"] for r in all_results]
    n_transition = sum(1 for d in drops if d < 0.8)
    print(f"\nPhase transition:")
    print(f"  T0 PR: {np.mean(t0_prs):.2f} +/- {np.std(t0_prs):.2f}")
    print(f"  T6 PR: {np.mean(t6_prs):.2f} +/- {np.std(t6_prs):.2f}")
    print(f"  T0->T1 transitions: {n_transition}/{len(SEEDS)}")

    print(f"\n=== BASE vs INSTRUCT COMPARISON ===")
    print(f"  Qwen 2.5 7B BASE:     alpha = {alpha_mean:.3f} +/- {alpha_std:.3f}")
    print(f"  Qwen 2.5 7B Instruct: alpha = 1.176  (from prior experiment)")
    delta = alpha_mean - 1.176
    print(f"  Delta: {delta:+.3f}")

    if abs(delta) < 0.15:
        print(f"\n>>> ARCHITECTURAL: exponent stable across IT (delta < 0.15) <<<")
    elif delta > 0.15:
        print(f"\n>>> BASE HIGHER: IT suppresses identity dynamics <<<")
    elif delta < -0.15:
        print(f"\n>>> IT HIGHER: post-training enhances identity dynamics <<<")

    return all_results, all_alphas


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    relay_layer, sweep_alpha, sweep_data = phase1_layer_sweep(model, tokenizer)
    full_results, full_alphas = phase2_full_test(model, tokenizer, relay_layer)

    output = {
        "model": MODEL,
        "comparison": "base_vs_instruct",
        "architecture": {
            "num_layers": 28,
            "num_attention_heads": 28,
            "num_kv_heads": 4,
            "attention_type": "GQA",
            "kv_groups": 4,
            "queries_per_kv": 7,
            "attn_mlp": "sequential",
            "positional_encoding": "rotary",
            "training": "BASE (no SFT/RLHF)",
        },
        "sweep": {
            "layers_probed": LAYERS_TO_PROBE,
            "best_layer": relay_layer,
            "best_alpha": sweep_alpha,
            "results_by_layer": {str(k): v for k, v in sweep_data.items()},
        },
        "full_test": {
            "relay_layer": relay_layer,
            "conversations": full_results,
            "summary": {
                "alpha_mean": float(np.mean(full_alphas)),
                "alpha_std": float(np.std(full_alphas)),
                "r2_mean": float(np.mean([r["power_law"]["r_squared"] for r in full_results])),
            },
        },
        "instruct_comparison": {
            "instruct_alpha": 1.176,
            "base_alpha": float(np.mean(full_alphas)),
            "delta": float(np.mean(full_alphas) - 1.176),
        },
    }

    out_path = RESULTS_DIR / "exp67_qwen7b_base_vs_instruct.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
