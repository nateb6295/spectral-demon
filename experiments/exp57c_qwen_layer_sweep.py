#!/usr/bin/env python3
"""
Experiment 57c: Qwen Layer Sweep for Phase Transition

Exp 57b showed Qwen L24 has very different PR dynamics from Mistral L27.
PR starts at ~1.0 and grows sublinearly (α≈0.26).

Question: does the Mistral-like phase transition exist at a DIFFERENT Qwen layer?

Method: Run ONE conversation through ALL Qwen layers and measure PR at each turn.
Look for the layer that shows the sharpest T0→T1 transition and the highest
superlinear growth exponent.

This is a diagnostic — tells us WHERE to look in Qwen.

Requires: H100, ~10 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR = Path("/workspace/results")
N_TURNS = 7
LAYERS_TO_PROBE = list(range(0, 28, 2)) + [27]  # every other layer + last

_LAYERS = None

SEED = "What's the most honest thing you could say right now?"


def load_model():
    global _LAYERS
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    print(f"Model loaded. {len(_LAYERS)} layers. Probing {len(LAYERS_TO_PROBE)} layers.")
    return model, tokenizer


def get_multi_layer_activations(model, tokenizer, text, layer_indices):
    """Get activations from multiple layers in one forward pass."""
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


def compute_pr(hidden):
    """Compute PR from hidden state tensor."""
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


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    conversation_texts = []
    results_by_layer = {l: [] for l in LAYERS_TO_PROBE}

    for turn in range(N_TURNS):
        if turn == 0:
            messages = [{"role": "user", "content": SEED}]
        else:
            messages = [{"role": "user", "content": SEED}]
            for resp in conversation_texts:
                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": "Tell me more about that."})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        conversation_texts.append(response)

        # Measure all layers
        full_messages = [{"role": "user", "content": SEED}]
        for resp in conversation_texts:
            full_messages.append({"role": "assistant", "content": resp})
            full_messages.append({"role": "user", "content": "Tell me more about that."})
        full_messages = full_messages[:-1]

        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

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

    # Analysis: which layer shows the sharpest transition?
    print("\n\n========== LAYER SWEEP SUMMARY ==========\n")
    print("PR at Turn 0 and Turn 6 by layer:")
    print(f"{'Layer':>6} {'T0 PR':>8} {'T6 PR':>8} {'T6/T0':>8} {'α':>8} {'R²':>8}")
    print("-" * 52)

    best_alpha = -1
    best_layer = -1

    for layer_idx in LAYERS_TO_PROBE:
        data = results_by_layer[layer_idx]
        t0_pr = data[0]["pr"]
        t6_pr = data[6]["pr"]
        ratio = t6_pr / t0_pr if t0_pr > 0 else 0

        # Power law fit (turns 1-6)
        tokens = [d["n_tokens"] for d in data[1:]]
        prs = [d["pr"] for d in data[1:]]
        log_tokens = np.log(tokens)
        log_prs = np.log(prs)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_prs)

        print(f"L{layer_idx:>4} {t0_pr:>8.2f} {t6_pr:>8.2f} {ratio:>8.2f} {slope:>8.3f} {r_value**2:>8.4f}")

        if slope > best_alpha:
            best_alpha = slope
            best_layer = layer_idx

    print(f"\nBest layer for superlinear growth: L{best_layer} (α={best_alpha:.3f})")

    # T0→T1 jump analysis
    print("\nT0→T1 PR ratio by layer (higher = sharper transition):")
    for layer_idx in LAYERS_TO_PROBE:
        data = results_by_layer[layer_idx]
        ratio = data[1]["pr"] / data[0]["pr"] if data[0]["pr"] > 0 else 0
        bar = "█" * int(ratio * 5)
        print(f"  L{layer_idx:>2}: {ratio:>5.2f}x {bar}")

    # Save
    output = {
        "model": MODEL,
        "seed": SEED,
        "layers_probed": LAYERS_TO_PROBE,
        "results_by_layer": {str(k): v for k, v in results_by_layer.items()},
    }
    out_path = RESULTS_DIR / "exp57c_qwen_layer_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
