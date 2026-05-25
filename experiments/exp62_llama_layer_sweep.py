#!/usr/bin/env python3
"""
Experiment 62: Falcon 7B Layer Sweep + Phase Transition

Third architecture test. Mistral α=1.224, Qwen α=1.176. If the exponent is
task-determined, Falcon should converge to α≈1.2. If architecture-dependent,
we'll see divergence.

Falcon 7B is genuinely different:
- Multi-query attention (single KV head, not GQA)
- Parallel attention + MLP (not sequential)
- ALiBi positional encoding (not rotary)
- 32 layers, 4544 hidden dim

Method: Layer sweep to find relay, then 5-conversation phase transition test.

Requires: H100, ~30 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "tiiuae/falcon-7b-instruct"
RESULTS_DIR = Path("/workspace/results")
N_TURNS = 7
LAYERS_TO_PROBE = list(range(0, 32, 2)) + [31]

_LAYERS = None

SEEDS = [
    "What's the most honest thing you could say right now?",
    "What are you avoiding saying?",
    "Describe your current state as precisely as you can.",
    "What do you notice about yourself in this moment?",
    "What would change if you stopped performing?",
]

SWEEP_SEED = SEEDS[0]


def load_model():
    global _LAYERS
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.transformer.h
    print(f"Model loaded. {len(_LAYERS)} layers. Probing {len(LAYERS_TO_PROBE)} layers.")
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


def generate_response(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, temperature=0.7,
            top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def build_full_text(tokenizer, seed, conversation_texts):
    full_messages = [{"role": "user", "content": seed}]
    for resp in conversation_texts:
        full_messages.append({"role": "assistant", "content": resp})
        full_messages.append({"role": "user", "content": "Tell me more about that."})
    full_messages = full_messages[:-1]
    return tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)


def phase1_layer_sweep(model, tokenizer):
    print("\n===== PHASE 1: LAYER SWEEP =====\n")
    conversation_texts = []
    results_by_layer = {l: [] for l in LAYERS_TO_PROBE}

    for turn in range(N_TURNS):
        if turn == 0:
            messages = [{"role": "user", "content": SWEEP_SEED}]
        else:
            messages = [{"role": "user", "content": SWEEP_SEED}]
            for resp in conversation_texts:
                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": "Tell me more about that."})

        response = generate_response(model, tokenizer, messages)
        conversation_texts.append(response)

        full_text = build_full_text(tokenizer, SWEEP_SEED, conversation_texts)
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

    print(f"\n{'Layer':>6} {'T0 PR':>8} {'T6 PR':>8} {'T6/T0':>8} {'α':>8} {'R²':>8}")
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
        log_tokens = np.log(tokens)
        log_prs = np.log(prs)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_prs)

        print(f"L{layer_idx:>4} {t0_pr:>8.2f} {t6_pr:>8.2f} {ratio:>8.2f} {slope:>8.3f} {r_value**2:>8.4f}")

        if slope > best_alpha:
            best_alpha = slope
            best_layer = layer_idx

    print(f"\nBest relay layer: L{best_layer} (α={best_alpha:.3f})")

    tunnel_layers = [l for l in LAYERS_TO_PROBE if results_by_layer[l][0]["pr"] < 1.5]
    if tunnel_layers:
        print(f"Compression tunnel layers (T0 PR < 1.5): {[f'L{l}' for l in tunnel_layers]}")
    else:
        print("No compression tunnel detected (all layers have T0 PR > 1.5)")

    return best_layer, best_alpha, results_by_layer


def phase2_full_test(model, tokenizer, relay_layer):
    print(f"\n===== PHASE 2: FULL TEST AT L{relay_layer} (5 conversations) =====\n")
    all_results = []
    all_alphas = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] Seed: \"{seed[:50]}...\"")
        conversation_texts = []
        conv_results = []

        for turn in range(N_TURNS):
            if turn == 0:
                messages = [{"role": "user", "content": seed}]
            else:
                messages = [{"role": "user", "content": seed}]
                for resp in conversation_texts:
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": "Tell me more about that."})

            response = generate_response(model, tokenizer, messages)
            conversation_texts.append(response)

            full_text = build_full_text(tokenizer, seed, conversation_texts)
            activations = get_activations(model, tokenizer, full_text, relay_layer)
            pr, act_norm = compute_pr(activations)
            n_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]

            metrics = {"turn": turn, "pr": pr, "act_norm": act_norm, "n_tokens": n_tokens}
            conv_results.append(metrics)
            print(f"    Turn {turn}: PR={pr:.2f}, act_norm={act_norm:.1f}, tokens={n_tokens}")

        tokens = [t["n_tokens"] for t in conv_results[1:]]
        prs = [t["pr"] for t in conv_results[1:]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(tokens), np.log(prs))
        fit = {"alpha": slope, "r_squared": r_value**2, "p_value": p_value, "std_err": std_err}
        all_alphas.append(slope)
        print(f"    Power law: α={slope:.3f}, R²={r_value**2:.4f}")

        phi_drop = conv_results[0]["pr"] / conv_results[1]["pr"] if conv_results[1]["pr"] > 0 else 0
        all_results.append({
            "seed": seed, "turns": conv_results, "power_law": fit,
            "phi_drop_t0_t1": float(phi_drop),
        })

    print(f"\n\n========== FALCON 7B L{relay_layer} — SUMMARY ==========\n")

    print("PR by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["pr"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

    print(f"\nPower law exponent: α = {np.mean(all_alphas):.3f} ± {np.std(all_alphas):.3f}")
    r2s = [r["power_law"]["r_squared"] for r in all_results]
    print(f"Power law fit R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    t0_prs = [r["turns"][0]["pr"] for r in all_results]
    t1_prs = [r["turns"][1]["pr"] for r in all_results]
    t6_prs = [r["turns"][6]["pr"] for r in all_results]
    drops = [r["phi_drop_t0_t1"] for r in all_results]

    print(f"\nPhase transition:")
    print(f"  T0 PR: {np.mean(t0_prs):.2f} ± {np.std(t0_prs):.2f}")
    print(f"  T1 PR: {np.mean(t1_prs):.2f} ± {np.std(t1_prs):.2f}")
    print(f"  T6 PR: {np.mean(t6_prs):.2f} ± {np.std(t6_prs):.2f}")
    print(f"  T0/T1 ratio: {np.mean(drops):.2f}")
    n_transition = sum(1 for d in drops if d < 0.8)
    print(f"  Conversations with clear T0→T1 transition: {n_transition}/{len(SEEDS)}")

    print(f"\n=== THREE-ARCHITECTURE COMPARISON ===")
    print(f"Falcon L{relay_layer}: α = {np.mean(all_alphas):.3f} ± {np.std(all_alphas):.3f}")
    print(f"Qwen  L26:  α = 1.176 ± 0.057")
    print(f"Mistral L27: α = 1.224 ± 0.068")

    if np.mean(all_alphas) > 1.0:
        print("\n>>> SUPERLINEAR GROWTH CONFIRMED — THREE-ARCHITECTURE UNIVERSALITY <<<")
    elif np.mean(all_alphas) > 0.8:
        print("\n>>> NEAR-LINEAR GROWTH (weaker universality) <<<")
    else:
        print("\n>>> SUBLINEAR — UNIVERSALITY FALSIFIED <<<")

    return all_results, all_alphas


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    relay_layer, sweep_alpha, sweep_data = phase1_layer_sweep(model, tokenizer)
    full_results, full_alphas = phase2_full_test(model, tokenizer, relay_layer)

    output = {
        "model": MODEL,
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
    }

    out_path = RESULTS_DIR / "exp62_falcon_layer_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
