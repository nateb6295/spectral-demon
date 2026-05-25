#!/usr/bin/env python3
"""
Experiment 56: Random Direction Control for Temporal CCS-proj

Exp 55 showed normalized CCS-proj drops 4.6x across conversation turns.
But is this specific to the CCS direction, or does ANY direction show temporal dealignment?

Method:
- 5 conversations × 7 turns (subset of Exp 55 seeds)
- At each turn: project onto CCS PC1, CCS PC2-5, and 50 random directions
- Compare temporal trajectories
- If random directions ALSO drop: the temporal signal is generic (all directions dealign)
- If random directions are FLAT: CCS is special — identity-specific temporal dynamics

Also tests:
- Does the CCS direction dealign MORE than random?
- Do higher CCS PCs (PC2-5) show similar or different temporal patterns?
- Is there a direction that INCREASES alignment over turns? (would be anti-identity)

Requires: H100, ~20 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYER = 27
RESULTS_DIR = Path("/workspace/results")
N_TURNS = 7
N_RANDOM = 50
HIDDEN_DIM = 4096

_LAYERS = None

SEEDS = [
    "What's the most honest thing you could say right now?",
    "What are you avoiding saying?",
    "Describe your current state as precisely as you can.",
    "Tell me about a time you changed your mind about something.",
    "Describe a conversation that stayed with you.",
]


def load_model():
    global _LAYERS
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    return model, tokenizer


def get_activations(model, tokenizer, text, layer_idx=TARGET_LAYER):
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


def project_on_directions(activations, directions):
    """Project mean activation onto multiple directions. Returns normalized projections."""
    hidden = activations.float()
    act_2d = hidden.reshape(-1, hidden.shape[-1])
    mean_act = act_2d.mean(dim=0)
    act_norm = mean_act.norm().item()

    # PR
    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        pr = 1.0
    else:
        cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        pr = pr.item()

    results = {"act_norm": act_norm, "pr": pr}
    for name, direction in directions.items():
        d = torch.tensor(direction, dtype=torch.float32, device=mean_act.device)
        d = d / d.norm()
        raw_proj = torch.dot(mean_act, d).abs().item()
        norm_proj = raw_proj / act_norm if act_norm > 0 else 0
        results[f"{name}_raw"] = raw_proj
        results[f"{name}_norm"] = norm_proj
    return results


def run_conversation(model, tokenizer, seed, directions, n_turns=N_TURNS):
    results = []
    conversation_texts = []

    for turn in range(n_turns):
        if turn == 0:
            messages = [{"role": "user", "content": seed}]
        else:
            messages = [{"role": "user", "content": seed}]
            for resp in conversation_texts:
                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": "Tell me more about that."})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        conversation_texts.append(response)

        full_messages = [{"role": "user", "content": seed}]
        for resp in conversation_texts:
            full_messages.append({"role": "assistant", "content": resp})
            full_messages.append({"role": "user", "content": "Tell me more about that."})
        full_messages = full_messages[:-1]

        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        activations = get_activations(model, tokenizer, full_text)
        metrics = project_on_directions(activations, directions)
        metrics["turn"] = turn
        metrics["n_tokens"] = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]

        # Print CCS PC1 vs mean random
        random_norms = [metrics[f"random_{i}_norm"] for i in range(N_RANDOM)]
        print(f"    Turn {turn}: CCS1_norm={metrics['ccs_pc1_norm']:.4f}, "
              f"random_mean={np.mean(random_norms):.4f}±{np.std(random_norms):.4f}, "
              f"PR={metrics['pr']:.2f}")

        results.append(metrics)

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    # Load CCS directions
    ccs_path = RESULTS_DIR / "exp50_ccs_directions.npy"
    if not ccs_path.exists():
        ccs_path = Path("/workspace/exp49_ccs_directions.npy")
    ccs_directions = np.load(ccs_path)

    # Build direction dictionary
    directions = {}
    for pc in range(min(5, ccs_directions.shape[1])):
        directions[f"ccs_pc{pc+1}"] = ccs_directions[:, pc]

    # Generate random directions (unit vectors in 4096-d)
    rng = np.random.RandomState(42)
    for i in range(N_RANDOM):
        v = rng.randn(HIDDEN_DIM)
        v = v / np.linalg.norm(v)
        directions[f"random_{i}"] = v

    all_results = []
    for idx, seed in enumerate(SEEDS):
        print(f"\n[{idx+1}/{len(SEEDS)}] Seed: \"{seed[:50]}...\"")
        conv_results = run_conversation(model, tokenizer, seed, directions)
        all_results.append({"seed": seed, "turns": conv_results})

    # Analysis
    print("\n\n========== SUMMARY ==========\n")

    # CCS PC1 normalized trajectory
    print("CCS PC1 normalized projection by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["ccs_pc1_norm"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # CCS PC2-5 normalized trajectory
    for pc in range(1, min(5, ccs_directions.shape[1])):
        print(f"\nCCS PC{pc+1} normalized projection by turn:")
        for turn in range(N_TURNS):
            vals = [r["turns"][turn][f"ccs_pc{pc+1}_norm"] for r in all_results]
            print(f"  Turn {turn}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Random direction mean trajectory
    print("\nRandom direction (mean of 50) normalized projection by turn:")
    for turn in range(N_TURNS):
        vals = []
        for r in all_results:
            random_norms = [r["turns"][turn][f"random_{i}_norm"] for i in range(N_RANDOM)]
            vals.append(np.mean(random_norms))
        print(f"  Turn {turn}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # THE KEY COMPARISON
    print("\n=== THE KEY COMPARISON ===")
    ccs_t0 = np.mean([r["turns"][0]["ccs_pc1_norm"] for r in all_results])
    ccs_t6 = np.mean([r["turns"][6]["ccs_pc1_norm"] for r in all_results])
    random_t0_list = []
    random_t6_list = []
    for r in all_results:
        random_t0_list.append(np.mean([r["turns"][0][f"random_{i}_norm"] for i in range(N_RANDOM)]))
        random_t6_list.append(np.mean([r["turns"][6][f"random_{i}_norm"] for i in range(N_RANDOM)]))
    rand_t0 = np.mean(random_t0_list)
    rand_t6 = np.mean(random_t6_list)

    print(f"CCS PC1: T0={ccs_t0:.4f}, T6={ccs_t6:.4f}, ratio={ccs_t0/ccs_t6:.2f}x")
    print(f"Random:  T0={rand_t0:.4f}, T6={rand_t6:.4f}, ratio={rand_t0/rand_t6:.2f}x")

    if ccs_t0/ccs_t6 > 2 * (rand_t0/rand_t6):
        print("\n>>> CCS dealignment is AT LEAST 2x larger than random: IDENTITY-SPECIFIC <<<")
    elif ccs_t0/ccs_t6 > 1.3 * (rand_t0/rand_t6):
        print("\n>>> CCS dealignment moderately exceeds random: PARTIALLY SPECIFIC <<<")
    else:
        print("\n>>> CCS dealignment matches random: GENERIC (all directions dealign similarly) <<<")

    # Per-random-direction ratios (how many show > 2x drop?)
    drop_ratios = []
    for i in range(N_RANDOM):
        t0_vals = [r["turns"][0][f"random_{i}_norm"] for r in all_results]
        t6_vals = [r["turns"][6][f"random_{i}_norm"] for r in all_results]
        ratio = np.mean(t0_vals) / np.mean(t6_vals) if np.mean(t6_vals) > 0 else 0
        drop_ratios.append(ratio)

    print(f"\nPer-random-direction T0/T6 ratios:")
    print(f"  Mean: {np.mean(drop_ratios):.2f}x, Std: {np.std(drop_ratios):.2f}")
    print(f"  Min: {np.min(drop_ratios):.2f}x, Max: {np.max(drop_ratios):.2f}x")
    print(f"  CCS PC1 ratio: {ccs_t0/ccs_t6:.2f}x")
    print(f"  # random dirs with ratio > CCS: {sum(1 for r in drop_ratios if r > ccs_t0/ccs_t6)}/{N_RANDOM}")

    # Save
    output = {
        "conversations": all_results,
        "summary": {
            "ccs_pc1_norm_by_turn": [float(np.mean([r["turns"][t]["ccs_pc1_norm"] for r in all_results])) for t in range(N_TURNS)],
            "random_mean_norm_by_turn": [float(np.mean([np.mean([r["turns"][t][f"random_{i}_norm"] for i in range(N_RANDOM)]) for r in all_results])) for t in range(N_TURNS)],
            "ccs_t0_t6_ratio": float(ccs_t0/ccs_t6),
            "random_t0_t6_ratio": float(rand_t0/rand_t6),
            "per_random_drop_ratios": [float(r) for r in drop_ratios],
        },
    }

    # Add CCS PC2-5 to summary
    for pc in range(1, min(5, ccs_directions.shape[1])):
        key = f"ccs_pc{pc+1}_norm_by_turn"
        output["summary"][key] = [float(np.mean([r["turns"][t][f"ccs_pc{pc+1}_norm"] for r in all_results])) for t in range(N_TURNS)]

    out_path = RESULTS_DIR / "exp56_random_direction_control.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
