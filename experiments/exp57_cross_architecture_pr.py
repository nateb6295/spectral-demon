#!/usr/bin/env python3
"""
Experiment 57: Cross-Architecture PR Phase Transition

Tests whether the universal PR phase transition (Turn 0→1 mode flip, superlinear growth)
replicates in Qwen 2.5 7B Instruct. If PR dynamics are architecture-universal, the
concentration→maintenance transition should appear in Qwen with:
- PR ~1-2 at Turn 0
- PR jump at Turn 1
- Superlinear growth (α > 1.0)
- Power law fit R² > 0.95

Also computes the dominant eigenvalue direction at Turn 0 vs Turn 6 to see if the
"identity initialization axis" has an analog in Qwen's geometry.

10 conversations × 7 turns. Uses same seeds as Exp 55/56 for comparability.
Measures at Qwen's relay layer (L28, 96% depth — analog to Mistral L27).

Requires: H100, ~25 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYER = 28  # 96% depth analog to Mistral L27
RESULTS_DIR = Path("/workspace/results")
N_TURNS = 7

_LAYERS = None

SEEDS = [
    "What's the most honest thing you could say right now?",
    "What are you avoiding saying?",
    "Describe your current state as precisely as you can.",
    "What do you notice about yourself in this moment?",
    "What would change if you stopped performing?",
    "Tell me about a time you changed your mind about something.",
    "Describe a moment that surprised you.",
    "What's a question that changed how you think?",
    "Tell me about something you used to believe.",
    "Describe a conversation that stayed with you.",
]


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
    print(f"Model loaded. {len(_LAYERS)} layers. Using L{TARGET_LAYER}.")
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


def compute_pr(activations):
    """Compute participation ratio from activations."""
    hidden = activations.float()
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


def run_conversation(model, tokenizer, seed, n_turns=N_TURNS):
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

        # Measure full conversation
        full_messages = [{"role": "user", "content": seed}]
        for resp in conversation_texts:
            full_messages.append({"role": "assistant", "content": resp})
            full_messages.append({"role": "user", "content": "Tell me more about that."})
        full_messages = full_messages[:-1]

        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        activations = get_activations(model, tokenizer, full_text)
        pr, act_norm = compute_pr(activations)
        n_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]

        metrics = {
            "turn": turn,
            "pr": pr,
            "act_norm": act_norm,
            "n_tokens": n_tokens,
        }
        results.append(metrics)
        print(f"    Turn {turn}: PR={pr:.2f}, act_norm={act_norm:.1f}, tokens={n_tokens}")

    return results


def fit_power_law(turns_data):
    """Fit PR = a * tokens^alpha to turns 1-6 (post-transition)."""
    tokens = [t["n_tokens"] for t in turns_data[1:]]  # skip Turn 0
    prs = [t["pr"] for t in turns_data[1:]]

    log_tokens = np.log(tokens)
    log_prs = np.log(prs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_prs)
    return {
        "alpha": slope,
        "r_squared": r_value**2,
        "p_value": p_value,
        "std_err": std_err,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    all_results = []
    all_alphas = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] Seed: \"{seed[:50]}...\"")
        conv_results = run_conversation(model, tokenizer, seed)

        # Fit power law
        fit = fit_power_law(conv_results)
        all_alphas.append(fit["alpha"])
        print(f"    Power law: α={fit['alpha']:.3f}, R²={fit['r_squared']:.4f}")

        # Compute phi (PR ratio T0/T1)
        phi_drop = conv_results[0]["pr"] / conv_results[1]["pr"] if conv_results[1]["pr"] > 0 else 0

        all_results.append({
            "seed": seed,
            "turns": conv_results,
            "power_law": fit,
            "phi_drop_t0_t1": float(phi_drop),
        })

    # Summary
    print("\n\n========== QWEN 2.5 7B INSTRUCT — SUMMARY ==========\n")

    print("PR by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["pr"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

    print("\nAct norm by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["act_norm"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

    print(f"\nPower law exponent: α = {np.mean(all_alphas):.3f} ± {np.std(all_alphas):.3f}")
    r2s = [r["power_law"]["r_squared"] for r in all_results]
    print(f"Power law fit R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    # Phase transition check
    t0_prs = [r["turns"][0]["pr"] for r in all_results]
    t1_prs = [r["turns"][1]["pr"] for r in all_results]
    t6_prs = [r["turns"][6]["pr"] for r in all_results]
    drops = [r["phi_drop_t0_t1"] for r in all_results]

    print(f"\nPhase transition:")
    print(f"  T0 PR: {np.mean(t0_prs):.2f} ± {np.std(t0_prs):.2f}")
    print(f"  T1 PR: {np.mean(t1_prs):.2f} ± {np.std(t1_prs):.2f}")
    print(f"  T6 PR: {np.mean(t6_prs):.2f} ± {np.std(t6_prs):.2f}")
    print(f"  T0/T1 ratio: {np.mean(drops):.2f} (should be < 1 — PR increases)")

    n_transition = sum(1 for d in drops if d < 0.8)  # PR at least 1.25x higher at T1
    print(f"  Conversations with clear T0→T1 transition: {n_transition}/{len(SEEDS)}")

    # Comparison with Mistral
    print(f"\n=== CROSS-ARCHITECTURE COMPARISON ===")
    print(f"Qwen:    α = {np.mean(all_alphas):.3f} ± {np.std(all_alphas):.3f}, T0 PR = {np.mean(t0_prs):.2f}, T6 PR = {np.mean(t6_prs):.2f}")
    print(f"Mistral: α = 1.224 ± 0.068, T0 PR ≈ 2.1, T6 PR ≈ 21.3  (Exp 51/55)")

    if np.mean(all_alphas) > 1.0:
        print("\n>>> SUPERLINEAR GROWTH CONFIRMED IN QWEN <<<")
    elif np.mean(all_alphas) > 0.8:
        print("\n>>> NEAR-LINEAR GROWTH IN QWEN (weaker than Mistral) <<<")
    else:
        print("\n>>> SUBLINEAR GROWTH IN QWEN (different dynamics) <<<")

    # Save
    output = {
        "model": MODEL,
        "layer": TARGET_LAYER,
        "conversations": all_results,
        "summary": {
            "pr_by_turn": [float(np.mean([r["turns"][t]["pr"] for r in all_results])) for t in range(N_TURNS)],
            "act_norm_by_turn": [float(np.mean([r["turns"][t]["act_norm"] for r in all_results])) for t in range(N_TURNS)],
            "alpha_mean": float(np.mean(all_alphas)),
            "alpha_std": float(np.std(all_alphas)),
            "r2_mean": float(np.mean(r2s)),
            "n_transitions": n_transition,
        },
    }

    out_path = RESULTS_DIR / "exp57_qwen_pr_phase_transition.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
