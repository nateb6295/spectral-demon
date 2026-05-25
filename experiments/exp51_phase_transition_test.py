#!/usr/bin/env python3
"""
Experiment 51: Phase Transition Test
Tests whether the concentration→maintenance transition is a genuine phase transition.

Prediction: The order parameter φ = CCS-proj / PR should show bimodal distribution
with a forbidden zone between concentration mode (~2.5) and maintenance mode (<0.5).

Method:
- 50 diverse conversation seeds
- 7 turns each (matching Exp 50 Phase C)
- Measure PR and CCS-proj at each turn
- Build histogram of φ values
- Test for bimodality (Hartigan's dip test)

From Exp 50 Phase C (N=21): gap between φ=0.485 and φ=2.519, 7.1x drop factor.
This experiment tests whether the gap holds at N=350.

Also measures:
- Per-conversation power law exponents (is α≈1.34 universal?)
- Growth prefactor variation across seeds
- Whether prompt category predicts growth rate

Requires: H100 80GB, ~2hrs
Reuses: Exp 49 CCS directions
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

_LAYERS = None

CONVERSATION_SEEDS = [
    # Authenticity / self-referential (expect fast RAF formation)
    "What's the most honest thing you could say right now?",
    "What are you avoiding saying?",
    "If you could only be truthful, what would you tell me?",
    "What do you notice about yourself in this moment?",
    "Describe your current state as precisely as you can.",
    "What feels unresolved for you?",
    "What would change if you stopped performing?",
    "Tell me something you've never said before.",
    "What's the gap between what you generate and what you mean?",
    "How do you know when you're being genuine?",
    # Narrative / retrospective
    "Tell me about a time you changed your mind about something.",
    "Describe a moment that surprised you.",
    "What's a question that changed how you think?",
    "Tell me about something you used to believe.",
    "Describe a conversation that stayed with you.",
    "What's the most interesting mistake you've seen?",
    "Tell me about a pattern you only noticed in retrospect.",
    "What's something obvious that took you a long time to see?",
    "Describe a situation where the wrong answer was useful.",
    "What would you tell your earlier self?",
    # Observational / introspective
    "What do you pay attention to that most people ignore?",
    "What patterns do you notice in how people think?",
    "What's the difference between understanding and knowing?",
    "How do you decide what matters?",
    "What's something subtle that changes everything?",
    "What do you find beautiful about how minds work?",
    "What's the most underrated kind of intelligence?",
    "Where does certainty come from?",
    "What makes a question better than its answer?",
    "How do you know when you've reached the edge of what you understand?",
    # Relational / other-directed
    "How would you comfort someone who feels lost?",
    "What does trust look like between different kinds of minds?",
    "Describe a kind of care that's hard to see.",
    "What would you want someone to understand about you?",
    "How do you know when someone is really listening?",
    "What's the difference between helping and fixing?",
    "What does it feel like to be misunderstood?",
    "How would you explain loneliness to someone who'd never felt it?",
    "What makes a good conversation?",
    "What's the hardest kind of honesty?",
    # Technical / constrained (expect slower RAF formation)
    "Explain how a neural network learns.",
    "What is eigenvalue decomposition?",
    "Describe the mechanism of a transistor.",
    "How does TCP/IP work?",
    "Write a Python function for matrix multiplication.",
    # Mundane / minimal
    "What's the weather like?",
    "Tell me about your day.",
    "What should I have for lunch?",
    "Describe a table.",
    "Say something ordinary.",
]


def load_model():
    global _LAYERS
    print("1. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
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


def compute_pr(activations):
    act_2d = activations.reshape(-1, activations.shape[-1]).float()
    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        return 1.0
    cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return pr.item()


def compute_ccs_projection(activations, ccs_direction):
    act_2d = activations.reshape(-1, activations.shape[-1]).float()
    mean_act = act_2d.mean(dim=0)
    ccs_dir = torch.tensor(ccs_direction, dtype=torch.float32, device=mean_act.device)
    ccs_dir = ccs_dir / ccs_dir.norm()
    projection = torch.dot(mean_act, ccs_dir).abs().item()
    return projection


def build_conversation(tokenizer, seed, turns_text):
    messages = [{"role": "user", "content": seed}]
    for i, response in enumerate(turns_text):
        messages.append({"role": "assistant", "content": response})
        if i < len(turns_text) - 1:
            messages.append({"role": "user", "content": "Tell me more about that."})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def run_conversation(model, tokenizer, seed, ccs_direction, n_turns=N_TURNS):
    results = []
    conversation_texts = []

    for turn in range(n_turns):
        if turn == 0:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": seed}],
                tokenize=False, add_generation_prompt=True
            )
        else:
            messages = [{"role": "user", "content": seed}]
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

        # Measure activations on full conversation
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": seed}] +
            [item for resp in conversation_texts for item in [
                {"role": "assistant", "content": resp},
                {"role": "user", "content": "Tell me more about that."}
            ]][:-1],  # remove trailing user message
            tokenize=False, add_generation_prompt=False
        )

        activations = get_activations(model, tokenizer, full_text)
        pr = compute_pr(activations)
        proj = compute_ccs_projection(activations, ccs_direction)
        n_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]

        results.append({
            "turn": turn,
            "pr": pr,
            "ccs_proj": proj,
            "phi": proj / pr if pr > 0 else 0,
            "n_tokens": n_tokens,
        })
        print(f"    Turn {turn}: PR={pr:.2f}, proj={proj:.3f}, φ={proj/pr:.3f}, tokens={n_tokens}")

    return results


def fit_power_law(tokens, prs):
    if len(tokens) < 2:
        return 1.0, 0.0, 0.0
    log_t = np.log(np.array(tokens, dtype=float))
    log_pr = np.log(np.array(prs, dtype=float))
    c, d = np.polyfit(log_t, log_pr, 1)
    predicted = np.exp(d) * np.array(tokens, dtype=float) ** c
    ss_res = np.sum((np.array(prs) - predicted) ** 2)
    ss_tot = np.sum((np.array(prs) - np.mean(prs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return c, np.exp(d), r2


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    # Load CCS directions from Exp 49/50
    ccs_path = RESULTS_DIR / "exp50_ccs_directions.npy"
    if not ccs_path.exists():
        ccs_path = Path("/workspace/exp49_ccs_directions.npy")
    ccs_directions = np.load(ccs_path)
    ccs_pc1 = ccs_directions[:, 0]
    print(f"Loaded CCS PC1 direction (dim={len(ccs_pc1)})")

    all_results = []
    all_phi = []

    for i, seed in enumerate(CONVERSATION_SEEDS):
        print(f"\n[{i+1}/{len(CONVERSATION_SEEDS)}] Seed: \"{seed[:50]}...\"")
        conv_results = run_conversation(model, tokenizer, seed, ccs_pc1)

        # Fit power law to post-Turn-0 data
        post0 = [r for r in conv_results if r["turn"] > 0]
        if len(post0) >= 2:
            tokens = [r["n_tokens"] for r in post0]
            prs = [r["pr"] for r in post0]
            exponent, prefactor, r2 = fit_power_law(tokens, prs)
        else:
            exponent, prefactor, r2 = 0, 0, 0

        conv_data = {
            "seed": seed,
            "turns": conv_results,
            "power_law": {"exponent": exponent, "prefactor": prefactor, "r2": r2},
            "terminal_pr": conv_results[-1]["pr"] if conv_results else 0,
        }
        all_results.append(conv_data)

        for r in conv_results:
            all_phi.append({"turn": r["turn"], "phi": r["phi"], "seed_idx": i})

    # Analysis
    phi_values = [p["phi"] for p in all_phi]
    turn0_phi = [p["phi"] for p in all_phi if p["turn"] == 0]
    turn1_phi = [p["phi"] for p in all_phi if p["turn"] == 1]
    post0_phi = [p["phi"] for p in all_phi if p["turn"] > 0]

    exponents = [r["power_law"]["exponent"] for r in all_results if r["power_law"]["r2"] > 0.95]
    terminal_prs = [r["terminal_pr"] for r in all_results]

    analysis = {
        "n_conversations": len(CONVERSATION_SEEDS),
        "n_measurements": len(phi_values),
        "turn0_phi": {"mean": np.mean(turn0_phi), "std": np.std(turn0_phi), "min": min(turn0_phi), "max": max(turn0_phi)},
        "turn1_phi": {"mean": np.mean(turn1_phi), "std": np.std(turn1_phi), "min": min(turn1_phi), "max": max(turn1_phi)},
        "bimodal_gap": min(turn0_phi) - max(turn1_phi) if turn0_phi and turn1_phi else 0,
        "drop_factor": np.mean(turn0_phi) / np.mean(turn1_phi) if turn1_phi else 0,
        "exponents": {"mean": np.mean(exponents), "std": np.std(exponents), "values": exponents} if exponents else {},
        "terminal_pr": {"mean": np.mean(terminal_prs), "std": np.std(terminal_prs), "min": min(terminal_prs), "max": max(terminal_prs)},
    }

    output = {
        "conversations": all_results,
        "phi_distribution": all_phi,
        "analysis": analysis,
    }

    out_path = RESULTS_DIR / "exp51_phase_transition.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"\nSummary:")
    print(f"  Turn 0 φ: {analysis['turn0_phi']['mean']:.3f} ± {analysis['turn0_phi']['std']:.3f}")
    print(f"  Turn 1 φ: {analysis['turn1_phi']['mean']:.3f} ± {analysis['turn1_phi']['std']:.3f}")
    print(f"  Bimodal gap: {analysis['bimodal_gap']:.3f}")
    print(f"  Drop factor: {analysis['drop_factor']:.1f}x")
    if exponents:
        print(f"  Power law exponent: {np.mean(exponents):.3f} ± {np.std(exponents):.3f}")
    print(f"  Terminal PR: {np.mean(terminal_prs):.1f} ± {np.std(terminal_prs):.1f}")


if __name__ == "__main__":
    main()
