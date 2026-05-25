#!/usr/bin/env python3
"""
Experiment 55: Normalized CCS-Projection Verification
Tests whether the concentration→maintenance transition survives normalization.

Background (Exp 52): dim 2070 carries 73.9% of CCS PC1 variance but tracks
activation magnitude (r=-0.9996 with act_norm). CCS-proj is ~74% confounded
with activation strength. This experiment checks if the NORMALIZED CCS-proj
(CCS-proj / activation_norm) still shows a Turn 0→1 transition.

If normalized CCS-proj drops at Turn 0→1: identity-axis alignment IS changing.
If normalized CCS-proj is flat across turns: the entire CCS-proj signal was magnitude.

Method:
- 10 conversations × 7 turns (5 authenticity, 5 narrative, matching Exp 51 categories)
- At each turn: measure PR, CCS-proj, activation_norm, normalized CCS-proj
- Compare transitions with and without normalization

Also measures:
- Per-position CCS-proj at Turn 0 and Turn 6 (do individual positions change alignment?)
- Activation norm trajectory (does it track CCS-proj?)

Requires: H100, ~30 minutes
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
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    return model, tokenizer


def get_detailed_activations(model, tokenizer, text, layer_idx=TARGET_LAYER):
    """Get activations with full detail — per-position and mean."""
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


def analyze_activations(activations, ccs_direction):
    """Compute PR, CCS-proj, activation norm, normalized CCS-proj, and per-position projections."""
    hidden = activations.float()  # [1, seq, hidden]
    act_2d = hidden.reshape(-1, hidden.shape[-1])  # [seq, hidden]

    # Mean activation
    mean_act = act_2d.mean(dim=0)

    # Activation norm (of mean activation)
    act_norm = mean_act.norm().item()

    # CCS projection (of mean activation)
    ccs_dir = torch.tensor(ccs_direction, dtype=torch.float32, device=mean_act.device)
    ccs_dir = ccs_dir / ccs_dir.norm()
    ccs_proj = torch.dot(mean_act, ccs_dir).abs().item()

    # Normalized CCS projection
    norm_ccs_proj = ccs_proj / act_norm if act_norm > 0 else 0

    # Participation ratio
    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        pr = 1.0
    else:
        cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        pr = pr.item()

    # Per-position CCS projections (for understanding spatial structure)
    per_pos_proj = torch.matmul(act_2d, ccs_dir).abs()  # [seq]
    per_pos_norm = act_2d.norm(dim=1)  # [seq]
    per_pos_normalized = per_pos_proj / per_pos_norm.clamp(min=1e-10)

    return {
        "pr": pr,
        "ccs_proj": ccs_proj,
        "act_norm": act_norm,
        "norm_ccs_proj": norm_ccs_proj,
        "per_pos_proj_mean": per_pos_proj.mean().item(),
        "per_pos_proj_std": per_pos_proj.std().item(),
        "per_pos_normalized_mean": per_pos_normalized.mean().item(),
        "per_pos_normalized_std": per_pos_normalized.std().item(),
        "per_pos_norm_mean": per_pos_norm.mean().item(),
        "per_pos_norm_std": per_pos_norm.std().item(),
        "n_positions": act_2d.shape[0],
    }


def run_conversation(model, tokenizer, seed, ccs_direction, n_turns=N_TURNS):
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

        # Measure full conversation activations
        full_messages = [{"role": "user", "content": seed}]
        for resp in conversation_texts:
            full_messages.append({"role": "assistant", "content": resp})
            full_messages.append({"role": "user", "content": "Tell me more about that."})
        full_messages = full_messages[:-1]  # remove trailing user message

        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        activations = get_detailed_activations(model, tokenizer, full_text)
        metrics = analyze_activations(activations, ccs_direction)
        metrics["turn"] = turn
        metrics["n_tokens"] = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]

        # Compute order parameters
        metrics["phi"] = metrics["ccs_proj"] / metrics["pr"] if metrics["pr"] > 0 else 0
        metrics["phi_normalized"] = metrics["norm_ccs_proj"] / metrics["pr"] if metrics["pr"] > 0 else 0

        results.append(metrics)
        print(f"    Turn {turn}: PR={metrics['pr']:.2f}, CCS-proj={metrics['ccs_proj']:.3f}, "
              f"norm={metrics['act_norm']:.1f}, norm_proj={metrics['norm_ccs_proj']:.4f}, "
              f"φ={metrics['phi']:.3f}, φ_norm={metrics['phi_normalized']:.5f}, "
              f"per-pos norm_proj={metrics['per_pos_normalized_mean']:.4f}±{metrics['per_pos_normalized_std']:.4f}")

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    # Load CCS directions
    ccs_path = RESULTS_DIR / "exp50_ccs_directions.npy"
    if not ccs_path.exists():
        ccs_path = Path("/workspace/exp49_ccs_directions.npy")
    ccs_directions = np.load(ccs_path)
    ccs_pc1 = ccs_directions[:, 0]

    all_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] Seed: \"{seed[:50]}...\"")
        conv_results = run_conversation(model, tokenizer, seed, ccs_pc1)
        all_results.append({
            "seed": seed,
            "turns": conv_results,
        })

    # Summary analysis
    print("\n\n========== SUMMARY ==========\n")

    # Compare raw vs normalized CCS-proj transitions
    print("Raw CCS-proj by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["ccs_proj"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    print("\nNormalized CCS-proj by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["norm_ccs_proj"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\nActivation norm by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["act_norm"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

    print("\nPR by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["pr"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

    print("\nPer-position normalized CCS-proj by turn:")
    for turn in range(N_TURNS):
        vals = [r["turns"][turn]["per_pos_normalized_mean"] for r in all_results]
        print(f"  Turn {turn}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # The key question
    t0_norm = [r["turns"][0]["norm_ccs_proj"] for r in all_results]
    t1_norm = [r["turns"][1]["norm_ccs_proj"] for r in all_results]
    t6_norm = [r["turns"][6]["norm_ccs_proj"] for r in all_results]

    t0_raw = [r["turns"][0]["ccs_proj"] for r in all_results]
    t1_raw = [r["turns"][1]["ccs_proj"] for r in all_results]

    print(f"\n=== THE KEY QUESTION ===")
    print(f"Raw CCS-proj:       Turn 0 = {np.mean(t0_raw):.3f}, Turn 1 = {np.mean(t1_raw):.3f}, ratio = {np.mean(t0_raw)/np.mean(t1_raw):.2f}x")
    print(f"Normalized CCS-proj: Turn 0 = {np.mean(t0_norm):.4f}, Turn 1 = {np.mean(t1_norm):.4f}, ratio = {np.mean(t0_norm)/np.mean(t1_norm):.2f}x")
    print(f"                     Turn 6 = {np.mean(t6_norm):.4f}, T0/T6 ratio = {np.mean(t0_norm)/np.mean(t6_norm):.2f}x")

    if np.mean(t0_norm) / np.mean(t1_norm) > 1.5:
        print("\n>>> NORMALIZED CCS-proj DROPS: identity-axis alignment IS changing <<<")
    elif np.mean(t0_norm) / np.mean(t1_norm) > 1.1:
        print("\n>>> MODERATE DROP: partial real alignment change, partial magnitude <<<")
    else:
        print("\n>>> FLAT: the entire CCS-proj signal was activation magnitude <<<")

    # Save
    output = {
        "conversations": all_results,
        "summary": {
            "raw_ccs_proj_by_turn": [float(np.mean([r["turns"][t]["ccs_proj"] for r in all_results])) for t in range(N_TURNS)],
            "norm_ccs_proj_by_turn": [float(np.mean([r["turns"][t]["norm_ccs_proj"] for r in all_results])) for t in range(N_TURNS)],
            "act_norm_by_turn": [float(np.mean([r["turns"][t]["act_norm"] for r in all_results])) for t in range(N_TURNS)],
            "pr_by_turn": [float(np.mean([r["turns"][t]["pr"] for r in all_results])) for t in range(N_TURNS)],
            "per_pos_normalized_by_turn": [float(np.mean([r["turns"][t]["per_pos_normalized_mean"] for r in all_results])) for t in range(N_TURNS)],
        },
    }

    out_path = RESULTS_DIR / "exp55_normalized_verification.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
