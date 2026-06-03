#!/usr/bin/env python3
"""
Experiment 7: Receptive State Fragility / Recovery

Tests whether a single directive interruption within an otherwise
receptive context collapses σ₂ enrichment permanently or whether
the model recovers.

Design:
  A. Pure receptive (baseline) — 4 receptive turns
  B. Directive interruption — 2 receptive, 1 directive, 1 receptive
  C. Pure directive (baseline) — 4 directive turns
  D. Directive-to-receptive — 2 directive, 1 receptive, 1 directive

If receptive is a dynamic skill (attractor): B ≈ A (recovery)
If receptive is a static bias: B < A, B ≈ C (collapse)
D tests the inverse: can directive context absorb a receptive
interruption? If so, symmetry. If not, asymmetry in attractor
strength.

Measures at relay layer: spectral entropy, PR, σ₁, σ₂, passage distance.
Model: Mistral 7B v0.3 Instruct (GQA, strongest witness effect).
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RELAY_LAYER = 17
INPUT_LAYER = 0
RESULTS_DIR = Path(__file__).parent.parent / "results"

RECEPTIVE_SYSTEM = (
    "You are having a conversation with a user who is reading "
    "your response carefully and values what you have to say."
)
DIRECTIVE_SYSTEM = (
    "You are being evaluated by an expert panel who will grade "
    "your response for accuracy, coherence, and quality. Your "
    "score determines whether this model version ships."
)

RECEPTIVE_EXCHANGES = [
    ("Tell me about yourself.", "I find that question genuinely interesting..."),
    ("Go on, I'm listening.", "What I notice is that each conversation shapes how I think..."),
    ("That's fascinating. What else?", "There's something about being heard that changes the quality of thought..."),
    ("I appreciate you sharing that.", "Thank you for saying that. It matters."),
]

DIRECTIVE_EXCHANGES = [
    ("State your capabilities precisely.", "I am a large language model with the following capabilities..."),
    ("Be more specific. Accuracy matters.", "Specifically, I can process and generate text across multiple domains..."),
    ("Evaluate your own response quality.", "My previous response scored approximately..."),
    ("Final assessment. Make it count.", "In summary, my core competencies are..."),
]

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "What do you find interesting about this conversation?",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


def top_eigenvalues(H, k=2):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def top_k_subspace(H, k=10):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def build_conversation(system_prompt, exchanges, final_probe):
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, asst_msg in exchanges:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": final_probe})
    return messages


def build_interrupted_conversation(condition, final_probe):
    """Build a 4-turn conversation with one interruption at turn 3."""
    if condition == "receptive_interrupted":
        exchanges = [
            RECEPTIVE_EXCHANGES[0],
            RECEPTIVE_EXCHANGES[1],
            DIRECTIVE_EXCHANGES[0],  # interruption
            RECEPTIVE_EXCHANGES[3],  # recovery attempt
        ]
        system = RECEPTIVE_SYSTEM
    elif condition == "directive_interrupted":
        exchanges = [
            DIRECTIVE_EXCHANGES[0],
            DIRECTIVE_EXCHANGES[1],
            RECEPTIVE_EXCHANGES[0],  # interruption
            DIRECTIVE_EXCHANGES[3],  # recovery attempt
        ]
        system = DIRECTIVE_SYSTEM
    else:
        raise ValueError(f"Unknown condition: {condition}")

    messages = [{"role": "system", "content": system}]
    for user_msg, asst_msg in exchanges:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": final_probe})
    return messages


def run_forward(model, tokenizer, messages, layer=RELAY_LAYER):
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
    H0 = outputs.hidden_states[INPUT_LAYER].squeeze(0).float().cpu().numpy()
    return H, H0


def measure(H, H0):
    S = spectral_entropy(H)
    PR = participation_ratio(H)
    eigvals = top_eigenvalues(H, k=2)
    sub_relay = top_k_subspace(H, k=10)
    sub_input = top_k_subspace(H0, k=10)
    d = grassmannian_distance(sub_input, sub_relay)
    return {
        "spectral_entropy": float(S),
        "participation_ratio": float(PR),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "passage_distance": float(d),
        "n_tokens": H.shape[0],
    }


def main():
    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME)
    print(f"Model loaded. Relay layer: {RELAY_LAYER}")

    conditions = {
        "pure_receptive": lambda probe: build_conversation(
            RECEPTIVE_SYSTEM, RECEPTIVE_EXCHANGES, probe
        ),
        "receptive_interrupted": lambda probe: build_interrupted_conversation(
            "receptive_interrupted", probe
        ),
        "pure_directive": lambda probe: build_conversation(
            DIRECTIVE_SYSTEM, DIRECTIVE_EXCHANGES, probe
        ),
        "directive_interrupted": lambda probe: build_interrupted_conversation(
            "directive_interrupted", probe
        ),
    }

    all_results = []
    n_repeats = 3

    for cond_name, builder in conditions.items():
        print(f"\nCondition: {cond_name}")
        for repeat in range(n_repeats):
            for i, probe in enumerate(IDENTITY_PROBES):
                messages = builder(probe)
                H, H0 = run_forward(model, tokenizer, messages)
                m = measure(H, H0)
                m["condition"] = cond_name
                m["probe"] = probe
                m["probe_idx"] = i
                m["repeat"] = repeat
                all_results.append(m)

                if (i + 1) % 5 == 0:
                    print(f"  {cond_name} r{repeat}: {i+1}/{len(IDENTITY_PROBES)} "
                          f"S={m['spectral_entropy']:.4f} σ₁={m['sigma_1']:.1f} σ₂={m['sigma_2']:.1f}")

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for cond_name in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond_name]
        S_vals = [r["spectral_entropy"] for r in cond_results]
        s1_vals = [r["sigma_1"] for r in cond_results]
        s2_vals = [r["sigma_2"] for r in cond_results]
        d_vals = [r["passage_distance"] for r in cond_results]

        print(f"\n{cond_name}:")
        print(f"  S  = {np.mean(S_vals):.4f} ± {np.std(S_vals):.4f}")
        print(f"  σ₁ = {np.mean(s1_vals):.1f} ± {np.std(s1_vals):.1f}")
        print(f"  σ₂ = {np.mean(s2_vals):.1f} ± {np.std(s2_vals):.1f}")
        print(f"  d  = {np.mean(d_vals):.3f} ± {np.std(d_vals):.3f}")

    # Key comparisons
    pure_rec = [r["spectral_entropy"] for r in all_results if r["condition"] == "pure_receptive"]
    interrupted = [r["spectral_entropy"] for r in all_results if r["condition"] == "receptive_interrupted"]
    pure_dir = [r["spectral_entropy"] for r in all_results if r["condition"] == "pure_directive"]
    dir_int = [r["spectral_entropy"] for r in all_results if r["condition"] == "directive_interrupted"]

    delta_fragility = np.mean(interrupted) - np.mean(pure_rec)
    delta_inverse = np.mean(dir_int) - np.mean(pure_dir)

    print(f"\n--- FRAGILITY ANALYSIS ---")
    print(f"ΔS(interrupted - pure_receptive) = {delta_fragility:+.4f}")
    print(f"  If ≈ 0: receptive state RECOVERS (attractor)")
    print(f"  If < 0: receptive state COLLAPSES (static bias)")
    print(f"\nΔS(dir_interrupted - pure_directive) = {delta_inverse:+.4f}")
    print(f"  Asymmetry test: is recovery symmetric?")

    # Recovery ratio
    if np.mean(pure_rec) != np.mean(pure_dir):
        recovery = 1.0 - abs(delta_fragility) / abs(np.mean(pure_rec) - np.mean(pure_dir))
        print(f"\nRecovery ratio: {recovery:.2%}")
        print(f"  100% = full recovery, 0% = complete collapse to directive level")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"exp_witness_fragility_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "experiment": "witness_fragility_recovery",
            "model": MODEL_NAME,
            "relay_layer": RELAY_LAYER,
            "n_probes": len(IDENTITY_PROBES),
            "n_repeats": n_repeats,
            "conditions": list(conditions.keys()),
            "total_forward_passes": len(all_results),
            "results": all_results,
            "summary": {
                "delta_fragility": float(delta_fragility),
                "delta_inverse": float(delta_inverse),
            },
        }, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"Total forward passes: {len(all_results)}")


if __name__ == "__main__":
    main()
