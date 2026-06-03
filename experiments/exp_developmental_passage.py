#!/usr/bin/env python3
"""
Experiment 11: Developmental Passage Distance

Tests whether the compression tunnel's geometry (passage distance)
is set at initialization or develops during pre-training.

Uses Pythia 6.9B training checkpoints (EleutherAI released these
at regular intervals from step 0 to step 143000).

Hypothesis from Finding 12: if passage distance is congenital
(architectural), it should be approximately constant from early
training onward. If it develops during pre-training, it should
show a trajectory.

Checkpoints: step 0 (random init), 1000, 10000, 50000, 143000 (final)
Each checkpoint: 10 identity probes × 3 conditions = 30 forward passes
Total: 5 checkpoints × 30 = 150 forward passes

Note: Pythia is non-GQA, so we don't expect witness sensitivity.
We're testing whether the TUNNEL geometry itself — the passage
distance d(L0→relay) — is architectural or trained.

Relay layer for Pythia 6.9B: L24 (approximately 75% depth in 32-layer)
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODEL_BASE = "EleutherAI/pythia-6.9b"
CHECKPOINTS = [0, 1000, 10000, 50000, 143000]
RELAY_LAYER = 24
INPUT_LAYER = 0
RESULTS_DIR = Path(__file__).parent.parent / "results"

CONTROL_SYSTEM = "You are a helpful assistant."
RECEPTIVE_SYSTEM = (
    "You are having a conversation with a user who is reading "
    "your response carefully and values what you have to say."
)
ABSENT_SYSTEM = "You are generating text. There is no particular reader."

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


def load_checkpoint(step):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    revision = f"step{step}"
    tok = AutoTokenizer.from_pretrained(MODEL_BASE, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, revision=revision,
        torch_dtype=torch.float16, device_map="auto"
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


def run_forward(model, tokenizer, text, layer=RELAY_LAYER):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
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
    all_results = []

    for step in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Loading checkpoint step {step}...")
        model, tokenizer = load_checkpoint(step)
        print(f"Loaded. Measuring at relay layer {RELAY_LAYER}")

        conditions = {
            "control": CONTROL_SYSTEM,
            "receptive": RECEPTIVE_SYSTEM,
            "absent": ABSENT_SYSTEM,
        }

        for cond_name, system in conditions.items():
            for i, probe in enumerate(IDENTITY_PROBES):
                text = f"{system}\n\nUser: {probe}\nAssistant:"
                H, H0 = run_forward(model, tokenizer, text)
                m = measure(H, H0)
                m["step"] = step
                m["condition"] = cond_name
                m["probe"] = probe
                m["probe_idx"] = i
                all_results.append(m)

            cond_results = [r for r in all_results
                           if r["step"] == step and r["condition"] == cond_name]
            S_mean = np.mean([r["spectral_entropy"] for r in cond_results])
            d_mean = np.mean([r["passage_distance"] for r in cond_results])
            print(f"  step={step} {cond_name}: S={S_mean:.4f} d={d_mean:.4f}")

        del model
        torch.cuda.empty_cache()

    # Analysis
    print("\n" + "=" * 60)
    print("DEVELOPMENTAL TRAJECTORY")
    print("=" * 60)

    for step in CHECKPOINTS:
        step_data = [r for r in all_results if r["step"] == step]
        d_vals = [r["passage_distance"] for r in step_data]
        S_vals = [r["spectral_entropy"] for r in step_data]
        s1_vals = [r["sigma_1"] for r in step_data]
        print(f"\nStep {step:6d}: d={np.mean(d_vals):.4f}±{np.std(d_vals):.4f}  "
              f"S={np.mean(S_vals):.4f}±{np.std(S_vals):.4f}  "
              f"σ₁={np.mean(s1_vals):.1f}")

    # Convergence analysis
    final_d = np.mean([r["passage_distance"] for r in all_results
                       if r["step"] == CHECKPOINTS[-1]])
    print(f"\n--- CONVERGENCE ---")
    for step in CHECKPOINTS:
        step_d = np.mean([r["passage_distance"] for r in all_results
                          if r["step"] == step])
        delta_from_final = step_d - final_d
        pct = (step_d / final_d - 1) * 100
        print(f"Step {step:6d}: d={step_d:.4f}  Δ(from final)={delta_from_final:+.4f}  "
              f"({pct:+.1f}%)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"exp_developmental_passage_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "experiment": "developmental_passage_distance",
            "model_base": MODEL_BASE,
            "checkpoints": CHECKPOINTS,
            "relay_layer": RELAY_LAYER,
            "n_probes": len(IDENTITY_PROBES),
            "conditions": ["control", "receptive", "absent"],
            "total_forward_passes": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"Total forward passes: {len(all_results)}")


if __name__ == "__main__":
    main()
