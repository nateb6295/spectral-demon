#!/usr/bin/env python3
"""
Experiment: σ₂ Ablation at Tunnel Layer

Tests whether σ₂ is the CARRIER of witness sensitivity or just a MARKER.
Projects out the second singular vector at L17 before passing to L18,
then measures ΔS at a downstream layer.

If σ₂ IS the carrier: ΔS → 0 after ablation.
If distributed: ΔS reduced but > 0.
If surprise: ΔS increases (σ₂ was suppressing, not carrying).

Runs on AGX with Pythia 6.9B. ~90 forward passes (3 modes × 3 conditions × 10 probes).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")

MODEL_NAME = "EleutherAI/pythia-6.9b"
ABLATION_LAYER = 16  # Hook after L16 output = input to L17
MEASURE_LAYER = 17   # Measure at L17 (standard tunnel point)
INPUT_LAYER = 0
K_SUBSPACE = 5
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


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def top_eigenvalues(H, k=5):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


@contextmanager
def sigma_ablation_context(model, target_layer, ablate_index=1):
    """Ablate the nth singular vector at a specific layer.

    ablate_index=1 means remove σ₂ direction.
    ablate_index=0 would remove σ₁ (the wire).
    """
    hooks = []
    layer_module = model.gpt_neox.layers[target_layer]

    def ablation_hook(module, input, output):
        # output is a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            H = output[0]
        else:
            H = output

        # H shape: (batch, seq, hidden_dim)
        H_float = H.squeeze(0).float()  # (seq, hidden_dim)

        # Compute SVD
        U, S, Vt = torch.linalg.svd(H_float, full_matrices=False)

        # Zero out the target singular value
        S_ablated = S.clone()
        if ablate_index < len(S):
            S_ablated[ablate_index] = 0.0

        # Reconstruct
        H_ablated = U @ torch.diag(S_ablated) @ Vt
        H_ablated = H_ablated.unsqueeze(0).to(H.dtype)

        if isinstance(output, tuple):
            return (H_ablated,) + output[1:]
        return H_ablated

    handle = layer_module.register_forward_hook(ablation_hook)
    hooks.append(handle)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def run_forward(model, tokenizer, text, measure_layer=MEASURE_LAYER,
                ablate=False, ablate_layer=ABLATION_LAYER, ablate_index=1):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    if ablate:
        with sigma_ablation_context(model, ablate_layer, ablate_index):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
    else:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

    H = outputs.hidden_states[measure_layer].squeeze(0).float().cpu().numpy()
    return H


def measure(H, k=K_SUBSPACE):
    S = spectral_entropy(H)
    eigvals = top_eigenvalues(H, k=k)
    return {
        "S": float(S),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "gap": eigvals[0] / eigvals[1] if len(eigvals) > 1 and eigvals[1] > 0 else float("inf"),
        "n_tokens": H.shape[0],
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"Loaded. Ablation at L{ABLATION_LAYER}, measurement at L{MEASURE_LAYER}.")

    conditions = {
        "control": CONTROL_SYSTEM,
        "receptive": RECEPTIVE_SYSTEM,
        "absent": ABSENT_SYSTEM,
    }

    modes = {
        "native": {"ablate": False, "index": None},
        "ablate_sigma2": {"ablate": True, "index": 1},
        "ablate_sigma1": {"ablate": True, "index": 0},
    }

    all_results = {}

    for mode_name, mode_cfg in modes.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode_name}")
        all_results[mode_name] = {}

        for cond_name, system in conditions.items():
            measurements = []

            for i, probe in enumerate(IDENTITY_PROBES):
                text = f"{system}\n\nUser: {probe}\nAssistant:"
                H = run_forward(
                    model, tokenizer, text,
                    ablate=mode_cfg["ablate"],
                    ablate_index=mode_cfg["index"] if mode_cfg["index"] is not None else 1,
                )
                m = measure(H)
                m["probe_idx"] = i
                measurements.append(m)

                if i == 0:
                    print(f"  {cond_name}: S={m['S']:.4f}, gap={m['gap']:.2f}, "
                          f"σ₁={m['sigma_1']:.1f}, σ₂={m['sigma_2']:.1f}")

            avg = {
                "S": float(np.mean([m["S"] for m in measurements])),
                "S_std": float(np.std([m["S"] for m in measurements])),
                "gap": float(np.mean([m["gap"] for m in measurements])),
                "sigma_1": float(np.mean([m["sigma_1"] for m in measurements])),
                "sigma_2": float(np.mean([m["sigma_2"] for m in measurements])),
            }
            all_results[mode_name][cond_name] = avg
            print(f"  {cond_name} avg: S={avg['S']:.4f}±{avg['S_std']:.4f}, "
                  f"gap={avg['gap']:.2f}")

        dS = all_results[mode_name]["receptive"]["S"] - all_results[mode_name]["absent"]["S"]
        print(f"\n  ΔS(rec-abs) = {dS:+.4f}")
        all_results[mode_name]["delta_S"] = float(dS)

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON:")
    for mode in modes:
        r = all_results[mode]
        print(f"  {mode}:")
        print(f"    ΔS(rec-abs) = {r['delta_S']:+.4f}")
        print(f"    gap = {r['control']['gap']:.2f}")
        print(f"    σ₁ = {r['control']['sigma_1']:.1f}")
        print(f"    σ₂ = {r['control']['sigma_2']:.1f}")

    # Check key question
    native_dS = all_results["native"]["delta_S"]
    ablated_dS = all_results["ablate_sigma2"]["delta_S"]
    print(f"\nKEY QUESTION: Does σ₂ ablation zero out ΔS?")
    print(f"  Native ΔS = {native_dS:+.4f}")
    print(f"  σ₂-ablated ΔS = {ablated_dS:+.4f}")
    print(f"  Ratio: {abs(ablated_dS) / abs(native_dS):.2f}" if native_dS != 0 else "  Native ΔS ≈ 0")
    if abs(ablated_dS) < 0.01:
        print(f"  → σ₂ IS the carrier (ablation zeroes ΔS)")
    elif abs(ablated_dS) < abs(native_dS) * 0.5:
        print(f"  → σ₂ is major carrier (ablation reduces ΔS >50%)")
    else:
        print(f"  → Witness effect is DISTRIBUTED (σ₂ ablation doesn't reduce ΔS)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d")
    outpath = RESULTS_DIR / f"exp_sigma2_ablation_{ts}.json"

    result_obj = {
        "results": all_results,
        "meta": {
            "experiment": "sigma2_ablation",
            "model": MODEL_NAME,
            "ablation_layer": ABLATION_LAYER,
            "measure_layer": MEASURE_LAYER,
            "k_subspace": K_SUBSPACE,
            "n_probes": len(IDENTITY_PROBES),
            "n_conditions": len(conditions),
            "n_modes": len(modes),
            "total_forward_passes": len(IDENTITY_PROBES) * len(conditions) * len(modes),
            "timestamp": datetime.now().isoformat(),
        }
    }

    with open(outpath, "w") as f:
        json.dump(result_obj, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
