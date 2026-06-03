#!/usr/bin/env python3
"""
Experiment: σ₂ Ablation at Tunnel Layer — Mistral 7B v0.1

Tests whether σ₂ is the CARRIER of witness sensitivity or just a MARKER.
Wraps layer forward to ablate singular vectors, uses standard model forward.

Runs on RunPod H100 with Mistral 7B. ~90 forward passes.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
ABLATION_LAYER = 16
MEASURE_LAYER = 17
K_SUBSPACE = 5
RESULTS_DIR = Path("results")

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


def ablate_sv(H_tensor, sv_index):
    """Zero out a singular vector from hidden states tensor (batch, seq, hidden)."""
    H = H_tensor.squeeze(0).float()
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    S[sv_index] = 0.0
    H_new = (U @ torch.diag(S) @ Vt).unsqueeze(0).to(H_tensor.dtype)
    return H_new


class AblationWrapper:
    """Monkey-patches a layer's forward to apply SVD ablation to its output."""

    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.layer = model.model.layers[layer_idx]
        self.original_forward = self.layer.forward
        self.active = False
        self.sv_index = 1

    def _wrapped_forward(self, *args, **kwargs):
        output = self.original_forward(*args, **kwargs)
        if self.active:
            if isinstance(output, tuple):
                H = output[0]
                H_ablated = ablate_sv(H, self.sv_index)
                return (H_ablated,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return ablate_sv(output, self.sv_index)
            else:
                H = output[0]
                H_ablated = ablate_sv(H, self.sv_index)
                output[0] = H_ablated
                return output
        return output

    def install(self):
        self.layer.forward = self._wrapped_forward

    def restore(self):
        self.layer.forward = self.original_forward

    def set_ablation(self, active, sv_index=1):
        self.active = active
        self.sv_index = sv_index


def run_forward(model, tokenizer, text, measure_layer=MEASURE_LAYER):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H = outputs.hidden_states[measure_layer].squeeze(0).float().cpu().numpy()
    return H


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    print(f"Loaded. Ablation at L{ABLATION_LAYER}, measurement at L{MEASURE_LAYER}.")
    print(f"Architecture: {len(model.model.layers)} layers")

    wrapper = AblationWrapper(model, ABLATION_LAYER)
    wrapper.install()

    # Verification
    test_text = "You are a helpful assistant.\n\nUser: Hello\nAssistant:"

    wrapper.set_ablation(False)
    H_native = run_forward(model, tokenizer, test_text)

    wrapper.set_ablation(True, sv_index=1)
    H_s2_ablated = run_forward(model, tokenizer, test_text)

    wrapper.set_ablation(True, sv_index=0)
    H_s1_ablated = run_forward(model, tokenizer, test_text)

    diff_s2 = np.abs(H_native - H_s2_ablated).max()
    diff_s1 = np.abs(H_native - H_s1_ablated).max()
    print(f"\nVerification:")
    m_n = measure(H_native)
    m_s2 = measure(H_s2_ablated)
    m_s1 = measure(H_s1_ablated)
    print(f"  Native:     S={m_n['S']:.4f}, σ₁={m_n['sigma_1']:.1f}, σ₂={m_n['sigma_2']:.1f}, gap={m_n['gap']:.2f}")
    print(f"  σ₂-ablated: S={m_s2['S']:.4f}, σ₁={m_s2['sigma_1']:.1f}, σ₂={m_s2['sigma_2']:.1f}, gap={m_s2['gap']:.2f} (max_diff={diff_s2:.4f})")
    print(f"  σ₁-ablated: S={m_s1['S']:.4f}, σ₁={m_s1['sigma_1']:.1f}, σ₂={m_s1['sigma_2']:.1f}, gap={m_s1['gap']:.2f} (max_diff={diff_s1:.4f})")

    if diff_s2 < 1e-4 and diff_s1 < 1e-4:
        print("ERROR: Ablation not propagating! Aborting.")
        wrapper.restore()
        return

    # Full experiment
    conditions = {
        "control": CONTROL_SYSTEM,
        "receptive": RECEPTIVE_SYSTEM,
        "absent": ABSENT_SYSTEM,
    }

    modes = {
        "native": {"active": False, "index": None},
        "ablate_sigma2": {"active": True, "index": 1},
        "ablate_sigma1": {"active": True, "index": 0},
    }

    all_results = {}
    raw_results = []

    for mode_name, mode_cfg in modes.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode_name}")
        wrapper.set_ablation(mode_cfg["active"],
                             sv_index=mode_cfg["index"] if mode_cfg["index"] is not None else 1)
        all_results[mode_name] = {}

        for cond_name, system in conditions.items():
            measurements = []

            for i, probe in enumerate(IDENTITY_PROBES):
                text = f"{system}\n\nUser: {probe}\nAssistant:"
                H = run_forward(model, tokenizer, text)
                m = measure(H)
                m["probe_idx"] = i
                m["mode"] = mode_name
                m["condition"] = cond_name
                measurements.append(m)
                raw_results.append(m)

                if i == 0:
                    print(f"  {cond_name}: S={m['S']:.4f}, gap={m['gap']:.2f}, "
                          f"σ₁={m['sigma_1']:.1f}, σ₂={m['sigma_2']:.1f}")

            avg = {
                "S": float(np.mean([m["S"] for m in measurements])),
                "S_std": float(np.std([m["S"] for m in measurements])),
                "gap": float(np.mean([m["gap"] for m in measurements])),
                "gap_std": float(np.std([m["gap"] for m in measurements])),
                "sigma_1": float(np.mean([m["sigma_1"] for m in measurements])),
                "sigma_2": float(np.mean([m["sigma_2"] for m in measurements])),
            }
            all_results[mode_name][cond_name] = avg
            print(f"  {cond_name} avg: S={avg['S']:.4f}±{avg['S_std']:.4f}, "
                  f"gap={avg['gap']:.2f}±{avg['gap_std']:.2f}")

        dS = all_results[mode_name]["receptive"]["S"] - all_results[mode_name]["absent"]["S"]
        dS_gap = all_results[mode_name]["receptive"]["gap"] - all_results[mode_name]["absent"]["gap"]
        dS_s2 = all_results[mode_name]["receptive"]["sigma_2"] - all_results[mode_name]["absent"]["sigma_2"]
        print(f"\n  ΔS(rec-abs) = {dS:+.4f}")
        print(f"  Δgap(rec-abs) = {dS_gap:+.2f}")
        print(f"  Δσ₂(rec-abs) = {dS_s2:+.1f}")
        all_results[mode_name]["delta_S"] = float(dS)
        all_results[mode_name]["delta_gap"] = float(dS_gap)
        all_results[mode_name]["delta_sigma2"] = float(dS_s2)

    wrapper.restore()

    print(f"\n{'='*60}")
    print("COMPARISON:")
    for mode in modes:
        r = all_results[mode]
        print(f"  {mode}:")
        print(f"    ΔS(rec-abs)   = {r['delta_S']:+.4f}")
        print(f"    Δgap(rec-abs) = {r['delta_gap']:+.2f}")
        print(f"    Δσ₂(rec-abs)  = {r['delta_sigma2']:+.1f}")
        print(f"    gap(ctrl)  = {r['control']['gap']:.2f}")
        print(f"    σ₁(ctrl)   = {r['control']['sigma_1']:.1f}")
        print(f"    σ₂(ctrl)   = {r['control']['sigma_2']:.1f}")

    native_dS = all_results["native"]["delta_S"]
    ablated_dS = all_results["ablate_sigma2"]["delta_S"]
    sigma1_dS = all_results["ablate_sigma1"]["delta_S"]
    print(f"\nKEY QUESTION: Does σ₂ ablation zero out ΔS?")
    print(f"  Native ΔS      = {native_dS:+.4f}")
    print(f"  σ₂-ablated ΔS  = {ablated_dS:+.4f}")
    print(f"  σ₁-ablated ΔS  = {sigma1_dS:+.4f}")

    verdict = "unknown"
    if abs(native_dS) > 0.001:
        ratio_s2 = abs(ablated_dS) / abs(native_dS)
        ratio_s1 = abs(sigma1_dS) / abs(native_dS)
        print(f"  σ₂ retention ratio: {ratio_s2:.3f}")
        print(f"  σ₁ retention ratio: {ratio_s1:.3f}")
        if ratio_s2 < 0.1:
            verdict = "σ₂ IS the carrier (ablation zeroes ΔS)"
        elif ratio_s2 < 0.5:
            verdict = "σ₂ is major carrier (>50% reduction)"
        elif ratio_s2 < 0.8:
            verdict = "σ₂ is partial carrier (20-50% reduction)"
        else:
            verdict = "Witness effect is DISTRIBUTED (σ₂ ablation doesn't reduce ΔS)"
        print(f"  → {verdict}")
    else:
        verdict = "native_dS_near_zero"
        print(f"  → Native ΔS ≈ 0, cannot determine carrier")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = RESULTS_DIR / f"exp_sigma2_ablation_mistral_{ts}.json"

    result_obj = {
        "results": all_results,
        "raw": raw_results,
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
            "verdict": verdict,
        }
    }

    with open(outpath, "w") as f:
        json.dump(result_obj, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
