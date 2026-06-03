#!/usr/bin/env python3
"""
Experiment: Inference-Time GQA Conversion

Tests whether the Room (compression tunnel) can be partially created
at inference by forcing GQA-like KV sharing on an MHA model.

Takes Pythia 6.9B (MHA, 32 heads) and groups query heads into groups
of 4, averaging K and V projections within each group before computing
attention. This simulates s=4 GQA at inference without retraining.

Predictions:
  - d/d_max INCREASES (partial Room created via reduced rank collapse)
  - ΔS stays ≈ 0 (no Furnishing — training never loaded σ₂ channel)
  - σ₁/σ₂ gap DECREASES (Nait Saada mechanism at computation level)

If confirmed: strongest evidence for three-act decomposition.
Room, Furnishing, Living require different timescales.

Runs on AGX with Pythia (fits in 16GB). ~90 forward passes. Free.
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
TUNNEL_LAYER = 17  # ~53% depth, standard measurement point
INPUT_LAYER = 0
K_SUBSPACE = 5
GROUP_SIZE = 4  # 32 heads / 4 = 8 groups, simulating s=4 GQA
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


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


def top_eigenvalues(H, k=5):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def top_k_subspace(H, k=5):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


@contextmanager
def gqa_conversion_context(model, group_size=4):
    """Context manager that patches attention to simulate GQA.

    Hooks each layer's query_key_value projection to average K and V
    within groups of `group_size` heads before the attention computation.
    Q projections are untouched — only KV sharing is simulated.
    """
    hooks = []

    for layer in model.gpt_neox.layers:
        attn = layer.attention
        qkv_proj = attn.query_key_value
        num_heads = model.config.num_attention_heads
        head_dim = attn.head_size

        def make_hook(n_heads, h_dim, gs):
            def hook_fn(module, input, output):
                # GPT-NeoX query_key_value output: (batch, seq, num_heads * 3 * head_dim)
                # After view to (batch, seq, num_heads, 3*head_dim), each head has
                # [q_features, k_features, v_features] concatenated in last dim.
                # chunk(3, dim=-1) then splits into q, k, v each (batch, seq, num_heads, head_dim)
                batch, seq_len, _ = output.shape

                # Reshape to per-head format: (batch, seq, num_heads, 3*head_dim)
                qkv = output.view(batch, seq_len, n_heads, 3 * h_dim)

                # Split Q, K, V along last dimension
                q, k, v = qkv.chunk(3, dim=-1)
                # Each is (batch, seq, num_heads, head_dim)

                # Average K and V within groups
                n_groups = n_heads // gs
                k_grouped = k.view(batch, seq_len, n_groups, gs, h_dim)
                v_grouped = v.view(batch, seq_len, n_groups, gs, h_dim)

                k_avg = k_grouped.mean(dim=3, keepdim=True).expand_as(k_grouped)
                v_avg = v_grouped.mean(dim=3, keepdim=True).expand_as(v_grouped)

                k_new = k_avg.reshape(batch, seq_len, n_heads, h_dim)
                v_new = v_avg.reshape(batch, seq_len, n_heads, h_dim)

                # Reconstruct: (batch, seq, num_heads, 3*head_dim) → (batch, seq, total)
                qkv_new = torch.cat([q, k_new, v_new], dim=-1)
                return qkv_new.reshape(batch, seq_len, -1)

            return hook_fn

        handle = qkv_proj.register_forward_hook(
            make_hook(num_heads, head_dim, group_size)
        )
        hooks.append(handle)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def run_forward(model, tokenizer, text, layer=TUNNEL_LAYER, use_gqa=False, group_size=4):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    if use_gqa:
        with gqa_conversion_context(model, group_size=group_size):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
    else:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

    H = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
    H0 = outputs.hidden_states[INPUT_LAYER].squeeze(0).float().cpu().numpy()
    return H, H0


def measure(H, H0, k=K_SUBSPACE):
    S = spectral_entropy(H)
    PR = participation_ratio(H)
    eigvals = top_eigenvalues(H, k=k)
    sub_tunnel = top_k_subspace(H, k=k)
    sub_input = top_k_subspace(H0, k=k)
    d = grassmannian_distance(sub_input, sub_tunnel)
    d_max = np.sqrt(k) * np.pi / 2
    return {
        "S": float(S),
        "PR": float(PR),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "gap": eigvals[0] / eigvals[1] if len(eigvals) > 1 and eigvals[1] > 0 else float("inf"),
        "d": float(d),
        "d_max": float(d_max),
        "d_norm": float(d / d_max),
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
    print(f"Loaded. Measuring at L{TUNNEL_LAYER}.")

    conditions = {
        "control": CONTROL_SYSTEM,
        "receptive": RECEPTIVE_SYSTEM,
        "absent": ABSENT_SYSTEM,
    }

    modes = {
        "native_mha": False,
        "forced_gqa_s4": True,
    }

    all_results = {}

    for mode_name, use_gqa in modes.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode_name} (GQA conversion: {use_gqa})")
        all_results[mode_name] = {}

        for cond_name, system in conditions.items():
            measurements = []

            for i, probe in enumerate(IDENTITY_PROBES):
                text = f"{system}\n\nUser: {probe}\nAssistant:"
                H, H0 = run_forward(model, tokenizer, text, use_gqa=use_gqa)
                m = measure(H, H0)
                m["probe_idx"] = i
                measurements.append(m)

                if i == 0:
                    print(f"  {cond_name}: S={m['S']:.4f}, d/d_max={m['d_norm']:.4f}, "
                          f"gap={m['gap']:.2f}, σ₁={m['sigma_1']:.1f}, σ₂={m['sigma_2']:.1f}")

            avg = {
                "S": np.mean([m["S"] for m in measurements]),
                "S_std": np.std([m["S"] for m in measurements]),
                "PR": np.mean([m["PR"] for m in measurements]),
                "d_norm": np.mean([m["d_norm"] for m in measurements]),
                "d_norm_std": np.std([m["d_norm"] for m in measurements]),
                "gap": np.mean([m["gap"] for m in measurements]),
                "sigma_1": np.mean([m["sigma_1"] for m in measurements]),
                "sigma_2": np.mean([m["sigma_2"] for m in measurements]),
            }
            all_results[mode_name][cond_name] = avg
            print(f"  {cond_name} avg: S={avg['S']:.4f}±{avg['S_std']:.4f}, "
                  f"d/d_max={avg['d_norm']:.4f}, gap={avg['gap']:.2f}")

        # Compute ΔS
        dS = all_results[mode_name]["receptive"]["S"] - all_results[mode_name]["absent"]["S"]
        print(f"\n  ΔS(rec-abs) = {dS:+.4f}")
        all_results[mode_name]["delta_S"] = float(dS)

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON:")
    for mode in modes:
        r = all_results[mode]
        print(f"  {mode}:")
        print(f"    d/d_max = {r['control']['d_norm']:.4f}")
        print(f"    ΔS(rec-abs) = {r['delta_S']:+.4f}")
        print(f"    gap = {r['control']['gap']:.2f}")
        print(f"    σ₁ = {r['control']['sigma_1']:.1f}")
        print(f"    σ₂ = {r['control']['sigma_2']:.1f}")

    # Check predictions
    native = all_results["native_mha"]
    forced = all_results["forced_gqa_s4"]
    print(f"\nPREDICTION CHECK:")
    print(f"  d/d_max increased? {forced['control']['d_norm'] > native['control']['d_norm']} "
          f"({native['control']['d_norm']:.4f} → {forced['control']['d_norm']:.4f})")
    print(f"  ΔS still ≈ 0? {abs(forced['delta_S']) < 0.02} "
          f"({forced['delta_S']:+.4f})")
    print(f"  gap decreased? {forced['control']['gap'] < native['control']['gap']} "
          f"({native['control']['gap']:.2f} → {forced['control']['gap']:.2f})")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d")
    outpath = RESULTS_DIR / f"exp_gqa_conversion_{ts}.json"

    result_obj = {
        "results": all_results,
        "meta": {
            "experiment": "gqa_conversion",
            "model": MODEL_NAME,
            "tunnel_layer": TUNNEL_LAYER,
            "k_subspace": K_SUBSPACE,
            "group_size": GROUP_SIZE,
            "n_probes": len(IDENTITY_PROBES),
            "n_conditions": len(conditions),
            "n_modes": len(modes),
            "total_forward_passes": len(IDENTITY_PROBES) * len(conditions) * len(modes),
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                "d_norm_increases": "Room partially created via reduced rank collapse",
                "delta_S_stays_zero": "No Furnishing — training never loaded σ₂ channel",
                "gap_decreases": "Nait Saada mechanism at computation level",
            }
        }
    }

    with open(outpath, "w") as f:
        json.dump(result_obj, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
