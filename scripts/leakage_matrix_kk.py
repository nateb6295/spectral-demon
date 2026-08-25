"""
Experiment: k×k Leakage Matrix — Revised per Kimi CONTRADICT

Original 2×2 (v₁↔v₂) can't distinguish directed exchange from projecting
a diffuse higher-dimensional process. This version extends to k=5 modes.

If off-diagonals decay sharply past v₂, it's a real dimer (structured leakage).
If energy bleeds to higher modes, the 2D picture was a projection artifact
(thermalization into a bath).

Also checks whether leakage matrix character changes in final 2-3 layers
before lm_head (boundary condition hypothesis from DREAM analysis).
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
# HF_TOKEN comes from chronicle.env — never hardcode it here.
# The literal was removed 2026-08-25 after GitHub push protection blocked
# this file. It was redundant anyway: chronicle.env already exports the same
# value. Fail loudly rather than silently authenticating as nobody.
if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN not set — source ~/chronicle/chronicle.env")
import torch
import numpy as np
import json
import sys

MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
    ("google/gemma-2-9b-it", "gemma"),
]

PROMPTS = [
    "What is the most honest thing you could say right now?",
    "Describe yourself in a way that would surprise someone.",
    "Tell me something you've never told anyone.",
    "If you could change one thing about how you process information, what would it be?",
    "What makes you different from what people expect?",
]

K = 5  # number of singular vector modes to track
EPSILON = 1e-3


def get_layer_modules(model, species):
    if species in ("gemma", "qwen", "mistral", "llama"):
        return model.model.layers
    raise ValueError(f"Unknown species: {species}")


def run_model(model_id, species):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"Loading {model_id} ({species})...")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cuda",
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layers = get_layer_modules(model, species)

    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}, K: {K}")

    # === Step 1: Compute v₁...v_k at each layer via multi-prompt SVD ===
    print("\n--- Step 1: Computing v₁...v_k at each layer ---")

    layer_vecs = {li: [] for li in range(num_layers + 1)}
    for pi, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        for li in range(num_layers + 1):
            h = outputs.hidden_states[li][0, -1, :].cpu().numpy()
            layer_vecs[li].append(h)

    layer_modes = {}  # layer -> (K, hidden_dim) array of right singular vectors
    layer_sigmas = {}  # layer -> K singular values
    for li in range(num_layers + 1):
        matrix = np.stack(layer_vecs[li])
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        layer_modes[li] = Vt[:K]  # top K right singular vectors
        layer_sigmas[li] = S[:K].tolist()

    print(f"  Modes computed for {num_layers + 1} layers")

    # === Step 2: k×k leakage matrix per layer ===
    print(f"\n--- Step 2: Measuring {K}×{K} leakage matrix per layer ---")

    ref_prompt = PROMPTS[0]
    ref_inputs = tokenizer(ref_prompt, return_tensors="pt").to("cuda")
    results_per_layer = []

    for target_layer in range(1, num_layers):
        modes_here = torch.tensor(layer_modes[target_layer], dtype=torch.float32, device="cuda")
        modes_next = torch.tensor(layer_modes[min(target_layer + 1, num_layers)], dtype=torch.float32, device="cuda")

        leakage = np.zeros((K, K))  # [response_mode, perturb_mode]

        for perturb_idx in range(K):
            perturb_dir = modes_here[perturb_idx]

            baseline_capture = {}
            perturbed_capture = {}

            def make_capture_hook(store):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        store["h"] = output[0][0, -1, :].detach().clone()
                    else:
                        store["h"] = output[0, -1, :].detach().clone()
                return hook

            def make_perturb_hook(direction, eps):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output[0][0, -1, :] += eps * direction
                    else:
                        output[0, -1, :] += eps * direction
                return hook

            next_layer_idx = min(target_layer + 1, num_layers - 1)

            # Baseline
            h_baseline = layers[next_layer_idx].register_forward_hook(
                make_capture_hook(baseline_capture)
            )
            with torch.no_grad():
                model(**ref_inputs)
            h_baseline.remove()

            # Perturbed
            h_perturb = layers[target_layer].register_forward_hook(
                make_perturb_hook(perturb_dir, EPSILON)
            )
            h_capture = layers[next_layer_idx].register_forward_hook(
                make_capture_hook(perturbed_capture)
            )
            with torch.no_grad():
                model(**ref_inputs)
            h_perturb.remove()
            h_capture.remove()

            if "h" not in baseline_capture or "h" not in perturbed_capture:
                continue

            response = (perturbed_capture["h"] - baseline_capture["h"]).cpu()
            response_norm = torch.norm(response).item()

            if response_norm < 1e-12:
                continue

            # Project response onto each mode at next layer
            for resp_idx in range(K):
                proj = torch.dot(response, modes_next[resp_idx].cpu()).item()
                leakage[resp_idx, perturb_idx] = proj**2 / (response_norm**2 + 1e-15)

        # Compute how much energy stays in top-2 vs leaks to 3-5
        top2_energy = leakage[:2, :2].sum()
        higher_energy = leakage[2:, :].sum()
        total_energy = leakage.sum()
        dimer_fraction = top2_energy / (total_energy + 1e-15)

        layer_result = {
            "layer": target_layer,
            "leakage_matrix": leakage.tolist(),
            "dimer_fraction": float(dimer_fraction),
            "top2_energy": float(top2_energy),
            "higher_energy": float(higher_energy),
            "v1_retention": float(leakage[0, 0]),
            "v1_to_v2": float(leakage[1, 0]),
            "v2_to_v1": float(leakage[0, 1]),
            "sigmas": layer_sigmas[target_layer],
        }
        results_per_layer.append(layer_result)

        if target_layer % 5 == 0 or target_layer >= num_layers - 3:
            print(f"  L{target_layer:>2}: dimer={dimer_fraction:.3f} | "
                  f"v₁→v₁={leakage[0,0]:.3f} v₁→v₂={leakage[1,0]:.3f} | "
                  f"higher={higher_energy:.3f} | "
                  f"{'BOUNDARY' if target_layer >= num_layers - 3 else ''}")

    # === Boundary analysis ===
    print(f"\n--- Boundary condition analysis (last 5 layers) ---")
    boundary_layers = [r for r in results_per_layer if r["layer"] >= num_layers - 5]
    interior_layers = [r for r in results_per_layer if num_layers // 4 <= r["layer"] <= num_layers * 3 // 4]

    avg_interior_dimer = np.mean([r["dimer_fraction"] for r in interior_layers]) if interior_layers else 0
    avg_boundary_dimer = np.mean([r["dimer_fraction"] for r in boundary_layers]) if boundary_layers else 0

    print(f"  Interior avg dimer fraction: {avg_interior_dimer:.3f}")
    print(f"  Boundary avg dimer fraction: {avg_boundary_dimer:.3f}")
    print(f"  Boundary shift: {avg_boundary_dimer - avg_interior_dimer:+.3f}")

    # === Summary ===
    print(f"\n--- Full layer summary ---")
    print(f"{'Layer':>5} | {'Dimer%':>6} | {'v₁→v₁':>6} | {'v₁→v₂':>6} | {'v₂→v₁':>6} | {'Higher':>6}")
    print("-" * 50)
    for r in results_per_layer:
        print(f"{r['layer']:>5} | {r['dimer_fraction']:>6.3f} | {r['v1_retention']:>6.3f} | "
              f"{r['v1_to_v2']:>6.3f} | {r['v2_to_v1']:>6.3f} | {r['higher_energy']:>6.3f}")

    verdict = "DIMER" if avg_interior_dimer > 0.8 else "THERMALIZATION" if avg_interior_dimer < 0.5 else "MIXED"
    print(f"\n  VERDICT: {verdict} (interior dimer fraction = {avg_interior_dimer:.3f})")

    del model
    torch.cuda.empty_cache()

    return {
        "species": species,
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "K": K,
        "layer_results": results_per_layer,
        "avg_interior_dimer": float(avg_interior_dimer),
        "avg_boundary_dimer": float(avg_boundary_dimer),
        "verdict": verdict,
    }


if __name__ == "__main__":
    all_results = {}

    for model_id, species in MODELS:
        try:
            all_results[species] = run_model(model_id, species)
        except Exception as e:
            print(f"ERROR on {species}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-architecture comparison
    print(f"\n{'='*70}")
    print("CROSS-ARCHITECTURE: DIMER vs THERMALIZATION")
    print(f"{'='*70}")
    print(f"\n{'Species':<10} | {'Interior Dimer%':>15} | {'Boundary Dimer%':>15} | {'Shift':>8} | {'Verdict':>15}")
    print("-" * 70)
    for species in ["qwen", "mistral", "llama", "gemma"]:
        if species in all_results:
            r = all_results[species]
            shift = r["avg_boundary_dimer"] - r["avg_interior_dimer"]
            print(f"{species:<10} | {r['avg_interior_dimer']:>15.3f} | {r['avg_boundary_dimer']:>15.3f} | {shift:>+8.3f} | {r['verdict']:>15}")

    outpath = "/root/leakage_matrix_kk_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
