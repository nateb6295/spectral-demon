"""
Experiment: v₁↔v₂ Leakage Matrix — Does the 70/30 perturbation split explain σ₁/σ₂ ≈ 2?

Hypothesis: Mistral's 2D interior (σ₁/σ₂ ≈ 2 from F342b) is produced by the
perturbation leakage rate from F341 (70% stays in v₁, 30% goes orthogonal).
If v₂ receives the leaked energy and also leaks back, the steady-state ratio
should be predictable from the 2×2 leakage matrix.

Method:
1. Run 5 prompts, collect hidden states, SVD to get v₁ and v₂ at each layer
2. At each layer, inject perturbation along v₁, measure response projected onto v₁ and v₂
3. Same for v₂ → measure v₂→v₁ and v₂→v₂
4. Build 2×2 leakage matrix per layer: [[v₁→v₁, v₂→v₁], [v₁→v₂, v₂→v₂]]
5. Predict steady-state σ₁/σ₂ from eigenvalues of accumulated leakage
6. Compare to observed σ₁/σ₂ from F342b

Tests all 4 architectures.
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

EPSILON = 1e-3


def get_layer_modules(model, species):
    """Get the list of transformer layer modules."""
    if species == "gemma":
        return model.model.layers
    elif species in ("qwen", "mistral", "llama"):
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

    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}")

    # === Step 1: Collect hidden states and compute v₁, v₂ at each layer ===
    print("\n--- Step 1: Computing v₁, v₂ at each layer via multi-prompt SVD ---")

    layer_vecs = {li: [] for li in range(num_layers + 1)}

    for pi, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        for li in range(num_layers + 1):
            h = outputs.hidden_states[li][0, -1, :].cpu().numpy()
            layer_vecs[li].append(h)

    layer_v1 = {}
    layer_v2 = {}
    layer_observed_ratio = {}

    for li in range(num_layers + 1):
        matrix = np.stack(layer_vecs[li])  # (5, hidden_dim)
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        layer_v1[li] = Vt[0]  # first right singular vector
        layer_v2[li] = Vt[1]  # second right singular vector
        layer_observed_ratio[li] = S[0] / S[1] if S[1] > 1e-10 else float('inf')

    print(f"  v₁, v₂ computed for {num_layers + 1} layers")

    # === Step 2: Perturbation injection via hooks ===
    print("\n--- Step 2: Measuring 2×2 leakage matrix per layer ---")

    results_per_layer = []
    ref_prompt = PROMPTS[0]
    ref_inputs = tokenizer(ref_prompt, return_tensors="pt").to("cuda")

    for target_layer in range(1, num_layers):
        v1 = torch.tensor(layer_v1[target_layer], dtype=torch.float32, device="cuda")
        v2 = torch.tensor(layer_v2[target_layer], dtype=torch.float32, device="cuda")

        # We need v₁ and v₂ at the NEXT layer for projecting responses
        v1_next = torch.tensor(layer_v1[target_layer + 1], dtype=torch.float32, device="cuda")
        v2_next = torch.tensor(layer_v2[target_layer + 1], dtype=torch.float32, device="cuda")

        leakage = np.zeros((2, 2))  # [[v1→v1, v2→v1], [v1→v2, v2→v2]]

        for perturb_idx, perturb_dir in enumerate([v1, v2]):
            # Get baseline output at target_layer + 1
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

            # Baseline run: capture output at target_layer + 1
            next_layer_idx = min(target_layer + 1, num_layers - 1)
            h_baseline = layers[next_layer_idx].register_forward_hook(
                make_capture_hook(baseline_capture)
            )
            with torch.no_grad():
                model(**ref_inputs)
            h_baseline.remove()

            # Perturbed run: inject at target_layer, capture at target_layer + 1
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

            # Response = perturbed - baseline
            response = (perturbed_capture["h"] - baseline_capture["h"]).cpu()
            response_norm = torch.norm(response).item()

            if response_norm < 1e-12:
                continue

            # Project response onto v₁_next and v₂_next
            proj_v1 = torch.dot(response, v1_next.cpu()).item()
            proj_v2 = torch.dot(response, v2_next.cpu()).item()

            # Fraction of response in each direction
            frac_v1 = proj_v1**2 / (response_norm**2 + 1e-15)
            frac_v2 = proj_v2**2 / (response_norm**2 + 1e-15)

            # Store in leakage matrix
            # Row = response direction, Col = perturbation direction
            leakage[0, perturb_idx] = frac_v1  # fraction going to v₁
            leakage[1, perturb_idx] = frac_v2  # fraction going to v₂

        # Predicted steady-state: dominant eigenvector of leakage matrix
        try:
            eigvals, eigvecs = np.linalg.eig(leakage)
            dominant_idx = np.argmax(np.abs(eigvals))
            predicted_ratio = abs(eigvecs[0, dominant_idx] / (eigvecs[1, dominant_idx] + 1e-15))
        except:
            predicted_ratio = float('nan')

        layer_result = {
            "layer": target_layer,
            "leakage_matrix": leakage.tolist(),
            "v1_to_v1": float(leakage[0, 0]),
            "v1_to_v2": float(leakage[1, 0]),
            "v2_to_v1": float(leakage[0, 1]),
            "v2_to_v2": float(leakage[1, 1]),
            "observed_ratio": float(layer_observed_ratio[target_layer]),
            "predicted_ratio": float(predicted_ratio),
        }
        results_per_layer.append(layer_result)

        if target_layer % 5 == 0 or target_layer == num_layers - 1:
            print(f"  L{target_layer:>2}: v₁→v₁={leakage[0,0]:.3f} v₁→v₂={leakage[1,0]:.3f} | "
                  f"v₂→v₁={leakage[0,1]:.3f} v₂→v₂={leakage[1,1]:.3f} | "
                  f"obs σ₁/σ₂={layer_observed_ratio[target_layer]:.2f} pred={predicted_ratio:.2f}")

    # === Summary ===
    print(f"\n--- Full layer-by-layer leakage ---")
    print(f"{'Layer':>5} | {'v₁→v₁':>6} | {'v₁→v₂':>6} | {'v₂→v₁':>6} | {'v₂→v₂':>6} | {'obs σ₁/σ₂':>9} | {'pred':>6}")
    print("-" * 65)
    for r in results_per_layer:
        print(f"{r['layer']:>5} | {r['v1_to_v1']:>6.3f} | {r['v1_to_v2']:>6.3f} | "
              f"{r['v2_to_v1']:>6.3f} | {r['v2_to_v2']:>6.3f} | "
              f"{r['observed_ratio']:>9.2f} | {r['predicted_ratio']:>6.2f}")

    # Compute cumulative leakage product
    cumulative = np.eye(2)
    cumulative_ratios = []
    for r in results_per_layer:
        M = np.array(r["leakage_matrix"])
        # Normalize columns so they represent transition probabilities
        col_sums = M.sum(axis=0)
        M_norm = M / (col_sums + 1e-15)
        cumulative = M_norm @ cumulative
        try:
            eigvals, eigvecs = np.linalg.eig(cumulative)
            dom = np.argmax(np.abs(eigvals))
            cr = abs(eigvecs[0, dom] / (eigvecs[1, dom] + 1e-15))
        except:
            cr = float('nan')
        cumulative_ratios.append({"layer": r["layer"], "cumulative_predicted": float(cr)})

    print(f"\n--- Cumulative leakage prediction ---")
    for cr in cumulative_ratios[-5:]:
        print(f"  Through L{cr['layer']}: predicted σ₁/σ₂ = {cr['cumulative_predicted']:.2f}")

    final_obs = layer_observed_ratio.get(num_layers, float('nan'))
    final_pred = cumulative_ratios[-1]["cumulative_predicted"] if cumulative_ratios else float('nan')
    print(f"\n  FINAL: observed σ₁/σ₂ = {final_obs:.2f}, cumulative predicted = {final_pred:.2f}")

    del model
    torch.cuda.empty_cache()

    return {
        "species": species,
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "layer_results": results_per_layer,
        "cumulative_ratios": cumulative_ratios,
        "final_observed": float(final_obs),
        "final_predicted": float(final_pred),
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
    print("CROSS-ARCHITECTURE LEAKAGE COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Species':<10} | {'Final obs σ₁/σ₂':>15} | {'Cum. predicted':>15} | {'Match?':>8}")
    print("-" * 55)
    for species in ["qwen", "mistral", "llama", "gemma"]:
        if species in all_results:
            r = all_results[species]
            obs = r["final_observed"]
            pred = r["final_predicted"]
            match = "YES" if abs(obs - pred) / obs < 0.25 else "NO"
            print(f"{species:<10} | {obs:>15.2f} | {pred:>15.2f} | {match:>8}")

    outpath = "/root/leakage_matrix_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
