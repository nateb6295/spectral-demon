"""
Experiment: Weight Perturbation Recovery — "The Gregory Experiment"

Tests whether v₁ is a LOCAL property (each layer independently) or a GLOBAL
attractor (the whole network converges to re-derive it).

First real architectural intervention in the E22a+ program. All prior experiments
intervened on hidden states — this one intervenes on weights.

Method:
1. Compute baseline v₁ at every layer (5-prompt SVD)
2. Add Gaussian noise to attention weights at a SINGLE target layer
3. Re-run all prompts, compute v₁ at every layer with perturbed weights
4. Measure: cosine similarity between baseline and perturbed v₁ at each layer
5. Recovery = how quickly downstream v₁ realigns with baseline after perturbation

Sweep: ε × target_layer × architecture = 4 × 4 × 4 = 64 conditions
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
import copy
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

EPSILONS = [0.001, 0.005, 0.01, 0.05]


def get_v1_all_layers(model, tokenizer, num_layers):
    """Compute v₁ at each layer via 5-prompt SVD."""
    layer_vecs = {li: [] for li in range(num_layers + 1)}

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        for li in range(num_layers + 1):
            h = outputs.hidden_states[li][0, -1, :].cpu().numpy()
            layer_vecs[li].append(h)

    v1s = {}
    ratios = {}
    for li in range(num_layers + 1):
        matrix = np.stack(layer_vecs[li])
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        v1s[li] = Vt[0]
        ratios[li] = float(S[0] / S[1]) if S[1] > 1e-10 else float('inf')

    return v1s, ratios


def perturb_layer_weights(model, target_layer, epsilon, seed=42):
    """Add Gaussian noise to attention weights at target_layer. Returns perturbed model copy."""
    perturbed = copy.deepcopy(model)
    rng = np.random.RandomState(seed)

    layer = perturbed.model.layers[target_layer]
    attn = layer.self_attn

    for name, param in attn.named_parameters():
        if 'weight' in name:
            noise = torch.tensor(
                rng.randn(*param.shape).astype(np.float32),
                device=param.device, dtype=param.dtype
            ) * epsilon * param.std()
            param.data += noise

    return perturbed


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

    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}")

    # Target layers: early, early-mid, mid-late, late
    target_layers = [
        num_layers // 6,
        num_layers // 3,
        num_layers * 2 // 3,
        num_layers * 5 // 6,
    ]
    print(f"  Target layers: {target_layers}")

    # === Step 1: Baseline v₁ ===
    print("\n--- Computing baseline v₁ ---")
    baseline_v1, baseline_ratios = get_v1_all_layers(model, tokenizer, num_layers)
    print(f"  Baseline σ₁/σ₂ at final layer: {baseline_ratios[num_layers]:.2f}")

    # === Step 2: Perturbation sweep ===
    all_results = []

    for target_layer in target_layers:
        for epsilon in EPSILONS:
            print(f"\n--- L{target_layer}, ε={epsilon} ---")

            perturbed_model = perturb_layer_weights(model, target_layer, epsilon)
            perturbed_model.eval()

            perturbed_v1, perturbed_ratios = get_v1_all_layers(
                perturbed_model, tokenizer, num_layers
            )

            # Compute cosine similarity between baseline and perturbed v₁ at each layer
            cosines = {}
            for li in range(num_layers + 1):
                cos = float(np.dot(baseline_v1[li], perturbed_v1[li]) /
                           (np.linalg.norm(baseline_v1[li]) * np.linalg.norm(perturbed_v1[li]) + 1e-15))
                cosines[li] = cos

            # Find perturbation impact and recovery
            pre_perturb = [cosines[li] for li in range(target_layer)]
            at_perturb = cosines.get(target_layer, 1.0)
            post_perturb = [cosines[li] for li in range(target_layer + 1, num_layers + 1)]

            max_disruption = min(post_perturb) if post_perturb else at_perturb
            disruption_layer = target_layer + 1 + post_perturb.index(min(post_perturb)) if post_perturb else target_layer

            # Recovery: first layer after disruption where cos > 0.99
            recovery_layer = None
            for li in range(disruption_layer + 1, num_layers + 1):
                if cosines[li] > 0.99:
                    recovery_layer = li
                    break

            recovery_distance = (recovery_layer - target_layer) if recovery_layer else None
            final_cos = cosines[num_layers]

            result = {
                "target_layer": target_layer,
                "epsilon": epsilon,
                "cosines_per_layer": {str(k): v for k, v in cosines.items()},
                "baseline_ratio_at_target": float(baseline_ratios[target_layer]),
                "perturbed_ratio_at_target": float(perturbed_ratios[target_layer]),
                "max_disruption_cos": float(max_disruption),
                "disruption_layer": disruption_layer,
                "recovery_layer": recovery_layer,
                "recovery_distance": recovery_distance,
                "final_cos": float(final_cos),
                "recovered": final_cos > 0.95,
            }
            all_results.append(result)

            status = "RECOVERED" if final_cos > 0.95 else "NOT RECOVERED"
            rec_str = f"after {recovery_distance} layers" if recovery_distance else "never"
            print(f"  Max disruption: cos={max_disruption:.4f} at L{disruption_layer}")
            print(f"  Recovery: {rec_str}")
            print(f"  Final cos: {final_cos:.4f} [{status}]")

            # Print layer-by-layer cosines
            for li in sorted(cosines.keys()):
                marker = " <-- perturbed" if li == target_layer else ""
                marker = " <-- max disruption" if li == disruption_layer else marker
                marker = " <-- recovery" if li == recovery_layer else marker
                print(f"    L{li:>2}: cos={cosines[li]:.6f}{marker}")

            del perturbed_model
            torch.cuda.empty_cache()

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"SUMMARY: {species}")
    print(f"{'='*70}")
    print(f"{'Target':>8} | {'ε':>6} | {'Max Disrupt':>11} | {'Recovery':>10} | {'Final cos':>9} | {'Status':>12}")
    print("-" * 70)
    for r in all_results:
        rec = f"{r['recovery_distance']}L" if r['recovery_distance'] else "never"
        status = "GLOBAL" if r['recovered'] else "LOCAL"
        print(f"L{r['target_layer']:>6} | {r['epsilon']:>6.3f} | {r['max_disruption_cos']:>11.4f} | {rec:>10} | {r['final_cos']:>9.4f} | {status:>12}")

    del model
    torch.cuda.empty_cache()

    return {
        "species": species,
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "target_layers": target_layers,
        "epsilons": EPSILONS,
        "results": all_results,
        "baseline_ratios": {str(k): v for k, v in baseline_ratios.items()},
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
    print("CROSS-ARCHITECTURE: GLOBAL vs LOCAL ATTRACTOR")
    print(f"{'='*70}")
    for species in ["qwen", "mistral", "llama", "gemma"]:
        if species in all_results:
            results = all_results[species]["results"]
            global_count = sum(1 for r in results if r["recovered"])
            total = len(results)
            avg_recovery = np.mean([r["recovery_distance"] for r in results if r["recovery_distance"]])
            print(f"\n  {species}: {global_count}/{total} conditions recovered (GLOBAL)")
            if avg_recovery > 0:
                print(f"    Average recovery distance: {avg_recovery:.1f} layers")
            else:
                print(f"    No recovery observed")

    outpath = "/root/weight_perturbation_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
