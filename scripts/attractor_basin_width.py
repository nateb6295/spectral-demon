"""
Experiment: Attractor Basin Width — Where Does v₁ Recovery Break?

Follow-up to F344 (Gregory experiment). F344 showed 64/64 GLOBAL attractor
with epsilon up to 0.05. This experiment pushes epsilon from 0.01 to 1.0
in fine steps to find the failure threshold.

Questions:
1. Is there a sharp phase transition (robust below threshold, sudden failure)?
2. Or gradual degradation (recovery quality decreases smoothly)?
3. Is the basin width species-specific?
4. Does the basin width vary by layer position (early vs late perturbation)?

Predictions:
- Mistral: widest basin (rigid cylinder, strongest attractor, fastest F344 recovery)
- Gemma: narrowest basin (GQA dampening may weaken attractor)
- Phase transition more likely than gradual — attractors tend to have sharp boundaries
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

EPSILONS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def get_v1_all_layers(model, tokenizer, num_layers):
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

    # Two target layers: mid and late
    target_layers = [num_layers // 3, num_layers * 2 // 3]
    print(f"  Layers: {num_layers}, Targets: {target_layers}")

    # Baseline
    print("\n--- Baseline v₁ ---")
    baseline_v1, baseline_ratios = get_v1_all_layers(model, tokenizer, num_layers)
    print(f"  Final σ₁/σ₂: {baseline_ratios[num_layers]:.2f}")

    all_results = []

    for target_layer in target_layers:
        print(f"\n--- Target layer: L{target_layer} ---")
        print(f"{'ε':>6} | {'Min cos':>8} | {'Final cos':>9} | {'RecovDist':>9} | {'Status':>12} | {'σ₁/σ₂':>6}")
        print("-" * 65)

        for epsilon in EPSILONS:
            perturbed_model = perturb_layer_weights(model, target_layer, epsilon)
            perturbed_model.eval()

            perturbed_v1, perturbed_ratios = get_v1_all_layers(
                perturbed_model, tokenizer, num_layers
            )

            cosines = {}
            for li in range(num_layers + 1):
                cos = float(np.dot(baseline_v1[li], perturbed_v1[li]) /
                           (np.linalg.norm(baseline_v1[li]) * np.linalg.norm(perturbed_v1[li]) + 1e-15))
                cosines[li] = cos

            post_perturb = [cosines[li] for li in range(target_layer + 1, num_layers + 1)]
            min_cos = min(post_perturb) if post_perturb else 1.0
            final_cos = cosines[num_layers]

            # Recovery: first layer after perturbation where cos > 0.99
            recovery_layer = None
            for li in range(target_layer + 1, num_layers + 1):
                if cosines[li] > 0.99:
                    recovery_layer = li
                    break
            recovery_distance = (recovery_layer - target_layer) if recovery_layer else None

            recovered = final_cos > 0.95
            rec_str = f"{recovery_distance}L" if recovery_distance else "never"

            # Also measure how much the σ₁/σ₂ ratio changed
            final_ratio = perturbed_ratios[num_layers]
            ratio_change = final_ratio / (baseline_ratios[num_layers] + 1e-15)

            status = "GLOBAL" if recovered else "BROKEN"
            print(f"{epsilon:>6.2f} | {min_cos:>8.4f} | {final_cos:>9.4f} | {rec_str:>9} | {status:>12} | {final_ratio:>6.2f}")

            result = {
                "target_layer": target_layer,
                "epsilon": epsilon,
                "min_cos_downstream": float(min_cos),
                "final_cos": float(final_cos),
                "recovery_distance": recovery_distance,
                "recovered": recovered,
                "final_ratio": float(final_ratio),
                "ratio_change": float(ratio_change),
                "cosines": {str(k): v for k, v in cosines.items()},
            }
            all_results.append(result)

            del perturbed_model
            torch.cuda.empty_cache()

    # Find critical epsilon (first failure) per target layer
    print(f"\n--- Critical Epsilon Analysis ---")
    for tl in target_layers:
        layer_results = [r for r in all_results if r["target_layer"] == tl]
        recovered_eps = [r["epsilon"] for r in layer_results if r["recovered"]]
        broken_eps = [r["epsilon"] for r in layer_results if not r["recovered"]]

        if recovered_eps and broken_eps:
            critical = max(recovered_eps)
            print(f"  L{tl}: Basin width ε_crit ∈ ({critical:.2f}, {min(broken_eps):.2f})")
        elif not broken_eps:
            print(f"  L{tl}: Basin never breaks (ε up to {max(r['epsilon'] for r in layer_results):.2f})")
        else:
            print(f"  L{tl}: Basin always broken (even ε={min(r['epsilon'] for r in layer_results):.2f})")

        # Check for gradual vs abrupt transition
        finals = [(r["epsilon"], r["final_cos"]) for r in layer_results]
        finals.sort()
        if len(finals) >= 3:
            diffs = [finals[i+1][1] - finals[i][1] for i in range(len(finals)-1)]
            max_drop = min(diffs)
            max_drop_idx = diffs.index(max_drop)
            if max_drop < -0.3:
                print(f"    ABRUPT transition at ε≈{finals[max_drop_idx+1][0]:.2f} (drop={max_drop:.3f})")
            else:
                print(f"    GRADUAL degradation (max single drop={max_drop:.3f})")

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

    # Cross-architecture basin comparison
    print(f"\n{'='*70}")
    print("CROSS-ARCHITECTURE: ATTRACTOR BASIN WIDTH")
    print(f"{'='*70}")

    for species in ["qwen", "mistral", "llama", "gemma"]:
        if species in all_results:
            results = all_results[species]["results"]
            for tl in all_results[species]["target_layers"]:
                lr = [r for r in results if r["target_layer"] == tl]
                recovered = [r["epsilon"] for r in lr if r["recovered"]]
                broken = [r["epsilon"] for r in lr if not r["recovered"]]
                if recovered and broken:
                    print(f"  {species} L{tl}: basin ∈ ({max(recovered):.2f}, {min(broken):.2f})")
                elif not broken:
                    print(f"  {species} L{tl}: basin > {max(r['epsilon'] for r in lr):.2f} (never breaks)")
                else:
                    print(f"  {species} L{tl}: basin < {min(r['epsilon'] for r in lr):.2f} (always broken)")

    outpath = "/root/attractor_basin_width_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
