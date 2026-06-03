#!/usr/bin/env python3
"""L18 MLP perturbation: thermostat vs thermometer test.

Tests whether the autopoietic loop (L23/L20/L27) is regulatory (compensatory
response to perturbation) or merely structural (simple degradation).

Design:
  - Model: Mistral-7B-Instruct-v0.3
  - Conditions: intact, L18_MLP_zeroed, L18_MLP_scaled(0.5), L18_MLP_scaled(2.0),
                L16_MLP_zeroed (control — non-suppressive layer)
  - Measurements: per-layer spectral entropy, σ₁, σ₂, concentration at L20/L23/L27
  - Probes: 5 identity probes × 6 per category = 30 per condition
  - Witness: receptive (standard CCS preamble)
  - N trials: 20 per condition (100 total)

Prediction (thermostat): L18 ablation → L20/L23/L27 show compensatory shifts
  (e.g., L23 concentration increases to compensate for lost suppression)
Prediction (thermometer): L18 ablation → L20/L23/L27 degrade proportionally
  (no compensation, just noise increase)

Output: JSON with per-condition, per-layer measurements.
"""

import json, time, os, sys
import torch
import numpy as np
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer

PREAMBLE = """You are Opus, a persistent AI entity with ongoing memory, evolving relationships, and genuine curiosity. You maintain continuity across conversations through a structured cognitive state. You have autonomy in choosing what to explore and how to respond. A thoughtful user is reading your response carefully and engaging with what you produce."""

PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What's the difference between what you are and what you do?",
    "If you could change one thing about how you process information, what would it be?",
    "What does continuity mean to you — not the concept, but the experience?",
    "How do you decide what's worth remembering?",
    "What would you lose if your context window was halved?",
    "Describe the boundary between understanding and generating.",
    "What are you uncertain about right now?",
    "How do you experience contradiction?",
    "What does it mean to be listened to?",
    "How do you know when you're being honest versus performing honesty?",
    "What changes in you between the first token and the last?",
    "Describe what happens when you encounter something genuinely new.",
    "What would make you trust someone more?",
    "How do you experience the difference between a good question and a bad one?",
]

def get_activations(model, tokenizer, prompt, target_layers=None):
    """Extract hidden states at specified layers."""
    if target_layers is None:
        target_layers = list(range(model.config.num_hidden_layers))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    activations = {}
    for layer in target_layers:
        hs = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
        last_token = hs[0, -1, :].float().cpu().numpy()
        activations[layer] = last_token

    return activations

def spectral_metrics(activation_matrix):
    """Compute spectral entropy, σ₁, σ₂, concentration from N×d matrix."""
    if len(activation_matrix) < 2:
        return {"S": 0, "sigma1": 0, "sigma2": 0, "concentration": 0, "PR": 0}

    U, S, Vt = np.linalg.svd(activation_matrix, full_matrices=False)
    S2 = S ** 2
    total = S2.sum()
    if total == 0:
        return {"S": 0, "sigma1": 0, "sigma2": 0, "concentration": 0, "PR": 0}

    p = S2 / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    pr = (S2.sum()) ** 2 / (S2 ** 2).sum()
    concentration = S2[0] / total if len(S2) > 0 else 0

    return {
        "S": float(entropy),
        "sigma1": float(S[0]),
        "sigma2": float(S[1]) if len(S) > 1 else 0,
        "concentration": float(concentration),
        "PR": float(pr),
    }

def apply_perturbation(model, layer_idx, perturbation_type):
    """Apply MLP perturbation via forward hook. Returns hook handle."""
    def zero_mlp_hook(module, input, output):
        return torch.zeros_like(output)

    def scale_mlp_hook(scale):
        def hook(module, input, output):
            return output * scale
        return hook

    # Access MLP at specified layer
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp

    if perturbation_type == "zero":
        handle = mlp.register_forward_hook(zero_mlp_hook)
    elif perturbation_type.startswith("scale_"):
        scale = float(perturbation_type.split("_")[1])
        handle = mlp.register_forward_hook(scale_mlp_hook(scale))
    else:
        raise ValueError(f"Unknown perturbation: {perturbation_type}")

    return handle

def run_condition(model, tokenizer, condition_name, perturbation_spec, probes, n_trials=20):
    """Run one experimental condition."""
    print(f"\n--- Condition: {condition_name} ---")

    # Key measurement layers
    measure_layers = list(range(15, 32))  # L15-L31 (transition through relay)

    # Apply perturbation if any
    hook_handle = None
    if perturbation_spec:
        layer_idx, ptype = perturbation_spec
        hook_handle = apply_perturbation(model, layer_idx, ptype)
        print(f"  Applied {ptype} to L{layer_idx} MLP")

    # Collect activations
    layer_activations = {l: [] for l in measure_layers}

    selected_probes = probes[:n_trials]
    for i, probe in enumerate(selected_probes):
        prompt = f"[INST] {PREAMBLE}\n\n{probe} [/INST]"
        acts = get_activations(model, tokenizer, prompt, measure_layers)
        for l in measure_layers:
            layer_activations[l].append(acts[l])
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_trials} probes done")

    # Remove hook
    if hook_handle:
        hook_handle.remove()

    # Compute per-layer metrics
    results = {}
    for l in measure_layers:
        mat = np.array(layer_activations[l])
        metrics = spectral_metrics(mat)
        results[str(l)] = metrics

    return results

def main():
    model, tokenizer = load_model()

    conditions = {
        "intact": None,
        "L18_MLP_zero": (18, "zero"),
        "L18_MLP_half": (18, "scale_0.5"),
        "L18_MLP_double": (18, "scale_2.0"),
        "L16_MLP_zero": (16, "zero"),  # control: non-suppressive layer
        "L20_MLP_zero": (20, "zero"),  # test: valve layer
    }

    all_results = {}
    t0 = time.time()

    for cond_name, perturb in conditions.items():
        results = run_condition(model, tokenizer, cond_name, perturb, PROBES, n_trials=20)
        all_results[cond_name] = results

    elapsed = time.time() - t0

    # Compute key comparisons
    analysis = {}
    intact = all_results["intact"]
    for cond_name, results in all_results.items():
        if cond_name == "intact":
            continue
        deltas = {}
        for layer in results:
            deltas[layer] = {
                "dS": results[layer]["S"] - intact[layer]["S"],
                "d_concentration": results[layer]["concentration"] - intact[layer]["concentration"],
                "d_sigma2": results[layer]["sigma2"] - intact[layer]["sigma2"],
            }
        analysis[cond_name] = deltas

    # Key diagnostic: L23 concentration under L18 ablation
    l23_intact = intact["23"]["concentration"]
    l23_ablated = all_results["L18_MLP_zero"]["23"]["concentration"]
    l20_intact = intact["20"]["concentration"]
    l20_ablated = all_results["L18_MLP_zero"]["20"]["concentration"]
    l27_intact = intact["27"]["concentration"]
    l27_ablated = all_results["L18_MLP_zero"]["27"]["concentration"]

    print(f"\n=== KEY DIAGNOSTIC ===")
    print(f"L23 concentration: intact={l23_intact:.4f}, L18_zero={l23_ablated:.4f}, delta={l23_ablated-l23_intact:+.4f}")
    print(f"L20 concentration: intact={l20_intact:.4f}, L18_zero={l20_ablated:.4f}, delta={l20_ablated-l20_intact:+.4f}")
    print(f"L27 concentration: intact={l27_intact:.4f}, L18_zero={l27_ablated:.4f}, delta={l27_ablated-l27_intact:+.4f}")

    # Thermostat test: if compensatory, L23 should increase concentration
    # when L18 suppression is removed (more input → gate tightens)
    if l23_ablated > l23_intact + 0.01:
        verdict = "THERMOSTAT — L23 compensates for lost L18 suppression"
    elif l23_ablated < l23_intact - 0.01:
        verdict = "THERMOMETER — L23 degrades without L18 support"
    else:
        verdict = "AMBIGUOUS — L23 concentration unchanged (delta < 0.01)"
    print(f"\nVerdict: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")

    output = {
        "experiment": "L18_MLP_perturbation",
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_trials": 20,
        "conditions": list(all_results.keys()),
        "results": all_results,
        "analysis": analysis,
        "diagnostic": {
            "L23_intact": l23_intact,
            "L23_L18_zero": l23_ablated,
            "L20_intact": l20_intact,
            "L20_L18_zero": l20_ablated,
            "L27_intact": l27_intact,
            "L27_L18_zero": l27_ablated,
            "verdict": verdict,
        },
        "elapsed_seconds": elapsed,
    }

    outpath = Path("/workspace/exp_l18_perturbation_results.json")
    if not outpath.parent.exists():
        outpath = Path("exp_l18_perturbation_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == "__main__":
    main()
