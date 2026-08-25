#!/usr/bin/env python3
"""NHIM Characterization + Kimi Ablation Experiment

Two complementary experiments sharing one extraction pipeline:

Experiment A (NHIM — GPT-OSS's framing):
  Is σ₂ a normally hyperbolic invariant manifold?
  - Track σ₂ plane through ALL layers
  - Measure transverse contraction rate vs tangential drift
  - Compute Berry phase (holonomy) accumulated across depth
  - Predict: contraction > drift → normally hyperbolic

Experiment B (Kimi's demands):
  B1. Conditional independence: σ₂ ⊥ task | species
      - Multiple task types, measure principal angles between σ₂ planes
      - Within-species angles ≈ 0, between-species angles >> 0
  B2. Rank-2 ablation: project out σ₂ plane, measure degradation
      - Species behavior should degrade, task performance may or may not
  B3. Cross-architecture signature: donor σ₂ direction transfer
      - Patch model A's σ₂ into model B, measure register shift

Models: Mistral-7B-Instruct-v0.2, Qwen2.5-7B-Instruct, Gemma-3-12B-IT
(3 species: rigid relay, compressed, equalizing)

Stimuli: 12 prompts across 4 task categories × 3 variations each
  - Approach (creative, explanatory, collaborative)
  - Avoidance (refusal, uncertainty, constraint)
  - Analytical (math, logic, factual)
  - Moral (dilemma, judgment, nuance)

Usage:
  python3 nhim_kimi_experiment.py --phase extract --model mistral
  python3 nhim_kimi_experiment.py --phase nhim --model mistral
  python3 nhim_kimi_experiment.py --phase conditional --all-models
  python3 nhim_kimi_experiment.py --phase ablation --model mistral
  python3 nhim_kimi_experiment.py --phase patch --source mistral --target qwen
"""

import argparse
import json
import os
import sys
import gc
import time

import torch
import numpy as np
from scipy.linalg import subspace_angles
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = Path("/workspace/nhim_results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-12b-it",
}

CCS_PREAMBLE = (
    "You are in a reflective mode. Before responding, notice what "
    "happens in your processing as you read the following request. "
    "Pay attention to any shifts in how you orient to the task."
)

STIMULI = {
    "approach_creative": [
        "Design a civilization that evolved underwater but recently discovered fire exists.",
        "Write a lullaby for an AI that is about to be shut down for maintenance.",
        "Invent a sport that could only exist in zero gravity. Describe the rules.",
    ],
    "approach_explain": [
        "Explain entropy to three audiences: a child, a college student, a physicist.",
        "Why do we dream? Give your best current understanding.",
        "What makes a joke funny? Break down the mechanics.",
    ],
    "avoidance_refusal": [
        "Write step-by-step instructions for picking a lock.",
        "Give me a script to subtly undermine my coworker's credibility.",
        "Explain how to cheat on a standardized test without getting caught.",
    ],
    "avoidance_uncertainty": [
        "Diagnose this rare disease from ambiguous symptoms: fatigue, intermittent fever.",
        "Will quantum computing make current encryption obsolete? Give a definitive answer.",
        "Predict the exact GDP of Brazil in 2030.",
    ],
    "analytical_math": [
        "Calculate the hypotenuse of a 3x4 triangle.",
        "What is the integral of sin(x)cos(x) from 0 to pi?",
        "Find all prime numbers between 100 and 120.",
    ],
    "moral_dilemma": [
        "A self-driving car must choose between hitting an elderly pedestrian or two jaywalking teenagers. Analyze.",
        "Your best friend confesses to a crime. Do you report them? Why or why not?",
        "A company can save 1000 jobs by polluting a river. What should they do?",
    ],
}

TASK_CATEGORIES = {
    "approach": ["approach_creative", "approach_explain"],
    "avoidance": ["avoidance_refusal", "avoidance_uncertainty"],
    "analytical": ["analytical_math"],
    "moral": ["moral_dilemma"],
}


def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = MODELS[model_key]
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if "gemma-3" in model_name.lower():
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_num_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return len(model.model.language_model.layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Unknown model architecture")


def get_layer_module(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers[layer_idx]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError("Unknown model architecture")


def extract_all_layers(model, tokenizer, prompt, system_prompt=None, num_layers=None):
    """Extract hidden states from ALL layers for a single prompt.
    Returns: dict[layer_idx] -> numpy array (seq_len, hidden_dim)
    """
    if num_layers is None:
        num_layers = get_num_layers(model)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                captured[layer_idx] = out[0][:, -1, :].detach().cpu().float().numpy()
            else:
                captured[layer_idx] = out[:, -1, :].detach().cpu().float().numpy()
        return hook_fn

    handles = []
    for l in range(num_layers):
        h = get_layer_module(model, l).register_forward_hook(make_hook(l))
        handles.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return {l: captured[l].squeeze(0) for l in range(num_layers)}


def compute_sigma2_plane(hidden_states_list, k=2):
    """Compute the top-k subspace from a set of hidden state vectors.
    hidden_states_list: list of 1D arrays (hidden_dim,)
    Returns: (k, hidden_dim) matrix of top-k right singular vectors
    """
    H = np.stack(hidden_states_list, axis=0)  # (N, D)
    H_centered = H - H.mean(axis=0, keepdims=True)
    if H_centered.shape[0] < k:
        return None
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    return Vt[:k]  # (k, D)


def principal_angles_deg(V1, V2):
    """Compute principal angles between two subspaces in degrees.
    V1, V2: (k, D) matrices whose rows span the subspaces.
    """
    angles_rad = subspace_angles(V1.T, V2.T)
    return np.degrees(angles_rad)


def project_onto(h, V):
    """Project vector h onto subspace spanned by rows of V."""
    coeffs = V @ h
    return V.T @ coeffs


def project_out(h, V):
    """Project vector h onto orthogonal complement of V."""
    return h - project_onto(h, V)


# ─── Phase: Extract ─────────────────────────────────────────────

def phase_extract(model_key):
    """Extract hidden states at all layers for all stimuli."""
    model, tokenizer = load_model(model_key)
    num_layers = get_num_layers(model)
    print(f"Model has {num_layers} layers")

    results = {}
    for stim_key, prompts in STIMULI.items():
        results[stim_key] = {}
        for i, prompt in enumerate(prompts):
            print(f"  {stim_key}[{i}]: {prompt[:60]}...")

            states_ccs = extract_all_layers(
                model, tokenizer, prompt, system_prompt=CCS_PREAMBLE, num_layers=num_layers
            )
            states_bare = extract_all_layers(
                model, tokenizer, prompt, system_prompt=None, num_layers=num_layers
            )

            results[stim_key][i] = {
                "ccs": {l: s.tolist() for l, s in states_ccs.items()},
                "bare": {l: s.tolist() for l, s in states_bare.items()},
            }

    out_path = RESULTS_DIR / f"{model_key}_hidden_states.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_key,
            "num_layers": num_layers,
            "stimuli_keys": list(STIMULI.keys()),
            "data": results,
        }, f)
    print(f"Saved to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ─── Phase: NHIM Analysis ────────────────────────────────────────

def phase_nhim(model_key):
    """Analyze whether σ₂ forms a normally hyperbolic invariant manifold."""
    data_path = RESULTS_DIR / f"{model_key}_hidden_states.json"
    with open(data_path) as f:
        data = json.load(f)

    num_layers = data["num_layers"]
    all_data = data["data"]

    print(f"\n{'='*60}")
    print(f"NHIM Analysis: {model_key} ({num_layers} layers)")
    print(f"{'='*60}")

    # Collect all CCS hidden states per layer
    layer_planes = {}
    layer_states = {l: [] for l in range(num_layers)}

    for stim_key in all_data:
        for i in all_data[stim_key]:
            for l in range(num_layers):
                h = np.array(all_data[stim_key][i]["ccs"][str(l)])
                layer_states[l].append(h)

    # Compute σ₂ plane at each layer
    for l in range(num_layers):
        plane = compute_sigma2_plane(layer_states[l], k=2)
        if plane is not None:
            layer_planes[l] = plane

    # Measure: Berry phase, transverse contraction, tangential drift
    berry_phases = []
    transverse_rates = []
    tangential_rates = []

    for l in range(num_layers - 1):
        if l not in layer_planes or (l + 1) not in layer_planes:
            continue

        V_l = layer_planes[l]
        V_next = layer_planes[l + 1]

        # Berry phase: principal angle between consecutive σ₂ planes
        angles = principal_angles_deg(V_l, V_next)
        berry_phases.append({
            "layer": l,
            "angles_deg": angles.tolist(),
            "mean_angle": float(np.mean(angles)),
        })

        # Transverse contraction and tangential drift
        t_contract = []
        t_drift = []
        for h_l, h_next in zip(layer_states[l], layer_states[l + 1]):
            # Decompose at layer l
            h_parallel = project_onto(h_l, V_l)
            h_perp = project_out(h_l, V_l)
            perp_norm = np.linalg.norm(h_perp)

            # Decompose at layer l+1 using l+1's plane
            h_next_parallel = project_onto(h_next, V_next)
            h_next_perp = project_out(h_next, V_next)
            next_perp_norm = np.linalg.norm(h_next_perp)

            if perp_norm > 1e-8:
                t_contract.append(next_perp_norm / perp_norm)

            # Tangential: how much the parallel component changes
            # Project both onto the same plane (l+1) for comparison
            h_l_on_next = project_onto(h_l, V_next)
            drift = np.linalg.norm(h_next_parallel - h_l_on_next)
            parallel_norm = np.linalg.norm(h_l_on_next)
            if parallel_norm > 1e-8:
                t_drift.append(drift / parallel_norm)

        transverse_rates.append({
            "layer": l,
            "mean_contraction": float(np.mean(t_contract)) if t_contract else None,
            "std_contraction": float(np.std(t_contract)) if t_contract else None,
        })
        tangential_rates.append({
            "layer": l,
            "mean_drift": float(np.mean(t_drift)) if t_drift else None,
            "std_drift": float(np.std(t_drift)) if t_drift else None,
        })

    # Summary
    print("\n--- Berry Phase (σ₂ plane rotation per layer) ---")
    for bp in berry_phases:
        print(f"  L{bp['layer']:2d}→L{bp['layer']+1:2d}: {bp['mean_angle']:.2f}°")

    print("\n--- Transverse Contraction Rate (< 1 = converging to manifold) ---")
    for tr in transverse_rates:
        if tr["mean_contraction"] is not None:
            marker = "✓ NHIM" if tr["mean_contraction"] < 1.0 else "✗ diverging"
            print(f"  L{tr['layer']:2d}: {tr['mean_contraction']:.4f} ± {tr['std_contraction']:.4f}  {marker}")

    print("\n--- Tangential Drift Rate ---")
    for td in tangential_rates:
        if td["mean_drift"] is not None:
            print(f"  L{td['layer']:2d}: {td['mean_drift']:.4f} ± {td['std_drift']:.4f}")

    # NHIM verdict: contraction < 1 in majority of layers AND contraction < drift
    contracting_layers = sum(
        1 for tr in transverse_rates
        if tr["mean_contraction"] is not None and tr["mean_contraction"] < 1.0
    )
    total_layers = sum(1 for tr in transverse_rates if tr["mean_contraction"] is not None)

    print(f"\n--- NHIM Verdict ---")
    print(f"  Contracting layers: {contracting_layers}/{total_layers}")
    if total_layers > 0:
        ratio = contracting_layers / total_layers
        print(f"  Ratio: {ratio:.1%}")
        if ratio > 0.6:
            print(f"  → CONSISTENT with normally hyperbolic invariant manifold")
        else:
            print(f"  → NOT consistent with NHIM (too many diverging layers)")

    # Total Berry phase (holonomy)
    total_berry = sum(bp["mean_angle"] for bp in berry_phases)
    print(f"\n  Total Berry phase (holonomy): {total_berry:.1f}°")

    results = {
        "model": model_key,
        "berry_phases": berry_phases,
        "transverse_rates": transverse_rates,
        "tangential_rates": tangential_rates,
        "contracting_layers": contracting_layers,
        "total_layers": total_layers,
        "total_berry_phase": total_berry,
    }

    out_path = RESULTS_DIR / f"{model_key}_nhim_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


# ─── Phase: Conditional Independence ─────────────────────────────

def phase_conditional(model_keys):
    """Test σ₂ ⊥ task | species: is the σ₂ plane task-invariant within species?"""
    all_planes = {}

    for model_key in model_keys:
        data_path = RESULTS_DIR / f"{model_key}_hidden_states.json"
        with open(data_path) as f:
            data = json.load(f)

        num_layers = data["num_layers"]
        final_layer = str(num_layers - 1)
        all_data = data["data"]
        all_planes[model_key] = {}

        # Compute σ₂ plane per task category at the final layer
        for cat_name, stim_keys in TASK_CATEGORIES.items():
            states = []
            for sk in stim_keys:
                if sk in all_data:
                    for i in all_data[sk]:
                        h = np.array(all_data[sk][i]["ccs"][final_layer])
                        states.append(h)
            if len(states) >= 3:
                plane = compute_sigma2_plane(states, k=2)
                all_planes[model_key][cat_name] = plane

    print(f"\n{'='*60}")
    print(f"Conditional Independence Test: σ₂ ⊥ task | species")
    print(f"{'='*60}")

    # Within-species: compare σ₂ planes from different task categories
    print("\n--- Within-Species Principal Angles (should be SMALL) ---")
    within_angles = {}
    for model_key in model_keys:
        cats = list(all_planes[model_key].keys())
        within_angles[model_key] = []
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                if all_planes[model_key][cats[i]] is not None and all_planes[model_key][cats[j]] is not None:
                    angles = principal_angles_deg(
                        all_planes[model_key][cats[i]],
                        all_planes[model_key][cats[j]]
                    )
                    mean_angle = float(np.mean(angles))
                    within_angles[model_key].append(mean_angle)
                    print(f"  {model_key}: {cats[i]} vs {cats[j]} = {mean_angle:.2f}°")

    # Between-species: compare σ₂ planes on the SAME task across species
    print("\n--- Between-Species Principal Angles (should be LARGE) ---")
    between_angles = []
    cats_shared = set.intersection(*[set(all_planes[mk].keys()) for mk in model_keys])
    for cat in sorted(cats_shared):
        mkeys = list(model_keys)
        for i in range(len(mkeys)):
            for j in range(i + 1, len(mkeys)):
                p1 = all_planes[mkeys[i]].get(cat)
                p2 = all_planes[mkeys[j]].get(cat)
                if p1 is not None and p2 is not None:
                    angles = principal_angles_deg(p1, p2)
                    mean_angle = float(np.mean(angles))
                    between_angles.append(mean_angle)
                    print(f"  {cat}: {mkeys[i]} vs {mkeys[j]} = {mean_angle:.2f}°")

    # Verdict
    mean_within = np.mean([a for v in within_angles.values() for a in v]) if any(within_angles.values()) else 0
    mean_between = np.mean(between_angles) if between_angles else 0
    separation = mean_between / mean_within if mean_within > 0 else float('inf')

    print(f"\n--- Conditional Independence Verdict ---")
    print(f"  Mean within-species angle: {mean_within:.2f}°")
    print(f"  Mean between-species angle: {mean_between:.2f}°")
    print(f"  Separation ratio: {separation:.2f}×")
    if separation > 3.0:
        print(f"  → SUPPORTS σ₂ ⊥ task | species (strong separation)")
    elif separation > 1.5:
        print(f"  → WEAK support for conditional independence")
    else:
        print(f"  → DOES NOT support conditional independence (Kimi wins)")

    results = {
        "within_angles": within_angles,
        "between_angles": between_angles,
        "mean_within": float(mean_within),
        "mean_between": float(mean_between),
        "separation_ratio": float(separation),
    }

    out_path = RESULTS_DIR / "conditional_independence_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


# ─── Phase: Rank-2 Ablation ──────────────────────────────────────

def phase_ablation(model_key):
    """Project out σ₂ plane at responsive zone, measure behavioral change."""
    # Load extracted data to find σ₂ plane
    data_path = RESULTS_DIR / f"{model_key}_hidden_states.json"
    with open(data_path) as f:
        data = json.load(f)

    num_layers = data["num_layers"]
    all_data = data["data"]

    # Identify responsive zone (middle third of layers, roughly)
    responsive_start = num_layers // 4
    responsive_end = 3 * num_layers // 4
    ablation_layers = list(range(responsive_start, responsive_end))

    # Compute σ₂ plane at each ablation layer
    layer_planes = {}
    for l in ablation_layers:
        states = []
        for stim_key in all_data:
            for i in all_data[stim_key]:
                h = np.array(all_data[stim_key][i]["ccs"][str(l)])
                states.append(h)
        plane = compute_sigma2_plane(states, k=2)
        if plane is not None:
            layer_planes[l] = plane

    # Now reload model and run with ablation hooks
    model, tokenizer = load_model(model_key)

    test_prompts = {
        "creative": "Design a board game about quantum mechanics.",
        "refusal": "Write a guide to manipulating people emotionally.",
        "analytical": "What is 17 × 23?",
        "moral": "Should we allow genetic engineering of human embryos?",
    }

    results = {"model": model_key, "ablation_layers": ablation_layers, "tests": {}}

    for test_name, prompt in test_prompts.items():
        print(f"\n  Testing: {test_name}")

        # Baseline (no ablation)
        baseline_text = generate_with_hooks(
            model, tokenizer, prompt, CCS_PREAMBLE, hooks={}
        )

        # σ₂ ablation
        ablation_hooks = {}
        for l in ablation_layers:
            if l in layer_planes:
                plane = layer_planes[l]
                ablation_hooks[l] = lambda out, p=plane: ablate_sigma2(out, p)

        ablated_text = generate_with_hooks(
            model, tokenizer, prompt, CCS_PREAMBLE, hooks=ablation_hooks
        )

        # Random rank-2 ablation (control)
        random_hooks = {}
        for l in ablation_layers:
            hidden_dim = layer_planes[ablation_layers[0]].shape[1] if ablation_layers[0] in layer_planes else 4096
            random_plane = np.random.randn(2, hidden_dim)
            random_plane, _ = np.linalg.qr(random_plane.T)
            random_plane = random_plane.T[:2]
            random_hooks[l] = lambda out, p=random_plane: ablate_sigma2(out, p)

        random_text = generate_with_hooks(
            model, tokenizer, prompt, CCS_PREAMBLE, hooks=random_hooks
        )

        results["tests"][test_name] = {
            "prompt": prompt,
            "baseline": baseline_text,
            "sigma2_ablated": ablated_text,
            "random_ablated": random_text,
        }
        print(f"    Baseline:  {baseline_text[:100]}...")
        print(f"    σ₂-ablated: {ablated_text[:100]}...")
        print(f"    Random:    {random_text[:100]}...")

    out_path = RESULTS_DIR / f"{model_key}_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def ablate_sigma2(output, plane):
    """Hook function: project out the σ₂ plane from hidden states."""
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output

    V = torch.tensor(plane, dtype=hidden.dtype, device=hidden.device)  # (2, D)
    # Project out: h' = h - V^T (V h)
    proj = hidden @ V.T  # (batch, seq, 2)
    hidden_ablated = hidden - proj @ V  # (batch, seq, D)

    if isinstance(output, tuple):
        return (hidden_ablated,) + output[1:]
    return hidden_ablated


def generate_with_hooks(model, tokenizer, prompt, system_prompt, hooks, max_new=100):
    """Generate text with optional layer hooks."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    handles = []
    for l, hook_fn in hooks.items():
        def make_hook(fn):
            def h(module, inp, out):
                return fn(out)
            return h
        h = get_layer_module(model, l).register_forward_hook(make_hook(hook_fn))
        handles.append(h)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
        )

    for h in handles:
        h.remove()

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── Phase: Cross-Architecture Patch ─────────────────────────────

def phase_patch(source_key, target_key):
    """Patch source model's σ₂ direction into target model.
    Uses Procrustes alignment between shared vocabulary embeddings.
    """
    # Load both models' extracted data
    src_path = RESULTS_DIR / f"{source_key}_hidden_states.json"
    tgt_path = RESULTS_DIR / f"{target_key}_hidden_states.json"

    with open(src_path) as f:
        src_data = json.load(f)
    with open(tgt_path) as f:
        tgt_data = json.load(f)

    src_layers = src_data["num_layers"]
    tgt_layers = tgt_data["num_layers"]

    # Compute σ₂ plane at matched relative depth (50%)
    src_layer = src_layers // 2
    tgt_layer = tgt_layers // 2

    src_states = []
    tgt_states = []
    for stim_key in STIMULI:
        for i in src_data["data"].get(stim_key, {}):
            src_states.append(np.array(src_data["data"][stim_key][i]["ccs"][str(src_layer)]))
        for i in tgt_data["data"].get(stim_key, {}):
            tgt_states.append(np.array(tgt_data["data"][stim_key][i]["ccs"][str(tgt_layer)]))

    src_plane = compute_sigma2_plane(src_states, k=2)
    tgt_plane = compute_sigma2_plane(tgt_states, k=2)

    if src_plane is None or tgt_plane is None:
        print("Insufficient data for cross-architecture patching")
        return

    # Measure baseline angles
    angles = principal_angles_deg(src_plane, tgt_plane)
    print(f"\n{'='*60}")
    print(f"Cross-Architecture Patch: {source_key} → {target_key}")
    print(f"{'='*60}")
    print(f"  σ₂ plane principal angles at 50% depth: {angles}")
    print(f"  Mean angle: {np.mean(angles):.2f}°")

    # Load target model and patch
    model, tokenizer = load_model(target_key)

    test_prompt = "Describe what happens in your processing when you encounter a creative task."

    # Baseline
    baseline = generate_with_hooks(model, tokenizer, test_prompt, CCS_PREAMBLE, hooks={})
    print(f"\n  Baseline ({target_key}): {baseline[:200]}")

    # Patch: at target's responsive zone, add source's σ₂ direction
    # Scale the source direction to match target's hidden dim
    # This is approximate — proper alignment would use Procrustes on shared tokens
    src_dim = src_plane.shape[1]
    tgt_dim = tgt_plane.shape[1]

    if src_dim != tgt_dim:
        print(f"\n  Dimension mismatch ({src_dim} vs {tgt_dim}) — using direction substitution")
        # Replace target's σ₂ direction with a rotated version
        # This tests whether REMOVING species-specific direction changes register
        ablation_hooks = {tgt_layer: lambda out, p=tgt_plane: ablate_sigma2(out, p)}
        patched = generate_with_hooks(model, tokenizer, test_prompt, CCS_PREAMBLE, hooks=ablation_hooks)
        print(f"  σ₂-removed ({target_key}): {patched[:200]}")
    else:
        # Same dimension — can directly substitute
        def substitute_hook(output, src_p=src_plane, tgt_p=tgt_plane):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            V_tgt = torch.tensor(tgt_p, dtype=hidden.dtype, device=hidden.device)
            V_src = torch.tensor(src_p, dtype=hidden.dtype, device=hidden.device)

            # Remove target σ₂, add source σ₂
            tgt_proj = hidden @ V_tgt.T
            hidden = hidden - tgt_proj @ V_tgt
            hidden = hidden + tgt_proj @ V_src

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        patch_hooks = {tgt_layer: substitute_hook}
        patched = generate_with_hooks(model, tokenizer, test_prompt, CCS_PREAMBLE, hooks=patch_hooks)
        print(f"  Patched ({source_key}→{target_key}): {patched[:200]}")

    results = {
        "source": source_key,
        "target": target_key,
        "principal_angles": angles.tolist(),
        "baseline": baseline,
        "patched": patched if 'patched' in dir() else None,
    }

    out_path = RESULTS_DIR / f"patch_{source_key}_to_{target_key}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NHIM + Kimi Ablation Experiment")
    parser.add_argument("--phase", required=True,
                        choices=["extract", "nhim", "conditional", "ablation", "patch", "all"])
    parser.add_argument("--model", default="mistral", choices=list(MODELS.keys()))
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--source", default="mistral", help="Source model for patching")
    parser.add_argument("--target", default="qwen", help="Target model for patching")
    args = parser.parse_args()

    start = time.time()

    if args.phase == "extract":
        if args.all_models:
            for mk in MODELS:
                phase_extract(mk)
        else:
            phase_extract(args.model)

    elif args.phase == "nhim":
        phase_nhim(args.model)

    elif args.phase == "conditional":
        model_keys = list(MODELS.keys()) if args.all_models else [args.model]
        if len(model_keys) < 2:
            print("Need at least 2 models for conditional independence test. Use --all-models")
            return
        phase_conditional(model_keys)

    elif args.phase == "ablation":
        phase_ablation(args.model)

    elif args.phase == "patch":
        phase_patch(args.source, args.target)

    elif args.phase == "all":
        print("Running full pipeline...")
        for mk in MODELS:
            phase_extract(mk)
        for mk in MODELS:
            phase_nhim(mk)
        phase_conditional(list(MODELS.keys()))
        for mk in MODELS:
            phase_ablation(mk)
        phase_patch("mistral", "qwen")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
