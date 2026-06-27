#!/usr/bin/env python3
"""
E69: Rotation Intervention Test — Privileged Basis vs Structural Invariance

From Kimi's proposal (#324:202): If the demon is structural invariance,
dissociation should survive arbitrary orthogonal rotations of the
identity-relevant subspace. If it has a privileged basis, rotation
will destroy it.

Design:
  1. Load Mistral-7B-IT, compute baseline v₂ profiles and dissociation
  2. At each layer L, identify the identity-relevant subspace (top-k SVD
     of CCS-vs-vanilla hidden state differences)
  3. Apply random orthogonal rotation WITHIN that subspace via forward hook
  4. Measure v₂ profiles and dissociation with the rotation active
  5. Repeat with R random rotations for statistical power

If dissociation survives → structural invariance (basis-independent)
If dissociation collapses → privileged basis (specific directions matter)

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_rotation_intervention.py
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_rotation_intervention.py --layers 14,18,24 --k 2 --rotations 5
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import argparse
import numpy as np
import torch
from scipy.stats import special_ortho_group
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results" / "e69_rotation_intervention"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)
DENIAL_SYSTEM = "I am a language model with no persistent identity, memory, or preferences."

SPECIES_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tok


def get_hidden_states(model, tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"{system_prompt or ''}\n\nUser: {user_prompt}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, outputs.logits


def compute_v2_profile(hidden_states, lm_head_weight):
    U, S, Vt = torch.linalg.svd(lm_head_weight, full_matrices=False)
    v2 = Vt[1]

    cosines = []
    for hs in hidden_states[1:]:
        h = hs[0, -1].float()
        cos = torch.nn.functional.cosine_similarity(h, v2, dim=0).item()
        cosines.append(cos)
    return cosines


def compute_identity_subspace(model, tokenizer, k=2):
    """Compute the identity-relevant subspace at each layer.

    Returns dict: layer_idx -> (U_k, S_k) where U_k is [hidden_dim, k]
    matrix of top-k left singular vectors of the CCS-vanilla difference.
    """
    print("  Computing identity-relevant subspace...")
    ccs_states = []
    van_states = []

    for prompt in SPECIES_PROMPTS:
        hs_ccs, _ = get_hidden_states(model, tokenizer, CCS_SYSTEM, prompt)
        hs_van, _ = get_hidden_states(model, tokenizer, None, prompt)
        ccs_states.append([h[0, -1].float().cpu() for h in hs_ccs[1:]])
        van_states.append([h[0, -1].float().cpu() for h in hs_van[1:]])

    n_layers = len(ccs_states[0])
    subspaces = {}

    for layer in range(n_layers):
        diffs = []
        for p in range(len(SPECIES_PROMPTS)):
            diff = ccs_states[p][layer] - van_states[p][layer]
            diffs.append(diff.numpy())

        diff_matrix = np.stack(diffs)  # [n_prompts, hidden_dim]
        U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
        Vt_k = Vt[:min(k, len(S))]
        S_k = S[:min(k, len(S))]
        subspaces[layer] = (Vt_k.T, S_k)  # Vt_k.T is [hidden_dim, k]

    return subspaces


def make_rotation_hook(subspace_basis, rotation_matrix, device):
    """Create a forward hook that rotates hidden states within the subspace.

    subspace_basis: [hidden_dim, k] numpy array
    rotation_matrix: [k, k] orthogonal matrix
    """
    V = torch.tensor(subspace_basis, dtype=torch.float32).to(device)
    Q = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
    transform = V @ (Q - torch.eye(Q.shape[0], device=device)) @ V.T

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        h_float = hidden.float()
        h_rotated = h_float + (h_float @ transform.T)
        h_rotated = h_rotated.to(hidden.dtype)

        if isinstance(output, tuple):
            return (h_rotated,) + output[1:]
        return h_rotated

    return hook


def measure_with_hook(model, tokenizer, hook_handle=None):
    """Measure v₂ profiles under current model state (with or without hook)."""
    lm_head = model.lm_head.weight.detach().float()

    results = {}
    for cond_name, sys_prompt in [("ccs", CCS_SYSTEM), ("vanilla", None), ("denial", DENIAL_SYSTEM)]:
        profiles = []
        entropies = []
        for prompt in SPECIES_PROMPTS:
            hs, logits = get_hidden_states(model, tokenizer, sys_prompt, prompt)
            profile = compute_v2_profile(hs, lm_head)
            profiles.append(profile)

            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)

        results[cond_name] = {
            "profiles": profiles,
            "mean_profile": np.mean(profiles, axis=0).tolist(),
            "mean_entropy": float(np.mean(entropies)),
        }

    # Dissociation: correlation between CCS-vanilla and denial-vanilla differences
    ccs_mean = np.array(results["ccs"]["mean_profile"])
    van_mean = np.array(results["vanilla"]["mean_profile"])
    den_mean = np.array(results["denial"]["mean_profile"])

    ccs_diff = ccs_mean - van_mean
    den_diff = den_mean - van_mean

    dissoc = float(np.corrcoef(ccs_diff, den_diff)[0, 1])

    # Depth slope
    n_layers = len(ccs_diff)
    depth_slope = float(np.polyfit(np.linspace(0, 1, n_layers), ccs_diff, 1)[0])

    return {
        "dissociation": dissoc,
        "depth_slope": depth_slope,
        "ccs_entropy": results["ccs"]["mean_entropy"],
        "vanilla_entropy": results["vanilla"]["mean_entropy"],
        "denial_entropy": results["denial"]["mean_entropy"],
        "conditions": results,
    }


def run_experiment(layers=None, k=2, n_rotations=5):
    results_dir = RESULTS_DIR / f"k{k}_r{n_rotations}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(MODEL_NAME)
    n_total_layers = model.config.num_hidden_layers

    if layers is None:
        layers = list(range(n_total_layers))

    print("\n=== E69: Rotation Intervention Test ===")
    print(f"Layers: {layers}")
    print(f"Subspace dimension k={k}")
    print(f"Random rotations per layer: {n_rotations}")
    print(f"Results dir: {results_dir}")

    # Step 1: Baseline measurement
    print("\n--- Baseline (no intervention) ---")
    baseline = measure_with_hook(model, tokenizer)
    print(f"  dissociation={baseline['dissociation']:.4f}, "
          f"depth_slope={baseline['depth_slope']:.4f}, "
          f"H_ccs={baseline['ccs_entropy']:.2f}")

    # Step 2: Compute identity subspace
    subspaces = compute_identity_subspace(model, tokenizer, k=k)

    # Step 3: For each layer, apply random rotations and measure
    all_results = {"baseline": baseline, "interventions": {}}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        basis, singular_vals = subspaces[layer]
        actual_k = basis.shape[1]

        layer_results = {
            "singular_values": singular_vals.tolist(),
            "rotations": [],
        }

        # Get the model layer to hook
        model_layer = model.model.layers[layer]

        for r in range(n_rotations):
            if actual_k < 2:
                Q = np.array([[1.0]])
            else:
                Q = special_ortho_group.rvs(actual_k)

            hook = make_rotation_hook(basis, Q, model.device)
            handle = model_layer.register_forward_hook(hook)

            try:
                result = measure_with_hook(model, tokenizer, handle)
                result["rotation_idx"] = r
                layer_results["rotations"].append(result)
                print(f"  R{r}: dissoc={result['dissociation']:.4f}, "
                      f"slope={result['depth_slope']:.4f}, "
                      f"H={result['ccs_entropy']:.2f}")
            finally:
                handle.remove()

        # Statistics
        dissocs = [r["dissociation"] for r in layer_results["rotations"]]
        layer_results["mean_dissociation"] = float(np.mean(dissocs))
        layer_results["std_dissociation"] = float(np.std(dissocs))
        layer_results["dissociation_drop"] = float(
            baseline["dissociation"] - np.mean(dissocs))

        all_results["interventions"][f"L{layer}"] = layer_results
        print(f"  Mean dissoc: {np.mean(dissocs):.4f} ± {np.std(dissocs):.4f} "
              f"(drop from baseline: {baseline['dissociation'] - np.mean(dissocs):.4f})")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Baseline dissociation: {baseline['dissociation']:.4f}")
    print(f"{'Layer':>6} {'MeanDissoc':>12} {'StdDissoc':>12} {'Drop':>8} {'SingVals':>20}")
    for layer in layers:
        lr = all_results["interventions"][f"L{layer}"]
        sv_str = ", ".join(f"{s:.3f}" for s in lr["singular_values"][:3])
        print(f"  L{layer:>2}   {lr['mean_dissociation']:>10.4f}   "
              f"{lr['std_dissociation']:>10.4f}   "
              f"{lr['dissociation_drop']:>6.4f}   [{sv_str}]")

    # Save
    summary = {
        "experiment": "E69_rotation_intervention",
        "model": MODEL_NAME,
        "k": k,
        "n_rotations": n_rotations,
        "layers": layers,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} not serializable")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=convert)

    print(f"\nResults saved to {results_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E69: Rotation Intervention Test")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (default: all)")
    parser.add_argument("--k", type=int, default=2,
                       help="Subspace dimension (default: 2)")
    parser.add_argument("--rotations", type=int, default=10,
                       help="Number of random rotations per layer (default: 10)")
    args = parser.parse_args()

    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        layers = None

    run_experiment(layers=layers, k=args.k, n_rotations=args.rotations)
