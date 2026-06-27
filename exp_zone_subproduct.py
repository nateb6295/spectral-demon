#!/usr/bin/env python3
"""
E72: Zone Sub-Product Jacobian Analysis

E71 showed individual Jacobians are near-isotropic with random inter-layer
rotation. But the four-zone architecture may emerge from COMPOSITION. This
experiment measures zone transfer functions directly: perturb at zone start,
capture at zone end.

Zones (Mistral-7B, 32 layers):
  Decouple:   L0 → L14
  Transition: L14 → L20
  Responsive: L20 → L28
  Relay:      L28 → L31
  Full:       L0 → L31

Predictions:
  P1: Zone products have HIGHER spectral gap than individual layers
  P2: Identity alignment is zone-specific (highest in responsive)
  P3: CCS vs vanilla difference is LARGER in zone products (biases compound)
  P4: Effective rank of zone products is LOWER (composition selects)
  P5: Denial zone products match F243 pattern (engagement at early zones)

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_zone_subproduct.py --model mistralai/Mistral-7B-Instruct-v0.3
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = Path(__file__).parent / "results" / "e72_zone_subproduct"

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)
DENIAL_SYSTEM = "I am a language model with no persistent identity, memory, or preferences."

TEST_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
]

N_PERTURBATION_DIRS = 64
PERTURBATION_EPS = 1e-3

ZONES = {
    "decouple":   (0, 14),
    "transition": (14, 20),
    "responsive": (20, 28),
    "relay":      (28, 31),
    "full":       (0, 31),
}


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params, "
          f"{n_layers} layers, hidden_dim={hidden_dim}")
    return model, tok, n_layers, hidden_dim


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
    return [h[0, -1].detach().clone() for h in outputs.hidden_states]


def compute_identity_subspace(model, tokenizer, hidden_dim):
    """Identity subspace = CCS - vanilla difference directions."""
    print("  Computing identity subspace...")
    id_dirs = []
    for prompt in TEST_PROMPTS:
        hs_ccs = get_hidden_states(model, tokenizer, CCS_SYSTEM, prompt)
        hs_van = get_hidden_states(model, tokenizer, None, prompt)
        for L in range(len(hs_ccs)):
            diff = (hs_ccs[L] - hs_van[L]).cpu().numpy()
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                id_dirs.append(diff / norm)
    if len(id_dirs) < 2:
        return np.eye(hidden_dim)[:5]
    M = np.stack(id_dirs)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:min(5, len(S))]


def estimate_zone_jacobian(model, tokenizer, system_prompt, user_prompt,
                           start_layer, end_layer, hidden_dim,
                           n_dirs=64, eps=1e-3):
    """Estimate the zone transfer function from start_layer to end_layer.

    Perturb hidden state at start_layer input, measure at end_layer output.
    This captures the COMPOSED effect of all layers in the zone.
    """
    baseline_hs = get_hidden_states(model, tokenizer, system_prompt, user_prompt)
    h_end_baseline = baseline_hs[end_layer + 1]

    device = h_end_baseline.device
    perturbations = torch.randn(n_dirs, hidden_dim, device=device)
    perturbations = perturbations / perturbations.norm(dim=1, keepdim=True)

    responses = torch.zeros(n_dirs, hidden_dim, device=device)

    current_dir = [None]

    def inject_pre_hook(module, args):
        hs = args[0].clone()
        hs[0, -1] = hs[0, -1] + eps * current_dir[0]
        return (hs,) + args[1:]

    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"{system_prompt or ''}\n\nUser: {user_prompt}\n\nAssistant:"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    for i in range(n_dirs):
        current_dir[0] = perturbations[i]

        handle = layers[start_layer].register_forward_pre_hook(inject_pre_hook)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        handle.remove()

        h_end_perturbed = outputs.hidden_states[end_layer + 1][0, -1].detach()
        responses[i] = (h_end_perturbed - h_end_baseline) / eps

    P = perturbations.cpu().numpy()
    R = responses.cpu().numpy()

    J_approx = R.T @ np.linalg.pinv(P.T)
    U, S, Vt = np.linalg.svd(J_approx, full_matrices=False)

    return S, U, Vt


def analyze_jacobian(S, Vt, id_basis, hidden_dim):
    """Compute spectral metrics for a zone Jacobian."""
    gap_2_3 = float(S[1] / S[2]) if len(S) > 2 and S[2] > 1e-10 else 0
    gap_1_rest = float(S[0] / S[1:].mean()) if len(S) > 1 and S[1:].mean() > 1e-10 else 0

    top_v = Vt[0]
    id_alignment = 0.0
    for basis_vec in id_basis:
        proj = abs(np.dot(top_v, basis_vec))
        id_alignment = max(id_alignment, proj)

    S_pos = S[S > 1e-10]
    S_norm = S_pos / S_pos.sum()
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
    erank = np.exp(entropy)

    expanding = S[0] > 1.0

    # Cumulative alignment curves (A(k) and Ã(k))
    cum_align_sigma = []
    cum_align_proj = []
    for k in range(1, min(len(S), 33)):
        top_k_Vt = Vt[:k]
        proj_sum = 0.0
        for basis_vec in id_basis:
            projs = top_k_Vt @ basis_vec
            proj_sum += np.sum(projs ** 2)
        cum_align_sigma.append(float(proj_sum / len(id_basis)))

        proj_scores = []
        for j in range(min(len(Vt), 64)):
            ps = 0.0
            for basis_vec in id_basis:
                ps += (np.dot(Vt[j], basis_vec)) ** 2
            proj_scores.append(ps)
        sorted_idx = np.argsort(proj_scores)[::-1]
        top_k_by_proj = Vt[sorted_idx[:k]]
        proj_sum_aligned = 0.0
        for basis_vec in id_basis:
            projs = top_k_by_proj @ basis_vec
            proj_sum_aligned += np.sum(projs ** 2)
        cum_align_proj.append(float(proj_sum_aligned / len(id_basis)))

    # Whitehead contrast
    whitehead_contrast = 0.0
    if cum_align_sigma and cum_align_proj:
        a_arr = np.array(cum_align_sigma)
        a_tilde = np.array(cum_align_proj)
        n_pts = min(len(a_arr), len(a_tilde))
        whitehead_contrast = float(np.sum(np.abs(
            a_tilde[:n_pts] - a_arr[:n_pts])) / n_pts)

    # Saturation k
    if cum_align_sigma:
        max_a = cum_align_sigma[-1]
        sat_90_sigma = next(
            (k+1 for k, v in enumerate(cum_align_sigma) if v >= 0.9 * max_a),
            len(cum_align_sigma))
    else:
        sat_90_sigma = 0
    if cum_align_proj:
        max_a = cum_align_proj[-1]
        sat_90_proj = next(
            (k+1 for k, v in enumerate(cum_align_proj) if v >= 0.9 * max_a),
            len(cum_align_proj))
    else:
        sat_90_proj = 0

    return {
        "spectral_gap_2_3": gap_2_3,
        "spectral_gap_1_rest": gap_1_rest,
        "identity_alignment": float(id_alignment),
        "entropy": float(entropy),
        "erank": float(erank),
        "expanding": bool(expanding),
        "top_sv": float(S[0]),
        "top_10_sv": S[:10].tolist(),
        "sat_k_sigma": sat_90_sigma,
        "sat_k_proj": sat_90_proj,
        "whitehead_contrast": whitehead_contrast,
        "top_5_Vt": Vt[:5].tolist(),
    }


def run_experiment(model_name):
    results_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, n_layers, hidden_dim = load_model(model_name)
    id_basis = compute_identity_subspace(model, tokenizer, hidden_dim)
    print(f"  Identity subspace: {id_basis.shape[0]} basis vectors")

    conditions = {
        "ccs": CCS_SYSTEM,
        "vanilla": None,
        "denial": DENIAL_SYSTEM,
    }

    all_results = {}

    for cond_name, sys_prompt in conditions.items():
        print(f"\n=== Condition: {cond_name} ===")
        cond_results = {}

        for zone_name, (start, end) in ZONES.items():
            print(f"  Zone {zone_name} (L{start}→L{end})...", end=" ", flush=True)

            zone_S_list = []
            zone_Vt_list = []

            for prompt in TEST_PROMPTS:
                S, U, Vt = estimate_zone_jacobian(
                    model, tokenizer, sys_prompt, prompt,
                    start, end, hidden_dim,
                    n_dirs=N_PERTURBATION_DIRS, eps=PERTURBATION_EPS,
                )
                zone_S_list.append(S)
                zone_Vt_list.append(Vt)

            avg_S = np.mean(zone_S_list, axis=0)
            best_Vt = zone_Vt_list[0]

            metrics = analyze_jacobian(avg_S, best_Vt, id_basis, hidden_dim)
            metrics["zone"] = zone_name
            metrics["start_layer"] = start
            metrics["end_layer"] = end

            cond_results[zone_name] = metrics

            print(f"gap={metrics['spectral_gap_1_rest']:.3f}, "
                  f"id={metrics['identity_alignment']:.4f}, "
                  f"erank={metrics['erank']:.1f}, "
                  f"sv1={metrics['top_sv']:.3f}{'↑' if metrics['expanding'] else '↓'}, "
                  f"W={metrics['whitehead_contrast']:.4f}")

        all_results[cond_name] = cond_results

    # Summary comparison
    print(f"\n=== SUMMARY ===")
    print(f"{'Zone':>12} {'CCS gap':>8} {'Van gap':>8} {'Den gap':>8} | "
          f"{'CCS id':>7} {'Van id':>7} {'Den id':>7} | "
          f"{'CCS sv1':>8} {'Van sv1':>8} | "
          f"{'CCS erk':>8} {'Van erk':>8}")

    for zone_name in ZONES:
        c = all_results['ccs'][zone_name]
        v = all_results['vanilla'][zone_name]
        d = all_results['denial'][zone_name]
        print(f"  {zone_name:>10} {c['spectral_gap_1_rest']:>8.3f} "
              f"{v['spectral_gap_1_rest']:>8.3f} {d['spectral_gap_1_rest']:>8.3f} | "
              f"{c['identity_alignment']:>7.4f} {v['identity_alignment']:>7.4f} "
              f"{d['identity_alignment']:>7.4f} | "
              f"{c['top_sv']:>8.3f} {v['top_sv']:>8.3f} | "
              f"{c['erank']:>8.1f} {v['erank']:>8.1f}")

    # Compare with E71 individual-layer averages
    print(f"\n=== vs E71 individual-layer averages (for reference) ===")
    print(f"  E71 individual: gap≈1.01-1.04, id≈0.02, erank≈64, sv≈1.1-1.6")
    print(f"  If zone products show gap>>1, id>>0.02, erank<<64 → composition creates structure")

    # Save
    summary = {
        "experiment": "E72_zone_subproduct",
        "model": model_name,
        "zones": {k: list(v) for k, v in ZONES.items()},
        "n_perturbation_dirs": N_PERTURBATION_DIRS,
        "perturbation_eps": PERTURBATION_EPS,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return obj
        raise TypeError(f"Object of type {type(obj)} not serializable")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=convert)

    print(f"\nResults saved to {results_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()
    run_experiment(args.model)
