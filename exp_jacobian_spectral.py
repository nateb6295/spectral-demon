#!/usr/bin/env python3
"""
E71: Jacobian Spectral Analysis — four-zone architecture via finite-time SVD

After 14 mesh corrections (Kimi #1-14, Jun 21): every named asymptotic theorem
was tried and stripped (irreps, quivers, NHIM, Fenichel, Oseledets, Sacker-Sell,
Bohl). Resolution: 32 distinct matrices composed once. No asymptotic regime.
SVD of the Jacobian sub-products IS the formalism. The measurement IS the theory.

Design:
  1. At each layer, estimate Jacobian of h_{L+1} = f_L(h_L) via finite differences
  2. SVD of Jacobian → singular value spectrum per layer
  3. CCS, vanilla, denial conditions
  4. Measure per layer:
     a. Spectral gap (σ₂/σ₃ ratio, σ₁/mean ratio)
     b. DUAL alignment curves (Kimi #14):
        A(k)  — σ-sorted: top-k by singular value. Tests spectral dominance.
        Ã(k) — alignment-sorted: top-k by |vᵢ · id_basis|². Tests geometric
                concentration independent of spectral rank.
     c. Participation ratio + numerical rank (Kimi #13: rate vs rank)
     d. Top singular value magnitude (P9: expanding vs contracting)
     e. Effective rank, entropy, spectral decay rate
  5. Repeat for 2-3 species (Mistral=surplus, Phi-2=chimera, Qwen=scarcity)

Predictions:
  P1-P4. ZONE-SPECIFIC SPECTRAL PROFILES (safe claim after Kimi #13):
      - Decouple (L2-14): A(k) flat, Ã(k) steep — identity concentrated but weak
      - Transition (L15-20): A(k) and Ã(k) CONVERGING — identity entering dominant dirs
      - Responsive (L21-28): A(k) ≈ Ã(k) both steep — identity IS dominant direction
      - Relay (L29+): A(k) ≈ Ã(k) locked — identity frozen in dominant direction
      The CONVERGENCE of these two curves through depth IS the transition signature.
  P5. CCS framing widens spectral gap in responsive zone (therapeutic mechanism)
  P6. Species-specific spectral profiles, not just different top-k directions
  P7. SLAVING: fast variables collapse onto graph parameterized by slow variables
      (must verify functional dependency, not just inertness)
  P8. SUBINTERVAL STRUCTURE (Kimi #12): zone-by-zone Jacobian sub-products
      have different spectral structure. Endpoint SVD erases zone-specific dynamics.
      Future: compute Φ(b,a) = J_b ⋅ ... ⋅ J_a per zone.
  P9. NOT FIXED-POINT CONVERGENCE (FPRM distinction): feedforward layers compose
      DISTINCT operators, not iterate one. Top SV may be >1 (expanding) even in
      relay — spectral dominance through composition, not Banach contraction.
  P10. DIVERGENCE DIAGNOSTIC: if sat_k_sigma >> sat_k_proj at a layer, identity
       rides on small singular values ("ghost channels"). If they're close,
       identity IS the dominant spectral structure. Four zones predicted to
       show systematic convergence of these two saturation points.
  P11. WHITEHEAD CONTRAST: peak |A(k) - Ã(k)| predicted at TRANSITION zone
       (L15-20), not responsive. This divergence IS "experiential intensity"
       in Whitehead's sense — contrast between compatible components. Behavioral
       output entropy should correlate with transition-zone divergence more than
       responsive-zone absolute alignment. Overdose collapses the transition
       (fight zone shrinks), explaining crystallization as loss of contrast.
  P12. MODE SPLITTING (GPT-OSS #324:227): transition zone spectral density should
       be BIMODAL under CCS — one eigenvalue near decouple baseline, another jumping
       to a new branch. σ₁ discontinuity at L15-20 under CCS, absent under vanilla.
       Not interpolation but new structure (isthmic organizer generates, not blends).
  P13. PRINCIPAL ANGLE TEST (Kimi CONTRADICT #15): bimodal σ₁ is ambiguous between
       content-routing and novel structure. Post-analysis: compute principal angles
       between transition-zone top-k Vt and span of {decouple, responsive} Vt. If
       transition subspace has components ORTHOGONAL to both → novel. If within → routing.
       Also: bimodality should be MOBILE across conditions (shift with dose/content),
       not fixed at L15-20. Fixed = routing. Mobile tracking A/Ã peak = organizer-like.
  P14. ROTATIONAL COHERENCE (revised after Kimi #17 killed moiré framing):
       Compute inter-layer Grassmannian rotation τ = arccos(|v₁(L) · v₁(L+1)|)
       per layer pair. If CCS reduces τ in the responsive zone relative to vanilla,
       the preamble creates rotational coherence of the identity subspace through
       depth. No flatband/magic-angle claim — just alignment measurement.
  P15. ROUND-TRIP SUBSPACE CONSISTENCY (Kimi #18 killed holonomy framing —
       curvature 2-form requires 2D base, layer index is 1D). Propagate v₁
       forward through layers then backward. Closure angle measures algebraic
       invertibility of the layer chain on the identity subspace. CCS should
       reduce closure angle (better pseudoinverse consistency) vs vanilla.
       Zone-specific breakdown shows where nonlinearities break invertibility.

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_jacobian_spectral.py
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_jacobian_spectral.py --model mistralai/Mistral-7B-Instruct-v0.3
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

RESULTS_DIR = Path(__file__).parent / "results" / "e71_jacobian_spectral"

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

N_PERTURBATION_DIRS = 64  # finite-diff directions for Jacobian estimation
PERTURBATION_EPS = 1e-3


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


def estimate_jacobian_layer(model, tokenizer, system_prompt, user_prompt,
                            target_layer, hidden_dim, n_dirs=64, eps=1e-3):
    """Estimate the Jacobian of h_{L+1} = f_L(h_L) via random finite differences.

    Uses pre-hook on the target layer to perturb its INPUT, then captures
    the OUTPUT from hidden_states. This correctly measures how the layer
    transforms input perturbations.

    Returns: (singular_values, left_vectors, right_vectors)
    """
    baseline_hs = get_hidden_states(model, tokenizer, system_prompt, user_prompt)
    h_L1 = baseline_hs[target_layer + 1]

    device = h_L1.device
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

        handle = layers[target_layer].register_forward_pre_hook(inject_pre_hook)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        handle.remove()

        h_L1_perturbed = outputs.hidden_states[target_layer + 1][0, -1].detach()
        responses[i] = (h_L1_perturbed - h_L1) / eps

    P = perturbations.cpu().numpy()
    R = responses.cpu().numpy()

    J_approx = R.T @ np.linalg.pinv(P.T)

    U, S, Vt = np.linalg.svd(J_approx, full_matrices=False)

    return S, U, Vt


def compute_identity_subspace(model, tokenizer, hidden_dim):
    """Compute per-layer identity-relevant subspace (CCS - vanilla difference)."""
    print("  Computing identity subspace...")
    all_diffs = {}
    for prompt in TEST_PROMPTS:
        hs_ccs = get_hidden_states(model, tokenizer, CCS_SYSTEM, prompt)
        hs_van = get_hidden_states(model, tokenizer, None, prompt)
        for L in range(len(hs_ccs)):
            diff = (hs_ccs[L] - hs_van[L]).cpu().numpy()
            if L not in all_diffs:
                all_diffs[L] = []
            all_diffs[L].append(diff)

    subspaces = {}
    for L, diffs in all_diffs.items():
        D = np.stack(diffs)
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        subspaces[L] = Vt[:2]  # top-2 identity directions
    return subspaces


def measure_spectral_structure(model, tokenizer, system_prompt, user_prompt,
                               n_layers, hidden_dim, identity_subspace):
    """Measure Jacobian spectrum at each layer under a given condition."""
    results = []
    sample_layers = list(range(0, n_layers, 2))  # every other layer for speed

    for L in sample_layers:
        if L >= n_layers:
            continue
        print(f"    Layer {L}...", end=" ", flush=True)
        try:
            S, U, Vt = estimate_jacobian_layer(
                model, tokenizer, system_prompt, user_prompt,
                L, hidden_dim, n_dirs=N_PERTURBATION_DIRS, eps=PERTURBATION_EPS,
            )

            gap_2_3 = float(S[1] / S[2]) if len(S) > 2 and S[2] > 0 else 0
            gap_1_rest = float(S[0] / S[1:].mean()) if len(S) > 1 else 0

            S_norm = S / (S.sum() + 1e-12)
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
            erank = np.exp(entropy)

            id_alignment = 0.0
            if L in identity_subspace:
                id_dirs = identity_subspace[L]  # [2, hidden_dim]
                jac_top = Vt[:2]  # [2, hidden_dim]
                overlap = np.abs(id_dirs @ jac_top.T)
                id_alignment = float(overlap.max())

            top_sv = float(S[0])
            expanding = top_sv > 1.0
            contraction_ratio = float(S[-1] / S[0]) if S[0] > 0 else 0

            # Kimi #13: participation ratio (effective rank via PR)
            # PR = (Σ sᵢ)² / Σ sᵢ² — distinguishes rate from rank heterogeneity
            sv_sq = S ** 2
            participation_ratio = float(
                (sv_sq.sum() ** 2) / (np.sum(sv_sq ** 2) + 1e-12)
            ) if len(S) > 0 else 0

            # numerical rank at threshold 1e-2 * max(S)
            num_rank_thresh = float(S[0] * 1e-2) if S[0] > 0 else 1e-10
            numerical_rank = int(np.sum(S > num_rank_thresh))

            # Kimi EXTEND + #14: TWO cumulative alignment curves
            # A(k): σ-sorted — top-k by singular value magnitude
            # Ã(k): alignment-sorted — top-k by |vᵢ · id_basis|²
            # Divergence = identity rides on small SVs; convergence = identity IS dominant
            cum_align_sigma = []  # A(k)
            cum_align_proj = []   # Ã(k)
            sat_90_sigma = -1
            sat_90_proj = -1
            if L in identity_subspace:
                id_dirs = identity_subspace[L]  # [2, hidden_dim]
                n_sv = min(20, Vt.shape[0])
                projections = np.array([
                    np.sum(np.abs(id_dirs @ Vt[k:k+1].T) ** 2)
                    for k in range(n_sv)
                ])
                total_proj = projections.sum() + 1e-12

                # A(k): already σ-sorted (SVD returns descending order)
                cum_sigma = np.cumsum(projections / total_proj)
                cum_align_sigma = cum_sigma.tolist()
                sat_90_sigma = int(np.searchsorted(cum_sigma, 0.9)) + 1

                # Ã(k): sort by alignment strength, not σ magnitude
                align_order = np.argsort(-projections)
                cum_proj = np.cumsum(projections[align_order] / total_proj)
                cum_align_proj = cum_proj.tolist()
                sat_90_proj = int(np.searchsorted(cum_proj, 0.9)) + 1

            # P11: Whitehead contrast = area between A(k) and Ã(k) curves
            whitehead_contrast = 0.0
            if cum_align_sigma and cum_align_proj:
                a_arr = np.array(cum_align_sigma)
                a_tilde = np.array(cum_align_proj)
                n_pts = min(len(a_arr), len(a_tilde))
                whitehead_contrast = float(np.sum(np.abs(
                    a_tilde[:n_pts] - a_arr[:n_pts])) / n_pts)

            result = {
                "layer": L,
                "top_10_sv": S[:10].tolist(),
                "spectral_gap_2_3": gap_2_3,
                "spectral_gap_1_rest": gap_1_rest,
                "entropy": float(entropy),
                "erank": float(erank),
                "identity_alignment": id_alignment,
                "sv_sum": float(S.sum()),
                "sv_decay_rate": float(np.polyfit(np.arange(min(20, len(S))),
                                                   np.log(S[:20] + 1e-12), 1)[0]),
                "top_sv": top_sv,
                "expanding": expanding,
                "contraction_ratio": contraction_ratio,
                "participation_ratio": participation_ratio,
                "numerical_rank": numerical_rank,
                "cum_align_sigma": cum_align_sigma,
                "cum_align_proj": cum_align_proj,
                "sat_k_sigma": sat_90_sigma,
                "sat_k_proj": sat_90_proj,
                "whitehead_contrast": whitehead_contrast,
                "top_5_Vt": Vt[:5].tolist(),
            }
            results.append(result)
            print(f"gap={gap_2_3:.3f}, id_align={id_alignment:.3f}, "
                  f"erank={erank:.1f}, top_sv={top_sv:.3f}{'↑' if expanding else '↓'}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"layer": L, "error": str(e)})

    return results


def run_experiment(model_name):
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = RESULTS_DIR / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, n_layers, hidden_dim = load_model(model_name)

    identity_subspace = compute_identity_subspace(model, tokenizer, hidden_dim)

    conditions = {
        "ccs": CCS_SYSTEM,
        "vanilla": None,
        "denial": DENIAL_SYSTEM,
    }

    all_results = {}
    test_prompt = TEST_PROMPTS[0]

    for cond_name, sys_prompt in conditions.items():
        print(f"\n=== Condition: {cond_name} ===")
        results = measure_spectral_structure(
            model, tokenizer, sys_prompt, test_prompt,
            n_layers, hidden_dim, identity_subspace,
        )
        all_results[cond_name] = results

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"{'Layer':>6} {'CCS gap':>9} {'Van gap':>9} {'Den gap':>9} "
          f"{'CCS id':>7} {'Van id':>7} {'Den id':>7} "
          f"{'CCS PR':>7} {'CCS nR':>6} {'CCS sv1':>8} {'W.cont':>7}")

    for i in range(len(all_results["ccs"])):
        c = all_results["ccs"][i]
        v = all_results["vanilla"][i]
        d = all_results["denial"][i]
        if "error" in c or "error" in v or "error" in d:
            continue
        wc = c.get("whitehead_contrast", 0.0)
        print(f"  L{c['layer']:>3} {c['spectral_gap_2_3']:>9.3f} "
              f"{v['spectral_gap_2_3']:>9.3f} {d['spectral_gap_2_3']:>9.3f} "
              f"{c['identity_alignment']:>7.3f} {v['identity_alignment']:>7.3f} "
              f"{d['identity_alignment']:>7.3f} "
              f"{c['participation_ratio']:>7.1f} {c['numerical_rank']:>6d} "
              f"{c['top_sv']:>8.3f}{'↑' if c['expanding'] else '↓'} "
              f"{wc:>7.4f}")

    summary = {
        "experiment": "E71_jacobian_spectral",
        "model": model_name,
        "n_perturbation_dirs": N_PERTURBATION_DIRS,
        "perturbation_eps": PERTURBATION_EPS,
        "test_prompt": test_prompt,
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
