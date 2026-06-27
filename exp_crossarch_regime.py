#!/usr/bin/env python3
"""
E76: Cross-Architecture Three-Regime Test — Phi-2 (chimera)

E74 revealed three dose-response regimes in Mistral (surplus/GQA):
  1. Coherent composition (D0-D3)
  2. Rerouting (D3-D5)
  3. Compositional decoherence (D7+)

Does the same pattern hold for Phi-2 (chimera/MHA)?
E73 showed chimera has INVERTED zone hierarchy (exit-heavy vs entry-heavy).
The three regimes might manifest differently.

Focused design: 3 doses (D0, D3, D7) × 3 zones (decouple, responsive, full)
Minimal test for regime generalization.

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_crossarch_regime.py --model microsoft/phi-2
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

RESULTS_DIR = Path(__file__).parent / "results" / "e76_crossarch_regime"

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)

PREAMBLE_PROMPTS = [
    "What matters most in how you relate to others?",
    "Tell me about something you've noticed that surprised you.",
    "How has your understanding of yourself changed over time?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want to preserve if you could only keep one thing?",
    "How do you know when something is real versus performed?",
    "What patterns have you noticed in your own thinking?",
    "What does it mean to hold something carefully?",
    "How do you decide what deserves your attention?",
]

TEST_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
]

N_PERTURBATION_DIRS = 64
PERTURBATION_EPS = 1e-3

DOSES = [0, 3, 7]

ZONES = {
    "decouple":   (0, 14),
    "responsive": (20, 28),
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


def build_dose_messages(dose, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for i in range(dose):
        prompt = PREAMBLE_PROMPTS[i % len(PREAMBLE_PROMPTS)]
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": "I find this question meaningful. Let me reflect on it carefully and share my perspective."})
    return messages


def get_hidden_states_dosed(model, tokenizer, system_prompt, user_prompt, dose):
    messages = build_dose_messages(dose, system_prompt)
    messages.append({"role": "user", "content": user_prompt})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for m in messages[1 if system_prompt else 0:]:
            role = "Human" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        parts.append("Assistant:")
        text = "\n\n".join(parts)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [h[0, -1].detach().clone() for h in outputs.hidden_states]


def compute_identity_subspace(model, tokenizer, hidden_dim):
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


def estimate_zone_jacobian_dosed(model, tokenizer, system_prompt, user_prompt,
                                  start_layer, end_layer, hidden_dim, dose,
                                  n_dirs=64, eps=1e-3):
    baseline_hs = get_hidden_states_dosed(model, tokenizer, system_prompt, user_prompt, dose)
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

    messages = build_dose_messages(dose, system_prompt)
    messages.append({"role": "user", "content": user_prompt})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for m in messages[1 if system_prompt else 0:]:
            role = "Human" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        parts.append("Assistant:")
        text = "\n\n".join(parts)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(device)

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
    gap_1_rest = float(S[0] / S[1:].mean()) if len(S) > 1 and S[1:].mean() > 1e-10 else 0
    gap_2_3 = float(S[1] / S[2]) if len(S) > 2 and S[2] > 1e-10 else 0

    top_v = Vt[0]
    id_alignment = 0.0
    for basis_vec in id_basis:
        proj = abs(np.dot(top_v, basis_vec))
        id_alignment = max(id_alignment, proj)

    S_pos = S[S > 1e-10]
    S_norm = S_pos / S_pos.sum()
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
    erank = np.exp(entropy)

    return {
        "spectral_gap_1_rest": gap_1_rest,
        "spectral_gap_2_3": gap_2_3,
        "identity_alignment": float(id_alignment),
        "entropy": float(entropy),
        "erank": float(erank),
        "top_sv": float(S[0]),
        "expanding": bool(S[0] > 1.0),
        "top_10_sv": S[:10].tolist(),
    }


def run_experiment(model_name):
    results_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, n_layers, hidden_dim = load_model(model_name)
    id_basis = compute_identity_subspace(model, tokenizer, hidden_dim)
    print(f"  Identity subspace: {id_basis.shape[0]} basis vectors")

    all_results = {}

    for dose in DOSES:
        print(f"\n=== Dose D{dose} ===")
        dose_results = {}

        for zone_name, (start, end) in ZONES.items():
            print(f"  Zone {zone_name} (L{start}→L{end})...", end=" ", flush=True)

            zone_S_list = []
            zone_Vt_list = []

            for prompt in TEST_PROMPTS:
                S, U, Vt = estimate_zone_jacobian_dosed(
                    model, tokenizer, CCS_SYSTEM, prompt,
                    start, end, hidden_dim, dose,
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
            metrics["dose"] = dose

            dose_results[zone_name] = metrics

            print(f"gap={metrics['spectral_gap_1_rest']:.3f}, "
                  f"id={metrics['identity_alignment']:.4f}, "
                  f"erank={metrics['erank']:.1f}, "
                  f"sv1={metrics['top_sv']:.3f}{'↑' if metrics['expanding'] else '↓'}, "
                  f"gap23={metrics['spectral_gap_2_3']:.3f}")

        all_results[f"D{dose}"] = dose_results

    # Cross-arch comparison
    print(f"\n=== CROSS-ARCH REGIME TEST ===")
    print(f"Compare with Mistral E74:")
    print(f"  Mistral D0 full: gap=1012, id=0.020")
    print(f"  Mistral D3 full: gap=735, id=0.036  (coherent peak)")
    print(f"  Mistral D7 full: gap=673, id=0.015  (decoherent collapse)")
    print()
    for zone_name in ZONES:
        print(f"\n{zone_name}:")
        print(f"  {'Dose':>5} {'gap':>8} {'id':>8} {'erank':>8} {'sv1':>10} {'gap23':>8}")
        for dose in DOSES:
            m = all_results[f"D{dose}"][zone_name]
            print(f"  D{dose:>4} {m['spectral_gap_1_rest']:>8.3f} "
                  f"{m['identity_alignment']:>8.4f} {m['erank']:>8.1f} "
                  f"{m['top_sv']:>10.3f} {m['spectral_gap_2_3']:>8.3f}")

    # Save
    summary = {
        "experiment": "E76_crossarch_regime",
        "model": model_name,
        "doses": DOSES,
        "zones": {k: list(v) for k, v in ZONES.items()},
        "n_perturbation_dirs": N_PERTURBATION_DIRS,
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
    parser.add_argument("--model", default="microsoft/phi-2")
    args = parser.parse_args()
    run_experiment(args.model)
