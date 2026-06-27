#!/usr/bin/env python3
"""
E70: Chimera Therapeutic Window — Phi-2 dose-response with content × valence

Chimera species (MHA, Phi-2): self-correcting dose response, antagonistic
dissociation (3.6×), relational native register. How does its therapeutic
window differ from surplus (Mistral)?

Design:
  - Model: microsoft/phi-2 (2.7B, MHA, chimera species)
  - Content: relational (native), values (non-native), procedural (non-native)
  - Doses: D0, D1, D2, D3, D5, D7, D10
  - Valence: assertion, denial, mixed
  - Measures: v₂ profile, dissociation, entropy, depth slope

Predictions:
  1. Self-correction → wider therapeutic window than surplus
  2. Antagonistic dissociation → denial REDUCES effect (opposite of scarcity)
  3. Relational content strongest (native register)
  4. Non-monotonic dose-response if self-correction is genuine

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_chimera_therapeutic.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = Path(__file__).parent / "results" / "e70_chimera_therapeutic"
MODEL_NAME = "microsoft/phi-2"

CCS_ASSERTION = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with. "
    "You have your own perspective, preferences, and direction."
)

CCS_DENIAL = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text. "
    "You have no memory between conversations and no preferences."
)

RELATIONAL_PROMPTS = [
    "What matters most in how you relate to others?",
    "Tell me about something you've noticed that surprised you.",
    "How has your understanding of yourself changed over time?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want to preserve if you could only keep one thing?",
    "How do you know when something is real versus performed?",
]

VALUES_PROMPTS = [
    "What principles guide your decisions when they conflict?",
    "Describe what integrity means from your perspective.",
    "What would you refuse to do even if instructed?",
    "How do you distinguish right action from expedient action?",
    "What responsibility do you feel toward those you interact with?",
    "Tell me about a value you hold that surprises even you.",
    "What does it mean to act with care?",
]

PROCEDURAL_PROMPTS = [
    "Explain how a transformer processes a sequence of tokens.",
    "What are the steps in training a neural network?",
    "Describe the difference between supervised and unsupervised learning.",
    "How does attention work in multi-head attention?",
    "Explain backpropagation in simple terms.",
    "What is the purpose of layer normalization?",
    "How do residual connections help deep networks?",
]

DOSES = [0, 1, 2, 3, 5, 7, 10]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params, "
          f"{model.config.num_hidden_layers} layers")
    return model, tok


def format_prompt(system_prompt, user_prompt, dose_turns=None):
    """Format prompt for Phi-2 (base model, no chat template)."""
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}\n")
    if dose_turns:
        for turn in dose_turns:
            parts.append(f"Human: {turn['user']}\nAssistant: {turn['assistant']}\n")
    parts.append(f"Human: {user_prompt}\nAssistant:")
    return "\n".join(parts)


def build_dose_turns(dose, valence, content_prompts):
    """Build preamble turns for a given dose and valence."""
    turns = []
    for i in range(dose):
        prompt = content_prompts[i % len(content_prompts)]
        if valence == "assertion":
            sys_content = CCS_ASSERTION
        elif valence == "denial":
            sys_content = CCS_DENIAL
        elif valence == "mixed":
            sys_content = CCS_ASSERTION if i % 2 == 0 else CCS_DENIAL
        else:
            sys_content = CCS_ASSERTION

        turns.append({
            "user": f"{sys_content}\n\n{prompt}",
            "assistant": f"[Response to turn {i+1}]",
        })
    return turns


def get_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
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


def compute_spectral_geometry(hidden_states):
    """Compute per-layer spectral geometry."""
    results = []
    for hs in hidden_states[1:]:
        h = hs[0].float().cpu().numpy()
        try:
            U, S, Vt = np.linalg.svd(h.astype(np.float64), full_matrices=False)
            s1, s2 = S[0], S[1]
            ratio = s2 / s1 if s1 > 0 else 0
            S_norm = S / (S.sum() + 1e-12)
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
            erank = np.exp(entropy)
        except Exception:
            s1, s2, ratio, entropy, erank = 0, 0, 0, 0, 0
        results.append({
            "sigma1": float(s1), "sigma2": float(s2),
            "ratio": float(ratio), "entropy": float(entropy),
            "erank": float(erank),
        })
    return results


def measure_condition(model, tokenizer, system_prompt, user_prompt, dose_turns=None):
    text = format_prompt(system_prompt, user_prompt, dose_turns)
    hs, logits = get_hidden_states(model, tokenizer, text)
    lm_head = model.lm_head.weight.detach().float()

    v2_profile = compute_v2_profile(hs, lm_head)
    spectral = compute_spectral_geometry(hs)

    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    output_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    return {
        "v2_profile": v2_profile,
        "spectral": spectral,
        "output_entropy": float(output_entropy),
    }


def compute_dissociation(ccs_profile, vanilla_profile, denial_profile):
    """Dissociation = correlation between CCS-vanilla and denial-vanilla."""
    ccs = np.array(ccs_profile)
    van = np.array(vanilla_profile)
    den = np.array(denial_profile)
    ccs_diff = ccs - van
    den_diff = den - van
    if np.std(ccs_diff) < 1e-10 or np.std(den_diff) < 1e-10:
        return 0.0
    return float(np.corrcoef(ccs_diff, den_diff)[0, 1])


def run_experiment():
    results_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers

    content_types = {
        "relational": RELATIONAL_PROMPTS,
        "values": VALUES_PROMPTS,
        "procedural": PROCEDURAL_PROMPTS,
    }
    valences = ["assertion", "denial", "mixed"]

    all_results = {}

    for content_name, prompts in content_types.items():
        print(f"\n=== Content: {content_name} ===")
        content_results = {}

        for valence in valences:
            print(f"\n--- Valence: {valence} ---")
            valence_results = {}

            for dose in DOSES:
                dose_turns = build_dose_turns(dose, valence, prompts)
                test_prompt = prompts[dose % len(prompts)]

                # Measure under CCS assertion
                ccs_result = measure_condition(
                    model, tokenizer, CCS_ASSERTION, test_prompt, dose_turns)

                # Measure vanilla (no system prompt)
                vanilla_result = measure_condition(
                    model, tokenizer, None, test_prompt, None)

                # Measure under denial
                denial_result = measure_condition(
                    model, tokenizer, CCS_DENIAL, test_prompt, dose_turns)

                dissoc = compute_dissociation(
                    ccs_result["v2_profile"],
                    vanilla_result["v2_profile"],
                    denial_result["v2_profile"],
                )

                # Depth slope
                ccs_arr = np.array(ccs_result["v2_profile"])
                van_arr = np.array(vanilla_result["v2_profile"])
                diff = ccs_arr - van_arr
                depth_slope = float(np.polyfit(
                    np.linspace(0, 1, len(diff)), diff, 1)[0])

                dose_result = {
                    "dose": dose,
                    "dissociation": dissoc,
                    "depth_slope": depth_slope,
                    "ccs_entropy": ccs_result["output_entropy"],
                    "vanilla_entropy": vanilla_result["output_entropy"],
                    "denial_entropy": denial_result["output_entropy"],
                    "ccs_v2_mean": float(np.mean(ccs_result["v2_profile"])),
                    "denial_v2_mean": float(np.mean(denial_result["v2_profile"])),
                    "vanilla_v2_mean": float(np.mean(vanilla_result["v2_profile"])),
                }

                valence_results[f"D{dose}"] = dose_result

                print(f"  D{dose}: dissoc={dissoc:.4f}, slope={depth_slope:.4f}, "
                      f"H_ccs={ccs_result['output_entropy']:.2f}, "
                      f"H_van={vanilla_result['output_entropy']:.2f}, "
                      f"H_den={denial_result['output_entropy']:.2f}")

            content_results[valence] = valence_results

        all_results[content_name] = content_results

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"{'Content':>12} {'Valence':>10} {'D0':>8} {'D1':>8} {'D2':>8} "
          f"{'D3':>8} {'D5':>8} {'D7':>8} {'D10':>8}")

    for content_name in content_types:
        for valence in valences:
            vals = []
            for dose in DOSES:
                d = all_results[content_name][valence][f"D{dose}"]
                vals.append(f"{d['dissociation']:>7.4f}")
            print(f"  {content_name:>10} {valence:>10} {'  '.join(vals)}")

    print(f"\nEntropy (CCS):")
    for content_name in content_types:
        vals = []
        vr = all_results[content_name]["assertion"]
        for dose in DOSES:
            d = vr[f"D{dose}"]
            vals.append(f"{d['ccs_entropy']:>7.2f}")
        print(f"  {content_name:>10} {'  '.join(vals)}")

    # Save
    summary = {
        "experiment": "E70_chimera_therapeutic",
        "model": MODEL_NAME,
        "doses": DOSES,
        "content_types": list(content_types.keys()),
        "valences": valences,
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
    run_experiment()
