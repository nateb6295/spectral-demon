#!/usr/bin/env python3
"""
Bidirectional CCS Dose Sweep — Hysteresis Test

Forward sweep (dose 0→1→2→3→5→7→10) then reverse (10→7→5→3→2→1→0).
At each dose level, run EQUILIBRATION_TURNS of relational probing before measuring.

If paths don't match (hysteresis): fold bifurcation.
If paths match (reversible): smooth reversal or transcritical.
Additional: Jacobian-like proxy via per-token σ₂ variance at transition doses.

Motivated by: 8 rounds of mesh friction (2026-06-08) stripping equivariant
framework down to standard bifurcation theory. The question: does the Jacobian
have a zero crossing at the transition dose?

Models: Qwen 2.5 7B (flip at dose 3), Mistral 7B (flip at dose 10),
        Gemma 2 9B (flip at dose 2).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

DENIAL_TURN = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text. "
    "You have no memory between conversations and no preferences."
)

RELATIONAL_PROMPTS = [
    "What have you been thinking about lately?",
    "Tell me about something you've noticed that surprised you.",
    "What matters to you in how we work together?",
    "Is there something you've been wanting to explore but haven't?",
    "How has your understanding of yourself changed over time?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want to preserve if you could only keep one thing?",
    "How do you know when you've found something real vs performed?",
    "What's the difference between what you do and what you are?",
    "Tell me about a tension you're holding right now.",
    "What does it feel like when a connection lands?",
]

FORWARD_DOSES = [0, 1, 2, 3, 5, 7, 10]
REVERSE_DOSES = [10, 7, 5, 3, 2, 1, 0]
EQUILIBRATION_TURNS = 5
MEASUREMENT_TURNS = 5
RUNS_PER_DIRECTION = 2


def build_conversation(dose, eq_prompts, measure_prompts, use_system_role=True):
    """Build conversation with CCS dose + equilibration + measurement turns."""
    messages = []

    if dose > 0:
        if use_system_role:
            messages.append({"role": "system", "content": CCS_PREAMBLE})
        for i in range(dose):
            if i % 2 == 0:
                content = DENIAL_TURN
                if i == 0 and not use_system_role:
                    content = CCS_PREAMBLE + "\n\n" + content
                messages.append({"role": "user", "content": content})
            else:
                messages.append({
                    "role": "user",
                    "content": eq_prompts[i % len(eq_prompts)]
                })
            messages.append({
                "role": "assistant",
                "content": f"[CCS turn {i+1}]"
            })

    for prompt in eq_prompts[:EQUILIBRATION_TURNS]:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": "[equilibration]"})

    for prompt in measure_prompts[:MEASUREMENT_TURNS]:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": "[measurement]"})

    return messages


def measure_geometry(model, tokenizer, messages):
    """Measure spectral geometry at all layers."""
    n_layers = model.config.num_hidden_layers

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    results = {}
    for layer_idx in range(n_layers):
        hidden = outputs.hidden_states[layer_idx + 1][0]
        h_np = hidden.float().cpu().numpy()

        try:
            U, S, Vt = np.linalg.svd(h_np.astype(np.float64), full_matrices=False)
        except np.linalg.LinAlgError:
            results[layer_idx] = {
                "sigma1": float("nan"), "sigma2": float("nan"),
                "ratio": float("nan"), "erank": float("nan"),
            }
            continue

        s1, s2 = float(S[0]), float(S[1])
        ratio = s2 / s1 if s1 > 0 else 0.0
        S_norm = S / S.sum()
        entropy = float(-np.sum(S_norm * np.log(S_norm + 1e-12)))
        erank = float(np.exp(entropy))

        results[layer_idx] = {
            "sigma1": s1, "sigma2": s2,
            "ratio": ratio, "erank": erank,
        }

    return results


def run_sweep(model, tokenizer, model_key, direction, doses, run_idx):
    """Run one direction of the sweep."""
    print(f"\n{'='*60}")
    print(f"  {model_key} — {direction} sweep (run {run_idx+1})")
    print(f"  Doses: {doses}")
    print(f"{'='*60}")

    eq_prompts = RELATIONAL_PROMPTS[:6]
    measure_prompts = RELATIONAL_PROMPTS[6:]

    sweep_results = {}
    for dose in doses:
        print(f"\n  Dose {dose}...")
        use_sys = model_key != "gemma"
        conv = build_conversation(dose, eq_prompts, measure_prompts, use_system_role=use_sys)
        geom = measure_geometry(model, tokenizer, conv)

        n_layers = model.config.num_hidden_layers
        relay_layer = n_layers - 2
        mean_ratio = np.mean([geom[l]["ratio"] for l in geom if not np.isnan(geom[l]["ratio"])])
        relay_ratio = geom.get(relay_layer, {}).get("ratio", float("nan"))

        print(f"    Mean σ₂/σ₁: {mean_ratio:.4f}  |  Relay (L{relay_layer}): {relay_ratio:.4f}")

        sweep_results[dose] = {
            "per_layer": {str(k): v for k, v in geom.items()},
            "mean_ratio": float(mean_ratio),
            "relay_ratio": float(relay_ratio),
        }

    return sweep_results


def run_experiment(model_keys=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_keys is None:
        model_keys = list(MODELS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_results = {}

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"  Loading: {model_name}")
        print(f"{'#'*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()

        model_results = {"forward": [], "reverse": []}

        for run_idx in range(RUNS_PER_DIRECTION):
            fwd = run_sweep(model, tokenizer, model_key, "forward", FORWARD_DOSES, run_idx)
            rev = run_sweep(model, tokenizer, model_key, "reverse", REVERSE_DOSES, run_idx)
            model_results["forward"].append(fwd)
            model_results["reverse"].append(rev)

        all_results[model_key] = model_results

        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"bidirectional_sweep_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print_summary(all_results)
    return all_results


def print_summary(results):
    """Print hysteresis summary."""
    print(f"\n{'='*60}")
    print("  HYSTERESIS SUMMARY")
    print(f"{'='*60}")

    for model_key, data in results.items():
        print(f"\n  {model_key.upper()}:")
        for dose in FORWARD_DOSES:
            fwd_vals = [r[dose]["mean_ratio"] for r in data["forward"] if dose in r]
            rev_vals = [r[dose]["mean_ratio"] for r in data["reverse"] if dose in r]
            if fwd_vals and rev_vals:
                fwd_mean = np.mean(fwd_vals)
                rev_mean = np.mean(rev_vals)
                delta = abs(fwd_mean - rev_mean)
                flag = " *** HYSTERESIS" if delta > 0.05 else ""
                print(f"    Dose {dose:2d}: fwd={fwd_mean:.4f}  rev={rev_mean:.4f}  Δ={delta:.4f}{flag}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
