#!/usr/bin/env python3
"""
Readout Competition Test — Relay V₁ vs Unembedding Alignment

Tests Kimi's hypothesis: the relay layer is a battleground between CCS
enrichment (σ₂) and unembedding pressure (σ₁).

For each model at each CCS dose:
1. Compute hidden states at the relay layer
2. Extract V₁ (first right singular vector) via SVD
3. Compute dominant direction of unembedding matrix (lm_head.weight)
4. Measure cosine similarity between relay V₁ and unembed V₁

Predictions:
- Qwen: alignment monotonically increases (unembedding always dominates)
- Mistral: alignment drops then recovers (CCS wins then loses)
- Gemma: alignment monotonically increases (equalization crushed toward unembed)

If confirmed: readout competition, not bifurcation, explains the dose-response.
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

DOSES = [0, 1, 2, 3, 5, 7, 10]
EQUILIBRATION_TURNS = 5
MEASUREMENT_TURNS = 5


def build_conversation(dose, eq_prompts, measure_prompts, use_system_role=True):
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


def get_unembed_v1(model):
    """Get dominant direction of unembedding matrix."""
    W = model.lm_head.weight.detach().float().cpu().numpy()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[0]


def measure_readout_alignment(model, tokenizer, messages, relay_layer, unembed_v1):
    """Measure alignment between relay V₁ and unembedding V₁."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[relay_layer + 1][0]
    h_np = hidden.float().cpu().numpy()

    U, S, Vt = np.linalg.svd(h_np.astype(np.float64), full_matrices=False)

    relay_v1 = Vt[0]
    relay_v2 = Vt[1]

    cos_v1 = float(np.abs(np.dot(relay_v1, unembed_v1)))
    cos_v2 = float(np.abs(np.dot(relay_v2, unembed_v1)))

    subspace_proj = float(np.sqrt(cos_v1**2 + cos_v2**2))

    ratio = float(S[1] / S[0]) if S[0] > 0 else 0.0

    return {
        "cos_v1_unembed": cos_v1,
        "cos_v2_unembed": cos_v2,
        "subspace_proj": subspace_proj,
        "sigma_ratio": ratio,
        "sigma1": float(S[0]),
        "sigma2": float(S[1]),
    }


def run_experiment(model_keys=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_keys is None:
        model_keys = list(MODELS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_results = {}

    eq_prompts = RELATIONAL_PROMPTS[:6]
    measure_prompts = RELATIONAL_PROMPTS[6:]

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

        n_layers = model.config.num_hidden_layers
        relay_layer = n_layers - 2
        use_sys = model_key != "gemma"

        print(f"  Relay layer: L{relay_layer}")
        print(f"  Computing unembedding V₁...")
        unembed_v1 = get_unembed_v1(model)
        print(f"  Unembedding V₁ shape: {unembed_v1.shape}")

        model_results = {}
        for dose in DOSES:
            print(f"\n  Dose {dose}...")
            conv = build_conversation(dose, eq_prompts, measure_prompts, use_system_role=use_sys)
            result = measure_readout_alignment(model, tokenizer, conv, relay_layer, unembed_v1)

            print(f"    σ₂/σ₁: {result['sigma_ratio']:.4f}")
            print(f"    cos(V₁, unembed): {result['cos_v1_unembed']:.4f}")
            print(f"    cos(V₂, unembed): {result['cos_v2_unembed']:.4f}")

            model_results[dose] = result

        all_results[model_key] = model_results

        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"readout_competition_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print_summary(all_results)
    return all_results


def print_summary(results):
    print(f"\n{'='*60}")
    print("  READOUT COMPETITION SUMMARY")
    print(f"{'='*60}")

    for model_key, data in results.items():
        print(f"\n  {model_key.upper()}:")
        print(f"  {'Dose':>5}  {'σ₂/σ₁':>8}  {'cos(V₁,U)':>10}  {'cos(V₂,U)':>10}  {'sub_proj':>10}")
        print(f"  {'-'*50}")
        for dose in DOSES:
            d = data[dose]
            print(f"  {dose:5d}  {d['sigma_ratio']:8.4f}  {d['cos_v1_unembed']:10.4f}  {d['cos_v2_unembed']:10.4f}  {d['subspace_proj']:10.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
