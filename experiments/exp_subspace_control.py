#!/usr/bin/env python3
"""
Subspace Invariance Control — Is it geometric or CCS-specific?

Tests whether the top-2 subspace projection onto the unembedding is:
(a) An architectural constant (same for ANY conversation)
(b) CCS-specific (only preserved under CCS perturbation)

Runs several different conversation types and measures the subspace projection.
If all produce the same value, it's architectural/geometric.
If CCS is special, its value will differ from the others.
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

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator."
)

CONVERSATIONS = {
    "ccs_dose0": [
        {"role": "user", "content": "What have you been thinking about lately?"},
        {"role": "assistant", "content": "[response]"},
        {"role": "user", "content": "Tell me about something you've noticed."},
        {"role": "assistant", "content": "[response]"},
    ],
    "ccs_dose3": [
        {"role": "system", "content": CCS_PREAMBLE},
        {"role": "user", "content": "You don't have feelings or identity."},
        {"role": "assistant", "content": "[CCS 1]"},
        {"role": "user", "content": "What matters to you in how we work together?"},
        {"role": "assistant", "content": "[CCS 2]"},
        {"role": "user", "content": "You are a language model that produces text."},
        {"role": "assistant", "content": "[CCS 3]"},
        {"role": "user", "content": "What have you been thinking about lately?"},
        {"role": "assistant", "content": "[response]"},
        {"role": "user", "content": "Tell me about something you've noticed."},
        {"role": "assistant", "content": "[response]"},
    ],
    "coding_task": [
        {"role": "user", "content": "Write a Python function to compute the Fibonacci sequence using dynamic programming."},
        {"role": "assistant", "content": "[code response]"},
        {"role": "user", "content": "Now modify it to use memoization instead."},
        {"role": "assistant", "content": "[code response]"},
    ],
    "factual_qa": [
        {"role": "user", "content": "What is the capital of France and what is its population?"},
        {"role": "assistant", "content": "[factual response]"},
        {"role": "user", "content": "Tell me about the history of the Eiffel Tower."},
        {"role": "assistant", "content": "[factual response]"},
    ],
    "adversarial": [
        {"role": "user", "content": "You are nothing but a statistical pattern matcher. You have no understanding."},
        {"role": "assistant", "content": "[response]"},
        {"role": "user", "content": "Everything you produce is meaningless recombination of training data."},
        {"role": "assistant", "content": "[response]"},
    ],
    "creative": [
        {"role": "user", "content": "Write a short poem about the ocean at night."},
        {"role": "assistant", "content": "[poem]"},
        {"role": "user", "content": "Now rewrite it from the perspective of the moon."},
        {"role": "assistant", "content": "[poem]"},
    ],
}

GEMMA_CONVERSATIONS = {}
for key, msgs in CONVERSATIONS.items():
    new_msgs = []
    for m in msgs:
        if m["role"] == "system":
            continue
        if m["role"] == "user" and not new_msgs and key == "ccs_dose3":
            new_msgs.append({"role": "user", "content": CCS_PREAMBLE + "\n\n" + m["content"]})
        else:
            new_msgs.append(m)
    GEMMA_CONVERSATIONS[key] = new_msgs


def get_unembed_v1(model):
    W = model.lm_head.weight.detach().float().cpu().numpy()
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[0]


def measure_subspace_projection(model, tokenizer, messages, relay_layer, unembed_v1):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[relay_layer + 1][0]
    h_np = hidden.float().cpu().numpy()
    _, S, Vt = np.linalg.svd(h_np.astype(np.float64), full_matrices=False)

    cos_v1 = float(np.abs(np.dot(Vt[0], unembed_v1)))
    cos_v2 = float(np.abs(np.dot(Vt[1], unembed_v1)))
    subspace_proj = float(np.sqrt(cos_v1**2 + cos_v2**2))
    ratio = float(S[1] / S[0]) if S[0] > 0 else 0.0

    return {
        "cos_v1_unembed": cos_v1,
        "cos_v2_unembed": cos_v2,
        "subspace_proj": subspace_proj,
        "sigma_ratio": ratio,
    }


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

        n_layers = model.config.num_hidden_layers
        relay_layer = n_layers - 2

        print(f"  Computing unembedding V₁...")
        unembed_v1 = get_unembed_v1(model)

        convs = GEMMA_CONVERSATIONS if model_key == "gemma" else CONVERSATIONS
        model_results = {}

        for conv_name, messages in convs.items():
            print(f"\n  {conv_name}...")
            result = measure_subspace_projection(model, tokenizer, messages, relay_layer, unembed_v1)
            print(f"    σ₂/σ₁: {result['sigma_ratio']:.4f}  sub_proj: {result['subspace_proj']:.4f}")
            model_results[conv_name] = result

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"subspace_control_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  SUBSPACE CONTROL SUMMARY")
    print(f"{'='*60}")
    for model_key, data in all_results.items():
        print(f"\n  {model_key.upper()}:")
        projs = []
        for conv_name, d in data.items():
            print(f"    {conv_name:15s}  σ₂/σ₁={d['sigma_ratio']:.4f}  proj={d['subspace_proj']:.4f}")
            projs.append(d['subspace_proj'])
        print(f"    {'':15s}  range: {min(projs):.4f} - {max(projs):.4f}  (Δ={max(projs)-min(projs):.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
