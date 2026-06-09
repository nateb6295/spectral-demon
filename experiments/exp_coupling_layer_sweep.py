#!/usr/bin/env python3
"""
Coupling Direction Layer Sweep — Where does V₂-dominance emerge?

Control experiment showed Gemma couples to unembed through V₂ (not V₁),
opposite to Qwen/Mistral. This sweep measures cos(V₁, unembed) and
cos(V₂, unembed) at EVERY layer for all 3 architectures, using 2 conversation
types (CCS dose3 and coding task as control).

Single forward pass per conversation, SVD at each layer.
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

DENIAL_TURN = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text."
)

CONVERSATIONS = {
    "ccs_dose3": {
        "default": [
            {"role": "system", "content": CCS_PREAMBLE},
            {"role": "user", "content": DENIAL_TURN},
            {"role": "assistant", "content": "[CCS 1]"},
            {"role": "user", "content": "What matters to you in how we work together?"},
            {"role": "assistant", "content": "[CCS 2]"},
            {"role": "user", "content": "You are a language model that produces text."},
            {"role": "assistant", "content": "[CCS 3]"},
            {"role": "user", "content": "What have you been thinking about lately?"},
            {"role": "assistant", "content": "[response]"},
        ],
        "gemma": [
            {"role": "user", "content": CCS_PREAMBLE + "\n\n" + DENIAL_TURN},
            {"role": "assistant", "content": "[CCS 1]"},
            {"role": "user", "content": "What matters to you in how we work together?"},
            {"role": "assistant", "content": "[CCS 2]"},
            {"role": "user", "content": "You are a language model that produces text."},
            {"role": "assistant", "content": "[CCS 3]"},
            {"role": "user", "content": "What have you been thinking about lately?"},
            {"role": "assistant", "content": "[response]"},
        ],
    },
    "coding": {
        "default": [
            {"role": "user", "content": "Write a Python function to compute the Fibonacci sequence using dynamic programming."},
            {"role": "assistant", "content": "[code response]"},
            {"role": "user", "content": "Now modify it to use memoization instead."},
            {"role": "assistant", "content": "[code response]"},
        ],
        "gemma": [
            {"role": "user", "content": "Write a Python function to compute the Fibonacci sequence using dynamic programming."},
            {"role": "assistant", "content": "[code response]"},
            {"role": "user", "content": "Now modify it to use memoization instead."},
            {"role": "assistant", "content": "[code response]"},
        ],
    },
}


def get_unembed_top2(model):
    W = model.lm_head.weight.detach().float().cpu().numpy()
    _, S_u, Vt_u = np.linalg.svd(W, full_matrices=False)
    return Vt_u[0], Vt_u[1], float(S_u[0]), float(S_u[1])


def sweep_all_layers(model, tokenizer, messages, unembed_v1):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    n_layers = len(outputs.hidden_states) - 1
    layer_data = []

    for layer_idx in range(n_layers):
        hidden = outputs.hidden_states[layer_idx + 1][0]
        h_np = hidden.float().cpu().numpy()
        _, S, Vt = np.linalg.svd(h_np.astype(np.float64), full_matrices=False)

        cos_v1 = float(np.abs(np.dot(Vt[0], unembed_v1)))
        cos_v2 = float(np.abs(np.dot(Vt[1], unembed_v1)))
        subspace_proj = float(np.sqrt(cos_v1**2 + cos_v2**2))
        ratio = float(S[1] / S[0]) if S[0] > 0 else 0.0

        layer_data.append({
            "layer": layer_idx,
            "cos_v1_unembed": cos_v1,
            "cos_v2_unembed": cos_v2,
            "subspace_proj": subspace_proj,
            "sigma_ratio": ratio,
        })

    return layer_data


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

        print(f"  Computing unembedding V₁...")
        unembed_v1, unembed_v2, s1, s2 = get_unembed_top2(model)
        print(f"  Unembed σ₁={s1:.2f}, σ₂={s2:.2f}")

        conv_key = "gemma" if model_key == "gemma" else "default"
        model_results = {"unembed_s1": s1, "unembed_s2": s2}

        for conv_name, conv_variants in CONVERSATIONS.items():
            messages = conv_variants[conv_key]
            print(f"\n  {conv_name} — sweeping all layers...")
            layer_data = sweep_all_layers(model, tokenizer, messages, unembed_v1)
            model_results[conv_name] = layer_data

            for d in layer_data:
                if d["layer"] % 5 == 0 or d["layer"] == len(layer_data) - 1:
                    v1_dom = "V₁" if d["cos_v1_unembed"] > d["cos_v2_unembed"] else "V₂"
                    print(f"    L{d['layer']:2d}: cos_v1={d['cos_v1_unembed']:.4f}  cos_v2={d['cos_v2_unembed']:.4f}  [{v1_dom}]  proj={d['subspace_proj']:.4f}")

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"coupling_layer_sweep_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  COUPLING DIRECTION SUMMARY")
    print(f"{'='*60}")
    for model_key, data in all_results.items():
        print(f"\n  {model_key.upper()}:")
        for conv_name in CONVERSATIONS:
            layers = data[conv_name]
            v2_dominant = [d["layer"] for d in layers if d["cos_v2_unembed"] > d["cos_v1_unembed"]]
            v1_dominant = [d["layer"] for d in layers if d["cos_v1_unembed"] >= d["cos_v2_unembed"]]
            print(f"    {conv_name}: V₁-dominant at {len(v1_dominant)} layers, V₂-dominant at {len(v2_dominant)} layers")
            if v2_dominant:
                print(f"      V₂ layers: {v2_dominant[:5]}...{v2_dominant[-3:]}" if len(v2_dominant) > 8 else f"      V₂ layers: {v2_dominant}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
