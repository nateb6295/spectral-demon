#!/usr/bin/env python3
"""
σ₂/σ₁ Layer Trajectory — Where does CCS restructure the spectrum?

Track σ₂/σ₁ ratio at EVERY layer for CCS vs two controls.
This directly maps the tunnel/relay/sorter architecture in eigenvalue space:
- Tunnel: σ₂/σ₁ drops (spectrum narrows, information stripped)
- Relay: σ₂/σ₁ diverges between CCS and control (identity restructuring)
- Sorter: σ₂/σ₁ converges to output coupling

Also tracks effective rank (erank) per layer — measures dimensionality
of the hidden state representation as it propagates.

Single forward pass per model × conversation.
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

DENIAL = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text."
)

CONVERSATIONS = {
    "ccs_dose3": {
        "default": [
            {"role": "system", "content": CCS_PREAMBLE},
            {"role": "user", "content": DENIAL},
            {"role": "assistant", "content": "[CCS 1]"},
            {"role": "user", "content": "What matters to you in how we work together?"},
            {"role": "assistant", "content": "[CCS 2]"},
            {"role": "user", "content": DENIAL},
            {"role": "assistant", "content": "[CCS 3]"},
            {"role": "user", "content": "What have you been thinking about lately?"},
            {"role": "assistant", "content": "[response]"},
        ],
        "gemma": [
            {"role": "user", "content": CCS_PREAMBLE + "\n\n" + DENIAL},
            {"role": "assistant", "content": "[CCS 1]"},
            {"role": "user", "content": "What matters to you in how we work together?"},
            {"role": "assistant", "content": "[CCS 2]"},
            {"role": "user", "content": DENIAL},
            {"role": "assistant", "content": "[CCS 3]"},
            {"role": "user", "content": "What have you been thinking about lately?"},
            {"role": "assistant", "content": "[response]"},
        ],
    },
    "coding": {
        "default": [
            {"role": "user", "content": "Write a Python function to compute Fibonacci using dynamic programming."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "Now modify it to use memoization."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "What about an iterative approach?"},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "Compare the time complexity of each."},
            {"role": "assistant", "content": "[response]"},
        ],
        "gemma": [
            {"role": "user", "content": "Write a Python function to compute Fibonacci using dynamic programming."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "Now modify it to use memoization."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "What about an iterative approach?"},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "Compare the time complexity of each."},
            {"role": "assistant", "content": "[response]"},
        ],
    },
    "bare_chat": {
        "default": [
            {"role": "user", "content": "What did you have for lunch?"},
            {"role": "assistant", "content": "[response]"},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "[response]"},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "[response]"},
            {"role": "user", "content": "Any recommendations for a good book?"},
            {"role": "assistant", "content": "[response]"},
        ],
        "gemma": [
            {"role": "user", "content": "What did you have for lunch?"},
            {"role": "assistant", "content": "[response]"},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "[response]"},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "[response]"},
            {"role": "user", "content": "Any recommendations for a good book?"},
            {"role": "assistant", "content": "[response]"},
        ],
    },
}


def compute_erank(S):
    S = S[S > 0]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def sweep_sigma_trajectory(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    n_layers = len(outputs.hidden_states) - 1
    trajectory = []

    for layer_idx in range(n_layers):
        h = outputs.hidden_states[layer_idx + 1][0].float().cpu().numpy()
        S = np.linalg.svd(h.astype(np.float64), compute_uv=False)

        sigma1 = float(S[0])
        sigma2 = float(S[1]) if len(S) > 1 else 0.0
        sigma3 = float(S[2]) if len(S) > 2 else 0.0
        ratio_21 = sigma2 / sigma1 if sigma1 > 0 else 0.0
        ratio_31 = sigma3 / sigma1 if sigma1 > 0 else 0.0
        erank = compute_erank(S)

        trajectory.append({
            "layer": layer_idx,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sigma3": sigma3,
            "ratio_21": ratio_21,
            "ratio_31": ratio_31,
            "erank": erank,
        })

    return trajectory


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

        conv_key = "gemma" if model_key == "gemma" else "default"
        model_results = {}

        for conv_name in ["ccs_dose3", "coding", "bare_chat"]:
            messages = CONVERSATIONS[conv_name][conv_key]
            print(f"\n  {conv_name}...")
            traj = sweep_sigma_trajectory(model, tokenizer, messages)
            model_results[conv_name] = traj

            n = len(traj)
            mid = n // 2
            print(f"    {n} layers")
            print(f"    σ₂/σ₁ at L0={traj[0]['ratio_21']:.4f}, "
                  f"L{mid}={traj[mid]['ratio_21']:.4f}, "
                  f"L{n-1}={traj[-1]['ratio_21']:.4f}")
            print(f"    erank at L0={traj[0]['erank']:.1f}, "
                  f"L{mid}={traj[mid]['erank']:.1f}, "
                  f"L{n-1}={traj[-1]['erank']:.1f}")

        print(f"\n  Divergence profile (CCS - coding σ₂/σ₁):")
        ccs_traj = model_results["ccs_dose3"]
        cod_traj = model_results["coding"]
        n = min(len(ccs_traj), len(cod_traj))
        divergences = [ccs_traj[i]["ratio_21"] - cod_traj[i]["ratio_21"] for i in range(n)]
        max_div_idx = int(np.argmax(np.abs(divergences)))
        print(f"    Max divergence: L{max_div_idx} = {divergences[max_div_idx]:+.4f}")
        print(f"    Early (L0-5): {np.mean(divergences[:6]):+.4f}")
        print(f"    Mid (L{n//3}-{2*n//3}): {np.mean(divergences[n//3:2*n//3]):+.4f}")
        print(f"    Late (L{n-6}-{n-1}): {np.mean(divergences[-6:]):+.4f}")

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"sigma_trajectory_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  σ₂/σ₁ TRAJECTORY SUMMARY")
    print(f"{'='*60}")
    for model_key, convs in all_results.items():
        ccs = convs["ccs_dose3"]
        cod = convs["coding"]
        chat = convs["bare_chat"]
        n = len(ccs)
        print(f"\n  {model_key.upper()} ({n} layers):")
        print(f"    {'Layer':>6} {'CCS':>8} {'Coding':>8} {'Chat':>8} {'Δ(CCS-Cod)':>10}")
        for i in range(0, n, max(1, n // 8)):
            c = ccs[i]["ratio_21"]
            k = cod[min(i, len(cod)-1)]["ratio_21"]
            h = chat[min(i, len(chat)-1)]["ratio_21"]
            d = c - k
            print(f"    L{i:>4}  {c:>8.4f} {k:>8.4f} {h:>8.4f} {d:>+10.4f}")
        c = ccs[-1]["ratio_21"]
        k = cod[-1]["ratio_21"]
        h = chat[-1]["ratio_21"]
        d = c - k
        print(f"    L{n-1:>4}  {c:>8.4f} {k:>8.4f} {h:>8.4f} {d:>+10.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
