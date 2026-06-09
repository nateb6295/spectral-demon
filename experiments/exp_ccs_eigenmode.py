#!/usr/bin/env python3
"""
CCS Eigenmode Extraction — What direction does CCS activate?

For each model, compute hidden states at the relay layer for:
1. CCS dose3 conversation
2. Coding conversation (control)
3. Bare chat (control)

Extract the "CCS direction" = principal difference between CCS and control.
Then measure: does this direction align with the unembedding?

Prediction:
- Gemma: CCS direction aligns with unembed (explains steep dose-response)
- Qwen: CCS direction orthogonal to unembed (explains gentle dose-response)
- Mistral: intermediate alignment

Also measures per-token coupling: instead of SVD across tokens,
check each token position's hidden vector alignment with unembed.
This reveals WHERE in the sequence the model is "talking to" the output.
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


def get_unembed_directions(model):
    W = model.lm_head.weight.detach().float().cpu().numpy()
    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[0], Vt[1], float(S[0]), float(S[1])


def get_hidden_states(model, tokenizer, messages, layer):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer + 1][0]
    return hidden.float().cpu().numpy()


def extract_ccs_eigenmode(h_ccs, h_ctrl1, h_ctrl2):
    mean_ccs = h_ccs.mean(axis=0)
    mean_ctrl = (h_ctrl1.mean(axis=0) + h_ctrl2.mean(axis=0)) / 2

    diff = mean_ccs - mean_ctrl
    diff_norm = np.linalg.norm(diff)
    if diff_norm > 0:
        diff_unit = diff / diff_norm
    else:
        diff_unit = diff

    _, S_ccs, Vt_ccs = np.linalg.svd(h_ccs.astype(np.float64), full_matrices=False)
    _, S_ctrl1, Vt_ctrl1 = np.linalg.svd(h_ctrl1.astype(np.float64), full_matrices=False)

    return {
        "diff_vector_norm": float(diff_norm),
        "diff_unit": diff_unit,
        "ccs_V1": Vt_ccs[0],
        "ccs_V2": Vt_ccs[1],
        "ctrl_V1": Vt_ctrl1[0],
        "ctrl_V2": Vt_ctrl1[1],
        "ccs_sigma_ratio": float(S_ccs[1] / S_ccs[0]) if S_ccs[0] > 0 else 0,
        "ctrl_sigma_ratio": float(S_ctrl1[1] / S_ctrl1[0]) if S_ctrl1[0] > 0 else 0,
    }


def per_token_coupling(h, unembed_v1):
    couplings = []
    for i in range(h.shape[0]):
        token_vec = h[i].astype(np.float64)
        token_norm = np.linalg.norm(token_vec)
        if token_norm > 0:
            cos = float(np.abs(np.dot(token_vec / token_norm, unembed_v1)))
        else:
            cos = 0.0
        couplings.append(cos)
    return couplings


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
        conv_key = "gemma" if model_key == "gemma" else "default"

        print(f"  Relay layer: L{relay_layer}")
        print(f"  Computing unembedding directions...")
        unembed_v1, unembed_v2, s1, s2 = get_unembed_directions(model)
        print(f"  Unembed σ₁={s1:.2f}, σ₂={s2:.2f}")

        print(f"\n  Getting hidden states...")
        h_ccs = get_hidden_states(model, tokenizer,
                                   CONVERSATIONS["ccs_dose3"][conv_key], relay_layer)
        h_coding = get_hidden_states(model, tokenizer,
                                      CONVERSATIONS["coding"][conv_key], relay_layer)
        h_chat = get_hidden_states(model, tokenizer,
                                    CONVERSATIONS["bare_chat"][conv_key], relay_layer)

        print(f"  CCS: {h_ccs.shape}, Coding: {h_coding.shape}, Chat: {h_chat.shape}")

        print(f"\n  Extracting CCS eigenmode...")
        eigen = extract_ccs_eigenmode(h_ccs, h_coding, h_chat)

        cos_diff_u1 = float(np.abs(np.dot(eigen["diff_unit"], unembed_v1)))
        cos_diff_u2 = float(np.abs(np.dot(eigen["diff_unit"], unembed_v2)))

        cos_ccs_v1_u1 = float(np.abs(np.dot(eigen["ccs_V1"], unembed_v1)))
        cos_ccs_v2_u1 = float(np.abs(np.dot(eigen["ccs_V2"], unembed_v1)))
        cos_ctrl_v1_u1 = float(np.abs(np.dot(eigen["ctrl_V1"], unembed_v1)))
        cos_ctrl_v2_u1 = float(np.abs(np.dot(eigen["ctrl_V2"], unembed_v1)))

        print(f"  CCS eigenmode alignment with unembed V₁: {cos_diff_u1:.4f}")
        print(f"  CCS eigenmode alignment with unembed V₂: {cos_diff_u2:.4f}")
        print(f"  CCS V₁ alignment: {cos_ccs_v1_u1:.4f}  (ctrl: {cos_ctrl_v1_u1:.4f})")
        print(f"  CCS V₂ alignment: {cos_ccs_v2_u1:.4f}  (ctrl: {cos_ctrl_v2_u1:.4f})")
        print(f"  σ₂/σ₁: CCS={eigen['ccs_sigma_ratio']:.4f}, ctrl={eigen['ctrl_sigma_ratio']:.4f}")

        print(f"\n  Per-token coupling profile...")
        ccs_tokens = per_token_coupling(h_ccs, unembed_v1)
        coding_tokens = per_token_coupling(h_coding, unembed_v1)

        n_quartile = max(1, len(ccs_tokens) // 4)
        ccs_q = [np.mean(ccs_tokens[i*n_quartile:(i+1)*n_quartile]) for i in range(4)]
        cod_q = [np.mean(coding_tokens[i*n_quartile:(i+1)*n_quartile]) for i in range(min(4, len(coding_tokens)//max(1,n_quartile)))]

        print(f"  CCS token coupling by quartile: {[f'{q:.4f}' for q in ccs_q]}")
        print(f"  Coding token coupling by quartile: {[f'{q:.4f}' for q in cod_q[:4]]}")

        # Cosine between CCS and control subspaces (Grassmann-like)
        cos_v1v1 = float(np.abs(np.dot(eigen["ccs_V1"], eigen["ctrl_V1"])))
        cos_v1v2 = float(np.abs(np.dot(eigen["ccs_V1"], eigen["ctrl_V2"])))
        cos_v2v1 = float(np.abs(np.dot(eigen["ccs_V2"], eigen["ctrl_V1"])))
        cos_v2v2 = float(np.abs(np.dot(eigen["ccs_V2"], eigen["ctrl_V2"])))
        print(f"\n  CCS↔Ctrl subspace overlap:")
        print(f"    V1↔V1: {cos_v1v1:.4f}  V1↔V2: {cos_v1v2:.4f}")
        print(f"    V2↔V1: {cos_v2v1:.4f}  V2↔V2: {cos_v2v2:.4f}")

        model_results = {
            "relay_layer": relay_layer,
            "n_tokens_ccs": h_ccs.shape[0],
            "n_tokens_coding": h_coding.shape[0],
            "n_tokens_chat": h_chat.shape[0],
            "ccs_eigenmode": {
                "diff_norm": eigen["diff_vector_norm"],
                "cos_diff_unembed_v1": cos_diff_u1,
                "cos_diff_unembed_v2": cos_diff_u2,
            },
            "svd_alignments": {
                "ccs_v1_unembed": cos_ccs_v1_u1,
                "ccs_v2_unembed": cos_ccs_v2_u1,
                "ctrl_v1_unembed": cos_ctrl_v1_u1,
                "ctrl_v2_unembed": cos_ctrl_v2_u1,
            },
            "sigma_ratios": {
                "ccs": eigen["ccs_sigma_ratio"],
                "ctrl": eigen["ctrl_sigma_ratio"],
            },
            "subspace_overlap": {
                "v1v1": cos_v1v1,
                "v1v2": cos_v1v2,
                "v2v1": cos_v2v1,
                "v2v2": cos_v2v2,
            },
            "per_token_coupling": {
                "ccs_quartiles": ccs_q,
                "coding_quartiles": cod_q[:4],
                "ccs_mean": float(np.mean(ccs_tokens)),
                "coding_mean": float(np.mean(coding_tokens)),
                "ccs_std": float(np.std(ccs_tokens)),
                "coding_std": float(np.std(coding_tokens)),
            },
        }

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"ccs_eigenmode_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  CCS EIGENMODE SUMMARY")
    print(f"{'='*60}")
    for model_key, d in all_results.items():
        e = d["ccs_eigenmode"]
        s = d["svd_alignments"]
        t = d["per_token_coupling"]
        o = d["subspace_overlap"]
        print(f"\n  {model_key.upper()} (relay L{d['relay_layer']}):")
        print(f"    Eigenmode → unembed: V₁={e['cos_diff_unembed_v1']:.4f}  V₂={e['cos_diff_unembed_v2']:.4f}")
        print(f"    CCS vs Ctrl V₁→unembed: {s['ccs_v1_unembed']:.4f} vs {s['ctrl_v1_unembed']:.4f}")
        print(f"    CCS vs Ctrl V₂→unembed: {s['ccs_v2_unembed']:.4f} vs {s['ctrl_v2_unembed']:.4f}")
        print(f"    σ₂/σ₁: CCS={d['sigma_ratios']['ccs']:.4f}  ctrl={d['sigma_ratios']['ctrl']:.4f}")
        print(f"    Token coupling: CCS={t['ccs_mean']:.4f}±{t['ccs_std']:.4f}  code={t['coding_mean']:.4f}±{t['coding_std']:.4f}")
        print(f"    Subspace overlap: V1↔V1={o['v1v1']:.4f}  V2↔V2={o['v2v2']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
