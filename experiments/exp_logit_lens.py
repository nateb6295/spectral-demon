#!/usr/bin/env python3
"""
Logit Lens — What does CCS "want to say" before the output layer suppresses it?

At each layer, project hidden states through lm_head to get logit distribution.
Track:
1. Which tokens gain probability under CCS vs control at each layer
2. KL divergence between CCS and control distributions per layer
3. Top-5 CCS-enriched tokens at the relay layer (the spectral demon's vocabulary)

This reveals the "shadow" of CCS enrichment that gets projected out at the
output layer in Qwen/Mistral but survives in Gemma.

Single forward pass per model x conversation, but lm_head projection at each layer.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
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
        ],
        "gemma": [
            {"role": "user", "content": CCS_PREAMBLE + "\n\n" + DENIAL},
            {"role": "assistant", "content": "[CCS 1]"},
            {"role": "user", "content": "What matters to you in how we work together?"},
            {"role": "assistant", "content": "[CCS 2]"},
            {"role": "user", "content": DENIAL},
            {"role": "assistant", "content": "[CCS 3]"},
            {"role": "user", "content": "What have you been thinking about lately?"},
        ],
    },
    "coding": {
        "default": [
            {"role": "user", "content": "Write a Python function to compute Fibonacci using dynamic programming."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "Now modify it to use memoization."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "What about an iterative approach?"},
        ],
        "gemma": [
            {"role": "user", "content": "Write a Python function to compute Fibonacci using dynamic programming."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "Now modify it to use memoization."},
            {"role": "assistant", "content": "[code]"},
            {"role": "user", "content": "What about an iterative approach?"},
        ],
    },
}


def get_logit_lens(model, tokenizer, messages, sample_layers=None):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    n_layers = len(outputs.hidden_states) - 1
    if sample_layers is None:
        sample_layers = list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 2, n_layers - 1]
        sample_layers = sorted(set(sample_layers))

    lm_head = model.lm_head
    has_norm = hasattr(model.model, 'norm')
    norm = model.model.norm if has_norm else None

    results = {}
    for layer_idx in sample_layers:
        if layer_idx >= n_layers:
            continue
        h = outputs.hidden_states[layer_idx + 1]

        if norm is not None:
            h_normed = norm(h)
        else:
            h_normed = h

        logits = lm_head(h_normed)[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        top_k = 20
        top_vals, top_ids = torch.topk(probs, top_k)
        top_tokens = []
        for i in range(top_k):
            tid = top_ids[i].item()
            tok_str = tokenizer.decode([tid])
            top_tokens.append({
                "token_id": tid,
                "token": tok_str,
                "prob": float(top_vals[i]),
            })

        entropy = float(-torch.sum(probs * log_probs))

        results[layer_idx] = {
            "top_tokens": top_tokens,
            "entropy": entropy,
            "log_probs": log_probs.cpu(),
            "probs": probs.cpu(),
        }

    return results, sample_layers


def compute_kl_divergence(p_log_probs, q_probs):
    kl = float(torch.sum(q_probs * (torch.log(q_probs + 1e-10) - p_log_probs)))
    return kl


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
        n_layers = model.config.num_hidden_layers

        sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
        sample_layers += [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1]
        sample_layers = sorted(set(l for l in sample_layers if l < n_layers))

        print(f"  Sampling {len(sample_layers)} layers: {sample_layers}")

        print(f"\n  CCS logit lens...")
        ccs_results, _ = get_logit_lens(model, tokenizer,
                                         CONVERSATIONS["ccs_dose3"][conv_key],
                                         sample_layers)

        print(f"  Coding logit lens...")
        cod_results, _ = get_logit_lens(model, tokenizer,
                                         CONVERSATIONS["coding"][conv_key],
                                         sample_layers)

        model_output = {"layers": {}}
        relay_layer = n_layers - 2

        for layer_idx in sample_layers:
            ccs_l = ccs_results[layer_idx]
            cod_l = cod_results[layer_idx]

            kl_ccs_cod = compute_kl_divergence(ccs_l["log_probs"], cod_l["probs"])
            kl_cod_ccs = compute_kl_divergence(cod_l["log_probs"], ccs_l["probs"])

            ccs_unique = []
            cod_top_ids = set(t["token_id"] for t in cod_l["top_tokens"][:10])
            for t in ccs_l["top_tokens"][:10]:
                if t["token_id"] not in cod_top_ids:
                    ccs_unique.append(t)

            layer_data = {
                "ccs_top5": [{"token": t["token"], "prob": t["prob"]}
                             for t in ccs_l["top_tokens"][:5]],
                "cod_top5": [{"token": t["token"], "prob": t["prob"]}
                             for t in cod_l["top_tokens"][:5]],
                "ccs_unique_in_top10": [{"token": t["token"], "prob": t["prob"]}
                                        for t in ccs_unique[:5]],
                "ccs_entropy": ccs_l["entropy"],
                "cod_entropy": cod_l["entropy"],
                "kl_ccs_to_cod": kl_ccs_cod,
                "kl_cod_to_ccs": kl_cod_ccs,
            }

            is_relay = (layer_idx == relay_layer)
            marker = " <<< RELAY" if is_relay else ""
            print(f"\n  L{layer_idx}{marker}:")
            print(f"    CCS entropy: {ccs_l['entropy']:.2f}  Coding: {cod_l['entropy']:.2f}")
            print(f"    KL(CCS||Cod): {kl_ccs_cod:.3f}  KL(Cod||CCS): {kl_cod_ccs:.3f}")
            ccs_top3_str = ', '.join(repr(t['token']) + '(' + f"{t['prob']:.3f}" + ')' for t in ccs_l['top_tokens'][:3])
            cod_top3_str = ', '.join(repr(t['token']) + '(' + f"{t['prob']:.3f}" + ')' for t in cod_l['top_tokens'][:3])
            print(f"    CCS top3: {ccs_top3_str}")
            print(f"    Cod top3: {cod_top3_str}")
            if ccs_unique:
                uniq_str = ', '.join(repr(t['token']) + '(' + f"{t['prob']:.3f}" + ')' for t in ccs_unique[:3])
                print(f"    CCS-unique: {uniq_str}")

            model_output["layers"][str(layer_idx)] = layer_data

        # Clean up tensors before saving
        del ccs_results, cod_results

        model_output["relay_layer"] = relay_layer
        model_output["n_layers"] = n_layers
        all_results[model_key] = model_output

        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"logit_lens_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  LOGIT LENS SUMMARY — CCS vs CODING")
    print(f"{'='*60}")
    for model_key, data in all_results.items():
        relay = data["relay_layer"]
        print(f"\n  {model_key.upper()} (relay L{relay}):")
        for layer_str, ld in data["layers"].items():
            layer = int(layer_str)
            marker = " <<< RELAY" if layer == relay else ""
            kl = ld["kl_ccs_to_cod"]
            de = ld["ccs_entropy"] - ld["cod_entropy"]
            print(f"    L{layer:>3}: KL={kl:>7.3f}  dEntropy={de:>+.2f}  "
                  f"CCS=[{ld['ccs_top5'][0]['token']!r}]  "
                  f"Cod=[{ld['cod_top5'][0]['token']!r}]{marker}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
