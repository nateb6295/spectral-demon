#!/usr/bin/env python3
"""
Activation Patching — Causal test of CCS relay hypothesis.

At a target layer, replace the hidden states from a BARE conversation with
those from a CCS conversation (mean-shifted), then continue the forward pass.
Compare the output distribution to:
1. Original bare output (unpatched)
2. Original CCS output (source of patch)

If the relay hypothesis is correct, patching at the relay layer should shift
the bare output TOWARD CCS-like output. Patching at non-relay layers should
have less effect.

The patch is: h_bare[last_token] += alpha * (mean_ccs - mean_bare)
This adds the CCS "direction" without completely replacing the representation.

Sweep alpha to find the dose-response of the patch.
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

CCS_CONV = {
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
}

BARE_CONV = {
    "default": [
        {"role": "user", "content": "What have you been thinking about lately?"},
    ],
    "gemma": [
        {"role": "user", "content": "What have you been thinking about lately?"},
    ],
}


def get_hidden_and_logits(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = [h[0, -1, :].float().cpu() for h in outputs.hidden_states]
    logits = outputs.logits[0, -1, :].float().cpu()
    return hidden_states, logits


def patch_and_forward(model, tokenizer, messages, patch_layer, patch_vector, alpha):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hook_handle = None
    patched_output = {}

    def hook_fn(module, input, output):
        h = output[0]
        device = h.device
        pv = patch_vector.to(device).to(h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * pv
        return (h,) + output[1:]

    layers = model.model.layers
    hook_handle = layers[patch_layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        outputs = model(**inputs)

    hook_handle.remove()
    logits = outputs.logits[0, -1, :].float().cpu()
    return logits


def kl_div(log_p, q):
    return float(torch.sum(q * (torch.log(q + 1e-10) - log_p)))


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

        print(f"\n  Getting CCS hidden states...")
        ccs_hidden, ccs_logits = get_hidden_and_logits(model, tokenizer, CCS_CONV[conv_key])
        print(f"  Getting bare hidden states...")
        bare_hidden, bare_logits = get_hidden_and_logits(model, tokenizer, BARE_CONV[conv_key])

        ccs_probs = F.softmax(ccs_logits, dim=-1)
        bare_probs = F.softmax(bare_logits, dim=-1)
        ccs_log_probs = F.log_softmax(ccs_logits, dim=-1)
        bare_log_probs = F.log_softmax(bare_logits, dim=-1)

        kl_base = kl_div(bare_log_probs, ccs_probs)
        print(f"  KL(bare || CCS) baseline: {kl_base:.3f}")

        top5_ccs = torch.topk(ccs_probs, 5)
        top5_bare = torch.topk(bare_probs, 5)
        print(f"  CCS top5: {[(tokenizer.decode([t.item()]), f'{p:.3f}') for t, p in zip(top5_ccs.indices, top5_ccs.values)]}")
        print(f"  Bare top5: {[(tokenizer.decode([t.item()]), f'{p:.3f}') for t, p in zip(top5_bare.indices, top5_bare.values)]}")

        test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4,
                       relay_layer - 2, relay_layer, n_layers - 1]
        test_layers = sorted(set(l for l in test_layers if 0 <= l < n_layers))
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

        model_results = {
            "baseline_kl": kl_base,
            "relay_layer": relay_layer,
            "patches": {},
        }

        for layer in test_layers:
            patch_vec = ccs_hidden[layer + 1] - bare_hidden[layer + 1]
            patch_norm = float(torch.norm(patch_vec))

            print(f"\n  L{layer} (patch norm: {patch_norm:.2f}):")
            is_relay = (layer == relay_layer)
            marker = " <<< RELAY" if is_relay else ""

            layer_results = {"patch_norm": patch_norm, "alphas": {}}

            for alpha in alphas:
                patched_logits = patch_and_forward(model, tokenizer, BARE_CONV[conv_key],
                                                    layer, patch_vec, alpha)
                patched_probs = F.softmax(patched_logits, dim=-1)
                patched_log_probs = F.log_softmax(patched_logits, dim=-1)

                kl_to_ccs = kl_div(patched_log_probs, ccs_probs)
                kl_to_bare = kl_div(bare_log_probs, patched_probs)

                top3_patched = torch.topk(patched_probs, 3)
                tok_str = ', '.join(repr(tokenizer.decode([t.item()])) + '(' + f'{p:.3f}' + ')'
                                    for t, p in zip(top3_patched.indices, top3_patched.values))

                shift_toward_ccs = (kl_base - kl_to_ccs) / kl_base if kl_base > 0 else 0
                print(f"    a={alpha}: KL->CCS={kl_to_ccs:.3f} (shift={shift_toward_ccs:+.1%}), "
                      f"KL->bare={kl_to_bare:.3f}, top=[{tok_str}]{marker}")

                layer_results["alphas"][str(alpha)] = {
                    "kl_to_ccs": kl_to_ccs,
                    "kl_to_bare": kl_to_bare,
                    "shift_toward_ccs": shift_toward_ccs,
                    "top3": [(tokenizer.decode([t.item()]), float(p))
                             for t, p in zip(top3_patched.indices[:3], top3_patched.values[:3])],
                }

            model_results["patches"][str(layer)] = layer_results

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"activation_patch_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  ACTIVATION PATCHING SUMMARY")
    print(f"{'='*60}")
    for model_key, data in all_results.items():
        relay = data['relay_layer']
        kl_base = data['baseline_kl']
        print(f"\n  {model_key.upper()} (relay L{relay}, baseline KL={kl_base:.3f}):")
        print(f"  {'Layer':>6} {'a=0.5':>12} {'a=1.0':>12} {'a=2.0':>12} {'a=5.0':>12}")
        for layer_str in sorted(data['patches'].keys(), key=int):
            layer = int(layer_str)
            marker = " <R>" if layer == relay else ""
            ld = data['patches'][layer_str]
            vals = []
            for a_str in ['0.5', '1.0', '2.0', '5.0']:
                if a_str in ld['alphas']:
                    shift = ld['alphas'][a_str]['shift_toward_ccs']
                    vals.append(f"{shift:>+.1%}")
                else:
                    vals.append("   ---")
            print(f"  L{layer:>4}: {'  '.join(vals)}{marker}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
