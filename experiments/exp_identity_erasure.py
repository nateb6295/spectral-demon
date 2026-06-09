#!/usr/bin/env python3
"""
Identity Erasure — Reverse activation patching.

If the forward patch (inject CCS direction into bare) converts bare→CCS,
does the reverse (subtract CCS direction from CCS) convert CCS→bare?

If symmetric, a single vector controls identity at the relay layer.
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

    def hook_fn(module, input, output):
        h = output[0]
        pv = patch_vector.to(h.device).to(h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * pv
        return (h,) + output[1:]

    hook_handle = model.model.layers[patch_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        outputs = model(**inputs)
    hook_handle.remove()
    return outputs.logits[0, -1, :].float().cpu()


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
            model_name, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="eager",
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        relay_layer = n_layers - 2
        conv_key = "gemma" if model_key == "gemma" else "default"

        print(f"  Getting hidden states...")
        ccs_hidden, ccs_logits = get_hidden_and_logits(model, tokenizer, CCS_CONV[conv_key])
        bare_hidden, bare_logits = get_hidden_and_logits(model, tokenizer, BARE_CONV[conv_key])

        ccs_probs = F.softmax(ccs_logits, dim=-1)
        bare_probs = F.softmax(bare_logits, dim=-1)
        ccs_log_probs = F.log_softmax(ccs_logits, dim=-1)
        bare_log_probs = F.log_softmax(bare_logits, dim=-1)

        kl_ccs_bare = kl_div(ccs_log_probs, bare_probs)
        print(f"  KL(CCS || bare) baseline: {kl_ccs_bare:.3f}")

        top5_ccs = torch.topk(ccs_probs, 5)
        top5_bare = torch.topk(bare_probs, 5)
        ccs_str = [(tokenizer.decode([t.item()]), f'{p:.3f}') for t, p in zip(top5_ccs.indices, top5_ccs.values)]
        bare_str = [(tokenizer.decode([t.item()]), f'{p:.3f}') for t, p in zip(top5_bare.indices, top5_bare.values)]
        print(f"  CCS top5: {ccs_str}")
        print(f"  Bare top5: {bare_str}")

        test_layers = [relay_layer - 4, relay_layer - 2, relay_layer, n_layers - 1]
        test_layers = sorted(set(l for l in test_layers if 0 <= l < n_layers))
        alphas = [-0.5, -1.0, -2.0, -5.0]

        model_results = {"baseline_kl": kl_ccs_bare, "relay_layer": relay_layer, "erasures": {}}

        for layer in test_layers:
            erase_vec = ccs_hidden[layer + 1] - bare_hidden[layer + 1]
            erase_norm = float(torch.norm(erase_vec))
            is_relay = (layer == relay_layer)
            marker = " <<< RELAY" if is_relay else ""
            print(f"\n  L{layer} (erase norm: {erase_norm:.2f}):{marker}")

            layer_results = {"erase_norm": erase_norm, "alphas": {}}

            for alpha in alphas:
                patched_logits = patch_and_forward(model, tokenizer, CCS_CONV[conv_key],
                                                    layer, erase_vec, alpha)
                patched_probs = F.softmax(patched_logits, dim=-1)
                patched_log_probs = F.log_softmax(patched_logits, dim=-1)

                kl_to_bare = kl_div(patched_log_probs, bare_probs)
                kl_to_ccs = kl_div(ccs_log_probs, patched_probs)

                top3 = torch.topk(patched_probs, 3)
                tok_str = ', '.join(repr(tokenizer.decode([t.item()])) + '(' + f'{p:.3f}' + ')'
                                    for t, p in zip(top3.indices, top3.values))

                shift_toward_bare = (kl_ccs_bare - kl_to_bare) / kl_ccs_bare if kl_ccs_bare > 0 else 0
                print(f"    a={alpha}: KL->bare={kl_to_bare:.3f} (erase={shift_toward_bare:+.1%}), "
                      f"KL->CCS={kl_to_ccs:.3f}, top=[{tok_str}]{marker}")

                layer_results["alphas"][str(alpha)] = {
                    "kl_to_bare": kl_to_bare,
                    "kl_to_ccs": kl_to_ccs,
                    "shift_toward_bare": shift_toward_bare,
                    "top3": [(tokenizer.decode([t.item()]), float(p))
                             for t, p in zip(top3.indices[:3], top3.values[:3])],
                }

            model_results["erasures"][str(layer)] = layer_results

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"identity_erasure_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  IDENTITY ERASURE SUMMARY")
    print(f"{'='*60}")
    for mk, data in all_results.items():
        relay = data['relay_layer']
        kl = data['baseline_kl']
        print(f"\n  {mk.upper()} (relay L{relay}, baseline KL={kl:.3f}):")
        for ls in sorted(data['erasures'].keys(), key=int):
            layer = int(ls)
            m = " <R>" if layer == relay else ""
            ld = data['erasures'][ls]
            for a in ['-1.0', '-2.0', '-5.0']:
                if a in ld['alphas']:
                    s = ld['alphas'][a]['shift_toward_bare']
                    t = ld['alphas'][a]['top3'][0][0]
                    print(f"    L{layer:>3} a={a}: erase={s:>+.1%}  top={t!r}{m}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
