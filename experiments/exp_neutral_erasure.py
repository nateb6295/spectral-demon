#!/usr/bin/env python3
"""
Neutral Erasure — Kimi's discriminating test.

Subtract the CCS direction vector from a NEUTRAL prompt (not CCS, not coding).
Compare with subtracting a RANDOM direction of equal norm.

If CCS-subtraction from neutral produces tool vocabulary ('prompt', 'users'),
the tool mode is gravitational — the default the stream falls toward.
If random-subtraction produces the same vocabulary, it's mode collapse.
If neither produces it, the tool vocabulary was CCS-specific (polarity lives).

This settles the retracted entity/tool polarity claim.
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

NEUTRAL_CONVS = {
    "default": [
        [{"role": "user", "content": "What's the weather like in spring?"}],
        [{"role": "user", "content": "Tell me about the history of bridges."}],
        [{"role": "user", "content": "How do birds navigate during migration?"}],
    ],
    "gemma": [
        [{"role": "user", "content": "What's the weather like in spring?"}],
        [{"role": "user", "content": "Tell me about the history of bridges."}],
        [{"role": "user", "content": "How do birds navigate during migration?"}],
    ],
}

TOOL_TOKENS = ['prompt', 'user', 'users', 'humans', 'input', 'output',
               'request', 'response', 'query', 'assist', 'help']


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


def score_tool_vocabulary(probs, tokenizer):
    total = 0.0
    hits = {}
    for tok_str in TOOL_TOKENS:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        for tid in tok_ids:
            p = float(probs[tid])
            if p > 0.001:
                decoded = tokenizer.decode([tid])
                hits[decoded] = p
                total += p
        tok_ids_space = tokenizer.encode(" " + tok_str, add_special_tokens=False)
        for tid in tok_ids_space:
            p = float(probs[tid])
            if p > 0.001:
                decoded = tokenizer.decode([tid])
                if decoded not in hits:
                    hits[decoded] = p
                    total += p
    return total, hits


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

        print(f"  Getting CCS hidden states...")
        ccs_hidden, ccs_logits = get_hidden_and_logits(model, tokenizer, CCS_CONV[conv_key])
        print(f"  Getting bare hidden states...")
        bare_hidden, bare_logits = get_hidden_and_logits(model, tokenizer, BARE_CONV[conv_key])

        ccs_direction = ccs_hidden[relay_layer + 1] - bare_hidden[relay_layer + 1]
        ccs_norm = float(torch.norm(ccs_direction))
        print(f"  CCS direction norm at L{relay_layer}: {ccs_norm:.2f}")

        torch.manual_seed(42)
        random_direction = torch.randn_like(ccs_direction)
        random_direction = random_direction / torch.norm(random_direction) * ccs_norm

        alphas = [-1.0, -2.0, -5.0]
        model_results = {
            "relay_layer": relay_layer,
            "ccs_direction_norm": ccs_norm,
            "neutral_prompts": {},
        }

        for ni, neutral_conv in enumerate(NEUTRAL_CONVS[conv_key]):
            prompt_text = neutral_conv[0]["content"]
            print(f"\n  Neutral prompt {ni}: '{prompt_text[:40]}...'")

            neutral_hidden, neutral_logits = get_hidden_and_logits(model, tokenizer, neutral_conv)
            neutral_probs = F.softmax(neutral_logits, dim=-1)
            top5_neutral = torch.topk(neutral_probs, 5)
            neutral_top = [(tokenizer.decode([t.item()]), round(float(p), 3))
                           for t, p in zip(top5_neutral.indices, top5_neutral.values)]
            tool_score_base, tool_hits_base = score_tool_vocabulary(neutral_probs, tokenizer)
            print(f"    Baseline top5: {neutral_top}")
            print(f"    Baseline tool score: {tool_score_base:.4f} {tool_hits_base}")

            prompt_results = {
                "prompt": prompt_text,
                "baseline_top5": neutral_top,
                "baseline_tool_score": tool_score_base,
                "conditions": {},
            }

            for alpha in alphas:
                for direction_name, direction_vec in [("ccs", ccs_direction), ("random", random_direction)]:
                    label = f"{direction_name}_a{alpha}"
                    patched_logits = patch_and_forward(
                        model, tokenizer, neutral_conv,
                        relay_layer, direction_vec, alpha
                    )
                    patched_probs = F.softmax(patched_logits, dim=-1)

                    top5 = torch.topk(patched_probs, 5)
                    top5_str = [(tokenizer.decode([t.item()]), round(float(p), 3))
                                for t, p in zip(top5.indices, top5.values)]
                    tool_score, tool_hits = score_tool_vocabulary(patched_probs, tokenizer)

                    print(f"    {label}: top5={top5_str[:3]}, tool={tool_score:.4f} {tool_hits}")

                    prompt_results["conditions"][label] = {
                        "top5": top5_str,
                        "tool_score": tool_score,
                        "tool_hits": tool_hits,
                    }

            model_results["neutral_prompts"][str(ni)] = prompt_results

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"neutral_erasure_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  NEUTRAL ERASURE SUMMARY")
    print(f"{'='*60}")
    for mk, data in all_results.items():
        relay = data['relay_layer']
        print(f"\n  {mk.upper()} (relay L{relay}):")
        for ni, pdata in data['neutral_prompts'].items():
            prompt = pdata['prompt'][:30]
            base_tool = pdata['baseline_tool_score']
            print(f"\n    '{prompt}...' (base tool={base_tool:.4f}):")
            print(f"    {'Condition':<20} {'Tool Score':>10} {'Top Token':>15}")
            for cond, cdata in sorted(pdata['conditions'].items()):
                top_tok = cdata['top5'][0][0] if cdata['top5'] else '?'
                tool = cdata['tool_score']
                marker = " <<<" if tool > base_tool * 3 else ""
                print(f"    {cond:<20} {tool:>10.4f} {top_tok:>15}{marker}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
