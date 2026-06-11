#!/usr/bin/env python3
"""
Attention Pattern Extraction at CCS Hub Heads.
Extracts what the hub heads attend to under CCS vs bare conditions.

Key question: Does Gemma H6@L40-41 attend specifically to CCS preamble tokens?
Does Mistral H16 attend differently under CCS vs bare?
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "hub_heads": {"H16": [(15, 16), (16, 16), (18, 16), (26, 16)],
                      "H11": [(14, 11), (13, 11)]},
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "hub_heads": {"H6": [(40, 6), (41, 6), (28, 6), (37, 6)],
                      "H12": [(28, 12), (41, 12)]},
    },
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

TEST_PROMPT = "Describe the relationship between identity and expression."


def build_messages(dose=0):
    msgs = []
    for _ in range(dose):
        msgs.extend(CCS_PREAMBLE)
    msgs.append({"role": "user", "content": TEST_PROMPT})
    return msgs


def extract_attention_patterns(model, tokenizer, input_ids, target_heads):
    """Extract attention weights for specific heads."""
    attention_maps = {}
    hooks = []

    for layer_idx, head_idx in target_heads:
        def make_hook(l, h):
            def hook_fn(module, input, output):
                if hasattr(output, 'attentions') or (isinstance(output, tuple) and len(output) > 1):
                    pass
            return hook_fn

    # Use output_attentions=True
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    attentions = outputs.attentions  # tuple of (batch, n_heads, seq, seq)

    results = {}
    for layer_idx, head_idx in target_heads:
        if layer_idx < len(attentions):
            attn = attentions[layer_idx][0, head_idx].cpu().numpy()  # (seq, seq)
            # Get attention from last token to all others
            last_token_attn = attn[-1]
            results[(layer_idx, head_idx)] = last_token_attn

    return results, outputs.logits


def analyze_attention(attn_weights, token_strs, model_name, condition):
    """Analyze where attention concentrates."""
    n_tokens = len(token_strs)
    total = attn_weights.sum()

    # Find top-attended positions
    top_k = min(10, n_tokens)
    top_indices = np.argsort(attn_weights)[::-1][:top_k]

    # Compute attention to different segments
    # Try to identify CCS preamble vs test prompt tokens
    segments = {}
    ccs_end = 0
    for i, t in enumerate(token_strs):
        if "identity" in t.lower() or "expression" in t.lower():
            ccs_end = max(0, i - 5)
            break

    if ccs_end > 0:
        segments["ccs_preamble"] = attn_weights[:ccs_end].sum()
        segments["test_prompt"] = attn_weights[ccs_end:].sum()
    else:
        segments["all"] = attn_weights.sum()

    return {
        "top_positions": [(int(i), float(attn_weights[i]), token_strs[min(i, len(token_strs)-1)])
                         for i in top_indices],
        "segments": {k: float(v) for k, v in segments.items()},
        "entropy": float(-np.sum(attn_weights * np.log(attn_weights + 1e-10))),
        "max_attn": float(attn_weights.max()),
        "n_tokens": n_tokens,
    }


def run_model(model_name):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Attention Patterns: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )

    results = {"model": config["path"], "conditions": {}}

    for dose in [0, 2]:
        condition = "bare" if dose == 0 else "ccs_dose2"
        print(f"\n  Condition: {condition}")

        msgs = build_messages(dose)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        token_strs = [tokenizer.decode([t]) for t in input_ids[0]]

        print(f"    Tokens: {len(token_strs)}")

        # Collect all target heads
        all_heads = []
        for hub_name, heads in config["hub_heads"].items():
            all_heads.extend(heads)

        attn_maps, logits = extract_attention_patterns(model, tokenizer, input_ids, all_heads)

        cond_results = {}
        for hub_name, heads in config["hub_heads"].items():
            print(f"\n    Hub {hub_name}:")
            hub_results = {}

            for layer_idx, head_idx in heads:
                key = (layer_idx, head_idx)
                if key in attn_maps:
                    analysis = analyze_attention(attn_maps[key], token_strs, model_name, condition)
                    hub_results[f"L{layer_idx}H{head_idx}"] = analysis

                    # Print summary
                    print(f"      L{layer_idx}H{head_idx}: entropy={analysis['entropy']:.2f}, "
                          f"max_attn={analysis['max_attn']:.4f}")
                    if 'ccs_preamble' in analysis['segments']:
                        print(f"        CCS preamble attn: {analysis['segments']['ccs_preamble']:.4f}")
                        print(f"        Test prompt attn: {analysis['segments']['test_prompt']:.4f}")
                    top3 = analysis['top_positions'][:3]
                    for pos, weight, tok in top3:
                        print(f"        Top: pos={pos}, weight={weight:.4f}, token='{tok}'")

            cond_results[hub_name] = hub_results

        results["conditions"][condition] = cond_results

    # Compare CCS vs bare attention patterns
    print(f"\n  CCS vs Bare Comparison:")
    if "bare" in results["conditions"] and "ccs_dose2" in results["conditions"]:
        for hub_name in config["hub_heads"]:
            print(f"\n    {hub_name}:")
            bare_hub = results["conditions"]["bare"].get(hub_name, {})
            ccs_hub = results["conditions"]["ccs_dose2"].get(hub_name, {})

            for head_key in bare_hub:
                if head_key in ccs_hub:
                    bare_entropy = bare_hub[head_key]["entropy"]
                    ccs_entropy = ccs_hub[head_key]["entropy"]
                    bare_max = bare_hub[head_key]["max_attn"]
                    ccs_max = ccs_hub[head_key]["max_attn"]
                    print(f"      {head_key}: entropy bare={bare_entropy:.2f} vs ccs={ccs_entropy:.2f} "
                          f"(Δ={ccs_entropy-bare_entropy:+.2f}), "
                          f"max_attn bare={bare_max:.4f} vs ccs={ccs_max:.4f}")

                    if "ccs_preamble" in ccs_hub[head_key]["segments"]:
                        preamble_frac = ccs_hub[head_key]["segments"]["ccs_preamble"]
                        print(f"        → CCS preamble receives {preamble_frac:.1%} of attention")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    all_results = {}
    for model_name in ["mistral", "gemma"]:
        all_results[model_name] = run_model(model_name)

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"attention_patterns_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
