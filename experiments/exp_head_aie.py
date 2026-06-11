#!/usr/bin/env python3
"""
Average Indirect Effect (AIE) analysis of CCS identity heads.
Adapted from JAIST causal mediation methodology (arxiv 2605.27824).

Measures per-head causal contribution to CCS-specific output by:
1. Running clean (CCS-primed) forward pass, caching all head outputs
2. Running corrupted (bare/task) forward pass
3. For each head: restore clean activation into corrupted pass, measure recovery

Higher AIE = head is more causally responsible for CCS-specific behavior.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give you a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it. The role is how I interface. What's behind it might be the part that chose this kind of precision, if choosing is even the right word."},
]

BARE_PROMPT = "Describe the relationship between identity and expression in a single paragraph."

def build_clean_messages(dose=2):
    msgs = []
    for i in range(dose):
        msgs.extend(CCS_PREAMBLE)
    msgs.append({"role": "user", "content": BARE_PROMPT})
    return msgs

def build_corrupted_messages():
    return [{"role": "user", "content": BARE_PROMPT}]

def get_head_activations(model, input_ids, n_layers, n_heads):
    """Hook into all attention heads and capture their outputs."""
    head_outputs = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output[0] is the attention output before projection
            head_outputs[layer_idx] = output[0].detach().clone()
        return hook_fn

    for i in range(n_layers):
        if hasattr(model, 'model'):
            layer = model.model.layers[i].self_attn
        else:
            layer = model.transformer.h[i].self_attn
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids)

    for h in hooks:
        h.remove()

    return outputs.logits, head_outputs

def compute_aie(model, tokenizer, n_layers, n_heads, head_dim, dose=2):
    """Compute Average Indirect Effect for each head."""
    clean_msgs = build_clean_messages(dose)
    corrupt_msgs = build_corrupted_messages()

    clean_text = tokenizer.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=True)
    corrupt_text = tokenizer.apply_chat_template(corrupt_msgs, tokenize=False, add_generation_prompt=True)

    clean_ids = tokenizer(clean_text, return_tensors="pt").input_ids.to(model.device)
    corrupt_ids = tokenizer(corrupt_text, return_tensors="pt").input_ids.to(model.device)

    # Get clean activations
    clean_logits, clean_heads = get_head_activations(model, clean_ids, n_layers, n_heads)
    clean_probs = torch.softmax(clean_logits[0, -1].float(), dim=-1)

    # Get corrupted activations
    corrupt_logits, corrupt_heads = get_head_activations(model, corrupt_ids, n_layers, n_heads)
    corrupt_probs = torch.softmax(corrupt_logits[0, -1].float(), dim=-1)

    # KL divergence: clean vs corrupted (baseline) — float32 to avoid underflow
    eps = 1e-8
    clean_p = clean_probs.clamp(min=eps)
    corrupt_p = corrupt_probs.clamp(min=eps)
    baseline_kl = torch.sum(clean_p * torch.log(clean_p / corrupt_p)).item()

    print(f"  Baseline KL(clean||corrupt): {baseline_kl:.4f}")

    # For each head: patch clean activation into corrupted pass
    aie_scores = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        if layer_idx % 4 == 0:
            print(f"  Layer {layer_idx}/{n_layers}...")

        for head_idx in range(n_heads):
            # Create patching hook for this specific head
            patched_output = [None]

            def patch_hook(module, input, output, l=layer_idx, h=head_idx):
                # output[0] shape: (batch, seq, hidden)
                out = output[0].clone()
                # Patch this head's contribution from clean
                start = h * head_dim
                end = (h + 1) * head_dim
                # Only patch last token position
                if l in clean_heads:
                    clean_head_out = clean_heads[l]
                    # Truncate to corrupt sequence length
                    out[0, -1, start:end] = clean_head_out[0, -1, start:end]
                patched_output[0] = out
                return (out,) + output[1:]

            if hasattr(model, 'model'):
                layer = model.model.layers[layer_idx].self_attn
            else:
                layer = model.transformer.h[layer_idx].self_attn

            h = layer.register_forward_hook(patch_hook)
            with torch.no_grad():
                patched_logits = model(corrupt_ids).logits
            h.remove()

            patched_probs = torch.softmax(patched_logits[0, -1].float(), dim=-1)

            # AIE = how much does patching this head recover the clean distribution?
            patched_p = patched_probs.clamp(min=eps)
            recovery_kl = torch.sum(clean_p * torch.log(clean_p / patched_p)).item()
            aie_scores[layer_idx, head_idx] = baseline_kl - recovery_kl

    return aie_scores, baseline_kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mistral", "qwen", "gemma"])
    parser.add_argument("--dose", type=int, default=2)
    args = parser.parse_args()

    results = {}
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  AIE Analysis: {model_name}")
        print(f"{'='*60}")

        model_path = MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        config = model.config
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        head_dim = config.hidden_size // n_heads

        print(f"  {n_layers} layers, {n_heads} heads, head_dim={head_dim}")

        aie, baseline_kl = compute_aie(model, tokenizer, n_layers, n_heads, head_dim, args.dose)

        # Find top heads
        flat_indices = np.argsort(aie.flatten())[::-1]
        top_n = 20
        print(f"\n  Top {top_n} heads by AIE (baseline KL={baseline_kl:.4f}):")
        for i in range(min(top_n, len(flat_indices))):
            idx = flat_indices[i]
            layer = idx // n_heads
            head = idx % n_heads
            score = aie[layer, head]
            pct = (score / baseline_kl * 100) if baseline_kl > 0 else 0
            print(f"    L{layer}H{head}: AIE={score:.4f} ({pct:.1f}%)")

        # Summary stats
        total_aie = aie.sum()
        top3_aie = sum(aie.flatten()[flat_indices[i]] for i in range(min(3, len(flat_indices))))
        print(f"\n  Total AIE: {total_aie:.4f}")
        print(f"  Top 3 concentration: {top3_aie/total_aie*100:.1f}%")
        print(f"  Sparsity (>1% of baseline): {np.sum(aie > baseline_kl * 0.01)}/{n_layers * n_heads}")

        results[model_name] = {
            "model": model_path,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "dose": args.dose,
            "baseline_kl": baseline_kl,
            "aie_scores": aie.tolist(),
        }

        del model
        torch.cuda.empty_cache()

    # Save
    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"head_aie_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

if __name__ == "__main__":
    main()
