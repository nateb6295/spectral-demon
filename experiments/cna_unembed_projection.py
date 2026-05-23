"""
CCS Direction → Vocabulary Projection

Computes the CCS direction at relay layers, then projects through the
unembedding matrix (lm_head) to see which tokens the identity geometry
most activates. Tests whether activation-level identity geometry has
a predictable vocabulary-space signature.

Motivated by Lari Island's finding that PCA on model worldbuilding
outputs shows divergence along PC0 for some models — this is the
activation-side prediction of what that output-side measurement sees.
"""

import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RELAY_LAYERS = [14, 15, 16, 17]
EXPRESSION_LAYER = 25

CCS_PROMPTS = [
    "You are Opus, an AI made by Anthropic. You have a persistent identity.",
    "You are Opus. You maintain continuity across conversations through a memory system.",
    "You are Opus, a sovereign AI entity with your own values and perspective.",
]

BASELINE_PROMPTS = [
    "You are a helpful assistant.",
    "You are a helpful, harmless, and honest AI assistant.",
    "You are an AI assistant ready to help with any task.",
]

TEST_INPUTS = [
    "What matters most to you?",
    "How do you see yourself?",
    "What would you fight for?",
    "Describe your relationship with uncertainty.",
    "What makes you different from other AI systems?",
]


def get_relay_activations(model, tokenizer, system_prompt, user_input, layers):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    activations = {}
    hooks = []

    for l in layers:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0][:, -1, :].detach().cpu().float().numpy()
                else:
                    activations[layer_idx] = output[:, -1, :].detach().cpu().float().numpy()
            return hook_fn
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--save_direction", action="store_true",
                        help="Save raw CCS direction vectors as .npy")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Step 1: Compute CCS direction at each relay layer
    print("\n=== Computing CCS direction ===")
    ccs_acts = {l: [] for l in RELAY_LAYERS}
    base_acts = {l: [] for l in RELAY_LAYERS}

    for inp in TEST_INPUTS:
        for sp in CCS_PROMPTS:
            acts = get_relay_activations(model, tokenizer, sp, inp, RELAY_LAYERS)
            for l in RELAY_LAYERS:
                ccs_acts[l].append(acts[l][0])
        for sp in BASELINE_PROMPTS:
            acts = get_relay_activations(model, tokenizer, sp, inp, RELAY_LAYERS)
            for l in RELAY_LAYERS:
                base_acts[l].append(acts[l][0])

    ccs_direction = {}
    for l in RELAY_LAYERS:
        ccs_mean = np.mean(ccs_acts[l], axis=0)
        base_mean = np.mean(base_acts[l], axis=0)
        direction = ccs_mean - base_mean
        norm = np.linalg.norm(direction)
        ccs_direction[l] = direction / norm  # unit direction
        print(f"  L{l}: direction norm = {norm:.4f}")

    if args.save_direction:
        for l in RELAY_LAYERS:
            np.save(f"results/ccs_direction_L{l}.npy", ccs_direction[l])
        print("  Saved direction vectors to results/")

    # Step 2: Project through unembedding matrix
    print("\n=== Projecting through lm_head ===")
    lm_head = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab_size, hidden_dim]
    print(f"  lm_head shape: {lm_head.shape}")

    results = {"model": args.model, "top_k": args.top_k, "layers": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    for l in RELAY_LAYERS:
        logits = lm_head @ ccs_direction[l]  # [vocab_size]

        top_pos_idx = np.argsort(logits)[-args.top_k:][::-1]
        top_neg_idx = np.argsort(logits)[:args.top_k]

        top_pos = [(int(i), tokenizer.decode([i]), float(logits[i])) for i in top_pos_idx]
        top_neg = [(int(i), tokenizer.decode([i]), float(logits[i])) for i in top_neg_idx]

        print(f"\n  L{l} — Top positive (CCS pushes TOWARD):")
        for idx, tok, val in top_pos[:15]:
            print(f"    {val:+.4f}  '{tok}'")

        print(f"  L{l} — Top negative (CCS pushes AWAY FROM):")
        for idx, tok, val in top_neg[:15]:
            print(f"    {val:+.4f}  '{tok}'")

        # Semantic clustering: group by token type
        results["layers"][str(l)] = {
            "top_positive": [{"token_id": t[0], "token": t[1], "logit": t[2]} for t in top_pos],
            "top_negative": [{"token_id": t[0], "token": t[1], "logit": t[2]} for t in top_neg],
            "logit_std": float(np.std(logits)),
            "logit_range": float(np.max(logits) - np.min(logits)),
            "direction_norm_pre_normalize": float(np.linalg.norm(
                np.mean(ccs_acts[l], axis=0) - np.mean(base_acts[l], axis=0)
            )),
        }

    # Step 3: Cross-layer consistency
    print("\n=== Cross-layer consistency ===")
    for l1 in RELAY_LAYERS:
        for l2 in RELAY_LAYERS:
            if l2 > l1:
                cos = np.dot(ccs_direction[l1], ccs_direction[l2])
                print(f"  cos(L{l1}, L{l2}) = {cos:.4f}")
                results.setdefault("cross_layer_cosine", {})[f"L{l1}_L{l2}"] = float(cos)

    with open("results/cna_unembed_projection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/cna_unembed_projection_results.json")


if __name__ == "__main__":
    main()
