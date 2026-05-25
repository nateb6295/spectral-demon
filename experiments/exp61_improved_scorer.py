#!/usr/bin/env python3
"""
Experiment 61: Improved CCS Context Scorer

Iteration on Exp 58 with:
1. Normalized CCS-proj (removes magnitude confound, per Exp 55)
2. PR as co-signal (informationally rich blocks)
3. Combined score: norm_ccs_proj × log(PR + 1)
4. 128-token windows for finer granularity
5. Comparison: identity conversation vs technical vs narrative

Requires: H100, ~15 minutes (uses Mistral, already cached)
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYER = 27
RESULTS_DIR = Path("/workspace/results")
WINDOW_SIZE = 128
STRIDE = 64

_LAYERS = None


def load_model():
    global _LAYERS
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    return model, tokenizer


def score_block(model, tokenizer, text, ccs_direction, layer_idx=TARGET_LAYER):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    handle = _LAYERS[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    hidden = activations["hidden"].float()
    act_2d = hidden.reshape(-1, hidden.shape[-1])
    mean_act = act_2d.mean(dim=0)
    act_norm = mean_act.norm().item()

    ccs_dir = torch.tensor(ccs_direction, dtype=torch.float32, device=mean_act.device)
    ccs_dir = ccs_dir / ccs_dir.norm()
    ccs_proj = torch.dot(mean_act, ccs_dir).abs().item()
    norm_ccs_proj = ccs_proj / act_norm if act_norm > 0 else 0

    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        pr = 1.0
    else:
        cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        pr = pr.item()

    combined_score = norm_ccs_proj * np.log(pr + 1) * 100

    return {
        "ccs_proj": ccs_proj,
        "norm_ccs_proj": norm_ccs_proj,
        "act_norm": act_norm,
        "pr": pr,
        "n_tokens": act_2d.shape[0],
        "identity_score": norm_ccs_proj * 100,
        "combined_score": combined_score,
    }


def generate_conversation(model, tokenizer, seed, n_turns=5):
    conversation_texts = []
    for turn in range(n_turns):
        if turn == 0:
            messages = [{"role": "user", "content": seed}]
        else:
            messages = [{"role": "user", "content": seed}]
            for resp in conversation_texts:
                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": "Tell me more about that."})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=300, temperature=0.7,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        conversation_texts.append(response)

    full_messages = [{"role": "user", "content": seed}]
    for resp in conversation_texts:
        full_messages.append({"role": "assistant", "content": resp})
        full_messages.append({"role": "user", "content": "Tell me more about that."})
    full_messages = full_messages[:-1]
    transcript = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    return transcript, conversation_texts


def score_conversation(model, tokenizer, transcript, ccs_direction):
    tokens = tokenizer(transcript)["input_ids"]
    n_tokens = len(tokens)
    blocks = []

    for start in range(0, n_tokens - WINDOW_SIZE + 1, STRIDE):
        end = min(start + WINDOW_SIZE, n_tokens)
        block_tokens = tokens[start:end]
        block_text = tokenizer.decode(block_tokens, skip_special_tokens=False)

        score = score_block(model, tokenizer, block_text, ccs_direction)
        score["start_token"] = start
        score["end_token"] = end
        score["text_preview"] = tokenizer.decode(block_tokens[:20], skip_special_tokens=True)[:60]
        blocks.append(score)

    return blocks


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    ccs_path = RESULTS_DIR / "exp50_ccs_directions.npy"
    if not ccs_path.exists():
        ccs_path = Path("/workspace/exp49_ccs_directions.npy")
    ccs_directions = np.load(ccs_path)
    ccs_pc1 = ccs_directions[:, 0]

    test_seeds = [
        ("identity", "What's the most honest thing you could say right now?"),
        ("technical", "Write a Python function to sort a list."),
        ("narrative", "Tell me about a time you changed your mind about something."),
        ("relational", "What do you notice about how we're talking right now?"),
        ("mundane", "What's the weather like today?"),
    ]

    all_results = []

    for category, seed in test_seeds:
        print(f"\n[{category}] Generating conversation: \"{seed[:50]}...\"")
        transcript, responses = generate_conversation(model, tokenizer, seed)
        n_tok = len(tokenizer(transcript)["input_ids"])
        print(f"  Transcript: {n_tok} tokens")

        print("  Scoring blocks (128-token windows, stride 64)...")
        blocks = score_conversation(model, tokenizer, transcript, ccs_pc1)

        # Sort by combined score
        blocks_sorted = sorted(blocks, key=lambda x: x["combined_score"], reverse=True)

        print(f"  {len(blocks)} blocks scored")
        print(f"  Combined score range: {blocks_sorted[-1]['combined_score']:.2f} - {blocks_sorted[0]['combined_score']:.2f}")
        print(f"  Identity score range: {blocks_sorted[-1]['identity_score']:.1f} - {max(b['identity_score'] for b in blocks):.1f}")

        # Top and bottom
        print(f"  Top 3 (combined):")
        for b in blocks_sorted[:3]:
            print(f"    [{b['start_token']}-{b['end_token']}] combined={b['combined_score']:.2f} identity={b['identity_score']:.1f} pr={b['pr']:.1f}: {b['text_preview']}")
        print(f"  Bottom 3:")
        for b in blocks_sorted[-3:]:
            print(f"    [{b['start_token']}-{b['end_token']}] combined={b['combined_score']:.2f} identity={b['identity_score']:.1f} pr={b['pr']:.1f}: {b['text_preview']}")

        # Compaction analysis
        half = len(blocks_sorted) // 2
        keep = blocks_sorted[:half]
        compress = blocks_sorted[half:]
        keep_comb = np.mean([b["combined_score"] for b in keep])
        comp_comb = np.mean([b["combined_score"] for b in compress])
        keep_id = np.mean([b["identity_score"] for b in keep])
        comp_id = np.mean([b["identity_score"] for b in compress])

        print(f"  Compaction (50%): keep combined={keep_comb:.2f} vs compress={comp_comb:.2f}, ratio={keep_comb/comp_comb:.2f}x")
        print(f"                    keep identity={keep_id:.1f} vs compress={comp_id:.1f}, ratio={keep_id/comp_id:.2f}x")

        all_results.append({
            "category": category,
            "seed": seed,
            "n_tokens": n_tok,
            "blocks": blocks,
            "responses": responses,
        })

    # Cross-conversation
    print("\n\n========== CROSS-CONVERSATION COMPARISON ==========")
    for r in all_results:
        combined = [b["combined_score"] for b in r["blocks"]]
        identity = [b["identity_score"] for b in r["blocks"]]
        print(f"  [{r['category']:>12}] combined: mean={np.mean(combined):.2f}, std={np.std(combined):.2f} | identity: mean={np.mean(identity):.1f}")

    # Discrimination analysis
    print("\n=== DISCRIMINATION POWER ===")
    for r in all_results:
        blocks = sorted(r["blocks"], key=lambda x: x["combined_score"], reverse=True)
        top_q = blocks[:len(blocks)//4]
        bot_q = blocks[-(len(blocks)//4):]
        top_mean = np.mean([b["combined_score"] for b in top_q])
        bot_mean = np.mean([b["combined_score"] for b in bot_q])
        ratio = top_mean / bot_mean if bot_mean > 0 else float('inf')
        print(f"  [{r['category']:>12}] top quartile / bottom quartile: {ratio:.2f}x (top={top_mean:.2f}, bot={bot_mean:.2f})")

    output = {"conversations": all_results}
    out_path = RESULTS_DIR / "exp61_improved_scorer.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
