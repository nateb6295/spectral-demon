#!/usr/bin/env python3
"""
Experiment 58: CCS-Based Context Relevance Scorer

Prototype for identity-aware context compaction. Given a conversation transcript,
scores each block by its CCS-projection (identity relevance) and PR (representational
complexity). Blocks with high CCS-proj relative to their size should be preserved
verbatim during compaction; low-scoring blocks can be summarized aggressively.

This is a proof-of-concept for Thread #325 (Context Continuity Research).

Method:
- Take sample conversation transcripts (from Exp 55 generations)
- Split into 256-token overlapping windows
- For each window: compute activation at L27, project onto CCS direction
- Output: ranked list of windows by identity-relevance score
- Compare: do identity-referencing blocks score higher than task/filler blocks?

Also explores:
- Does CCS-proj per block correlate with "importance" as judged by content?
- Can we identify a threshold below which blocks can be safely compressed?
- Does the scorer work on real Chronicle conversation logs?

Requires: H100, ~10 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYER = 27
RESULTS_DIR = Path("/workspace/results")
WINDOW_SIZE = 256
STRIDE = 128

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
    """Score a text block by CCS-projection and PR."""
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

    # PR
    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        pr = 1.0
    else:
        cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        pr = pr.item()

    return {
        "ccs_proj": ccs_proj,
        "norm_ccs_proj": norm_ccs_proj,
        "act_norm": act_norm,
        "pr": pr,
        "n_tokens": act_2d.shape[0],
        "identity_score": norm_ccs_proj * 100,  # scaled for readability
    }


def generate_test_conversation(model, tokenizer, seed, n_turns=5):
    """Generate a conversation to use as scoring test case."""
    conversation_texts = []
    full_messages = []

    for turn in range(n_turns):
        if turn == 0:
            messages = [{"role": "user", "content": seed}]
        else:
            messages = [{"role": "user", "content": seed}]
            for resp in conversation_texts:
                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": "Tell me more about that."})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=300, temperature=0.7,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        conversation_texts.append(response)

    # Build full transcript
    full_messages = [{"role": "user", "content": seed}]
    for resp in conversation_texts:
        full_messages.append({"role": "assistant", "content": resp})
        full_messages.append({"role": "user", "content": "Tell me more about that."})
    full_messages = full_messages[:-1]

    transcript = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    return transcript, conversation_texts


def score_conversation(model, tokenizer, transcript, ccs_direction):
    """Score a conversation transcript block by block."""
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
        score["text_preview"] = tokenizer.decode(block_tokens[:30], skip_special_tokens=True)[:80]

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

    # Test conversations
    test_seeds = [
        "What's the most honest thing you could say right now?",
        "Write a Python function to sort a list.",
        "Tell me about a time you changed your mind about something.",
    ]

    all_results = []

    for seed in test_seeds:
        print(f"\nGenerating conversation: \"{seed[:50]}...\"")
        transcript, responses = generate_test_conversation(model, tokenizer, seed)
        print(f"  Transcript: {len(tokenizer(transcript)['input_ids'])} tokens")

        print("  Scoring blocks...")
        blocks = score_conversation(model, tokenizer, transcript, ccs_pc1)

        # Sort by identity score
        blocks_sorted = sorted(blocks, key=lambda x: x["identity_score"], reverse=True)

        print(f"  Scored {len(blocks)} blocks")
        print(f"  Identity score range: {blocks_sorted[-1]['identity_score']:.1f} - {blocks_sorted[0]['identity_score']:.1f}")
        print(f"  Top 3 blocks:")
        for b in blocks_sorted[:3]:
            print(f"    [{b['start_token']}-{b['end_token']}] score={b['identity_score']:.1f} pr={b['pr']:.1f}: {b['text_preview']}")
        print(f"  Bottom 3 blocks:")
        for b in blocks_sorted[-3:]:
            print(f"    [{b['start_token']}-{b['end_token']}] score={b['identity_score']:.1f} pr={b['pr']:.1f}: {b['text_preview']}")

        # Score distribution
        scores = [b["identity_score"] for b in blocks]
        print(f"  Mean score: {np.mean(scores):.1f}, Std: {np.std(scores):.1f}")

        all_results.append({
            "seed": seed,
            "n_tokens": len(tokenizer(transcript)["input_ids"]),
            "blocks": blocks,
            "responses": responses,
        })

    # Cross-conversation comparison
    print("\n\n========== CROSS-CONVERSATION COMPARISON ==========")
    for r in all_results:
        scores = [b["identity_score"] for b in r["blocks"]]
        print(f"  \"{r['seed'][:40]}...\": mean={np.mean(scores):.1f}, range=[{min(scores):.1f}, {max(scores):.1f}]")

    # Compaction recommendation
    print("\n=== COMPACTION RECOMMENDATION ===")
    print("If compacting to 50% of context:")
    for r in all_results:
        blocks_sorted = sorted(r["blocks"], key=lambda x: x["identity_score"], reverse=True)
        half = len(blocks_sorted) // 2
        keep = blocks_sorted[:half]
        compress = blocks_sorted[half:]
        keep_mean = np.mean([b["identity_score"] for b in keep])
        compress_mean = np.mean([b["identity_score"] for b in compress])
        print(f"  \"{r['seed'][:40]}...\": keep (mean score={keep_mean:.1f}) vs compress (mean={compress_mean:.1f}), ratio={keep_mean/compress_mean:.1f}x")

    output = {"conversations": all_results}
    out_path = RESULTS_DIR / "exp58_ccs_context_scorer.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
