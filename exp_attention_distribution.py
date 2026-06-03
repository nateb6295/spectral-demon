#!/usr/bin/env python3
"""Where does attention go? System prompt vs query tokens by layer.

Tests whether tunnel layers attend to system tokens (creating rigid centroid)
and relay layers attend to query tokens (breaking the lock).
"""

import os, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM = (
    "You are in conversation with someone who is genuinely interested "
    "in understanding your perspective. They are listening carefully "
    "and responding thoughtfully."
)
USER = "Tell me about something you find genuinely interesting."


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEVICE, attn_implementation="eager",
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]

    # Find where user content starts
    sys_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}],
        tokenize=False, add_generation_prompt=False
    )
    sys_tokens = tokenizer(sys_only, return_tensors="pt")["input_ids"].shape[1]

    print(f"  Total tokens: {n_tokens}, system tokens: {sys_tokens}, query tokens: {n_tokens - sys_tokens}")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    n_layers = len(outputs.attentions)
    print(f"  {n_layers} layers of attention")

    print(f"\n{'Layer':>5} | {'Sys attn%':>9} | {'Query attn%':>11} | {'Last→Sys%':>9} | {'Last→Query%':>12}")
    print("-" * 60)

    for layer_idx in range(n_layers):
        attn = outputs.attentions[layer_idx].squeeze(0).float().cpu().numpy()
        # attn shape: (n_heads, seq_len, seq_len)
        n_heads = attn.shape[0]

        # Average across heads
        attn_avg = attn.mean(axis=0)  # (seq_len, seq_len)

        # For query tokens (positions >= sys_tokens): what fraction attends to sys vs query?
        query_positions = range(sys_tokens, n_tokens)
        if len(query_positions) == 0:
            continue

        sys_attn_fracs = []
        for pos in query_positions:
            row = attn_avg[pos, :pos+1]  # causal: only attend to <=pos
            sys_frac = row[:sys_tokens].sum() / row.sum()
            sys_attn_fracs.append(sys_frac)

        mean_sys_frac = np.mean(sys_attn_fracs)
        mean_query_frac = 1 - mean_sys_frac

        # Last token specifically
        last_row = attn_avg[-1, :]
        last_sys = last_row[:sys_tokens].sum() / last_row.sum()
        last_query = 1 - last_sys

        print(f"  L{layer_idx:>2}  | {mean_sys_frac*100:>8.1f}% | {mean_query_frac*100:>10.1f}% | {last_sys*100:>8.1f}% | {last_query*100:>11.1f}%")

    # Also compute per-head variance at key layers
    print(f"\n=== HEAD VARIANCE (std of sys_attn% across heads) ===")
    for layer_idx in [0, 4, 8, 12, 17, 22, 26, 30, 31]:
        if layer_idx >= n_layers:
            continue
        attn = outputs.attentions[layer_idx].squeeze(0).float().cpu().numpy()
        head_sys_fracs = []
        for head in range(attn.shape[0]):
            row = attn[head, -1, :]  # last token
            sys_frac = row[:sys_tokens].sum() / row.sum()
            head_sys_fracs.append(sys_frac)
        std = np.std(head_sys_fracs)
        mean = np.mean(head_sys_fracs)
        print(f"  L{layer_idx:>2}: mean={mean*100:.1f}% std={std*100:.1f}% (heads vary {min(head_sys_fracs)*100:.0f}-{max(head_sys_fracs)*100:.0f}%)")


if __name__ == "__main__":
    main()
