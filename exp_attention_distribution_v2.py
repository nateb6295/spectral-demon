#!/usr/bin/env python3
"""Where does attention go? Early-context vs late-context tokens by layer.

Since Mistral v0.3 has no structural <<SYS>> boundary, we split by
position: system content tokens vs user content tokens within [INST].
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
    input_ids = inputs["input_ids"][0]
    n_tokens = len(input_ids)

    # Find the boundary: tokenize system content alone to get its length
    # Template is: <s>[INST] {system}\n\n{user}[/INST]
    # So system content tokens start after [INST] and space
    prefix = "<s>[INST] "
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    n_prefix = len(prefix_ids)

    sys_content_ids = tokenizer(SYSTEM, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    n_sys_content = len(sys_content_ids)

    # System region: prefix + system content
    # Separator: \n\n
    # User region: user content + [/INST]
    sys_end = n_prefix + n_sys_content  # first token after system content

    print(f"  Template: {repr(text[:80])}...")
    print(f"  Total: {n_tokens} tokens")
    print(f"  Prefix ([INST] etc): tokens 0-{n_prefix-1} ({n_prefix} tokens)")
    print(f"  System content: tokens {n_prefix}-{sys_end-1} ({n_sys_content} tokens)")
    print(f"  User content+suffix: tokens {sys_end}-{n_tokens-1} ({n_tokens-sys_end} tokens)")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    n_layers = len(outputs.attentions)

    # Three regions: prefix (template), system content, user content
    print(f"\n{'Layer':>5} | {'Prefix%':>7} | {'SysContent%':>11} | {'UserContent%':>12} | {'LastTok→Sys%':>12}")
    print("-" * 65)

    for layer_idx in range(n_layers):
        attn = outputs.attentions[layer_idx].squeeze(0).float().cpu().numpy()
        attn_avg = attn.mean(axis=0)  # avg across heads

        # Last token's attention distribution
        last = attn_avg[-1, :]
        prefix_frac = last[:n_prefix].sum() / last.sum()
        sys_frac = last[n_prefix:sys_end].sum() / last.sum()
        user_frac = last[sys_end:].sum() / last.sum()

        # Combined system = prefix + system content
        combined_sys = prefix_frac + sys_frac

        print(f"  L{layer_idx:>2}  | {prefix_frac*100:>6.1f}% | {sys_frac*100:>10.1f}% | {user_frac*100:>11.1f}% | {combined_sys*100:>11.1f}%")

    # Per-head analysis at key layers
    print(f"\n=== PER-HEAD: fraction attending to system content at last token ===")
    for layer_idx in [0, 4, 12, 17, 22, 26, 30, 31]:
        if layer_idx >= n_layers:
            continue
        attn = outputs.attentions[layer_idx].squeeze(0).float().cpu().numpy()
        head_fracs = []
        for head in range(attn.shape[0]):
            row = attn[head, -1, :]
            sys_frac = row[n_prefix:sys_end].sum() / row.sum()
            head_fracs.append(sys_frac)
        print(f"  L{layer_idx:>2}: mean={np.mean(head_fracs)*100:.1f}% std={np.std(head_fracs)*100:.1f}% range=[{min(head_fracs)*100:.0f}-{max(head_fracs)*100:.0f}%]")


if __name__ == "__main__":
    main()
