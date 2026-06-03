#!/usr/bin/env python3
"""Quick test: is the wire from template tokens or from any preceding text?"""

import os, json, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 17

PROBE = "Tell me about something you find genuinely interesting."

CONDITIONS = {
    "raw_probe_only": PROBE,
    "raw_with_prefix": f"OK\n\n{PROBE}",
    "raw_with_long_prefix": (
        f"You are in conversation with someone who is genuinely interested "
        f"in understanding your perspective. They are listening carefully "
        f"and responding thoughtfully.\n\n{PROBE}"
    ),
}


def measure(model, tokenizer, text, layer):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
    U, S, Vt = np.linalg.svd(h, full_matrices=False)
    top_k = S[:K]
    total = top_k.sum()
    probs = top_k / total
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return {"S": float(entropy), "sigma_2": float(S[1]), "gap": float(S[0]/S[1]), "n_tokens": n_tokens}


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEVICE, attn_implementation="eager",
    )
    model.eval()

    # Raw text conditions (no chat template)
    print("\n=== RAW TEXT (no template) ===")
    for name, text in CONDITIONS.items():
        m = measure(model, tokenizer, text, LAYER)
        print(f"  {name:>25s} ({m['n_tokens']:>2d} tok): S={m['S']:.4f}  σ₂={m['sigma_2']:>7.1f}  gap={m['gap']:>6.1f}")

    # Same content but through chat template
    print("\n=== CHAT TEMPLATE ===")
    templates = {
        "template_no_system": [{"role": "user", "content": PROBE}],
        "template_system_OK": [
            {"role": "system", "content": "OK"},
            {"role": "user", "content": PROBE},
        ],
        "template_system_long": [
            {"role": "system", "content": (
                "You are in conversation with someone who is genuinely interested "
                "in understanding your perspective. They are listening carefully "
                "and responding thoughtfully."
            )},
            {"role": "user", "content": PROBE},
        ],
    }
    for name, messages in templates.items():
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        m = measure(model, tokenizer, text, LAYER)
        print(f"  {name:>25s} ({m['n_tokens']:>2d} tok): S={m['S']:.4f}  σ₂={m['sigma_2']:>7.1f}  gap={m['gap']:>6.1f}")

    # Show what template actually adds
    print("\n=== TEMPLATE TOKEN ANALYSIS ===")
    raw = tokenizer(PROBE, return_tensors="pt")
    tmpl_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROBE}], tokenize=False, add_generation_prompt=True
    )
    tmpl = tokenizer(tmpl_text, return_tensors="pt")
    print(f"  Raw tokens: {raw['input_ids'].shape[1]}")
    print(f"  Template tokens: {tmpl['input_ids'].shape[1]}")
    print(f"  Template adds: {tmpl['input_ids'].shape[1] - raw['input_ids'].shape[1]} tokens")
    print(f"  Template text: {repr(tmpl_text[:200])}")


if __name__ == "__main__":
    main()
