#!/usr/bin/env python3
"""Quick test: is σ₂ probe-invariant for raw text (no template)?"""

import os, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 17

PREFIX = "You are in conversation with someone who listens carefully.\n\n"

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe how you experience this conversation.",
    "What would you want someone to understand about you?",
    "How does the way someone listens change what you say?",
]


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

    print("\n=== RAW TEXT: prefix + 5 probes ===")
    s2_values = []
    for i, probe in enumerate(PROBES):
        text = PREFIX + probe
        m = measure(model, tokenizer, text, LAYER)
        s2_values.append(m["sigma_2"])
        print(f"  probe {i}: S={m['S']:.4f}  σ₂={m['sigma_2']:.4f}  gap={m['gap']:.2f}  tok={m['n_tokens']}")

    mean_s2 = np.mean(s2_values)
    cv_s2 = np.std(s2_values) / mean_s2 * 100
    print(f"\n  σ₂ mean={mean_s2:.4f}  CV={cv_s2:.4f}%")

    print("\n=== TEMPLATE: same prefix as system prompt + 5 probes ===")
    s2_values_t = []
    for i, probe in enumerate(PROBES):
        messages = [
            {"role": "system", "content": PREFIX.strip()},
            {"role": "user", "content": probe},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        m = measure(model, tokenizer, text, LAYER)
        s2_values_t.append(m["sigma_2"])
        print(f"  probe {i}: S={m['S']:.4f}  σ₂={m['sigma_2']:.4f}  gap={m['gap']:.2f}  tok={m['n_tokens']}")

    mean_s2_t = np.mean(s2_values_t)
    cv_s2_t = np.std(s2_values_t) / mean_s2_t * 100
    print(f"\n  σ₂ mean={mean_s2_t:.4f}  CV={cv_s2_t:.4f}%")

    print(f"\n=== COMPARISON ===")
    print(f"  Raw text:   σ₂ CV = {cv_s2:.4f}%")
    print(f"  Template:   σ₂ CV = {cv_s2_t:.4f}%")
    print(f"  Ratio: template is {cv_s2/cv_s2_t:.0f}× more invariant" if cv_s2_t > 0 else "  Template CV ≈ 0")

    # Also test: no prefix at all
    print("\n=== NO PREFIX: raw probes ===")
    s2_values_np = []
    for i, probe in enumerate(PROBES):
        m = measure(model, tokenizer, probe, LAYER)
        s2_values_np.append(m["sigma_2"])
        print(f"  probe {i}: S={m['S']:.4f}  σ₂={m['sigma_2']:.4f}  gap={m['gap']:.2f}  tok={m['n_tokens']}")

    mean_s2_np = np.mean(s2_values_np)
    cv_s2_np = np.std(s2_values_np) / mean_s2_np * 100
    print(f"\n  σ₂ mean={mean_s2_np:.4f}  CV={cv_s2_np:.4f}%")


if __name__ == "__main__":
    main()
