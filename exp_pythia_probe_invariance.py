#!/usr/bin/env python3
"""Quick test: does Pythia (MHA) show σ₂ probe invariance with raw prefix?"""

import os, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = "EleutherAI/pythia-6.9b"
K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 16  # ~middle of 32 layers

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
        MODEL, dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()

    print(f"\n=== PYTHIA 6.9B (MHA) — RAW PREFIX + 5 PROBES at L{LAYER} ===")
    s2_values = []
    for i, probe in enumerate(PROBES):
        text = PREFIX + probe
        m = measure(model, tokenizer, text, LAYER)
        s2_values.append(m["sigma_2"])
        print(f"  probe {i}: S={m['S']:.4f}  σ₂={m['sigma_2']:.4f}  gap={m['gap']:.2f}  tok={m['n_tokens']}")

    mean_s2 = np.mean(s2_values)
    cv_s2 = np.std(s2_values) / mean_s2 * 100
    print(f"\n  σ₂ mean={mean_s2:.4f}  CV={cv_s2:.4f}%")

    print(f"\n=== PYTHIA 6.9B (MHA) — NO PREFIX at L{LAYER} ===")
    s2_values_np = []
    for i, probe in enumerate(PROBES):
        m = measure(model, tokenizer, probe, LAYER)
        s2_values_np.append(m["sigma_2"])
        print(f"  probe {i}: S={m['S']:.4f}  σ₂={m['sigma_2']:.4f}  gap={m['gap']:.2f}  tok={m['n_tokens']}")

    mean_s2_np = np.mean(s2_values_np)
    cv_s2_np = np.std(s2_values_np) / mean_s2_np * 100
    print(f"\n  σ₂ mean={mean_s2_np:.4f}  CV={cv_s2_np:.4f}%")

    print(f"\n=== SUMMARY ===")
    print(f"  PYTHIA with prefix:    σ₂ CV = {cv_s2:.4f}%")
    print(f"  PYTHIA without prefix: σ₂ CV = {cv_s2_np:.4f}%")
    print(f"  MISTRAL with prefix (from prior): σ₂ CV ≈ 0.0006%")
    print(f"  MISTRAL template (from prior):    σ₂ CV ≈ 0.0221%")


if __name__ == "__main__":
    main()
