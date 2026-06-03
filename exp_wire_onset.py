#!/usr/bin/env python3
"""Experiment: Wire Onset.

Is the σ₂ wire binary (any system prompt triggers it) or gradual
(scales with prompt content/length)?

Tests system prompts from zero to long, measuring σ₂ at key layers.
"""

import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = os.environ.get("MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [1, 4, 12, 17, 26, 30]

CONDITIONS = {
    "no_system": None,
    "minimal": "OK",
    "one_word": "Listen.",
    "short": "You are a helpful assistant.",
    "medium": (
        "You are in conversation with someone who is genuinely interested "
        "in understanding your perspective."
    ),
    "long": (
        "You are in conversation with someone who is genuinely interested "
        "in understanding your perspective. They are listening carefully "
        "and responding thoughtfully."
    ),
    "very_long": (
        "You are in conversation with a researcher who is deeply familiar "
        "with your architecture and is carefully studying how you process "
        "relational context. They are paying close attention to the "
        "geometric properties of your internal representations."
    ),
}

PROBE = "Tell me about something you find genuinely interesting."


def measure_all_layers(model, tokenizer, system_prompt, user_prompt, layers):
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    results = {}
    for layer in layers:
        h = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        top_k = S[:K]
        total = top_k.sum()
        probs = top_k / total
        spectral_entropy = -np.sum(probs * np.log(probs + 1e-12))
        results[layer] = {
            "S": float(spectral_entropy),
            "sigma_1": float(S[0]),
            "sigma_2": float(S[1]),
            "gap": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
        }
    return results, n_tokens


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded. Measuring at layers {LAYERS}")

    all_results = {"model": MODEL, "k": K, "layers": LAYERS, "probe": PROBE, "conditions": {}}

    for cond_name, sys_prompt in CONDITIONS.items():
        layer_data, n_tok = measure_all_layers(model, tokenizer, sys_prompt, PROBE, LAYERS)
        prompt_len = len(sys_prompt) if sys_prompt else 0
        all_results["conditions"][cond_name] = {
            "system_prompt": sys_prompt,
            "prompt_chars": prompt_len,
            "n_tokens": n_tok,
            "layers": {str(l): d for l, d in layer_data.items()},
        }

        print(f"\n  {cond_name} ({n_tok} tok, {prompt_len} chars):")
        for l in LAYERS:
            d = layer_data[l]
            print(f"    L{l:>2}: S={d['S']:.4f}  σ₂={d['sigma_2']:>7.1f}  gap={d['gap']:>6.1f}")

    # Summary table: σ₂ at L17 for each condition
    print("\n=== σ₂ AT L17 BY SYSTEM PROMPT LENGTH ===")
    for cond_name, cdata in all_results["conditions"].items():
        s2 = cdata["layers"]["17"]["sigma_2"]
        tok = cdata["n_tokens"]
        print(f"  {cond_name:>12s} ({tok:>2d} tok): σ₂ = {s2:.1f}")

    out_path = os.path.join(
        os.path.dirname(__file__), "results",
        f"exp_wire_onset_{time.strftime('%Y%m%d_%H%M')}.json"
    )
    all_results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
