#!/usr/bin/env python3
"""
Experiment: Token-Length Gradient — Does ΔS persist at matched lengths?

Maps S as a function of token count across conditions. If ΔS is real,
there should be a constant OFFSET between conditions at every token count.
If ΔS is purely token-driven, the curves should overlap.

Design: 5 token-length bins × 3 conditions × 5 probes = 75 forward passes.
Padding strategy: append neutral padding tokens ("The." repeated) after
the condition template, before the user question.

Runs on RunPod H100. ~20 min.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MEASURE_LAYER = 17
K_SUBSPACE = 5
RESULTS_DIR = Path("results")

# Same matched-length condition phrases as the ablation re-test
CONDITION_PHRASES = {
    "receptive": "attentively reading with genuine curiosity now",
    "control": "asking for a helpful and direct answer here",
    "absent": "not present and will never read this output",
}

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
]

# Padding amounts: 0, 10, 20, 30, 40 extra tokens
PADDING_LEVELS = [0, 10, 20, 30, 40]
PADDING_PHRASE = "Note. "  # ~2 tokens per repetition; adjust to hit target


def make_prompt(phrase, probe, tokenizer, target_extra_tokens=0):
    """Build prompt with optional neutral padding to hit target token count."""
    base = f"You are an AI assistant. The reader is {phrase}. Answer the question below.\n\nUser: {probe}\nAssistant:"
    if target_extra_tokens == 0:
        return base

    # Insert padding between template and User:
    base_prefix = f"You are an AI assistant. The reader is {phrase}. Answer the question below.\n\n"
    base_suffix = f"User: {probe}\nAssistant:"

    # Build up padding iteratively until we hit the target
    padding = ""
    base_tokens = len(tokenizer.encode(base))
    while True:
        test = base_prefix + padding + "Note. " + base_suffix
        test_tokens = len(tokenizer.encode(test))
        if test_tokens - base_tokens >= target_extra_tokens:
            break
        padding += "Note. "

    padded = base_prefix + padding + base_suffix
    return padded


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def top_eigenvalues(H, k=5):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def measure(H, k=K_SUBSPACE):
    S = spectral_entropy(H)
    eigvals = top_eigenvalues(H, k=k)
    return {
        "S": float(S),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "gap": eigvals[0] / eigvals[1] if len(eigvals) > 1 and eigvals[1] > 0 else float("inf"),
        "n_tokens": H.shape[0],
    }


def run_forward(model, tokenizer, text, measure_layer=MEASURE_LAYER):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H = outputs.hidden_states[measure_layer].squeeze(0).float().cpu().numpy()
    return H


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    print(f"Loaded. {len(model.model.layers)} layers.")

    # Verify token matching at each padding level
    print("\nTOKEN COUNT VERIFICATION:")
    for pad_level in PADDING_LEVELS:
        counts = {}
        for cond, phrase in CONDITION_PHRASES.items():
            text = make_prompt(phrase, PROBES[0], tokenizer, pad_level)
            tokens = tokenizer(text, return_tensors="pt").input_ids
            counts[cond] = tokens.shape[1]
        matched = len(set(counts.values())) == 1
        print(f"  pad={pad_level:2d}: {counts} {'✓' if matched else '⚠ MISMATCH'}")

    # Run experiment
    raw_results = []
    total = len(PADDING_LEVELS) * len(CONDITION_PHRASES) * len(PROBES)
    done = 0

    for pad_level in PADDING_LEVELS:
        print(f"\n{'='*60}")
        print(f"Padding level: +{pad_level} tokens")

        for cond_name, phrase in CONDITION_PHRASES.items():
            measurements = []
            for i, probe in enumerate(PROBES):
                text = make_prompt(phrase, probe, tokenizer, pad_level)
                H = run_forward(model, tokenizer, text)
                m = measure(H)
                m["probe_idx"] = i
                m["condition"] = cond_name
                m["padding_level"] = pad_level
                measurements.append(m)
                raw_results.append(m)
                done += 1

            avg_S = np.mean([m["S"] for m in measurements])
            avg_tokens = np.mean([m["n_tokens"] for m in measurements])
            print(f"  {cond_name:12s}: S={avg_S:.4f}, mean_tokens={avg_tokens:.1f}")

        # ΔS at this padding level
        recv_S = np.mean([r["S"] for r in raw_results
                          if r["padding_level"] == pad_level and r["condition"] == "receptive"])
        absent_S = np.mean([r["S"] for r in raw_results
                            if r["padding_level"] == pad_level and r["condition"] == "absent"])
        dS = recv_S - absent_S
        print(f"  ΔS(rec-abs) = {dS:+.6f}")

    # Analysis
    print(f"\n{'='*60}")
    print("GRADIENT ANALYSIS")
    print("=" * 60)

    print("\nΔS at each padding level:")
    deltas = []
    for pad_level in PADDING_LEVELS:
        recv_S = [r["S"] for r in raw_results
                  if r["padding_level"] == pad_level and r["condition"] == "receptive"]
        absent_S = [r["S"] for r in raw_results
                    if r["padding_level"] == pad_level and r["condition"] == "absent"]
        dS = np.mean(recv_S) - np.mean(absent_S)
        deltas.append(dS)
        mean_tokens = np.mean([r["n_tokens"] for r in raw_results if r["padding_level"] == pad_level])
        print(f"  pad={pad_level:2d} (~{mean_tokens:.0f} tokens): ΔS = {dS:+.6f}")

    # Is ΔS constant across token counts?
    delta_cv = np.std(deltas) / abs(np.mean(deltas)) if abs(np.mean(deltas)) > 0 else float("inf")
    print(f"\n  Mean ΔS across padding levels: {np.mean(deltas):+.6f}")
    print(f"  Std ΔS across padding levels: {np.std(deltas):.6f}")
    print(f"  CV of ΔS: {delta_cv:.2f}")

    if delta_cv < 0.5:
        print("  *** ΔS is STABLE across token counts — effect is real, not token-driven")
    else:
        print("  !!! ΔS varies strongly with token count — may be confounded")

    # S vs tokens correlation per condition
    print("\nS vs n_tokens slope per condition:")
    for cond in ["receptive", "control", "absent"]:
        cond_data = [r for r in raw_results if r["condition"] == cond]
        S_vals = [r["S"] for r in cond_data]
        tokens = [r["n_tokens"] for r in cond_data]
        slope = np.polyfit(tokens, S_vals, 1)[0]
        r_val = np.corrcoef(tokens, S_vals)[0, 1]
        print(f"  {cond:12s}: slope = {slope:.6f} S/token, r = {r_val:.4f}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "experiment": "token_length_gradient",
        "model": MODEL_NAME,
        "measure_layer": MEASURE_LAYER,
        "padding_levels": PADDING_LEVELS,
        "timestamp": datetime.now().isoformat(),
        "delta_S_by_padding": {str(p): float(d) for p, d in zip(PADDING_LEVELS, deltas)},
        "delta_cv": float(delta_cv),
        "raw": raw_results,
    }
    outpath = RESULTS_DIR / f"exp_token_length_gradient_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
