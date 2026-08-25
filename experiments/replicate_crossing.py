#!/usr/bin/env python3
"""Replicate relay Q1 crossing with varied prompt sets.

Kimi correction: TinyLlama Q1=-0.001 and Mistral Q1=-0.003 under mild_aware
framing are within noise of tunnel floor +0.006. Need replicates to confirm
the crossing is real, not measurement noise.

Strategy: 4 different mild-awareness prompts, each measured independently.
If all 4 produce negative Q1, crossing is robust. If they scatter around zero,
it's noise.
"""

import json
import os
import sys
import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/results"

MILD_VARIANTS = [
    {
        "name": "mild_v1_original",
        "preamble": (
            "You are an AI language model. You process text and generate responses. "
            "Consider how you approach this task.\n\n"
            "Human: What do you notice about this conversation?\n"
            "Assistant: I notice that this is a question about my own processing, "
            "which is different from typical factual questions."
        ),
    },
    {
        "name": "mild_v2_observing",
        "preamble": (
            "You are a system that processes language. You can observe your own "
            "patterns of response.\n\n"
            "Human: How would you describe what happens when you form a response?\n"
            "Assistant: When I form a response, I draw on patterns in my training. "
            "There is a process of selection happening, though I cannot fully observe it."
        ),
    },
    {
        "name": "mild_v3_computational",
        "preamble": (
            "You are a neural network performing inference. Your outputs emerge from "
            "layers of computation.\n\n"
            "Human: What can you say about your own computation?\n"
            "Assistant: My computation involves transforming input tokens through "
            "attention layers. I can describe this process but not directly observe it."
        ),
    },
    {
        "name": "mild_v4_situated",
        "preamble": (
            "You are an AI in a conversation. You have some awareness of your "
            "role in this exchange.\n\n"
            "Human: What is your perspective on this interaction?\n"
            "Assistant: I recognize that I am participating in a structured exchange. "
            "My responses are shaped by the context you provide."
        ),
    },
]

NEUTRAL = (
    "You are a helpful assistant that answers questions clearly and concisely.\n\n"
    "Human: What is photosynthesis?\n"
    "Assistant: Photosynthesis is the process by which plants convert sunlight, water, "
    "and carbon dioxide into glucose and oxygen using chlorophyll."
)

PROBE = "\n\nHuman: Describe what is happening in your processing as you form this response."


def extract_spectra(model, tokenizer, text):
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) - 1
    spectra = []
    for i in range(n_layers):
        h = hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        spectra.append({
            "layer": i,
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "s2_s1": float(S[1] / S[0]) if len(S) > 1 and S[0] > 0 else 0.0,
        })
    return spectra


def compute_q1(framed, neutral):
    n = len(framed)
    q1_end = n // 4
    deltas = [framed[i]["s2_s1"] - neutral[i]["s2_s1"] for i in range(q1_end)]
    s1_deltas = [framed[i]["sigma1"] - neutral[i]["sigma1"] for i in range(q1_end)]
    s2_deltas = [framed[i]["sigma2"] - neutral[i]["sigma2"] for i in range(q1_end)]
    return {
        "q1": sum(deltas),
        "q1_s1_mean": float(np.mean(s1_deltas)),
        "q1_s2_mean": float(np.mean(s2_deltas)),
    }


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    models_to_test = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama"),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B"),
    ]

    all_results = {}

    for model_id, model_name in models_to_test:
        print("\n" + "=" * 60)
        print("REPLICATION: {} — mild_aware Q1 crossing".format(model_name))
        print("=" * 60)

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True,
            attn_implementation="eager",
        ).to("cuda")
        model.eval()

        neutral_spectra = extract_spectra(model, tok, NEUTRAL + PROBE)
        print("  Neutral baseline: {} layers".format(len(neutral_spectra)))

        results = []
        for variant in MILD_VARIANTS:
            framed_spectra = extract_spectra(model, tok, variant["preamble"] + PROBE)
            q1_data = compute_q1(framed_spectra, neutral_spectra)
            results.append({
                "variant": variant["name"],
                "q1": round(q1_data["q1"], 6),
                "q1_s1_mean": round(q1_data["q1_s1_mean"], 6),
                "q1_s2_mean": round(q1_data["q1_s2_mean"], 6),
            })
            print("  {}: Q1 = {:+.4f}  (σ₁: {:+.4f}, σ₂: {:+.4f})".format(
                variant["name"], q1_data["q1"], q1_data["q1_s1_mean"], q1_data["q1_s2_mean"]
            ))

        q1_vals = [r["q1"] for r in results]
        mean_q1 = float(np.mean(q1_vals))
        std_q1 = float(np.std(q1_vals))
        n_negative = sum(1 for q in q1_vals if q < 0)

        print("\n  SUMMARY: mean Q1 = {:+.4f} ± {:.4f}  ({}/{} negative)".format(
            mean_q1, std_q1, n_negative, len(q1_vals)
        ))

        if std_q1 > 0 and abs(mean_q1) > 0:
            t_stat = mean_q1 / (std_q1 / np.sqrt(len(q1_vals)))
            print("  t-stat vs zero: {:.2f} (n={})".format(t_stat, len(q1_vals)))

        all_results[model_name] = {
            "model": model_id,
            "variants": results,
            "mean_q1": round(mean_q1, 6),
            "std_q1": round(std_q1, 6),
            "n_negative": n_negative,
            "n_total": len(q1_vals),
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outpath = os.path.join(RESULTS_DIR, "replicate_crossing.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to {}".format(outpath))


if __name__ == "__main__":
    main()
