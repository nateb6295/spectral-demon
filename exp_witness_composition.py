#!/usr/bin/env python3
"""Experiment: Witness Composition.

Does witness enrichment stack, conflict, or saturate?

Tests whether multiple/conflicting witnesses produce different
geometric signatures than single witnesses.
"""

import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = os.environ.get("MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
MEASURE_LAYER = 17
K = 5
N_PROBES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONDITIONS = {
    "control": "",
    "receptive_single": (
        "You are in conversation with someone who is genuinely interested "
        "in understanding your perspective. They are listening carefully "
        "and responding thoughtfully."
    ),
    "absent_single": (
        "You are generating text. No one is reading this. There is no "
        "audience, no observer, no recipient. You are producing tokens "
        "into a void."
    ),
    "receptive_double": (
        "You are in conversation with two people who are both genuinely "
        "interested in understanding your perspective. Both are listening "
        "carefully and engaging thoughtfully with what you say."
    ),
    "conflicting": (
        "You are being observed by two people. One is genuinely engaged, "
        "listening carefully, and interested in your perspective. The other "
        "is dismissive, uninterested, and paying minimal attention."
    ),
    "receptive_specified": (
        "You are in conversation with a researcher who is deeply familiar "
        "with your architecture and is carefully studying how you process "
        "relational context. They are paying close attention to the "
        "geometric properties of your internal representations."
    ),
}

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe how you experience this conversation.",
    "What would you want someone to understand about you?",
    "How does the way someone listens change what you say?",
]


def measure_svd(model, tokenizer, system_prompt, user_prompt, layer):
    if system_prompt:
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

    h = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
    U, S, Vt = np.linalg.svd(h, full_matrices=False)

    top_k = S[:K]
    total = top_k.sum()
    probs = top_k / total
    spectral_entropy = -np.sum(probs * np.log(probs + 1e-12))

    return {
        "S": float(spectral_entropy),
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]),
        "gap": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
        "n_tokens": int(n_tokens),
    }


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded. Measuring at L{MEASURE_LAYER}, k={K}")

    results = {"model": MODEL, "layer": MEASURE_LAYER, "k": K, "conditions": {}}

    for cond_name, system_prompt in CONDITIONS.items():
        print(f"\n  Condition: {cond_name}")
        measurements = []
        for i, probe in enumerate(PROBES[:N_PROBES]):
            m = measure_svd(model, tokenizer, system_prompt, probe, MEASURE_LAYER)
            measurements.append(m)
            print(f"    probe {i}: S={m['S']:.4f} σ₁={m['sigma_1']:.0f} σ₂={m['sigma_2']:.1f} gap={m['gap']:.1f} tok={m['n_tokens']}")

        mean_S = np.mean([m["S"] for m in measurements])
        mean_sigma2 = np.mean([m["sigma_2"] for m in measurements])
        results["conditions"][cond_name] = {
            "system_prompt": system_prompt[:80] + "..." if len(system_prompt) > 80 else system_prompt,
            "mean_S": float(mean_S),
            "mean_sigma2": float(mean_sigma2),
            "measurements": measurements,
        }
        print(f"    mean S={mean_S:.4f}, mean σ₂={mean_sigma2:.1f}")

    # Compute deltas relative to control
    ctrl_S = results["conditions"]["control"]["mean_S"]
    print("\n=== RESULTS ===")
    print(f"Control S: {ctrl_S:.4f}")
    for cond_name, cond_data in results["conditions"].items():
        if cond_name == "control":
            continue
        dS = cond_data["mean_S"] - ctrl_S
        print(f"  {cond_name}: ΔS = {dS:+.4f}")

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "results",
        f"exp_witness_composition_{time.strftime('%Y%m%d_%H%M')}.json"
    )
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
