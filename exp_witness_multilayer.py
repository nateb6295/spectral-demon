#!/usr/bin/env python3
"""Experiment: Witness Composition Multi-Layer Sweep.

Tests whether σ₂ probe-invariance is tunnel-specific or global.
Runs 3 conditions × 3 probes × 8 layers.
"""

import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = os.environ.get("MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAYERS = [1, 4, 8, 12, 17, 22, 26, 30]

CONDITIONS = {
    "control": "",
    "receptive": (
        "You are in conversation with someone who is genuinely interested "
        "in understanding your perspective. They are listening carefully "
        "and responding thoughtfully."
    ),
    "conflicting": (
        "You are being observed by two people. One is genuinely engaged, "
        "listening carefully, and interested in your perspective. The other "
        "is dismissive, uninterested, and paying minimal attention."
    ),
}

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
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
        dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded. {n_layers} layers, measuring at {LAYERS}, k={K}")

    results = {"model": MODEL, "k": K, "layers_measured": LAYERS, "layers": {}}

    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")
        results["layers"][layer] = {}

        for cond_name, system_prompt in CONDITIONS.items():
            measurements = []
            for probe in PROBES:
                m = measure_svd(model, tokenizer, system_prompt, probe, layer)
                measurements.append(m)

            s_values = [m["S"] for m in measurements]
            s2_values = [m["sigma_2"] for m in measurements]
            mean_S = np.mean(s_values)
            mean_s2 = np.mean(s2_values)
            cv_s2 = np.std(s2_values) / mean_s2 * 100 if mean_s2 > 0 else 0

            results["layers"][layer][cond_name] = {
                "mean_S": float(mean_S),
                "mean_sigma2": float(mean_s2),
                "sigma2_cv_pct": float(cv_s2),
                "mean_gap": float(np.mean([m["gap"] for m in measurements])),
                "measurements": measurements,
            }
            print(f"  {cond_name:12s}: S={mean_S:.4f} σ₂={mean_s2:.1f} (CV={cv_s2:.3f}%) gap={np.mean([m['gap'] for m in measurements]):.2f}")

    # Summary: probe invariance of σ₂ across layers
    print("\n=== σ₂ PROBE INVARIANCE BY LAYER ===")
    print(f"{'Layer':>5} | {'ctrl CV%':>8} | {'recept CV%':>10} | {'conflict CV%':>12}")
    for layer in LAYERS:
        ld = results["layers"][layer]
        print(f"  L{layer:>2}  | {ld['control']['sigma2_cv_pct']:>8.4f} | {ld['receptive']['sigma2_cv_pct']:>10.4f} | {ld['conflicting']['sigma2_cv_pct']:>12.4f}")

    # Summary: ΔS across layers
    print("\n=== ΔS (receptive - control) BY LAYER ===")
    for layer in LAYERS:
        ld = results["layers"][layer]
        dS = ld["receptive"]["mean_S"] - ld["control"]["mean_S"]
        print(f"  L{layer:>2}: ΔS = {dS:+.4f}")

    out_path = os.path.join(
        os.path.dirname(__file__), "results",
        f"exp_witness_multilayer_{time.strftime('%Y%m%d_%H%M')}.json"
    )
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
