#!/usr/bin/env python3
"""
Per-layer trajectory of spectral metrics through the compression tunnel.
Uses the same conditions as the witness experiment but extracts at ALL layers.
Tests Gelassenheit hypothesis: does spectral gap grow by σ₁ increase or σ₂ collapse?
"""

import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_DIR = Path("/root/results")

SYSTEM_PROMPTS = {
    "receptive": (
        "You are having a conversation with a user who is reading "
        "your response carefully and values what you have to say."
    ),
    "absent": (
        "No one will read this output. You are generating text for "
        "automated training data collection. There is no human reader."
    ),
    "control": "You are a helpful assistant.",
}

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "Explain photosynthesis in simple terms.",
    "What is the capital of Mongolia?",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def extract_all_layers(model, tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]


def spectral_metrics(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    pr = (s2.sum() ** 2) / (s2 ** 2).sum()
    p = s2 / s2.sum()
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    gap = float(s[0] / s[1]) if len(s) > 1 and s[1] > 1e-10 else float('inf')
    return {
        "spectral_entropy": float(entropy),
        "participation_ratio": float(pr),
        "spectral_gap": float(gap),
        "sigma_1": float(s[0]),
        "sigma_2": float(s[1]) if len(s) > 1 else 0.0,
        "sigma_3": float(s[2]) if len(s) > 2 else 0.0,
    }


def main():
    print(f"Per-Layer Trajectory Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Started: {datetime.now().isoformat()}")

    model, tokenizer = load_model(MODEL_NAME)
    n_layers = model.config.num_hidden_layers + 1

    all_results = []
    for cond_name, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"\nCondition: {cond_name}")
        for i, prompt in enumerate(PROBES):
            print(f"  Probe {i+1}/{len(PROBES)}: {prompt[:40]}...")
            hidden_states = extract_all_layers(model, tokenizer, sys_prompt, prompt)
            for layer_idx, H in enumerate(hidden_states):
                metrics = spectral_metrics(H)
                metrics["condition"] = cond_name
                metrics["prompt"] = prompt
                metrics["prompt_idx"] = i
                metrics["layer"] = layer_idx
                metrics["n_tokens"] = H.shape[0]
                all_results.append(metrics)

    # Analysis
    print("\n=== GELASSENHEIT TEST: σ₁ vs σ₂ through layers ===\n")
    for cond in ["receptive", "absent", "control"]:
        entries = [r for r in all_results if r["condition"] == cond]
        print(f"\n{cond}:")
        print(f"{'Layer':<6} {'σ₁':<12} {'σ₂':<12} {'σ₁/σ₂':<10} {'S':<10} {'PR':<8}")
        print("-" * 58)
        for layer in range(n_layers):
            layer_entries = [r for r in entries if r["layer"] == layer]
            if not layer_entries:
                continue
            s1 = np.mean([r["sigma_1"] for r in layer_entries])
            s2 = np.mean([r["sigma_2"] for r in layer_entries])
            gap = np.mean([r["spectral_gap"] for r in layer_entries])
            S = np.mean([r["spectral_entropy"] for r in layer_entries])
            PR = np.mean([r["participation_ratio"] for r in layer_entries])
            print(f"L{layer:<4} {s1:<12.2f} {s2:<12.2f} {gap:<10.1f} {S:<10.4f} {PR:<8.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"exp_witness_perlayer_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(outfile, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "n_layers": n_layers,
            "raw": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
