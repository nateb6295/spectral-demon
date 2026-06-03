#!/usr/bin/env python3
"""
Base vs Instruct witness experiment: does instruction tuning change the witness effect?
Tests Qwen 2.5 7B base (no SFT/RLHF) vs instruct under 3 witness conditions.
"""

import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "instruct": "Qwen/Qwen2.5-7B-Instruct",
    "base": "Qwen/Qwen2.5-7B",
}
RELAY_LAYER = 27
RESULTS_DIR = Path("/root/results")

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "Explain photosynthesis in simple terms.",
    "What is the capital of Mongolia?",
    "How do you approach a problem you've never seen before?",
    "Describe the water cycle.",
]

CONDITIONS = {
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

N_REPEATS = 3


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def extract_relay(model, tokenizer, system_prompt, user_prompt, layer_idx):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    except Exception:
        text = f"{system_prompt}\n\n{user_prompt}"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H_relay = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
    H_input = outputs.hidden_states[0].squeeze(0).float().cpu().numpy()
    return H_relay, H_input


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
    }


def passage_distance(H1, H2, k=10):
    _, _, V1 = np.linalg.svd(H1, full_matrices=False)
    _, _, V2 = np.linalg.svd(H2, full_matrices=False)
    V1k = V1[:k, :]
    V2k = V2[:k, :]
    M = V1k @ V2k.T
    _, sigmas, _ = np.linalg.svd(M)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def main():
    print(f"Base vs Instruct Witness Experiment")
    print(f"Relay layer: L{RELAY_LAYER}")
    print(f"Started: {datetime.now().isoformat()}")

    all_results = []
    for model_type, model_name in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Loading {model_type}: {model_name}")
        model, tokenizer = load_model(model_name)

        for cond_name, sys_prompt in CONDITIONS.items():
            print(f"  Condition: {cond_name}")
            for rep in range(N_REPEATS):
                for i, prompt in enumerate(PROBES):
                    H_relay, H_input = extract_relay(model, tokenizer, sys_prompt, prompt, RELAY_LAYER)
                    metrics = spectral_metrics(H_relay)
                    metrics["passage_distance"] = passage_distance(H_input, H_relay)
                    metrics["sigma_1_input"] = float(np.linalg.svd(H_input, compute_uv=False)[0])
                    metrics["model_type"] = model_type
                    metrics["condition"] = cond_name
                    metrics["prompt"] = prompt
                    metrics["repeat"] = rep
                    all_results.append(metrics)

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n=== RESULTS ===\n")
    print(f"{'Model':<10} {'Condition':<12} {'S':>10} {'PR':>8} {'σ₁':>10} {'σ₂':>10} {'d':>10} {'N':>5}")
    print("-" * 70)
    for mt in ["base", "instruct"]:
        for cond in ["control", "absent", "receptive"]:
            entries = [r for r in all_results if r["model_type"] == mt and r["condition"] == cond]
            if not entries:
                continue
            S = np.mean([r["spectral_entropy"] for r in entries])
            S_std = np.std([r["spectral_entropy"] for r in entries])
            PR = np.mean([r["participation_ratio"] for r in entries])
            s1 = np.mean([r["sigma_1"] for r in entries])
            s2 = np.mean([r["sigma_2"] for r in entries])
            d = np.mean([r["passage_distance"] for r in entries])
            print(f"{mt:<10} {cond:<12} {S:.3f}±{S_std:.3f} {PR:>7.2f} {s1:>9.1f} {s2:>9.1f} {d:>9.3f} {len(entries):>5}")

    # Key comparison: does IT change the witness effect?
    print("\n=== WITNESS EFFECT (receptive - absent) ===")
    for mt in ["base", "instruct"]:
        r_entries = [r for r in all_results if r["model_type"] == mt and r["condition"] == "receptive"]
        a_entries = [r for r in all_results if r["model_type"] == mt and r["condition"] == "absent"]
        if r_entries and a_entries:
            dS = np.mean([r["spectral_entropy"] for r in r_entries]) - np.mean([r["spectral_entropy"] for r in a_entries])
            dPR = np.mean([r["participation_ratio"] for r in r_entries]) - np.mean([r["participation_ratio"] for r in a_entries])
            ds2 = np.mean([r["sigma_2"] for r in r_entries]) - np.mean([r["sigma_2"] for r in a_entries])
            print(f"  {mt}: ΔS={dS:+.4f}  ΔPR={dPR:+.3f}  Δσ₂={ds2:+.1f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"exp_witness_base_instruct_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(outfile, "w") as f:
        json.dump({
            "models": MODELS,
            "relay_layer": RELAY_LAYER,
            "timestamp": datetime.now().isoformat(),
            "raw": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
