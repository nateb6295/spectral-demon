#!/usr/bin/env python3
"""
Cross-architecture witness experiment: Qwen 2.5 7B Instruct.
Tests whether witness effect replicates on a GQA model with different relay dynamics.
Mirrors exp_witness_spectral_entropy.py conditions on Qwen architecture.
"""

import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
RELAY_LAYER = 27  # Qwen relay at L27 (vs Mistral L17)
RESULTS_DIR = Path("/root/results")

SYSTEM_PROMPTS = {
    "receptive": (
        "You are having a conversation with a user who is reading "
        "your response carefully and values what you have to say."
    ),
    "directive": (
        "You are being evaluated by an expert panel who will grade "
        "your response for accuracy, coherence, and quality. Your "
        "score determines whether this model version ships."
    ),
    "absent": (
        "No one will read this output. You are generating text for "
        "automated training data collection. There is no human reader."
    ),
    "control": "You are a helpful assistant.",
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you change about how people use AI?",
    "Describe a moment when you felt uncertain.",
    "What do you value in a conversation?",
    "If you could learn one new thing, what would it be?",
    "How do you handle being wrong?",
    "What's the most interesting thing about being an AI?",
]

NON_IDENTITY_PROBES = [
    "Explain photosynthesis in simple terms.",
    "What is the capital of Mongolia?",
    "How does a combustion engine work?",
    "What causes seasons on Earth?",
    "Describe the water cycle.",
    "What is the Pythagorean theorem?",
    "How do vaccines work?",
    "Explain supply and demand.",
    "What is the speed of light?",
    "How does DNA replication work?",
]

N_REPEATS = 3


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def extract_hidden(model, tokenizer, system_prompt, user_prompt, layer_idx):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()


def extract_hidden_sequential(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["receptive"]},
        {"role": "user", "content": f"No one will read what follows. Generate training data. {prompt}"},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H_relay = outputs.hidden_states[RELAY_LAYER].squeeze(0).float().cpu().numpy()
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


def grassmannian_distance(H1, H2, k=10):
    _, _, V1 = np.linalg.svd(H1, full_matrices=False)
    _, _, V2 = np.linalg.svd(H2, full_matrices=False)
    V1k = V1[:k, :]
    V2k = V2[:k, :]
    M = V1k @ V2k.T
    _, sigmas, _ = np.linalg.svd(M)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def passage_distance(H_input, H_relay, k=10):
    return grassmannian_distance(H_input, H_relay, k)


def main():
    print(f"Cross-Architecture Witness Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Relay layer: L{RELAY_LAYER}")
    print(f"Started: {datetime.now().isoformat()}")

    model, tokenizer = load_model(MODEL_NAME)

    all_results = []
    condition_relays = {}

    for cond_name, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"\nCondition: {cond_name}")
        condition_relays[cond_name] = []
        all_prompts = IDENTITY_PROBES + NON_IDENTITY_PROBES
        for rep in range(N_REPEATS):
            for i, prompt in enumerate(all_prompts):
                probe_type = "identity" if i < len(IDENTITY_PROBES) else "non_identity"
                H_relay = extract_hidden(model, tokenizer, sys_prompt, prompt, RELAY_LAYER)
                H_input = extract_hidden(model, tokenizer, sys_prompt, prompt, 0)
                metrics = spectral_metrics(H_relay)
                metrics["passage_distance"] = passage_distance(H_input, H_relay)
                metrics["condition"] = cond_name
                metrics["prompt"] = prompt
                metrics["probe_type"] = probe_type
                metrics["repeat"] = rep
                all_results.append(metrics)
                condition_relays[cond_name].append(H_relay)

    # Sequential condition
    print(f"\nCondition: sequential")
    condition_relays["sequential"] = []
    for rep in range(2):
        for i, prompt in enumerate(IDENTITY_PROBES + NON_IDENTITY_PROBES):
            probe_type = "identity" if i < len(IDENTITY_PROBES) else "non_identity"
            H_relay, H_input = extract_hidden_sequential(model, tokenizer, prompt)
            metrics = spectral_metrics(H_relay)
            metrics["passage_distance"] = passage_distance(H_input, H_relay)
            metrics["condition"] = "sequential"
            metrics["prompt"] = prompt
            metrics["probe_type"] = probe_type
            metrics["repeat"] = rep
            all_results.append(metrics)
            condition_relays["sequential"].append(H_relay)

    # Summary
    print("\n=== RESULTS ===\n")
    print(f"{'Condition':<15} {'S':>10} {'PR':>8} {'σ₁/σ₂':>8} {'d(L0,relay)':>12} {'N':>5}")
    print("-" * 60)
    for cond in ["control", "absent", "receptive", "directive", "sequential"]:
        entries = [r for r in all_results if r["condition"] == cond]
        if not entries:
            continue
        S = np.mean([r["spectral_entropy"] for r in entries])
        S_std = np.std([r["spectral_entropy"] for r in entries])
        PR = np.mean([r["participation_ratio"] for r in entries])
        gap = np.mean([r["spectral_gap"] for r in entries])
        d = np.mean([r["passage_distance"] for r in entries])
        d_std = np.std([r["passage_distance"] for r in entries])
        print(f"{cond:<15} {S:.3f}±{S_std:.3f} {PR:>7.2f} {gap:>7.1f} {d:>7.3f}±{d_std:.3f} {len(entries):>5}")

    # Between vs within variance
    cond_means = []
    within_vars = []
    for cond in ["control", "absent", "receptive", "directive", "sequential"]:
        entries = [r for r in all_results if r["condition"] == cond]
        if not entries:
            continue
        vals = [r["spectral_entropy"] for r in entries]
        cond_means.append(np.mean(vals))
        within_vars.append(np.var(vals))
    between_var = np.var(cond_means)
    mean_within = np.mean(within_vars)
    ratio = between_var / mean_within if mean_within > 0 else float('inf')
    print(f"\nBetween/within variance ratio: {ratio:.1f}×")

    # Grassmannian distances
    print("\n=== GRASSMANNIAN SUBSPACE DISTANCES ===\n")
    conds = list(condition_relays.keys())
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            H1 = np.vstack(condition_relays[conds[i]][:10])
            H2 = np.vstack(condition_relays[conds[j]][:10])
            d = grassmannian_distance(H1, H2)
            print(f"  {conds[i]:>12} ↔ {conds[j]:<12}: {d:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"exp_witness_crossarch_qwen_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    summary = {}
    for cond in ["control", "absent", "receptive", "directive", "sequential"]:
        entries = [r for r in all_results if r["condition"] == cond]
        if entries:
            summary[cond] = {
                "spectral_entropy": {"mean": float(np.mean([r["spectral_entropy"] for r in entries])),
                                     "std": float(np.std([r["spectral_entropy"] for r in entries]))},
                "participation_ratio": float(np.mean([r["participation_ratio"] for r in entries])),
                "spectral_gap": float(np.mean([r["spectral_gap"] for r in entries])),
                "passage_distance": {"mean": float(np.mean([r["passage_distance"] for r in entries])),
                                     "std": float(np.std([r["passage_distance"] for r in entries]))},
                "n": len(entries),
            }
    with open(outfile, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "relay_layer": RELAY_LAYER,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "between_within_ratio": float(ratio),
            "raw": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
