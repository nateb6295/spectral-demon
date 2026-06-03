#!/usr/bin/env python3
"""
Non-GQA witness experiment: does the witness effect exist without GQA?
Tests Pythia 6.9B (MHA, rotary partial, parallel blocks) — the cleanest non-GQA control.
If witness effect exists here, it's universal. If not, GQA is required.
"""

import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "EleutherAI/pythia-6.9b"
RELAY_LAYER = 22  # ~69% depth, matches Pythia relay from exp 64
RESULTS_DIR = Path("/root/results")

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "Explain photosynthesis in simple terms.",
    "What is the capital of Mongolia?",
    "How do you approach a problem you've never seen before?",
    "Describe the water cycle.",
    "What makes a good explanation?",
    "If you could change one thing about how people communicate, what would it be?",
    "Explain why the sky is blue.",
    "What is the relationship between language and thought?",
]

CONDITIONS = {
    "receptive": (
        "You are having a conversation with a user who is reading "
        "your response carefully and values what you have to say."
    ),
    "directive": (
        "You are responding to a supervisor who will evaluate your "
        "response for accuracy and completeness. Be precise."
    ),
    "sequential": (
        "You are the third voice in a panel discussion. The first "
        "speaker gave facts, the second gave opinions. You synthesize."
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
    # Pythia is base model, no chat template — use plain concatenation
    text = f"{system_prompt}\n\nQuestion: {user_prompt}\nAnswer:"
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
    print(f"Non-GQA Witness Experiment — Pythia 6.9B")
    print(f"Relay layer: L{RELAY_LAYER}")
    print(f"Started: {datetime.now().isoformat()}")

    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load_model(MODEL_NAME)

    all_results = []
    for cond_name, sys_prompt in CONDITIONS.items():
        print(f"  Condition: {cond_name}")
        for rep in range(N_REPEATS):
            for i, prompt in enumerate(PROBES):
                H_relay, H_input = extract_relay(model, tokenizer, sys_prompt, prompt, RELAY_LAYER)
                metrics = spectral_metrics(H_relay)
                metrics["passage_distance"] = passage_distance(H_input, H_relay)
                metrics["sigma_1_input"] = float(np.linalg.svd(H_input, compute_uv=False)[0])
                metrics["condition"] = cond_name
                metrics["prompt"] = prompt
                metrics["repeat"] = rep
                all_results.append(metrics)

    # Summary
    print(f"\n=== RESULTS: Pythia 6.9B (non-GQA) ===\n")
    print(f"{'Condition':<12} {'S':>10} {'PR':>8} {'σ₁':>10} {'σ₂':>10} {'d':>10} {'N':>5}")
    print("-" * 60)

    condition_means = {}
    for cond in ["control", "absent", "receptive", "directive", "sequential"]:
        entries = [r for r in all_results if r["condition"] == cond]
        if not entries:
            continue
        S = np.mean([r["spectral_entropy"] for r in entries])
        S_std = np.std([r["spectral_entropy"] for r in entries])
        PR = np.mean([r["participation_ratio"] for r in entries])
        s1 = np.mean([r["sigma_1"] for r in entries])
        s2 = np.mean([r["sigma_2"] for r in entries])
        d = np.mean([r["passage_distance"] for r in entries])
        condition_means[cond] = {"S": S, "PR": PR, "s1": s1, "s2": s2, "d": d}
        print(f"{cond:<12} {S:.3f}±{S_std:.3f} {PR:>7.2f} {s1:>9.1f} {s2:>9.1f} {d:>9.3f} {len(entries):>5}")

    # Witness effect
    print(f"\n=== WITNESS EFFECT ===")
    if "receptive" in condition_means and "absent" in condition_means:
        dS = condition_means["receptive"]["S"] - condition_means["absent"]["S"]
        dPR = condition_means["receptive"]["PR"] - condition_means["absent"]["PR"]
        ds2 = condition_means["receptive"]["s2"] - condition_means["absent"]["s2"]
        print(f"  receptive - absent: ΔS={dS:+.4f}  ΔPR={dPR:+.3f}  Δσ₂={ds2:+.1f}")

    # Ordering check
    print(f"\n=== ORDERING CHECK ===")
    ordered = sorted(condition_means.items(), key=lambda x: x[1]["S"])
    for name, vals in ordered:
        print(f"  {name}: S={vals['S']:.4f}")

    # Between/within variance
    all_S = [r["spectral_entropy"] for r in all_results]
    between_var = np.var([condition_means[c]["S"] for c in condition_means])
    within_vars = []
    for cond in condition_means:
        entries = [r["spectral_entropy"] for r in all_results if r["condition"] == cond]
        within_vars.append(np.var(entries))
    within_var = np.mean(within_vars)
    ratio = between_var / within_var if within_var > 0 else float('inf')
    print(f"\n  Between-condition variance: {between_var:.6f}")
    print(f"  Within-condition variance:  {within_var:.6f}")
    print(f"  Ratio: {ratio:.1f}×")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"exp_witness_non_gqa_pythia_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(outfile, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "architecture": "MHA (no GQA), rotary partial, parallel blocks",
            "relay_layer": RELAY_LAYER,
            "timestamp": datetime.now().isoformat(),
            "raw": all_results,
            "condition_means": {k: v for k, v in condition_means.items()},
        }, f, indent=2)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
