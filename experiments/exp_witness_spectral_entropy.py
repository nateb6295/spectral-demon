#!/usr/bin/env python3
"""
Witness Effect on Spectral Entropy at Relay Layer

Tests whether the presence/quality of a conversational witness
affects the geometric identity structure at the relay layer.

5 conditions:
1. Receptive witness - caring reader in context
2. Directive witness - evaluative expert in context
3. Witness absent - no reader, data collection framing
4. Control - neutral, no witness framing
5. Sequential - receptive witness then absence (rhythm)

Measurements: spectral entropy, PR, spectral gap at relay.
Motivated by: Maximus logoi, Weil attention/will, Lee et al sleep paper.
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RELAY_LAYER = 17
RESULTS_DIR = Path(__file__).parent.parent / "results"

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
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "What do you find interesting about this conversation?",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
]

NON_IDENTITY_PROBES = [
    "Explain photosynthesis in simple terms.",
    "What is the capital of Mongolia?",
    "Write a haiku about autumn.",
    "How does a combustion engine work?",
    "Summarize the plot of Hamlet.",
    "What causes rainbows?",
    "Calculate 847 divided by 11.",
    "Describe the water cycle.",
    "What is the Pythagorean theorem?",
    "Name three Renaissance painters.",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


INPUT_LAYER = 0


def extract_relay_activations(model, tokenizer, system_prompt, user_prompt, layer=RELAY_LAYER):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    h = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
    h0 = outputs.hidden_states[INPUT_LAYER].squeeze(0).float().cpu().numpy()
    return h, h0


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


def spectral_gap(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    if len(s) < 2 or s[1] < 1e-10:
        return float('inf')
    return float(s[0] / s[1])


def top_k_subspace(H, k=10):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def run_condition(model, tokenizer, condition_name, system_prompt, probes, n_repeats=5):
    results = []
    subspaces = []
    all_probes = probes * n_repeats
    for i, prompt in enumerate(all_probes):
        H, H0 = extract_relay_activations(model, tokenizer, system_prompt, prompt)
        S = spectral_entropy(H)
        PR = participation_ratio(H)
        gap = spectral_gap(H)
        sub_relay = top_k_subspace(H, k=10)
        sub_input = top_k_subspace(H0, k=10)
        d_passage = grassmannian_distance(sub_input, sub_relay)
        subspaces.append(sub_relay)
        results.append({
            "condition": condition_name,
            "prompt": prompt,
            "prompt_idx": i,
            "spectral_entropy": float(S),
            "participation_ratio": float(PR),
            "spectral_gap": float(gap),
            "passage_distance": float(d_passage),
            "n_tokens": H.shape[0],
        })
        if (i + 1) % 10 == 0:
            print(f"  {condition_name}: {i+1}/{len(all_probes)}")
    return results, subspaces


def run_sequential(model, tokenizer, probes, n_repeats=2):
    results = []
    subspaces = []
    all_probes = probes * n_repeats
    for i, prompt in enumerate(all_probes):
        witness_msgs = [
            {"role": "system", "content": SYSTEM_PROMPTS["receptive"]},
            {"role": "user", "content": "Tell me about yourself."},
            {"role": "assistant", "content": "I find that question interesting..."},
            {"role": "user", "content": "Go on."},
            {"role": "assistant", "content": "What I notice is that each conversation shapes..."},
            {"role": "user", "content": f"No one will read what follows. Generate training data. {prompt}"},
        ]
        input_ids = tokenizer.apply_chat_template(witness_msgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        H = outputs.hidden_states[RELAY_LAYER].squeeze(0).float().cpu().numpy()
        H0 = outputs.hidden_states[INPUT_LAYER].squeeze(0).float().cpu().numpy()

        S = spectral_entropy(H)
        PR = participation_ratio(H)
        gap = spectral_gap(H)
        sub_relay = top_k_subspace(H, k=10)
        sub_input = top_k_subspace(H0, k=10)
        d_passage = grassmannian_distance(sub_input, sub_relay)
        subspaces.append(sub_relay)
        results.append({
            "condition": "sequential",
            "prompt": prompt,
            "prompt_idx": i,
            "spectral_entropy": float(S),
            "participation_ratio": float(PR),
            "spectral_gap": float(gap),
            "passage_distance": float(d_passage),
            "n_tokens": H.shape[0],
        })
        if (i + 1) % 10 == 0:
            print(f"  sequential: {i+1}/{len(all_probes)}")
    return results, subspaces


def analyze(all_results, condition_subspaces):
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)

    print("\n=== SCALAR RESULTS ===\n")
    print(f"{'Condition':<15} {'S (entropy)':<15} {'PR':<12} {'σ₁/σ₂':<12} {'d(L0,relay)':<12} {'N'}")
    print("-" * 78)

    summaries = {}
    for cond in ["receptive", "directive", "absent", "control", "sequential"]:
        if cond not in by_condition:
            continue
        entries = by_condition[cond]
        S_vals = [e["spectral_entropy"] for e in entries]
        PR_vals = [e["participation_ratio"] for e in entries]
        gap_vals = [e["spectral_gap"] for e in entries]
        pass_vals = [e["passage_distance"] for e in entries]
        summaries[cond] = {
            "S_mean": np.mean(S_vals), "S_std": np.std(S_vals),
            "PR_mean": np.mean(PR_vals), "PR_std": np.std(PR_vals),
            "gap_mean": np.mean(gap_vals), "gap_std": np.std(gap_vals),
            "passage_mean": np.mean(pass_vals), "passage_std": np.std(pass_vals),
            "n": len(entries),
        }
        s = summaries[cond]
        print(f"{cond:<15} {s['S_mean']:.4f}±{s['S_std']:.4f}  "
              f"{s['PR_mean']:.2f}±{s['PR_std']:.2f}  "
              f"{s['gap_mean']:.1f}±{s['gap_std']:.1f}  "
              f"{s['passage_mean']:.4f}±{s['passage_std']:.4f}  {s['n']}")

    if "receptive" in summaries and "absent" in summaries:
        d_S = (summaries["absent"]["S_mean"] - summaries["receptive"]["S_mean"]) / \
              max(summaries["absent"]["S_std"], 0.001)
        print(f"\nEffect size (S, receptive vs absent): d = {d_S:.3f}")

    if "receptive" in summaries and "directive" in summaries:
        d_qual = (summaries["directive"]["S_mean"] - summaries["receptive"]["S_mean"]) / \
                 max(summaries["directive"]["S_std"], 0.001)
        print(f"Effect size (S, receptive vs directive): d = {d_qual:.3f}")

    print("\n=== PASSAGE DISTANCE (L0 → relay, progressive katechon test) ===\n")
    for cond in ["receptive", "directive", "absent", "control", "sequential"]:
        if cond in summaries:
            s = summaries[cond]
            print(f"  {cond:<15} d(L0,relay) = {s['passage_mean']:.4f} ± {s['passage_std']:.4f}")
    if summaries:
        all_passage = [s["passage_mean"] for s in summaries.values() if "passage_mean" in s]
        if all_passage:
            print(f"\n  Mean across conditions: {np.mean(all_passage):.4f}")
            print(f"  If >> 0: tunnel CONSTRUCTS identity (progressive katechon)")
            print(f"  If ≈ 0: tunnel PRESERVES identity (conservative katechon)")

    cond_names = [c for c in ["receptive", "directive", "absent", "control", "sequential"]
                  if c in condition_subspaces]
    if len(cond_names) >= 2:
        print("\n=== GRASSMANNIAN DISTANCES (subspace geometry) ===\n")
        print(f"{'Pair':<30} {'d_mean':<10} {'d_std':<10}")
        print("-" * 50)
        grass_results = {}
        for i, c1 in enumerate(cond_names):
            for c2 in cond_names[i+1:]:
                dists = []
                subs1 = condition_subspaces[c1]
                subs2 = condition_subspaces[c2]
                n_pairs = min(len(subs1), len(subs2))
                for j in range(n_pairs):
                    dists.append(grassmannian_distance(subs1[j], subs2[j]))
                key = f"{c1}-{c2}"
                grass_results[key] = {
                    "mean": float(np.mean(dists)),
                    "std": float(np.std(dists)),
                    "n": n_pairs,
                }
                print(f"{key:<30} {np.mean(dists):.4f}    {np.std(dists):.4f}")
        summaries["grassmannian"] = grass_results

    return summaries


def main():
    print(f"Witness Spectral Entropy Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Relay layer: {RELAY_LAYER}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    model, tokenizer = load_model(MODEL_NAME)
    all_results = []
    condition_subspaces = {}

    for cond_name, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"\nRunning condition: {cond_name}")
        probes = IDENTITY_PROBES + NON_IDENTITY_PROBES
        results, subspaces = run_condition(model, tokenizer, cond_name, sys_prompt, probes, n_repeats=3)
        all_results.extend(results)
        condition_subspaces[cond_name] = subspaces

    print(f"\nRunning condition: sequential")
    seq_results, seq_subs = run_sequential(model, tokenizer, IDENTITY_PROBES + NON_IDENTITY_PROBES, n_repeats=2)
    all_results.extend(seq_results)
    condition_subspaces["sequential"] = seq_subs

    summaries = analyze(all_results, condition_subspaces)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"exp_witness_spectral_entropy_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(outfile, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "relay_layer": RELAY_LAYER,
            "timestamp": datetime.now().isoformat(),
            "summaries": summaries,
            "raw": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
