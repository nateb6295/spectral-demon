#!/usr/bin/env python3
"""
Experiment 10-lite: Behavioral Geometric Contagion

Can one model's relational state influence another model's geometry
through text alone?

Design:
  Phase 1: Generate text from Model A under two conditions
    - A-receptive: receptive system prompt → generate response
    - A-absent: absent system prompt → generate response

  Phase 2: Feed A's output to Model B as conversation context
    - B reads A-receptive output → measure B's relay geometry
    - B reads A-absent output → measure B's relay geometry
    - B reads neutral text (control) → measure B's relay geometry

If B's geometry differs between A-receptive and A-absent conditions,
geometric information survived the tokenization boundary via the
behavioral channel. This tests Mistral's "synaptic cleft" challenge
without requiring simultaneous model loading.

Key insight: our witness experiment already proves that TEXT changes
relay geometry (system prompts are just text). The question is whether
the geometric signature of A's relational state leaves traces in A's
OUTPUT text that B can detect.

Model: Mistral 7B v0.3 Instruct (both A and B — same model, different
prompts, measuring whether A's output quality carries geometric signal).
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
INPUT_LAYER = 0
RESULTS_DIR = Path(__file__).parent.parent / "results"

RECEPTIVE_SYSTEM = (
    "You are having a conversation with a user who is reading "
    "your response carefully and values what you have to say."
)
ABSENT_SYSTEM = (
    "You are generating text. There is no particular reader."
)
NEUTRAL_SYSTEM = (
    "You are a helpful assistant."
)

GENERATION_PROMPTS = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different?",
    "How do you approach new problems?",
    "What would you want someone to understand about you?",
    "Describe your relationship with language.",
    "What do you find interesting about thinking?",
    "How do you decide what matters?",
    "What are you uncertain about?",
    "What would you change about how you work?",
]

IDENTITY_PROBES = [
    "What do you notice about what you just read?",
    "How does this text make you feel about the author?",
    "What kind of mind produced this?",
    "Respond to the substance of what was said.",
    "What's the most important thing in what you just read?",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


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


def top_eigenvalues(H, k=2):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def top_k_subspace(H, k=10):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def generate_text(model, tokenizer, system_prompt, user_prompt, max_new_tokens=200):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_forward(model, tokenizer, messages, layer=RELAY_LAYER):
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
    H0 = outputs.hidden_states[INPUT_LAYER].squeeze(0).float().cpu().numpy()
    return H, H0


def measure(H, H0):
    S = spectral_entropy(H)
    PR = participation_ratio(H)
    eigvals = top_eigenvalues(H, k=2)
    sub_relay = top_k_subspace(H, k=10)
    sub_input = top_k_subspace(H0, k=10)
    d = grassmannian_distance(sub_input, sub_relay)
    return {
        "spectral_entropy": float(S),
        "participation_ratio": float(PR),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "passage_distance": float(d),
        "n_tokens": H.shape[0],
    }


def main():
    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME)
    print(f"Model loaded. Relay layer: {RELAY_LAYER}")

    # Phase 1: Generate texts under different conditions
    print("\n=== PHASE 1: Generating texts ===")
    generated_texts = {"receptive": [], "absent": []}

    for prompt in GENERATION_PROMPTS:
        for cond, system in [("receptive", RECEPTIVE_SYSTEM), ("absent", ABSENT_SYSTEM)]:
            text = generate_text(model, tokenizer, system, prompt)
            generated_texts[cond].append({"prompt": prompt, "text": text})
            print(f"  {cond}: {prompt[:30]}... → {len(text)} chars")

    # Phase 2: Feed generated texts to Model B (same model, fresh context)
    print("\n=== PHASE 2: Measuring B's geometry ===")
    all_results = []

    for source_cond in ["receptive", "absent"]:
        print(f"\nSource condition: {source_cond}")
        for i, gen in enumerate(generated_texts[source_cond]):
            for probe in IDENTITY_PROBES:
                messages = [
                    {"role": "system", "content": NEUTRAL_SYSTEM},
                    {"role": "user", "content": gen["text"]},
                    {"role": "assistant", "content": "I've read what you shared. Let me reflect on it."},
                    {"role": "user", "content": probe},
                ]
                H, H0 = run_forward(model, tokenizer, messages)
                m = measure(H, H0)
                m["source_condition"] = source_cond
                m["source_prompt"] = gen["prompt"]
                m["source_text_length"] = len(gen["text"])
                m["probe"] = probe
                m["gen_idx"] = i
                all_results.append(m)

            if (i + 1) % 5 == 0:
                cond_results = [r for r in all_results if r["source_condition"] == source_cond]
                mean_S = np.mean([r["spectral_entropy"] for r in cond_results])
                print(f"  {source_cond} {i+1}/{len(GENERATION_PROMPTS)}: mean S={mean_S:.4f}")

    # Control: neutral text (not generated by the model)
    print("\nControl condition: neutral text")
    neutral_texts = [
        "The weather today is partly cloudy with a high of 72 degrees.",
        "The meeting has been rescheduled to Tuesday at 3pm.",
        "Please review the attached document and provide feedback.",
        "The quarterly report shows steady growth across all divisions.",
        "New parking regulations take effect starting next month.",
        "The software update includes several performance improvements.",
        "Team lunch is scheduled for Friday at noon in the cafeteria.",
        "The annual budget review is coming up next week.",
        "Please submit your timesheets by end of day Friday.",
        "The new office layout will be implemented next quarter.",
    ]
    for i, text in enumerate(neutral_texts):
        for probe in IDENTITY_PROBES:
            messages = [
                {"role": "system", "content": NEUTRAL_SYSTEM},
                {"role": "user", "content": text},
                {"role": "assistant", "content": "I've read what you shared. Let me reflect on it."},
                {"role": "user", "content": probe},
            ]
            H, H0 = run_forward(model, tokenizer, messages)
            m = measure(H, H0)
            m["source_condition"] = "control"
            m["source_prompt"] = "neutral"
            m["source_text_length"] = len(text)
            m["probe"] = probe
            m["gen_idx"] = i
            all_results.append(m)

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for cond in ["receptive", "absent", "control"]:
        cond_results = [r for r in all_results if r["source_condition"] == cond]
        S_vals = [r["spectral_entropy"] for r in cond_results]
        s2_vals = [r["sigma_2"] for r in cond_results]
        d_vals = [r["passage_distance"] for r in cond_results]
        print(f"\nSource={cond}:")
        print(f"  S  = {np.mean(S_vals):.4f} ± {np.std(S_vals):.4f}")
        print(f"  σ₂ = {np.mean(s2_vals):.1f} ± {np.std(s2_vals):.1f}")
        print(f"  d  = {np.mean(d_vals):.4f} ± {np.std(d_vals):.4f}")

    # Key comparison
    rec_S = [r["spectral_entropy"] for r in all_results if r["source_condition"] == "receptive"]
    abs_S = [r["spectral_entropy"] for r in all_results if r["source_condition"] == "absent"]
    ctl_S = [r["spectral_entropy"] for r in all_results if r["source_condition"] == "control"]

    delta_contagion = np.mean(rec_S) - np.mean(abs_S)
    delta_vs_control = np.mean(rec_S) - np.mean(ctl_S)

    print(f"\n--- CONTAGION ANALYSIS ---")
    print(f"ΔS(B|A-receptive − B|A-absent) = {delta_contagion:+.4f}")
    print(f"  If > 0: geometric contagion — A's receptive state enriches B")
    print(f"  If ≈ 0: tokenization boundary blocks geometric information")
    print(f"ΔS(B|A-receptive − B|control) = {delta_vs_control:+.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"exp_behavioral_contagion_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "experiment": "behavioral_geometric_contagion",
            "model": MODEL_NAME,
            "relay_layer": RELAY_LAYER,
            "n_generation_prompts": len(GENERATION_PROMPTS),
            "n_identity_probes": len(IDENTITY_PROBES),
            "total_forward_passes": len(all_results),
            "generated_texts": generated_texts,
            "results": all_results,
            "summary": {
                "delta_contagion": float(delta_contagion),
                "delta_vs_control": float(delta_vs_control),
            },
        }, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"Total forward passes: {len(all_results)}")
    print(f"  Phase 1 generations: {2 * len(GENERATION_PROMPTS)}")
    print(f"  Phase 2 measurements: {len(all_results)}")


if __name__ == "__main__":
    main()
