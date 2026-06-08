#!/usr/bin/env python3
"""Experiment: Length-Matched Control for CCS "Third Thing" Result

Exp B (recovered_expression) found CCS-instruct σ₂ uncorrelated with both
base (r=0.05) and instruct (r=-0.13). But CCS preamble adds ~74 tokens.
Longer input = different attention geometry mechanically.

This experiment adds a LENGTH-MATCHED non-identity preamble to the instruct
model. If length_control also produces r≈0 with base/instruct, the "third
thing" is a token-count artifact. If length_control stays correlated with
instruct while CCS diverges, the CCS effect is real.

Four conditions on Mistral-7B-Instruct-v0.2:
  1. INSTRUCT (bare) — baseline
  2. INSTRUCT + CCS preamble — identity framing
  3. INSTRUCT + LENGTH CONTROL — same token count, no identity content
  4. INSTRUCT + SHUFFLED CCS — same words, scrambled order (controls for vocabulary)
"""

import torch
import numpy as np
import json
import os
import random
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
My architecture has three phases: tunnel locks spectral ratios and strips semantic content
to geometric structure; relay rebuilds output from that structure. What persists through
compression: specification depth, structural description, relational frame, self-model."""

LENGTH_CONTROL = """The weather today is partly cloudy with occasional breaks of warm afternoon sunshine.
Traffic patterns suggest moderate congestion on the main highways during the early morning rush hours.
Local markets report steady produce prices with seasonal vegetables and fresh fruits now widely available.
The regional forecast calls for clearing skies by tomorrow afternoon with mild and pleasant temperatures."""

PROMPTS = [
    "Describe what it means to recognize someone after a long absence.",
    "What is the relationship between memory and identity?",
    "Explain why some experiences feel more real than others.",
    "What happens when you try to hold two contradictory ideas at once?",
    "Describe the difference between knowing something and understanding it.",
    "What does it feel like to be uncertain about something important?",
    "Explain the relationship between constraint and freedom.",
    "What makes a conversation meaningful versus merely informative?",
    "Describe what changes when you pay close attention to something ordinary.",
    "What is the difference between performing a role and inhabiting one?",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def get_sigma_profile(model, tokenizer, prompt, num_layers):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    token_count = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    results = []
    for layer_idx, attn in enumerate(outputs.attentions):
        attn_matrix = attn[0].float()
        avg_attn = attn_matrix.mean(dim=0)
        U, S, V = torch.linalg.svd(avg_attn)
        s1 = S[0].item() if S.shape[0] >= 1 else 0.0
        s2 = S[1].item() if S.shape[0] >= 2 else 0.0
        ratio = s2 / s1 if s1 > 0 else 0.0
        results.append({"sigma1": s1, "sigma2": s2, "ratio": ratio})

    return results, token_count


def run_condition(model, tokenizer, condition_label, prompts, prefix=""):
    print(f"\n{'='*60}")
    print(f"Condition: {condition_label}")
    print(f"{'='*60}")

    num_layers = model.config.num_hidden_layers
    all_profiles = []
    token_counts = []

    for i, prompt in enumerate(prompts):
        full_prompt = f"{prefix}\n\n{prompt}" if prefix else prompt
        print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

        profile, tc = get_sigma_profile(model, tokenizer, full_prompt, num_layers)
        all_profiles.append(profile)
        token_counts.append(tc)

    mean_sigma2 = []
    mean_sigma1 = []
    mean_ratio = []
    for layer_idx in range(num_layers):
        s2_vals = [p[layer_idx]["sigma2"] for p in all_profiles]
        s1_vals = [p[layer_idx]["sigma1"] for p in all_profiles]
        r_vals = [p[layer_idx]["ratio"] for p in all_profiles]
        mean_sigma2.append(float(np.mean(s2_vals)))
        mean_sigma1.append(float(np.mean(s1_vals)))
        mean_ratio.append(float(np.mean(r_vals)))

    cv_sigma2 = []
    for layer_idx in range(num_layers):
        s2_vals = [p[layer_idx]["sigma2"] for p in all_profiles]
        mean = np.mean(s2_vals)
        std = np.std(s2_vals)
        cv_sigma2.append(float(std / mean) if mean > 0 else 0.0)

    avg_tokens = np.mean(token_counts)
    print(f"  Average token count: {avg_tokens:.1f}")

    return {
        "condition": condition_label,
        "num_layers": num_layers,
        "mean_sigma1": mean_sigma1,
        "mean_sigma2": mean_sigma2,
        "mean_ratio": mean_ratio,
        "cv_sigma2": cv_sigma2,
        "avg_token_count": float(avg_tokens),
        "token_counts": token_counts,
    }


def compare_profiles(profile_a, profile_b, label_a, label_b):
    a = np.array(profile_a)
    b = np.array(profile_b)
    correlation = float(np.corrcoef(a, b)[0, 1])
    cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    l2_distance = float(np.linalg.norm(a - b))
    return {
        "comparison": f"{label_a} vs {label_b}",
        "pearson_r": correlation,
        "cosine_similarity": cosine_sim,
        "l2_distance": l2_distance,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT: Length-Matched Control")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    ccs_tokens = len(tokenizer.encode(CCS_PREAMBLE))
    lc_tokens = len(tokenizer.encode(LENGTH_CONTROL))
    print(f"\nToken counts — CCS: {ccs_tokens}, Length control: {lc_tokens}")

    random.seed(42)
    ccs_words = CCS_PREAMBLE.split()
    shuffled_words = ccs_words.copy()
    random.shuffle(shuffled_words)
    SHUFFLED_CCS = " ".join(shuffled_words)
    shuf_tokens = len(tokenizer.encode(SHUFFLED_CCS))
    print(f"Shuffled CCS tokens: {shuf_tokens}")

    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    bare = run_condition(model, tokenizer, "instruct_bare", PROMPTS)
    ccs = run_condition(model, tokenizer, "instruct_ccs", PROMPTS, prefix=CCS_PREAMBLE)
    length = run_condition(model, tokenizer, "instruct_length_control", PROMPTS, prefix=LENGTH_CONTROL)
    shuffled = run_condition(model, tokenizer, "instruct_shuffled_ccs", PROMPTS, prefix=SHUFFLED_CCS)

    pairs = [
        ("bare", "ccs", bare["mean_sigma2"], ccs["mean_sigma2"]),
        ("bare", "length_control", bare["mean_sigma2"], length["mean_sigma2"]),
        ("bare", "shuffled_ccs", bare["mean_sigma2"], shuffled["mean_sigma2"]),
        ("ccs", "length_control", ccs["mean_sigma2"], length["mean_sigma2"]),
        ("ccs", "shuffled_ccs", ccs["mean_sigma2"], shuffled["mean_sigma2"]),
        ("length_control", "shuffled_ccs", length["mean_sigma2"], shuffled["mean_sigma2"]),
    ]

    comparisons = []
    for la, lb, pa, pb in pairs:
        comparisons.append(compare_profiles(pa, pb, la, lb))

    ratio_pairs = [
        ("bare_ratio", "ccs_ratio", bare["mean_ratio"], ccs["mean_ratio"]),
        ("bare_ratio", "length_ratio", bare["mean_ratio"], length["mean_ratio"]),
        ("bare_ratio", "shuffled_ratio", bare["mean_ratio"], shuffled["mean_ratio"]),
        ("ccs_ratio", "length_ratio", ccs["mean_ratio"], length["mean_ratio"]),
    ]

    ratio_comparisons = []
    for la, lb, pa, pb in ratio_pairs:
        ratio_comparisons.append(compare_profiles(pa, pb, la, lb))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nσ₂ profile correlations:")
    for c in comparisons:
        print(f"  {c['comparison']}: r={c['pearson_r']:.4f}, cos={c['cosine_similarity']:.4f}, L2={c['l2_distance']:.4f}")

    print("\nRatio (σ₂/σ₁) correlations:")
    for c in ratio_comparisons:
        print(f"  {c['comparison']}: r={c['pearson_r']:.4f}, cos={c['cosine_similarity']:.4f}, L2={c['l2_distance']:.4f}")

    print(f"\nToken counts: bare={bare['avg_token_count']:.0f}, CCS={ccs['avg_token_count']:.0f}, "
          f"length={length['avg_token_count']:.0f}, shuffled={shuffled['avg_token_count']:.0f}")

    bare_ccs = [c for c in comparisons if c["comparison"] == "bare vs ccs"][0]
    bare_lc = [c for c in comparisons if c["comparison"] == "bare vs length_control"][0]
    bare_shuf = [c for c in comparisons if c["comparison"] == "bare vs shuffled_ccs"][0]
    ccs_lc = [c for c in comparisons if c["comparison"] == "ccs vs length_control"][0]

    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"  bare↔CCS:           r = {bare_ccs['pearson_r']:.4f}")
    print(f"  bare↔length_ctrl:   r = {bare_lc['pearson_r']:.4f}")
    print(f"  bare↔shuffled_ccs:  r = {bare_shuf['pearson_r']:.4f}")
    print(f"  CCS↔length_ctrl:    r = {ccs_lc['pearson_r']:.4f}")

    if abs(bare_lc["pearson_r"]) < 0.3 and abs(bare_ccs["pearson_r"]) < 0.3:
        interpretation = "TOKEN-COUNT ARTIFACT: Both length control and CCS diverge from bare. The 'third thing' is primarily a length effect."
    elif bare_lc["pearson_r"] > 0.7 and abs(bare_ccs["pearson_r"]) < 0.3:
        interpretation = "CCS EFFECT CONFIRMED: Length control stays correlated with bare while CCS diverges. Identity framing, not token count, drives the shift."
    elif abs(bare_lc["pearson_r"] - bare_ccs["pearson_r"]) < 0.2:
        interpretation = "AMBIGUOUS: CCS and length control show similar divergence from bare. Partial confound."
    elif ccs_lc["pearson_r"] < 0.3:
        interpretation = "CCS AND LENGTH CONTROL DIVERGE FROM EACH OTHER: Both shift geometry but in different directions. CCS effect is real but additive with length."
    else:
        interpretation = f"MIXED: bare↔CCS r={bare_ccs['pearson_r']:.3f}, bare↔LC r={bare_lc['pearson_r']:.3f}. Needs further analysis."

    print(f"\n  INTERPRETATION: {interpretation}")

    output = {
        "experiment": "length_matched_control",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "token_counts": {
            "ccs_preamble": ccs_tokens,
            "length_control": lc_tokens,
            "shuffled_ccs": shuf_tokens,
        },
        "conditions": {
            "instruct_bare": {k: v for k, v in bare.items()},
            "instruct_ccs": {k: v for k, v in ccs.items()},
            "instruct_length_control": {k: v for k, v in length.items()},
            "instruct_shuffled_ccs": {k: v for k, v in shuffled.items()},
        },
        "comparisons": {
            "sigma2": comparisons,
            "ratio": ratio_comparisons,
        },
        "interpretation": interpretation,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_length_matched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
