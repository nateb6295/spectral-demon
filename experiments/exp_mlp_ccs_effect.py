#!/usr/bin/env python3
"""Experiment: Where CCS Acts — MLP Hidden States

Attention geometry is frozen (per-head r=0.958, aggregate r>0.9 across 4 archs).
If CCS changes behavior without changing attention, it must act through the MLPs
or output projection.

Compare MLP hidden state geometry across four conditions:
  1. Bare instruct (no preamble)
  2. CCS preamble (identity framing)
  3. Length-matched control (weather forecast, 74 tokens)
  4. Shuffled CCS (same words, scrambled)

For each layer, extract the MLP output activations (hidden states after the
MLP sublayer), compute SVD, extract σ₂. Compare profiles.

If CCS changes MLP σ₂ while weather doesn't → CCS acts through MLPs.
If both change MLP σ₂ equally → still a length effect, even at MLP level.
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


def get_mlp_sigma2(model, tokenizer, prompt, num_layers):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[i] is the output of layer i (0 = embedding, 1..N = after each transformer block)
    sigma2_by_layer = []
    sigma1_by_layer = []
    for layer_idx in range(1, num_layers + 1):
        hs = outputs.hidden_states[layer_idx][0].float()  # (seq_len, hidden_dim)
        U, S, V = torch.linalg.svd(hs, full_matrices=False)
        s1 = S[0].item() if S.shape[0] >= 1 else 0.0
        s2 = S[1].item() if S.shape[0] >= 2 else 0.0
        sigma1_by_layer.append(s1)
        sigma2_by_layer.append(s2)

    return sigma1_by_layer, sigma2_by_layer


def run_condition(model, tokenizer, label, prompts, prefix=""):
    print(f"\n  Condition: {label}")
    num_layers = model.config.num_hidden_layers

    all_s1 = []
    all_s2 = []
    for i, prompt in enumerate(prompts):
        full_prompt = f"{prefix}\n\n{prompt}" if prefix else prompt
        print(f"    Prompt {i+1}/{len(prompts)}: {prompt[:45]}...")
        s1, s2 = get_mlp_sigma2(model, tokenizer, full_prompt, num_layers)
        all_s1.append(s1)
        all_s2.append(s2)

    mean_s1 = [float(np.mean([p[l] for p in all_s1])) for l in range(num_layers)]
    mean_s2 = [float(np.mean([p[l] for p in all_s2])) for l in range(num_layers)]
    mean_ratio = [s2 / s1 if s1 > 0 else 0.0 for s1, s2 in zip(mean_s1, mean_s2)]

    cv_s2 = []
    for l in range(num_layers):
        vals = [p[l] for p in all_s2]
        m = np.mean(vals)
        cv_s2.append(float(np.std(vals) / m) if m > 0 else 0.0)

    return {
        "condition": label,
        "mean_sigma1": mean_s1,
        "mean_sigma2": mean_s2,
        "mean_ratio": mean_ratio,
        "cv_sigma2": cv_s2,
    }


def compare(a, b):
    va = np.array(a)
    vb = np.array(b)
    if np.std(va) < 1e-10 or np.std(vb) < 1e-10:
        r = 1.0 if np.allclose(va, vb) else 0.0
    else:
        r = float(np.corrcoef(va, vb)[0, 1])
    cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))
    l2 = float(np.linalg.norm(va - vb))
    return {"pearson_r": r, "cosine_similarity": cos, "l2_distance": l2}


def main():
    print("=" * 60)
    print("EXPERIMENT: Where CCS Acts — MLP Hidden States")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    random.seed(42)
    ccs_words = CCS_PREAMBLE.split()
    shuffled_words = ccs_words.copy()
    random.shuffle(shuffled_words)
    SHUFFLED_CCS = " ".join(shuffled_words)

    bare = run_condition(model, tokenizer, "bare", PROMPTS)
    ccs = run_condition(model, tokenizer, "ccs", PROMPTS, prefix=CCS_PREAMBLE)
    weather = run_condition(model, tokenizer, "weather", PROMPTS, prefix=LENGTH_CONTROL)
    shuffled = run_condition(model, tokenizer, "shuffled", PROMPTS, prefix=SHUFFLED_CCS)

    print(f"\n{'='*60}")
    print("MLP σ₂ PROFILE COMPARISONS")
    print(f"{'='*60}")

    pairs = [
        ("bare", "ccs", bare, ccs),
        ("bare", "weather", bare, weather),
        ("bare", "shuffled", bare, shuffled),
        ("ccs", "weather", ccs, weather),
        ("ccs", "shuffled", ccs, shuffled),
    ]

    comparisons = []
    for la, lb, a, b in pairs:
        c = compare(a["mean_sigma2"], b["mean_sigma2"])
        c["comparison"] = f"{la} vs {lb}"
        comparisons.append(c)
        print(f"  {la:>10} vs {lb:<10}: r={c['pearson_r']:.4f}, cos={c['cosine_similarity']:.4f}, L2={c['l2_distance']:.4f}")

    print(f"\nRatio comparisons:")
    ratio_comps = []
    for la, lb, a, b in pairs:
        c = compare(a["mean_ratio"], b["mean_ratio"])
        c["comparison"] = f"{la}_ratio vs {lb}_ratio"
        ratio_comps.append(c)
        print(f"  {la:>10} vs {lb:<10}: r={c['pearson_r']:.4f}")

    print(f"\nCV comparisons:")
    cv_comps = []
    for la, lb, a, b in pairs:
        c = compare(a["cv_sigma2"], b["cv_sigma2"])
        c["comparison"] = f"{la}_cv vs {lb}_cv"
        cv_comps.append(c)
        print(f"  {la:>10} vs {lb:<10}: r={c['pearson_r']:.4f}")

    # Key diagnostic
    bare_ccs = [c for c in comparisons if c["comparison"] == "bare vs ccs"][0]
    bare_weather = [c for c in comparisons if c["comparison"] == "bare vs weather"][0]
    ccs_weather = [c for c in comparisons if c["comparison"] == "ccs vs weather"][0]

    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"  bare↔CCS:     r = {bare_ccs['pearson_r']:.4f}")
    print(f"  bare↔weather: r = {bare_weather['pearson_r']:.4f}")
    print(f"  CCS↔weather:  r = {ccs_weather['pearson_r']:.4f}")

    if abs(bare_weather['pearson_r']) < 0.3 and abs(bare_ccs['pearson_r']) < 0.3:
        interp = "MLP: BOTH DIVERGE — length effect dominates MLP geometry too."
    elif bare_weather['pearson_r'] > 0.7 and abs(bare_ccs['pearson_r']) < 0.5:
        interp = "MLP: CCS DIVERGES, WEATHER DOESN'T — CCS acts through MLP geometry!"
    elif ccs_weather['pearson_r'] > 0.9:
        interp = "MLP: CCS ≈ WEATHER — length effect, not identity content."
    elif ccs_weather['pearson_r'] < 0.5 and bare_ccs['pearson_r'] < 0.5:
        interp = "MLP: CCS AND WEATHER DIVERGE DIFFERENTLY — content matters at MLP level!"
    else:
        interp = f"MLP: MIXED — bare↔CCS r={bare_ccs['pearson_r']:.3f}, bare↔weather r={bare_weather['pearson_r']:.3f}"

    print(f"\n  INTERPRETATION: {interp}")

    output = {
        "experiment": "mlp_ccs_effect",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "conditions": {
            "bare": bare,
            "ccs": ccs,
            "weather": weather,
            "shuffled": shuffled,
        },
        "comparisons": {
            "sigma2": comparisons,
            "ratio": ratio_comps,
            "cv": cv_comps,
        },
        "interpretation": interp,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_mlp_ccs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
