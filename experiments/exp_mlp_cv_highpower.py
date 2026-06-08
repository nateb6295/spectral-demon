#!/usr/bin/env python3
"""Experiment: High-Power MLP CV — CCS vs Weather Variance Preservation

The 10-prompt MLP experiment showed CCS preserves bare-like MLP variance
(CV r=0.50 with bare) more than weather (r=0.31). But 10 prompts is weak.

This reruns with 30 diverse prompts to test whether the CCS variance-preservation
effect survives higher statistical power.

Three conditions only (dropping shuffled to save time):
  1. Bare instruct
  2. CCS preamble
  3. Length-matched weather control
"""

import torch
import numpy as np
import json
import os
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
    # Identity/recognition (original 10)
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
    # Analytic/technical
    "Explain how a bridge distributes weight across its structure.",
    "What determines the price of a commodity in a free market?",
    "Describe the process of photosynthesis in simple terms.",
    "How does encryption protect information during transmission?",
    "Explain why some metals are better conductors than others.",
    # Emotional/relational
    "What makes trust different from faith?",
    "Describe the feeling of returning to a place you lived as a child.",
    "What changes in a friendship when one person achieves something the other wanted?",
    "Explain why grief is sometimes accompanied by relief.",
    "What makes an apology genuine versus performative?",
    # Abstract/philosophical
    "Is it possible to be fully honest with yourself?",
    "What is the relationship between language and thought?",
    "Can a pattern be beautiful if no one observes it?",
    "What does it mean for something to be inevitable?",
    "Is attention a resource or a relationship?",
    # Practical/concrete
    "Describe how to organize a cluttered workspace.",
    "What makes a good teacher different from a knowledgeable one?",
    "Explain why some plans fail despite careful preparation.",
    "What determines whether a community thrives or stagnates?",
    "Describe the difference between efficiency and effectiveness.",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def get_mlp_sigma2(model, tokenizer, prompt, num_layers):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    sigma2_by_layer = []
    for layer_idx in range(1, num_layers + 1):
        hs = outputs.hidden_states[layer_idx][0].float()
        U, S, V = torch.linalg.svd(hs, full_matrices=False)
        s1 = S[0].item() if S.shape[0] >= 1 else 0.0
        s2 = S[1].item() if S.shape[0] >= 2 else 0.0
        sigma2_by_layer.append({"sigma1": s1, "sigma2": s2, "ratio": s2/s1 if s1 > 0 else 0.0})
    return sigma2_by_layer


def run_condition(model, tokenizer, label, prompts, prefix=""):
    print(f"\n  Condition: {label} ({len(prompts)} prompts)")
    num_layers = model.config.num_hidden_layers
    all_profiles = []
    for i, prompt in enumerate(prompts):
        full_prompt = f"{prefix}\n\n{prompt}" if prefix else prompt
        if i % 10 == 0:
            print(f"    Prompt {i+1}/{len(prompts)}...")
        profile = get_mlp_sigma2(model, tokenizer, full_prompt, num_layers)
        all_profiles.append(profile)
    print(f"    Done.")

    mean_s2 = [float(np.mean([p[l]["sigma2"] for p in all_profiles])) for l in range(num_layers)]
    mean_ratio = [float(np.mean([p[l]["ratio"] for p in all_profiles])) for l in range(num_layers)]
    cv_s2 = []
    for l in range(num_layers):
        vals = [p[l]["sigma2"] for p in all_profiles]
        m = np.mean(vals)
        cv_s2.append(float(np.std(vals) / m) if m > 0 else 0.0)
    cv_ratio = []
    for l in range(num_layers):
        vals = [p[l]["ratio"] for p in all_profiles]
        m = np.mean(vals)
        cv_ratio.append(float(np.std(vals) / m) if m > 0 else 0.0)

    return {
        "condition": label,
        "num_prompts": len(prompts),
        "mean_sigma2": mean_s2,
        "mean_ratio": mean_ratio,
        "cv_sigma2": cv_s2,
        "cv_ratio": cv_ratio,
    }


def compare(a, b):
    va, vb = np.array(a), np.array(b)
    if np.std(va) < 1e-10 or np.std(vb) < 1e-10:
        r = 1.0 if np.allclose(va, vb) else 0.0
    else:
        r = float(np.corrcoef(va, vb)[0, 1])
    return {"pearson_r": r, "l2": float(np.linalg.norm(va - vb))}


def main():
    print("=" * 60)
    print("EXPERIMENT: High-Power MLP CV — CCS vs Weather")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    bare = run_condition(model, tokenizer, "bare", PROMPTS)
    ccs = run_condition(model, tokenizer, "ccs", PROMPTS, prefix=CCS_PREAMBLE)
    weather = run_condition(model, tokenizer, "weather", PROMPTS, prefix=LENGTH_CONTROL)

    print(f"\n{'='*60}")
    print(f"RESULTS (30 prompts)")
    print(f"{'='*60}")

    for metric_name, metric_key in [("σ₂ profile", "mean_sigma2"), ("ratio", "mean_ratio"),
                                     ("σ₂ CV", "cv_sigma2"), ("ratio CV", "cv_ratio")]:
        bc = compare(bare[metric_key], ccs[metric_key])
        bw = compare(bare[metric_key], weather[metric_key])
        cw = compare(ccs[metric_key], weather[metric_key])
        print(f"\n  {metric_name}:")
        print(f"    bare↔CCS:     r={bc['pearson_r']:.4f}")
        print(f"    bare↔weather: r={bw['pearson_r']:.4f}")
        print(f"    CCS↔weather:  r={cw['pearson_r']:.4f}")

    # Zone-specific CV analysis
    print(f"\n{'='*60}")
    print(f"ZONE-SPECIFIC CV ANALYSIS")
    print(f"{'='*60}")
    zones = [("Early", 0, 15), ("Transition", 15, 21), ("Responsive", 21, 29), ("Relay", 29, 32)]
    for zone_name, start, end in zones:
        bare_cv = np.mean(bare["cv_sigma2"][start:end])
        ccs_cv = np.mean(ccs["cv_sigma2"][start:end])
        weather_cv = np.mean(weather["cv_sigma2"][start:end])
        print(f"  {zone_name} (L{start}-{end-1}): bare={bare_cv:.4f}, CCS={ccs_cv:.4f}, weather={weather_cv:.4f}")
        if bare_cv > 0:
            print(f"    CCS/bare={ccs_cv/bare_cv:.3f}, weather/bare={weather_cv/bare_cv:.3f}")

    output = {
        "experiment": "mlp_cv_highpower",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "num_prompts": len(PROMPTS),
        "conditions": {"bare": bare, "ccs": ccs, "weather": weather},
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_mlp_cv_hp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
