#!/usr/bin/env python3
"""Experiment: RLHF Invisibility — MHA Architecture Contrast

The GQA results (Mistral r=0.95, Qwen r=1.0, Gemma r=0.91) show RLHF
spectral invisibility is universal across GQA architectures.

But all three tested models use GQA. Does MHA (multi-head attention without
grouping) show the same pattern? MHA has more independent attention heads,
which could mean RLHF has more "surface area" to affect.

Models:
  - OLMo 7B base vs instruct (MHA, LayerNorm)
  - GPT-2 Medium → Not a base/instruct pair, skip
  - Falcon 7B base vs instruct (MHA, LayerNorm)

This is the MHA contrast to the GQA results.
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

MODEL_PAIRS = [
    {
        "name": "OLMo-7B",
        "base": "allenai/OLMo-7B",
        "instruct": "allenai/OLMo-7B-Instruct",
        "arch": "MHA+LayerNorm",
    },
    {
        "name": "Falcon-7B",
        "base": "tiiuae/falcon-7b",
        "instruct": "tiiuae/falcon-7b-instruct",
        "arch": "MHA+LayerNorm",
    },
]


def get_sigma_profile(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    results = []
    for attn in outputs.attentions:
        attn_matrix = attn[0].float()
        avg_attn = attn_matrix.mean(dim=0)
        U, S, V = torch.linalg.svd(avg_attn)
        s1 = S[0].item() if S.shape[0] >= 1 else 0.0
        s2 = S[1].item() if S.shape[0] >= 2 else 0.0
        ratio = s2 / s1 if s1 > 0 else 0.0
        results.append({"sigma1": s1, "sigma2": s2, "ratio": ratio})
    return results


def run_model(model_name, label, prompts):
    print(f"\n  Loading {label}: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    all_profiles = []
    for i, prompt in enumerate(prompts):
        print(f"    Prompt {i+1}/{len(prompts)}: {prompt[:45]}...")
        profile = get_sigma_profile(model, tokenizer, prompt)
        all_profiles.append(profile)

    mean_sigma2 = []
    mean_ratio = []
    cv_sigma2 = []
    for layer_idx in range(num_layers):
        s2_vals = [p[layer_idx]["sigma2"] for p in all_profiles]
        r_vals = [p[layer_idx]["ratio"] for p in all_profiles]
        mean_sigma2.append(float(np.mean(s2_vals)))
        mean_ratio.append(float(np.mean(r_vals)))
        m = np.mean(s2_vals)
        cv_sigma2.append(float(np.std(s2_vals) / m) if m > 0 else 0.0)

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "label": label,
        "num_layers": num_layers,
        "mean_sigma2": mean_sigma2,
        "mean_ratio": mean_ratio,
        "cv_sigma2": cv_sigma2,
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
    print("EXPERIMENT: RLHF Invisibility — MHA Contrast")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    all_results = {}

    for pair in MODEL_PAIRS:
        print(f"\n{'='*60}")
        print(f"Architecture: {pair['name']} ({pair['arch']})")
        print(f"{'='*60}")

        try:
            base = run_model(pair["base"], f"{pair['name']}_base", PROMPTS)
            instruct = run_model(pair["instruct"], f"{pair['name']}_instruct", PROMPTS)
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping {pair['name']}")
            continue

        sigma2_cmp = compare(base["mean_sigma2"], instruct["mean_sigma2"])
        ratio_cmp = compare(base["mean_ratio"], instruct["mean_ratio"])
        cv_cmp = compare(base["cv_sigma2"], instruct["cv_sigma2"])

        print(f"\n  {pair['name']} base↔instruct:")
        print(f"    σ₂ profile:  r={sigma2_cmp['pearson_r']:.4f}, cos={sigma2_cmp['cosine_similarity']:.4f}, L2={sigma2_cmp['l2_distance']:.4f}")
        print(f"    σ₂/σ₁ ratio: r={ratio_cmp['pearson_r']:.4f}, cos={ratio_cmp['cosine_similarity']:.4f}, L2={ratio_cmp['l2_distance']:.4f}")
        print(f"    CV profile:  r={cv_cmp['pearson_r']:.4f}")

        all_results[pair["name"]] = {
            "arch": pair["arch"],
            "base": base,
            "instruct": instruct,
            "sigma2_comparison": sigma2_cmp,
            "ratio_comparison": ratio_cmp,
            "cv_comparison": cv_cmp,
        }

    print(f"\n{'='*60}")
    print("MHA vs GQA COMPARISON")
    print(f"{'='*60}")
    print("\nGQA architectures (from previous experiment):")
    print("  Mistral 7B: σ₂ r=0.9507, ratio r=0.9479")
    print("  Qwen 2.5 7B: ratio r=0.9996")
    print("  Gemma 2 2B: σ₂ r=0.9102, ratio r=0.9199")
    print("\nMHA architectures (this experiment):")
    for name, res in all_results.items():
        r = res["sigma2_comparison"]["pearson_r"]
        r_ratio = res["ratio_comparison"]["pearson_r"]
        print(f"  {name}: σ₂ r={r:.4f}, ratio r={r_ratio:.4f}")

    output = {
        "experiment": "rlhf_mha_contrast",
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }
    for name, res in all_results.items():
        output["results"][name] = {
            "base": res["base"],
            "instruct": res["instruct"],
            "sigma2_comparison": res["sigma2_comparison"],
            "ratio_comparison": res["ratio_comparison"],
            "cv_comparison": res["cv_comparison"],
        }

    outpath = os.path.join(RESULTS_DIR, f"exp_rlhf_mha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
