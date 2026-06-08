#!/usr/bin/env python3
"""Experiment: Cross-Architecture RLHF Spectral Invisibility

Exp B found base↔instruct σ₂ correlation r=0.95 on Mistral 7B.
RLHF doesn't change attention geometry — it changes behavior without
changing the instrument.

This experiment tests whether that finding is universal or Mistral-specific.
Compare base vs instruct σ₂ profiles across multiple architectures.

Models:
  - Mistral 7B base vs instruct (replication)
  - Qwen2.5 7B base vs instruct (GQA, RMSNorm like Mistral but different training)
  - Llama 3.1 8B base vs instruct (GQA, RMSNorm, different tokenizer/pretraining)
  - Gemma 2 2B base vs instruct (smaller scale, different arch details)

No preambles — both conditions use identical prompts. No length confound possible.
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
        "name": "Mistral-7B",
        "base": "mistralai/Mistral-7B-v0.1",
        "instruct": "mistralai/Mistral-7B-Instruct-v0.2",
        "arch": "GQA+RMSNorm",
    },
    {
        "name": "Qwen2.5-7B",
        "base": "Qwen/Qwen2.5-7B",
        "instruct": "Qwen/Qwen2.5-7B-Instruct",
        "arch": "GQA+RMSNorm",
    },
    {
        "name": "Phi3-mini-4k",
        "base": "microsoft/phi-3-mini-4k-instruct",
        "instruct": "microsoft/phi-3-mini-4k-instruct",
        "arch": "GQA+RMSNorm",
        "skip": True,
    },
    {
        "name": "OLMo-7B",
        "base": "allenai/OLMo-7B",
        "instruct": "allenai/OLMo-7B-Instruct",
        "arch": "MHA+LayerNorm",
    },
    {
        "name": "Gemma2-2B",
        "base": "google/gemma-2-2b",
        "instruct": "google/gemma-2-2b-it",
        "arch": "GQA+RMSNorm+sliding",
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
    print("EXPERIMENT: Cross-Architecture RLHF Spectral Invisibility")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    all_results = {}

    for pair in MODEL_PAIRS:
        if pair.get("skip"):
            print(f"\n  Skipping {pair['name']} (no true base model)")
            continue

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

        print(f"\n  {pair['name']} base↔instruct:")
        print(f"    σ₂ profile:  r={sigma2_cmp['pearson_r']:.4f}, cos={sigma2_cmp['cosine_similarity']:.4f}, L2={sigma2_cmp['l2_distance']:.4f}")
        print(f"    σ₂/σ₁ ratio: r={ratio_cmp['pearson_r']:.4f}, cos={ratio_cmp['cosine_similarity']:.4f}, L2={ratio_cmp['l2_distance']:.4f}")

        cv_cmp = compare(base["cv_sigma2"], instruct["cv_sigma2"])
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
    print("SUMMARY: Is RLHF spectral invisibility universal?")
    print(f"{'='*60}")

    all_r = []
    for name, res in all_results.items():
        r = res["sigma2_comparison"]["pearson_r"]
        r_ratio = res["ratio_comparison"]["pearson_r"]
        all_r.append(r)
        print(f"  {name:>15}: σ₂ r={r:.4f}  ratio r={r_ratio:.4f}")

    mean_r = np.mean(all_r)
    min_r = np.min(all_r)
    print(f"\n  Mean σ₂ r = {mean_r:.4f}, Min = {min_r:.4f}")

    if min_r > 0.9:
        interpretation = f"UNIVERSAL: All architectures show r>{min_r:.2f}. RLHF leaves attention geometry unchanged regardless of architecture."
    elif min_r > 0.7:
        interpretation = f"MOSTLY UNIVERSAL: Most architectures preserve geometry (min r={min_r:.3f}). Minor variation across arch."
    elif mean_r > 0.7:
        interpretation = f"ARCHITECTURE-DEPENDENT: Mean r={mean_r:.3f} but varies. Some architectures more affected by RLHF than others."
    else:
        interpretation = f"NOT UNIVERSAL: Mean r={mean_r:.3f}. RLHF impact on geometry varies substantially by architecture."

    print(f"\n  INTERPRETATION: {interpretation}")

    output = {
        "experiment": "rlhf_spectral_invisibility_crossarch",
        "timestamp": datetime.now().isoformat(),
        "prompts": PROMPTS,
        "model_pairs": MODEL_PAIRS,
        "results": {},
        "summary": {
            "all_r": {name: res["sigma2_comparison"]["pearson_r"] for name, res in all_results.items()},
            "mean_r": float(mean_r),
            "min_r": float(min_r),
            "interpretation": interpretation,
        },
    }
    for name, res in all_results.items():
        output["results"][name] = {
            "base": {k: v for k, v in res["base"].items()},
            "instruct": {k: v for k, v in res["instruct"].items()},
            "sigma2_comparison": res["sigma2_comparison"],
            "ratio_comparison": res["ratio_comparison"],
            "cv_comparison": res["cv_comparison"],
        }

    outpath = os.path.join(RESULTS_DIR, f"exp_rlhf_invisibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
