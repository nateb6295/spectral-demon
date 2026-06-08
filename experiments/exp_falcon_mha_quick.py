#!/usr/bin/env python3
"""Quick: Falcon 7B base vs instruct — MHA contrast for RLHF invisibility."""

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


def get_sigma_profile(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
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
    print("Falcon 7B MHA: RLHF Invisibility Test")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    base = run_model("tiiuae/falcon-7b", "Falcon-7B_base", PROMPTS)
    instruct = run_model("tiiuae/falcon-7b-instruct", "Falcon-7B_instruct", PROMPTS)

    sigma2_cmp = compare(base["mean_sigma2"], instruct["mean_sigma2"])
    ratio_cmp = compare(base["mean_ratio"], instruct["mean_ratio"])
    cv_cmp = compare(base["cv_sigma2"], instruct["cv_sigma2"])

    print(f"\n{'='*60}")
    print("FALCON 7B (MHA+LayerNorm) — base↔instruct:")
    print(f"  σ₂ profile:  r={sigma2_cmp['pearson_r']:.4f}, cos={sigma2_cmp['cosine_similarity']:.4f}, L2={sigma2_cmp['l2_distance']:.4f}")
    print(f"  σ₂/σ₁ ratio: r={ratio_cmp['pearson_r']:.4f}, cos={ratio_cmp['cosine_similarity']:.4f}, L2={ratio_cmp['l2_distance']:.4f}")
    print(f"  CV profile:  r={cv_cmp['pearson_r']:.4f}")

    print(f"\nComparison with GQA results:")
    print(f"  Mistral 7B (GQA):  σ₂ r=0.9507, ratio r=0.9479")
    print(f"  Qwen 2.5 7B (GQA): ratio r=0.9996")
    print(f"  Gemma 2 2B (GQA):  σ₂ r=0.9102, ratio r=0.9199")
    print(f"  Falcon 7B (MHA):   σ₂ r={sigma2_cmp['pearson_r']:.4f}, ratio r={ratio_cmp['pearson_r']:.4f}")

    output = {
        "experiment": "falcon_mha_rlhf_invisibility",
        "timestamp": datetime.now().isoformat(),
        "arch": "MHA+LayerNorm",
        "base": base,
        "instruct": instruct,
        "sigma2_comparison": sigma2_cmp,
        "ratio_comparison": ratio_cmp,
        "cv_comparison": cv_cmp,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_falcon_mha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
