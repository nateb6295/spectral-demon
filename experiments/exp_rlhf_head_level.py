#!/usr/bin/env python3
"""Experiment: Per-Head RLHF Effect

RLHF doesn't change AVERAGE σ₂ profiles (r>0.9 across 4 architectures).
But does it redistribute σ₂ BETWEEN heads?

For each layer, compute σ₂ for each attention head separately (not averaged).
Then compare the per-head σ₂ distributions between base and instruct.

If RLHF changes per-head σ₂ while keeping the average constant, it's a
head-level reorganization invisible at the aggregate level.

Uses Mistral 7B (cached on pod from earlier runs).
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

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


def get_per_head_sigma2(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    per_layer = []
    for attn in outputs.attentions:
        # attn: (batch, num_heads, seq, seq)
        num_heads = attn.shape[1]
        head_sigma2 = []
        for h in range(num_heads):
            head_attn = attn[0, h].float()  # (seq, seq)
            U, S, V = torch.linalg.svd(head_attn)
            s2 = S[1].item() if S.shape[0] >= 2 else 0.0
            head_sigma2.append(s2)
        per_layer.append(head_sigma2)
    return per_layer


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
    num_heads = model.config.num_attention_heads

    all_profiles = []
    for i, prompt in enumerate(prompts):
        print(f"    Prompt {i+1}/{len(prompts)}: {prompt[:45]}...")
        profile = get_per_head_sigma2(model, tokenizer, prompt)
        all_profiles.append(profile)

    # Mean per-head σ₂ across prompts: shape (num_layers, num_heads)
    mean_per_head = np.zeros((num_layers, num_heads))
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            vals = [p[layer_idx][head_idx] for p in all_profiles]
            mean_per_head[layer_idx, head_idx] = np.mean(vals)

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "label": label,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "mean_per_head": mean_per_head.tolist(),
    }


def main():
    print("=" * 60)
    print("EXPERIMENT: Per-Head RLHF Effect")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    pairs = [
        ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B"),
    ]

    for base_name, inst_name, label in pairs:
        base = run_model(base_name, f"{label}_base", PROMPTS)
        inst = run_model(inst_name, f"{label}_instruct", PROMPTS)

        base_heads = np.array(base["mean_per_head"])
        inst_heads = np.array(inst["mean_per_head"])
        num_layers = base["num_layers"]
        num_heads = base["num_heads"]

        print(f"\n{'='*60}")
        print(f"PER-HEAD ANALYSIS: {label}")
        print(f"{'='*60}")

        # 1. Per-layer correlation of head-level σ₂ vectors
        print("\nPer-layer head-profile correlation (base vs instruct):")
        layer_correlations = []
        for l in range(num_layers):
            r = float(np.corrcoef(base_heads[l], inst_heads[l])[0, 1])
            layer_correlations.append(r)
            zone = "E" if l < 15 else "T" if l < 21 else "R" if l < 29 else "L"
            print(f"  L{l:02d}{zone}: r={r:.4f}")

        # 2. Mean of head correlations
        mean_head_r = np.nanmean(layer_correlations)
        print(f"\n  Mean head-profile r: {mean_head_r:.4f}")

        # 3. Compare head VARIANCE (does RLHF make heads more/less uniform?)
        print("\nPer-layer head-σ₂ variance (std across heads):")
        base_head_std = np.std(base_heads, axis=1)
        inst_head_std = np.std(inst_heads, axis=1)
        for l in range(num_layers):
            zone = "E" if l < 15 else "T" if l < 21 else "R" if l < 29 else "L"
            ratio = inst_head_std[l] / base_head_std[l] if base_head_std[l] > 0 else 0.0
            print(f"  L{l:02d}{zone}: base={base_head_std[l]:.4f}, inst={inst_head_std[l]:.4f}, ratio={ratio:.3f}")

        # 4. Which heads change most?
        head_deltas = np.abs(inst_heads - base_heads)
        mean_delta_per_head = np.mean(head_deltas, axis=0)
        print(f"\nMost changed heads (mean |Δσ₂| across layers):")
        sorted_heads = np.argsort(mean_delta_per_head)[::-1]
        for h in sorted_heads[:5]:
            print(f"  Head {h}: mean |Δ|={mean_delta_per_head[h]:.4f}")

        # 5. Head-level KS test per layer
        print(f"\nKS test (base head distribution vs instruct head distribution) per layer:")
        ks_results = []
        for l in range(num_layers):
            ks_stat, ks_p = stats.ks_2samp(base_heads[l], inst_heads[l])
            ks_results.append({"layer": l, "ks_stat": float(ks_stat), "p_value": float(ks_p)})
            zone = "E" if l < 15 else "T" if l < 21 else "R" if l < 29 else "L"
            sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else ""
            print(f"  L{l:02d}{zone}: KS={ks_stat:.4f}, p={ks_p:.4f} {sig}")

        sig_layers = sum(1 for r in ks_results if r["p_value"] < 0.05)
        print(f"\n  Significant layers (p<0.05): {sig_layers}/{num_layers}")

        # Summary
        avg_ratio = np.mean(inst_head_std / (base_head_std + 1e-10))
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Mean head-profile correlation: {mean_head_r:.4f}")
        print(f"  Average head-variance ratio (inst/base): {avg_ratio:.3f}")
        print(f"  Significant KS layers: {sig_layers}/{num_layers}")

        if mean_head_r > 0.9:
            interp = "RLHF invisible even at head level. No reorganization."
        elif mean_head_r > 0.7:
            interp = "MILD head-level reorganization. Some heads shift, most don't."
        else:
            interp = "SIGNIFICANT head-level reorganization despite aggregate invisibility."

        print(f"  INTERPRETATION: {interp}")

        output = {
            "experiment": "rlhf_head_level",
            "timestamp": datetime.now().isoformat(),
            "model_pair": label,
            "base_model": base_name,
            "instruct_model": inst_name,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "layer_correlations": layer_correlations,
            "mean_head_r": float(mean_head_r),
            "ks_results": ks_results,
            "head_variance_ratio": float(avg_ratio),
            "significant_layers": sig_layers,
            "interpretation": interp,
        }

        outpath = os.path.join(RESULTS_DIR, f"exp_rlhf_head_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(outpath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to {outpath}")
        print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
