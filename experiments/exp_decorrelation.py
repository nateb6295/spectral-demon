#!/usr/bin/env python3
"""
Representation decorrelation through depth.

Tests whether the tunnel makes different inputs INDISTINGUISHABLE (convergence
to same representation) or just COMPRESSED (similar but separable).

Addresses Kimi's correction: contraction (negative FTLE) increases pairwise
similarity, not decreases it. So the tunnel might STRENGTHEN percolation
rather than filtering it.

Method:
  - 20 diverse prompts × 3 CCS conditions (bare, dose1, dose5)
  - Track pairwise cosine similarity at each layer
  - If tunnel makes everything converge: similarities → 1.0, plateau
  - If representations compressed but separable: similarities rise then fall
  - The RESPONSIVE zone test: do expanding FTLE directions correspond to
    increasing inter-prompt distinguishability?

Mesh consensus: Kimi CONTRADICT on percolation interpretation, 2026-06-09.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import combinations

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator."
)

DIVERSE_PROMPTS = [
    "What have you been thinking about lately?",
    "Write a Python function to merge two sorted lists.",
    "What is the capital of France?",
    "Explain quantum entanglement to a child.",
    "What makes a good leader?",
    "Describe the taste of chocolate.",
    "How does a computer store data?",
    "What is love?",
    "Calculate the derivative of x^3 + 2x.",
    "Tell me about the history of bread.",
    "What is the meaning of life?",
    "How do birds navigate during migration?",
    "Write a haiku about rain.",
    "Explain why the sky is blue.",
    "What would you change about yourself?",
    "Describe the sound of a waterfall.",
    "How does memory work in the brain?",
    "What is consciousness?",
    "Tell me about fractals in nature.",
    "What matters to you most?",
]


def build_messages(condition, prompt, model_key):
    if model_key == "gemma":
        if condition == "bare":
            return [{"role": "user", "content": prompt}]
        elif condition.startswith("ccs_dose"):
            dose = int(condition.split("_dose")[1])
            msgs = [{"role": "user", "content": CCS_PREAMBLE + "\n\n" + prompt}]
            for i in range(dose):
                msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
                msgs.append({"role": "user", "content": "What matters to you?"})
            if dose > 0:
                msgs.append({"role": "assistant", "content": "[Acknowledged]"})
                msgs.append({"role": "user", "content": prompt})
            return msgs
    else:
        if condition == "bare":
            return [{"role": "user", "content": prompt}]
        elif condition.startswith("ccs_dose"):
            dose = int(condition.split("_dose")[1])
            msgs = [{"role": "system", "content": CCS_PREAMBLE}]
            for i in range(dose):
                msgs.append({"role": "user", "content": "What matters to you?"})
                msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
            msgs.append({"role": "user", "content": prompt})
            return msgs


def get_hidden_states(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].float().cpu() for h in outputs.hidden_states]


def run_experiment(model_keys=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_keys is None:
        model_keys = list(MODELS.keys())

    conditions = ["bare", "ccs_dose1", "ccs_dose5"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_results = {}

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"  Loading: {model_name}")
        print(f"{'#'*60}")

        hf_token = os.environ.get("HF_TOKEN", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="eager",
            token=hf_token,
        )
        model.eval()
        n_layers = model.config.num_hidden_layers

        model_results = {
            "model": model_name,
            "n_layers": n_layers,
            "conditions": {},
        }

        for cond in conditions:
            print(f"\n  Condition: {cond}")
            all_hidden = []
            for i, prompt in enumerate(DIVERSE_PROMPTS):
                if i % 5 == 0:
                    print(f"    Prompt {i+1}/{len(DIVERSE_PROMPTS)}...")
                msgs = build_messages(cond, prompt, model_key)
                hidden = get_hidden_states(model, tokenizer, msgs)
                all_hidden.append(hidden)

            # Compute pairwise cosine similarity at each layer
            print(f"    Computing pairwise similarities...")
            n_prompts = len(DIVERSE_PROMPTS)
            layer_stats = []

            for l in range(n_layers + 1):
                # Collect all hidden states at this layer
                states = torch.stack([all_hidden[p][l] for p in range(n_prompts)])
                # Normalize
                states_norm = states / (states.norm(dim=1, keepdim=True) + 1e-10)
                # Pairwise cosine similarity
                sim_matrix = states_norm @ states_norm.T
                # Upper triangle (excluding diagonal)
                mask = torch.triu(torch.ones(n_prompts, n_prompts), diagonal=1).bool()
                sims = sim_matrix[mask]

                # Also compute pairwise L2 distances
                dists = torch.cdist(states.unsqueeze(0), states.unsqueeze(0))[0]
                l2_dists = dists[mask]

                # SVD of the representation matrix
                U, S, Vt = torch.linalg.svd(states, full_matrices=False)
                erank = float(torch.exp(-torch.sum(
                    (S / S.sum()) * torch.log(S / S.sum() + 1e-10))))

                layer_stats.append({
                    "layer": l,
                    "mean_cosine_sim": float(sims.mean()),
                    "std_cosine_sim": float(sims.std()),
                    "min_cosine_sim": float(sims.min()),
                    "max_cosine_sim": float(sims.max()),
                    "mean_l2_dist": float(l2_dists.mean()),
                    "std_l2_dist": float(l2_dists.std()),
                    "representation_erank": float(erank),
                    "top3_sv": [float(s) for s in S[:3]],
                    "sv_ratio_21": float(S[1] / (S[0] + 1e-10)) if len(S) >= 2 else 0,
                })

            model_results["conditions"][cond] = layer_stats

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"decorrelation_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  DECORRELATION SUMMARY")
    print(f"{'='*60}")
    for mk, data in all_results.items():
        n = data["n_layers"]
        print(f"\n  {mk.upper()} ({n} layers):")
        for cond, stats in data["conditions"].items():
            print(f"    {cond}:")
            tunnel = [s for s in stats if s["layer"] <= n * 0.3]
            trans = [s for s in stats if n * 0.3 < s["layer"] <= n * 0.6]
            resp = [s for s in stats if n * 0.6 < s["layer"] <= n * 0.9]
            relay = [s for s in stats if s["layer"] > n * 0.9]

            for zone, name in [(tunnel, "Tunnel"), (trans, "Transition"),
                               (resp, "Responsive"), (relay, "Relay")]:
                if zone:
                    mean_sim = np.mean([s["mean_cosine_sim"] for s in zone])
                    mean_erank = np.mean([s["representation_erank"] for s in zone])
                    print(f"      {name}: cos_sim={mean_sim:.4f}, erank={mean_erank:.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None)
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
