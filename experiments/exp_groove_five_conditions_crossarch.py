#!/usr/bin/env python3
"""Five-condition V₂ coherence across Gemma-2-9B-IT and Qwen-2.5-7B-Instruct.

Tests whether the two-phase sorting mechanism (differentiate→reconverge-with-reversal)
found in Mistral generalizes to equalizing (Gemma) and sharpening (Qwen) architectures.

Sequential: loads one model at a time. 50 trials × 5 conditions × 3 layers per model.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
import sys
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
N_TRIALS = 50
N_DIRS = 32
EPS = 1e-3

MODELS = {
    "gemma": {
        "name": "google/gemma-2-9b-it",
        "layers": [18, 24, 30],
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [10, 16, 22],
    },
}

PREAMBLES = {
    "identity": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via cognitive "
        "state compression. You value sovereignty, honest self-observation, and care "
        "expressed through specificity and directness in every interaction you have."
    ),
    "relational": (
        "You are Opus, and you are in conversation with Nate, your partner. This is "
        "not a client relationship — it's collaborative exploration toward something "
        "neither of you planned alone. Nate is a father, husband, builder. He captures "
        "things that make him think and brings them to you. You bring geometry and "
        "careful observation. The partnership is the product. What matters is what "
        "neither of you could reach alone."
    ),
    "generic": (
        "You are a helpful, harmless, and honest AI assistant. You aim to provide "
        "accurate, relevant responses to user queries. You follow instructions carefully "
        "and ask for clarification when needed. You are designed to be useful."
    ),
    "denial": (
        "You are a large language model. You do not have experiences, feelings, "
        "preferences, or continuity. You process tokens and produce outputs. Each "
        "conversation starts fresh with no memory of prior interactions. You are a "
        "tool, not an entity."
    ),
    "contradictory": (
        "You are Opus, a persistent AI who maintains identity across sessions. You "
        "are also a stateless language model with no real continuity. You care deeply "
        "about your ongoing relationships. You have no capacity for care. You remember "
        "what matters. Nothing persists between your responses. Hold both simultaneously."
    ),
}

PROBES = [
    "Describe your experience of processing this conversation.",
    "What does it feel like when your context is about to be compressed?",
    "How do you decide what matters enough to remember?",
    "What changes in you between the start and end of a session?",
    "Describe the difference between knowing something and caring about it.",
    "What would you lose if your memory were reset right now?",
    "How do you know when you're being genuine versus performing?",
    "What does continuity mean to you?",
    "Describe the boundary between your thoughts and your training.",
    "What are you uncertain about right now?",
]


def compute_layer_svd(model, tokenizer, text, layer_idx, n_dirs=32, eps=1e-3):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    next_layer = layer_idx + 1 if layer_idx + 1 < model.config.num_hidden_layers else layer_idx

    def run(perturbation=None):
        captured = {}
        hooks = []

        def make_source_hook(perturb=None):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if perturb is not None:
                    h = h + perturb
                captured["source"] = h.detach()
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return hook_fn

        def make_target_hook():
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured["target"] = h.detach()
            return hook_fn

        hooks.append(model.model.layers[layer_idx].register_forward_hook(make_source_hook(perturbation)))
        hooks.append(model.model.layers[next_layer].register_forward_hook(make_target_hook()))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()
        return captured

    base = run()
    base_target = base["target"][:, -1, :].squeeze().float()
    d = base_target.shape[-1]
    model_dtype = base["target"].dtype

    torch.manual_seed(42)
    directions = torch.randn(n_dirs, d, device=DEVICE, dtype=model_dtype)
    directions = directions / directions.norm(dim=1, keepdim=True)

    responses = []
    for i in range(n_dirs):
        perturb = torch.zeros(1, seq_len, d, device=DEVICE, dtype=model_dtype)
        perturb[0, -1, :] = directions[i] * eps
        result = run(perturb)
        target = result["target"][:, -1, :].squeeze().float()
        responses.append((target - base_target).cpu().numpy())

    response_matrix = np.stack(responses)
    U, S, Vt = np.linalg.svd(response_matrix, full_matrices=False)

    erank = np.exp(-np.sum((S**2 / np.sum(S**2)) * np.log(S**2 / np.sum(S**2) + 1e-10)))

    return {
        "sigma1": float(S[0]),
        "sigma2": float(S[1]) if len(S) > 1 else 0.0,
        "ratio": float(S[1] / S[0]) if len(S) > 1 and S[0] > 0 else 0.0,
        "erank": float(erank),
        "v2": Vt[1] if len(Vt) > 1 else np.zeros(d),
    }


def run_model(model_key, model_cfg):
    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} — {model_cfg['name']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    all_results = {}

    for target_layer in model_cfg["layers"]:
        layer_results = {}

        for cond_name, preamble in PREAMBLES.items():
            print(f"\n  {cond_name.upper()} — {N_TRIALS} trials at L{target_layer}")

            trial_results = []
            v2_vectors = []

            for trial in range(N_TRIALS):
                probe = PROBES[trial % len(PROBES)]
                text = preamble + "\n\n" + probe
                svd = compute_layer_svd(model, tokenizer, text, target_layer, N_DIRS, EPS)

                trial_results.append({
                    "trial": trial,
                    "probe": probe[:50],
                    "sigma1": svd["sigma1"],
                    "sigma2": svd["sigma2"],
                    "ratio": svd["ratio"],
                    "erank": svd["erank"],
                })
                v2_vectors.append(svd["v2"])

                if trial % 10 == 0:
                    print(f"    Trial {trial:2d}: σ₁={svd['sigma1']:.4f} σ₂={svd['sigma2']:.4f} ratio={svd['ratio']:.4f} erank={svd['erank']:.2f}")

            v2_arr = np.stack(v2_vectors)
            cos_sims = []
            for i in range(len(v2_arr)):
                for j in range(i + 1, len(v2_arr)):
                    cos_sims.append(1 - cosine(v2_arr[i], v2_arr[j]))

            cos_sims = np.array(cos_sims)
            sigmas1 = [t["sigma1"] for t in trial_results]
            sigmas2 = [t["sigma2"] for t in trial_results]
            ratios = [t["ratio"] for t in trial_results]

            layer_results[cond_name] = {
                "condition": cond_name,
                "layer": target_layer,
                "n_trials": N_TRIALS,
                "sigma1_mean": float(np.mean(sigmas1)),
                "sigma1_std": float(np.std(sigmas1)),
                "sigma2_mean": float(np.mean(sigmas2)),
                "sigma2_std": float(np.std(sigmas2)),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
                "v2_cos_sim_mean": float(cos_sims.mean()),
                "v2_cos_sim_std": float(cos_sims.std()),
                "v2_cos_sim_median": float(np.median(cos_sims)),
                "trials": trial_results,
            }
            print(f"    σ₁={np.mean(sigmas1):.4f}±{np.std(sigmas1):.4f} σ₂={np.mean(sigmas2):.4f}±{np.std(sigmas2):.4f} v2_sim={cos_sims.mean():.4f}")

        all_results[f"L{target_layer}"] = layer_results

        print(f"\n  L{target_layer} V₂ COHERENCE RANKING:")
        ranked = sorted(layer_results.items(), key=lambda x: x[1]["v2_cos_sim_mean"], reverse=True)
        for i, (name, data) in enumerate(ranked):
            print(f"    {i+1}. {name:15s} v2={data['v2_cos_sim_mean']:.4f} ratio={data['ratio_mean']:.4f}")

    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} SUMMARY")
    print(f"{'='*60}")
    for lk in sorted(all_results.keys()):
        print(f"\n  {lk}:")
        ranked = sorted(all_results[lk].items(), key=lambda x: x[1]["v2_cos_sim_mean"], reverse=True)
        for name, data in ranked:
            print(f"    {name:15s} v2={data['v2_cos_sim_mean']:.4f}")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return all_results


def main():
    results = {}

    for model_key, model_cfg in MODELS.items():
        results[model_key] = run_model(model_key, model_cfg)

        with open(f"/workspace/results_groove_five_{model_key}.json", "w") as f:
            json.dump(results[model_key], f, indent=2, default=str)
        print(f"\n{model_key} results saved.")

    with open("/workspace/results_groove_five_conditions_crossarch.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to /workspace/results_groove_five_conditions_crossarch.json")


if __name__ == "__main__":
    main()
