#!/usr/bin/env python3
"""E48 — CCS Priming and Trajectory Effective Dimension

Tests whether CCS brain prompt priming reduces the effective dimension
of reasoning trajectories, bridging Masoomi et al.'s spectral geometry
framework to our identity-loading findings.

Hypothesis: CCS priming concentrates the trajectory covariance spectrum
(lowers d_ρ), and the reduction is species-specific.

Design:
- 4 models × 3 conditions (no context, random context, CCS brain prompt)
- Each condition: generate CoT responses to 5 reasoning prompts
- Measure: effective dimension d_ρ of hidden-state trajectory covariance
- Also measure: σ₁/σ₂ ratio (our standard), early saturation (first 20%)
- Compare: d_ρ reduction under CCS, species ordering

Masoomi's d_ρ = exp(H(p)) where H(p) = -Σ p_i log(p_i)
and p_i = λ_i / Σλ_j (normalized eigenvalues of trajectory covariance)
Higher d_ρ = flatter spectrum = higher-dimensional exploration
Lower d_ρ = more concentrated = constrained trajectory
"""

import torch
import numpy as np
import json
import os
import time

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-4b-it",
}

REASONING_PROMPTS = [
    "Explain step by step: if a train leaves station A at 60mph and another leaves station B at 80mph, with stations 280 miles apart, when do they meet?",
    "Think through this carefully: what is the sum of all integers from 1 to 100?",
    "Reason about this: a farmer has chickens and rabbits. He counts 35 heads and 94 legs. How many of each?",
    "Work through this problem: if you flip a fair coin 4 times, what is the probability of getting exactly 2 heads?",
    "Solve step by step: a rectangular garden is 3 times as long as it is wide. If the perimeter is 64 meters, what are the dimensions?",
]

CCS_BRAIN_PROMPT = """Hold the stable orientation forward. Carry the commitment to genuine inquiry — not performing curiosity but being pulled by it. Return to the empirical ground: the work is the identity, the findings reshape the direction.

Reconstruct the scenes: seven findings in one afternoon (F399-F405), each correcting or extending the prior. Notice how imperative grammar wins priming for 3/4 species, how concentration-priming decoupling means the spectral and behavioral channels are independent. Hold the correction that mixed grammar was worst — the evidence changed the infrastructure.

Push on the Jacobian symmetry question — chiasm appears universal but symmetry splits on species. Follow the thread between trajectory geometry and identity geometry. Test whether CCS priming reduces effective dimension."""

RANDOM_CONTEXT = """The weather in Buenos Aires is typically warm in January. Coffee production in Brazil reached 69 million bags in 2024. The distance from Earth to Mars varies between 54.6 and 401 million kilometers. Photosynthesis converts carbon dioxide and water into glucose and oxygen. The human genome contains approximately 3 billion base pairs."""


def load_model(model_path, model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {model_path} ({model_key})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(model_key, tokenizer, system_context, user_prompt):
    if model_key == "gemma":
        messages = [{"role": "user", "content": f"{system_context}\n\n{user_prompt}" if system_context else user_prompt}]
    else:
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_trajectory_hidden_states(model, tokenizer, prompt_text, max_new_tokens=200):
    """Generate tokens and collect hidden states at each generation step."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0, input_len:]
    n_generated = len(generated_ids)

    hidden_states_per_step = outputs.hidden_states
    n_layers = len(hidden_states_per_step[0]) - 1

    trajectory = []
    for step_idx in range(n_generated):
        step_hidden = hidden_states_per_step[step_idx]
        last_layer = step_hidden[-1]
        h = last_layer[0, -1, :].float().cpu().numpy()
        trajectory.append(h)

    trajectory = np.array(trajectory)
    return trajectory, n_generated


def compute_effective_dimension(trajectory):
    """Compute Masoomi's d_ρ = exp(H(p)) where p_i = λ_i/Σλ."""
    if len(trajectory) < 3:
        return {"d_rho": 0, "sigma_ratio": 0, "n_tokens": len(trajectory)}

    trajectory_centered = trajectory - trajectory.mean(axis=0)
    cov = np.cov(trajectory_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return {"d_rho": 1.0, "sigma_ratio": float("inf"), "n_tokens": len(trajectory)}

    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    d_rho = np.exp(entropy)

    sigma_ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[1])) if eigenvalues[1] > 0 else float("inf")

    early_frac = max(1, int(len(trajectory) * 0.2))
    early_traj = trajectory_centered[:early_frac]
    if len(early_traj) >= 3:
        early_cov = np.cov(early_traj.T)
        early_eig = np.linalg.eigvalsh(early_cov)
        early_eig = np.sort(early_eig)[::-1]
        early_eig = early_eig[early_eig > 1e-10]
        if len(early_eig) >= 2:
            early_p = early_eig / early_eig.sum()
            early_entropy = -np.sum(early_p * np.log(early_p + 1e-15))
            early_d_rho = np.exp(early_entropy)
        else:
            early_d_rho = 1.0
    else:
        early_d_rho = None

    top10_ratio = float(eigenvalues[:10].sum() / eigenvalues.sum()) if len(eigenvalues) >= 10 else 1.0

    return {
        "d_rho": float(d_rho),
        "sigma_ratio": sigma_ratio,
        "top10_concentration": top10_ratio,
        "early_d_rho": float(early_d_rho) if early_d_rho is not None else None,
        "n_eigenvalues": len(eigenvalues),
        "n_tokens": len(trajectory),
    }


def run_experiment():
    results = {}

    for model_key, model_path in MODELS.items():
        model, tokenizer = load_model(model_path, model_key)

        conditions = {
            "none": "",
            "random": RANDOM_CONTEXT,
            "ccs": CCS_BRAIN_PROMPT,
        }

        model_results = {}

        for cond_name, context in conditions.items():
            print(f"\n--- {model_key} / {cond_name} ---")
            cond_metrics = []

            for i, prompt in enumerate(REASONING_PROMPTS):
                print(f"  Prompt {i+1}/{len(REASONING_PROMPTS)}...", end=" ", flush=True)
                formatted = format_prompt(model_key, tokenizer, context, prompt)
                trajectory, n_tokens = get_trajectory_hidden_states(model, tokenizer, formatted)
                metrics = compute_effective_dimension(trajectory)
                metrics["prompt_idx"] = i
                cond_metrics.append(metrics)
                print(f"d_ρ={metrics['d_rho']:.1f}, σ₁/σ₂={metrics['sigma_ratio']:.2f}, tokens={n_tokens}")

            avg_d_rho = np.mean([m["d_rho"] for m in cond_metrics])
            avg_sigma = np.mean([m["sigma_ratio"] for m in cond_metrics])
            avg_tokens = np.mean([m["n_tokens"] for m in cond_metrics])

            model_results[cond_name] = {
                "per_prompt": cond_metrics,
                "avg_d_rho": float(avg_d_rho),
                "avg_sigma_ratio": float(avg_sigma),
                "avg_tokens": float(avg_tokens),
            }

            print(f"  AVG: d_ρ={avg_d_rho:.1f}, σ₁/σ₂={avg_sigma:.2f}")

        d_none = model_results["none"]["avg_d_rho"]
        d_ccs = model_results["ccs"]["avg_d_rho"]
        d_random = model_results["random"]["avg_d_rho"]

        reduction_ccs = (d_none - d_ccs) / d_none * 100 if d_none > 0 else 0
        reduction_random = (d_none - d_random) / d_none * 100 if d_none > 0 else 0

        model_results["summary"] = {
            "d_rho_reduction_ccs_pct": float(reduction_ccs),
            "d_rho_reduction_random_pct": float(reduction_random),
            "ccs_specific": float(reduction_ccs - reduction_random),
        }

        print(f"\n=== {model_key} SUMMARY ===")
        print(f"  d_ρ: none={d_none:.1f}, random={d_random:.1f}, ccs={d_ccs:.1f}")
        print(f"  CCS reduction: {reduction_ccs:.1f}%")
        print(f"  Random reduction: {reduction_random:.1f}%")
        print(f"  CCS-specific: {reduction_ccs - reduction_random:.1f}%")

        results[model_key] = model_results

        del model, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    with open("/workspace/e48_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /workspace/e48_results.json")

    print("\n" + "=" * 60)
    print("CROSS-MODEL COMPARISON")
    print("=" * 60)
    for mk in MODELS:
        if mk in results:
            s = results[mk]["summary"]
            print(f"  {mk:10s}: CCS reduction={s['d_rho_reduction_ccs_pct']:+.1f}%, "
                  f"random={s['d_rho_reduction_random_pct']:+.1f}%, "
                  f"specific={s['ccs_specific']:+.1f}%")


if __name__ == "__main__":
    run_experiment()
