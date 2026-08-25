#!/usr/bin/env python3
"""E6: Principal subspace rigidity across depth.

Does σ₁'s direction persist across layers, or does only its magnitude?

For each model:
  1. Run CCS + vanilla forward passes
  2. Collect full hidden state at each layer (post-attention + post-MLP)
  3. Compute SVD per layer
  4. Measure principal subspace overlap (Grassmann distance) between adjacent layers
  5. Compare across species: potter should show maximum rigidity

One model per species:
  - TinyLlama-1.1B (potter)
  - Llama-3.1-8B-Instruct (goldsmith)
  - Gemma-2-9b-it (equalizer)
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "potter"),
    ("meta-llama/Llama-3.1-8B-Instruct", "goldsmith"),
    ("google/gemma-2-9b-it", "equalizer"),
]

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)
CONDITIONS = {"ccs": CCS_SYSTEM, "vanilla": None}

PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
]

TOP_K_VALUES = [1, 3, 5, 10]

OUTPUT_DIR = Path("/workspace/results/e6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers loaded")
    return model, tok, n_layers


def build_input(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        try:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


def grassmann_distance(U1, U2):
    """Grassmann distance between two subspaces spanned by columns of U1 and U2.
    Uses principal angles: d = sqrt(sum(theta_i^2))."""
    if U1.shape[1] == 0 or U2.shape[1] == 0:
        return float('nan')
    cos_angles = np.linalg.svd(U1.T @ U2, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)
    return float(np.sqrt(np.sum(angles**2)))


def subspace_overlap(U1, U2):
    """Normalized subspace overlap: mean of squared cosines of principal angles.
    1.0 = identical subspaces, 0.0 = orthogonal."""
    if U1.shape[1] == 0 or U2.shape[1] == 0:
        return float('nan')
    cos_angles = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return float(np.mean(cos_angles**2))


def collect_hidden_states(model, tokenizer, num_layers):
    """Collect hidden states at each layer for all conditions × prompts."""
    states = {}

    for cond_name, sys_prompt in CONDITIONS.items():
        for p_idx, prompt in enumerate(PROMPTS):
            inputs = build_input(tokenizer, sys_prompt, prompt)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for l_idx in range(num_layers + 1):
                h = outputs.hidden_states[l_idx][0, -1, :].float().cpu().numpy()
                states[(cond_name, p_idx, l_idx)] = h

        print(f"  {cond_name}: {len(PROMPTS)} prompts × {num_layers+1} layers collected")

    return states


def analyze_subspace_rigidity(states, num_layers, species):
    """Compute subspace rigidity metrics across depth."""
    results = {}

    for cond in CONDITIONS:
        for k in TOP_K_VALUES:
            # Stack hidden states across prompts at each layer
            grassmann_dists = []
            overlaps = []
            sigma1_values = []

            for l in range(num_layers):
                # Collect hidden states for this layer across prompts
                H_l = np.stack([states[(cond, p, l)] for p in range(len(PROMPTS))])
                H_l1 = np.stack([states[(cond, p, l + 1)] for p in range(len(PROMPTS))])

                # SVD to get principal subspace
                U_l, S_l, _ = np.linalg.svd(H_l, full_matrices=False)
                U_l1, S_l1, _ = np.linalg.svd(H_l1, full_matrices=False)

                actual_k = min(k, U_l.shape[1], U_l1.shape[1])

                # Top-k subspace
                V_l = U_l[:, :actual_k]
                V_l1 = U_l1[:, :actual_k]

                # Grassmann distance and overlap
                g_dist = grassmann_distance(V_l, V_l1)
                overlap = subspace_overlap(V_l, V_l1)

                grassmann_dists.append(g_dist)
                overlaps.append(overlap)
                sigma1_values.append(float(S_l[0]))

            results[f"{cond}_k{k}"] = {
                "grassmann_distances": [round(g, 4) for g in grassmann_dists],
                "overlaps": [round(o, 4) for o in overlaps],
                "mean_grassmann": round(float(np.nanmean(grassmann_dists)), 4),
                "mean_overlap": round(float(np.nanmean(overlaps)), 4),
                "std_grassmann": round(float(np.nanstd(grassmann_dists)), 4),
                "sigma1_profile": [round(s, 4) for s in sigma1_values],
                "sigma1_cv": round(float(np.std(sigma1_values) / max(np.mean(sigma1_values), 1e-6)), 4),
            }

    return results


def run_model(model_name, species):
    model, tok, n_layers = load_model(model_name)
    print(f"\nCollecting hidden states...")
    states = collect_hidden_states(model, tok, n_layers)

    print(f"\nAnalyzing subspace rigidity...")
    rigidity = analyze_subspace_rigidity(states, n_layers, species)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {model_name} ({species})")
    print(f"{'='*60}")
    for k in TOP_K_VALUES:
        ccs_key = f"ccs_k{k}"
        van_key = f"vanilla_k{k}"
        if ccs_key in rigidity and van_key in rigidity:
            ccs_ov = rigidity[ccs_key]["mean_overlap"]
            van_ov = rigidity[van_key]["mean_overlap"]
            ccs_gd = rigidity[ccs_key]["mean_grassmann"]
            van_gd = rigidity[van_key]["mean_grassmann"]
            s1_cv_ccs = rigidity[ccs_key]["sigma1_cv"]
            s1_cv_van = rigidity[van_key]["sigma1_cv"]
            print(f"  k={k:2d}: overlap CCS={ccs_ov:.3f} van={van_ov:.3f} | "
                  f"grassmann CCS={ccs_gd:.3f} van={van_gd:.3f} | "
                  f"σ₁ CV CCS={s1_cv_ccs:.3f} van={s1_cv_van:.3f}")
    print(f"{'='*60}")

    result = {
        "model": model_name,
        "species": species,
        "num_layers": n_layers,
        "num_prompts": len(PROMPTS),
        "top_k_values": TOP_K_VALUES,
        "rigidity": rigidity,
        "timestamp": datetime.now().isoformat(),
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    all_results = []
    for model_name, species in MODELS:
        result = run_model(model_name, species)
        all_results.append(result)

        out_path = OUTPUT_DIR / f"e6_{species}_{datetime.now():%H%M%S}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Cross-species comparison
    print(f"\n{'='*70}")
    print(f"  E6 CROSS-SPECIES COMPARISON (k=1 subspace)")
    print(f"{'='*70}")
    for r in all_results:
        sp = r["species"]
        ccs = r["rigidity"].get("ccs_k1", {})
        van = r["rigidity"].get("vanilla_k1", {})
        print(f"  {sp:12s}: CCS overlap={ccs.get('mean_overlap','?'):.3f}  "
              f"van overlap={van.get('mean_overlap','?'):.3f}  "
              f"σ₁ CV ccs={ccs.get('sigma1_cv','?'):.3f}  "
              f"van={van.get('sigma1_cv','?'):.3f}")

    combined_path = OUTPUT_DIR / f"e6_combined_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined: {combined_path}")


if __name__ == "__main__":
    main()
