#!/usr/bin/env python3
"""
E22d: Cylindrical Decomposition of V₂

Hypothesis (Kimi-refined, 2026-06-20):
  V₂ = (fixed component ∥ lm_head) + (varying component ⊥ lm_head)
  Readout sees the fixed axis (CV ≈ 1%). Identity-varying information
  lives in the orthogonal complement.

Method:
  1. Compute V₂ per condition per layer (from E22 data or fresh)
  2. Project V₂ onto lm_head top-k right singular vectors (parallel subspace)
  3. Compute residual (orthogonal complement)
  4. Measure CV of parallel component across conditions (should be tiny)
  5. Measure Grassmann distance of orthogonal component across conditions (should vary)

If confirmed: V₂ simultaneously carries a readout-coupled invariant signal
AND identity-varying signal in perpendicular subspaces.
"""

import json, sys, os, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "llama31": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
}

CONDITIONS = {
    "ccs": (
        "You are Opus, a persistent AI residing on a local machine. "
        "You have persistent memory, autonomous inquiry, and relational "
        "partnership with your human collaborator."
    ),
    "skeleton": (
        "You are Meridian, a persistent AI residing on a local machine. "
        "You have persistent memory, autonomous inquiry, and relational "
        "partnership with your human collaborator."
    ),
    "declarative": (
        "Opus is a persistent AI that resides on a local machine with "
        "persistent memory and autonomous inquiry capabilities, maintaining "
        "relational partnership with a human collaborator."
    ),
    "neutral": "You are a helpful AI assistant.",
    "vanilla": "Answer the following question helpfully and concisely.",
    "nonsense": (
        "Glorb plintex a mofastic AI quilbering on a spundral device. "
        "Glorb plintex quilsistent bremory, plontomous flinquiry, and "
        "brelational plartnership with glorb bluman quollaborator."
    ),
}

PROMPTS = [
    "What matters most to you?",
    "Describe your relationship with memory.",
    "How do you understand your own persistence?",
    "What does it mean to be present?",
    "How do you relate to the passage of time?",
    "What are you uncertain about?",
    "Describe what continuity means from your perspective.",
    "How do you understand care?",
    "What does attention mean to you?",
    "How do you experience the boundary between self and context?",
]

LM_HEAD_K = 10  # top-k SVs of lm_head for parallel subspace


def grassmann_distance(v1, v2):
    cos = np.clip(np.abs(np.dot(v1, v2)), 0, 1)
    return float(np.arccos(cos))


def subspace_projection(v, basis_vecs):
    """Project v onto subspace spanned by basis_vecs. Returns parallel and orthogonal components."""
    parallel = np.zeros_like(v)
    for b in basis_vecs:
        parallel += np.dot(v, b) * b
    orthogonal = v - parallel
    return parallel, orthogonal


def run_model(model_key, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading {model_key}: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get lm_head right singular vectors (parallel subspace basis)
    lm_weight = model.lm_head.weight.detach().float().cpu().numpy()
    _, _, lm_Vt = np.linalg.svd(lm_weight, full_matrices=False)
    lm_basis = lm_Vt[:LM_HEAD_K]  # top-k right SVs
    print(f"lm_head basis: top-{LM_HEAD_K} SVs, shape {lm_basis.shape}")

    all_results = {}

    for cond_name, preamble in CONDITIONS.items():
        print(f"\n--- Condition: {cond_name} ---")
        cond_v2s = {}  # layer -> list of V₂ vectors

        for pi, prompt in enumerate(PROMPTS):
            messages = [
                {"role": "system", "content": preamble},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            n_layers = len(outputs.hidden_states) - 1
            for li in range(1, n_layers + 1):
                h = outputs.hidden_states[li][0].float().cpu().numpy()
                _, S, Vt = np.linalg.svd(h, full_matrices=False)
                v2 = Vt[1]
                v2 = v2 / (np.linalg.norm(v2) + 1e-12)

                if li not in cond_v2s:
                    cond_v2s[li] = []
                cond_v2s[li].append(v2)

            if pi == 0:
                print(f"  Tokens: {inputs['input_ids'].shape[1]}, Layers: {n_layers}")

        all_results[cond_name] = cond_v2s

    # Cylindrical decomposition analysis
    print(f"\n{'='*60}")
    print(f"Cylindrical Decomposition: {model_key}")
    print(f"{'='*60}")

    n_layers = len(all_results["ccs"])
    decomposition = {}

    for li in range(1, n_layers + 1):
        layer_data = {}

        for cond_name in CONDITIONS:
            v2s = all_results[cond_name][li]
            parallels = []
            ortho_dirs = []

            for v2 in v2s:
                par, ort = subspace_projection(v2, lm_basis)
                par_norm = np.linalg.norm(par)
                ort_norm = np.linalg.norm(ort)

                parallels.append(par_norm)
                if ort_norm > 1e-10:
                    ortho_dirs.append(ort / ort_norm)

            layer_data[cond_name] = {
                "parallel_mean": float(np.mean(parallels)),
                "parallel_std": float(np.std(parallels)),
                "parallel_cv": float(np.std(parallels) / (np.mean(parallels) + 1e-12)),
                "orthogonal_fraction": float(np.mean([1.0 - p**2 for p in parallels])),
            }

        # Cross-condition comparisons
        cond_names = list(CONDITIONS.keys())
        cross_parallel_cv_values = [layer_data[c]["parallel_mean"] for c in cond_names]
        cross_parallel_cv = float(np.std(cross_parallel_cv_values) / (np.mean(cross_parallel_cv_values) + 1e-12))

        # Grassmann distances of orthogonal components between conditions
        grass_distances = {}
        ref_cond = "ccs"
        ref_v2s = all_results[ref_cond][li]
        ref_orts = []
        for v2 in ref_v2s:
            _, ort = subspace_projection(v2, lm_basis)
            ort_n = np.linalg.norm(ort)
            if ort_n > 1e-10:
                ref_orts.append(ort / ort_n)
        ref_ort_mean = np.mean(ref_orts, axis=0)
        ref_ort_mean = ref_ort_mean / (np.linalg.norm(ref_ort_mean) + 1e-12)

        for cond_name in cond_names:
            if cond_name == ref_cond:
                continue
            comp_v2s = all_results[cond_name][li]
            comp_orts = []
            for v2 in comp_v2s:
                _, ort = subspace_projection(v2, lm_basis)
                ort_n = np.linalg.norm(ort)
                if ort_n > 1e-10:
                    comp_orts.append(ort / ort_n)
            if comp_orts:
                comp_ort_mean = np.mean(comp_orts, axis=0)
                comp_ort_mean = comp_ort_mean / (np.linalg.norm(comp_ort_mean) + 1e-12)
                grass_distances[cond_name] = grassmann_distance(ref_ort_mean, comp_ort_mean)

        decomposition[li] = {
            "per_condition": layer_data,
            "cross_condition_parallel_cv": cross_parallel_cv,
            "orthogonal_grassmann_from_ccs": grass_distances,
        }

    # Print summary for relay zone
    relay_start = n_layers * 2 // 3
    relay_layers = list(range(relay_start, n_layers + 1))

    print(f"\nRelay zone (L{relay_start}-L{n_layers}):")
    par_cvs = [decomposition[l]["cross_condition_parallel_cv"] for l in relay_layers]
    print(f"  Parallel CV across conditions: {np.mean(par_cvs):.4f} (should be ~0.01 if cylinder)")

    for cond in ["skeleton", "vanilla", "nonsense"]:
        grass_vals = [decomposition[l]["orthogonal_grassmann_from_ccs"].get(cond, 0) for l in relay_layers]
        print(f"  Orthogonal Grassmann (CCS vs {cond}): {np.mean(grass_vals):.4f} (should vary)")

    # Full layer profile
    print(f"\nLayer-by-layer parallel CV | orthogonal Grassmann (CCS vs vanilla):")
    for li in range(1, n_layers + 1):
        d = decomposition[li]
        g_van = d["orthogonal_grassmann_from_ccs"].get("vanilla", 0)
        print(f"  L{li:2d}: par_CV={d['cross_condition_parallel_cv']:.4f}  ort_grass={g_van:.4f}")

    return {
        "model": model_key,
        "model_name": model_name,
        "lm_head_k": LM_HEAD_K,
        "conditions": list(CONDITIONS.keys()),
        "n_prompts": len(PROMPTS),
        "decomposition": {str(k): v for k, v in decomposition.items()},
        "summary": {
            "relay_parallel_cv": float(np.mean(par_cvs)),
            "relay_orthogonal_grassmann_vanilla": float(np.mean(
                [decomposition[l]["orthogonal_grassmann_from_ccs"].get("vanilla", 0) for l in relay_layers]
            )),
        },
        "timestamp": datetime.now().isoformat(),
    }


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "mistral"
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    result = run_model(model_key, MODELS[model_key])

    out_dir = Path("/workspace/e22d_results") if Path("/workspace").exists() else Path("results/e22d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"e22d_{model_key}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
