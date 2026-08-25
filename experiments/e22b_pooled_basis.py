#!/usr/bin/env python3
"""
E22b: Pooled-Basis Pathway Alignment

Addresses Kimi's basis-shift critique of E22: V₂ is condition-specific PCA,
so cross-condition alignment comparisons involve different coordinate systems.

Fix: compute V₂ from POOLED activations (all conditions together), then
project condition-specific activations onto the shared V₂ to measure
condition-specific alignment with architectural targets.

This separates:
  - "Does V₂ direction change across conditions?" (E22 conflation)
  - "Do condition-specific activations project differently onto a fixed probe?" (E22b)

Can run on existing E22 data OR re-extract from models.
"""

import json, sys, os
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = Path(os.environ.get("E22_RESULTS_DIR",
    str(Path(__file__).parent.parent / "results" / "e22")))


def pooled_basis_from_model(model_key, model_name):
    """Re-extract hidden states and compute pooled-basis V₂."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Import preambles and prompts from E22
    sys.path.insert(0, str(Path(__file__).parent))
    from e22_mlp_pathway_alignment import (
        CCS_PREAMBLE, SKELETON_PREAMBLE, NEUTRAL_PREAMBLE, PROMPTS,
        build_input, extract_hidden_states, get_component_svds, TOP_K
    )

    conditions = {
        "ccs": CCS_PREAMBLE,
        "skeleton": SKELETON_PREAMBLE,
        "neutral": NEUTRAL_PREAMBLE,
        "vanilla": None,
    }

    print(f"Loading {model_key}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map={"": "cuda:0"}, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers")

    # Get component SVDs (same as E22)
    print("  Extracting component SVDs...")
    (lm_V, lm_S), mlp_svds, attn_svds = get_component_svds(model, TOP_K)

    # Collect ALL hidden states across ALL conditions
    print("  Collecting hidden states (all conditions)...")
    all_hidden = {li: [] for li in range(n_layers + 1)}
    cond_hidden = {}

    for cond_name, preamble in conditions.items():
        cond_states = {li: [] for li in range(n_layers + 1)}
        for pi, prompt in enumerate(PROMPTS):
            input_ids = build_input(tokenizer, preamble, prompt)
            input_ids = input_ids.to(model.device)
            states = extract_hidden_states(model, input_ids)

            for li, h in enumerate(states):
                last_tok = h[-1:, :]  # last token
                all_hidden[li].append(last_tok)
                cond_states[li].append(last_tok)

        cond_hidden[cond_name] = cond_states
        print(f"    {cond_name}: {len(PROMPTS)} prompts done")

    # Compute POOLED V₂ per layer
    print("  Computing pooled V₂ per layer...")
    pooled_v2 = {}
    for li in range(n_layers + 1):
        stacked = np.vstack(all_hidden[li]).astype(np.float64)
        try:
            U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
            pooled_v2[li] = {"v2": Vt[1], "sigma1": float(S[0]), "sigma2": float(S[1])}
        except np.linalg.LinAlgError:
            pooled_v2[li] = None

    # Measure condition-specific projections onto SHARED V₂
    print("  Computing condition-specific alignment on shared basis...")
    results = {"model": model_key, "n_layers": n_layers, "method": "pooled_basis"}
    cond_results = {}

    for cond_name in conditions:
        lm_profile, mlp_profile, attn_profile, resid_profile = [], [], [], []
        s2_profile = []

        for li in range(1, n_layers + 1):  # skip embedding layer
            if pooled_v2[li] is None:
                lm_profile.append(0.0)
                mlp_profile.append(0.0)
                attn_profile.append(0.0)
                resid_profile.append(0.0)
                s2_profile.append(0.0)
                continue

            v2 = pooled_v2[li]["v2"]

            # Condition-specific σ₂ projected onto shared V₂
            cond_stack = np.vstack(cond_hidden[cond_name][li]).astype(np.float64)
            projections = cond_stack @ v2
            s2_profile.append(float(np.std(projections)))

            # Shared V₂ alignment with architectural targets
            if lm_V is not None:
                cosines = [float(np.abs(np.dot(v2, lm_V[:, j])))
                          for j in range(min(TOP_K, lm_V.shape[1]))]
                lm_profile.append(float(max(cosines)))
            else:
                lm_profile.append(0.0)

            mlp_idx = min(li - 1, len(mlp_svds) - 1)
            mlp_V, _ = mlp_svds[mlp_idx]
            if mlp_V is not None and v2.shape[0] == mlp_V.shape[0]:
                cosines = [float(np.abs(np.dot(v2, mlp_V[:, j])))
                          for j in range(min(TOP_K, mlp_V.shape[1]))]
                mlp_profile.append(float(max(cosines)))
            else:
                mlp_profile.append(0.0)

            attn_V, _ = attn_svds[mlp_idx]
            if attn_V is not None and v2.shape[0] == attn_V.shape[0]:
                cosines = [float(np.abs(np.dot(v2, attn_V[:, j])))
                          for j in range(min(TOP_K, attn_V.shape[1]))]
                attn_profile.append(float(max(cosines)))
            else:
                attn_profile.append(0.0)

            if li < n_layers:
                v2_next = pooled_v2[li + 1]["v2"] if pooled_v2[li + 1] else v2
                resid_profile.append(float(np.abs(np.dot(v2, v2_next))))
            else:
                resid_profile.append(0.0)

        cond_results[cond_name] = {
            "lm_alignment_profile": lm_profile,
            "mlp_alignment_profile": mlp_profile,
            "attn_alignment_profile": attn_profile,
            "residual_alignment_profile": resid_profile,
            "sigma2_projection_profile": s2_profile,
        }

    results["conditions"] = cond_results

    # Pooled V₂ stability across layers
    results["pooled_sigma2_profile"] = [
        pooled_v2[li]["sigma2"] if pooled_v2[li] else 0.0
        for li in range(1, n_layers + 1)
    ]

    # V₂ stability: cosine between pooled and condition-specific V₂
    v2_stability = {}
    for cond_name in conditions:
        cosines_per_layer = []
        for li in range(1, n_layers + 1):
            if pooled_v2[li] is None:
                cosines_per_layer.append(0.0)
                continue
            pv2 = pooled_v2[li]["v2"]
            cond_stack = np.vstack(cond_hidden[cond_name][li]).astype(np.float64)
            try:
                _, _, Vt_c = np.linalg.svd(cond_stack, full_matrices=False)
                cv2 = Vt_c[1]
                cosines_per_layer.append(float(np.abs(np.dot(pv2, cv2))))
            except:
                cosines_per_layer.append(0.0)
        v2_stability[cond_name] = cosines_per_layer
    results["v2_pooled_vs_conditioned"] = v2_stability

    out_path = RESULTS_DIR / f"e22b_{model_key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    MODELS = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "yi9b": "01-ai/Yi-1.5-9B-Chat",
        "qwen25": "Qwen/Qwen2.5-7B-Instruct",
        "llama31": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    }

    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    for key in targets:
        if key in MODELS:
            pooled_basis_from_model(key, MODELS[key])
        elif key == "all":
            for k, v in MODELS.items():
                pooled_basis_from_model(k, v)
