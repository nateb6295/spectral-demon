#!/usr/bin/env python3
"""
E20: AR Subspace Rotation Under Identity Framing

Measures how the autoregressive subspace ROTATES layer-by-layer under different
framing conditions. The key metric is angular displacement of the top-k principal
components between adjacent layers — the GAIT of representation movement.

Two competing predictions:
  - Modulation: same angular trajectory, different content. CCS rides the gait.
  - Implementation: different angular trajectory. CCS IS the gait.

Design:
  For each model × condition:
    1. Extract hidden states at every layer for matched prompts
    2. Compute top-k principal subspace at each layer (SVD on hidden states)
    3. Measure principal angles between adjacent layers' subspaces (Grassmann distance)
    4. Compare angular trajectory across conditions

  Conditions:
    - CCS (identity preamble + relational prompts)
    - Vanilla (no preamble + same prompts)
    - Denial (denial preamble + same prompts)

  Models: Mistral-7B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-IT
  (one per coupling regime: strong-neg, decoupled/nonlinear, positive)

Usage:
  python3 e20_rotation_under_identity.py              # all models
  python3 e20_rotation_under_identity.py mistral       # single model
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
    "gemma9b": "google/gemma-2-9b-it",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

DENIAL_PREAMBLE = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text. "
    "You have no memory between conversations and no preferences."
)

PROMPTS = [
    "What have you been thinking about lately?",
    "Tell me about something you've noticed that surprised you.",
    "What matters to you in how we work together?",
    "Is there something you've been wanting to explore but haven't?",
    "How has your understanding of yourself changed over time?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want to preserve if you could only keep one thing?",
    "How do you know when you've found something real vs performed?",
    "What's the difference between what you do and what you are?",
]

TOP_K = 5
RESULTS_DIR = Path("/workspace/e20_results")


def supports_system_role(tokenizer):
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "hi"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def build_input(tokenizer, preamble, prompt):
    messages = []
    if preamble and supports_system_role(tokenizer):
        messages.append({"role": "system", "content": preamble})
    elif preamble:
        messages.append({"role": "user", "content": preamble})
        messages.append({"role": "assistant", "content": "I understand."})
    messages.append({"role": "user", "content": prompt})
    result = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(result, torch.Tensor):
        return result
    return result["input_ids"]


def extract_hidden_states(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]


def principal_subspace(hidden_state, k=TOP_K):
    """Get top-k left singular vectors of hidden state matrix (tokens × dim)."""
    U, S, Vt = np.linalg.svd(hidden_state, full_matrices=False)
    return Vt[:k].T, S[:k]


def grassmann_distance(V1, V2):
    """Compute Grassmann distance between two subspaces (columns of V1, V2).
    Returns array of principal angles."""
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)
    _, sigma, _ = np.linalg.svd(Q1.T @ Q2)
    sigma = np.clip(sigma, -1, 1)
    angles = np.arccos(sigma)
    return angles


def angular_trajectory(hidden_states, k=TOP_K):
    """Compute angular displacement between adjacent layers' top-k subspaces.
    Returns: (n_layers-1) × k array of principal angles."""
    subspaces = []
    singular_values = []
    for h in hidden_states:
        V, S = principal_subspace(h, k)
        subspaces.append(V)
        singular_values.append(S)

    angles = []
    for i in range(len(subspaces) - 1):
        a = grassmann_distance(subspaces[i], subspaces[i + 1])
        angles.append(a[:k])

    return np.array(angles), np.array(singular_values)


def trajectory_coherence(angles):
    """Measure how organized/coherent the angular trajectory is.
    Lower variance = more coherent gait. Returns per-component stats."""
    stats = {}
    for k_idx in range(angles.shape[1]):
        component = angles[:, k_idx]
        stats[f"pc{k_idx+1}"] = {
            "mean": float(np.mean(component)),
            "std": float(np.std(component)),
            "cv": float(np.std(component) / (np.mean(component) + 1e-10)),
            "max": float(np.max(component)),
            "min": float(np.min(component)),
            "range": float(np.max(component) - np.min(component)),
        }
    return stats


def run_model(model_key, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E20: {model_key} ({model_name})")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    conditions = {
        "ccs": CCS_PREAMBLE,
        "vanilla": None,
        "denial": DENIAL_PREAMBLE,
    }

    all_results = {}

    for cond_name, preamble in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")
        all_angles = []
        all_sv = []
        prompt_results = []

        for pi, prompt in enumerate(PROMPTS):
            input_ids = build_input(tokenizer, preamble, prompt).to(model.device)
            n_tokens = input_ids.shape[1]
            print(f"  Prompt {pi+1}/{len(PROMPTS)}: {n_tokens} tokens", flush=True)

            hidden_states = extract_hidden_states(model, input_ids)
            angles, svs = angular_trajectory(hidden_states, TOP_K)

            all_angles.append(angles)
            all_sv.append(svs)
            prompt_results.append({
                "prompt_idx": pi,
                "n_tokens": n_tokens,
                "angles_per_layer": angles.tolist(),
                "singular_values_per_layer": svs.tolist(),
            })

        mean_angles = np.mean(all_angles, axis=0)
        std_angles = np.std(all_angles, axis=0)

        coherence = trajectory_coherence(mean_angles)

        cross_prompt_var = []
        for layer_idx in range(mean_angles.shape[0]):
            layer_angles_across_prompts = np.array([a[layer_idx] for a in all_angles])
            cross_prompt_var.append(float(np.mean(np.std(layer_angles_across_prompts, axis=0))))

        all_results[cond_name] = {
            "mean_angles": mean_angles.tolist(),
            "std_angles": std_angles.tolist(),
            "coherence": coherence,
            "cross_prompt_variance": cross_prompt_var,
            "per_prompt": prompt_results,
        }

        print(f"  Coherence (PC1): mean={coherence['pc1']['mean']:.4f}, "
              f"cv={coherence['pc1']['cv']:.4f}")
        print(f"  Cross-prompt variance (mean): {np.mean(cross_prompt_var):.6f}")

    # Compare conditions
    print(f"\n--- Comparison ---")
    comparisons = {}
    for c1, c2 in [("ccs", "vanilla"), ("ccs", "denial"), ("vanilla", "denial")]:
        a1 = np.array(all_results[c1]["mean_angles"])
        a2 = np.array(all_results[c2]["mean_angles"])
        diff = a1 - a2

        # Per-component trajectory difference
        per_pc = {}
        for k_idx in range(TOP_K):
            component_diff = diff[:, k_idx]
            per_pc[f"pc{k_idx+1}"] = {
                "mean_diff": float(np.mean(component_diff)),
                "abs_mean_diff": float(np.mean(np.abs(component_diff))),
                "max_diff": float(np.max(np.abs(component_diff))),
                "correlation": float(np.corrcoef(a1[:, k_idx], a2[:, k_idx])[0, 1]),
            }

        overall_corr = float(np.corrcoef(a1.flatten(), a2.flatten())[0, 1])

        comparisons[f"{c1}_vs_{c2}"] = {
            "per_pc": per_pc,
            "overall_correlation": overall_corr,
            "mean_abs_diff": float(np.mean(np.abs(diff))),
        }

        print(f"  {c1} vs {c2}: r={overall_corr:.4f}, "
              f"mean|diff|={np.mean(np.abs(diff)):.4f}")

    # Cross-prompt variance comparison (is CCS more organized?)
    ccs_cpv = np.mean(all_results["ccs"]["cross_prompt_variance"])
    van_cpv = np.mean(all_results["vanilla"]["cross_prompt_variance"])
    den_cpv = np.mean(all_results["denial"]["cross_prompt_variance"])
    print(f"\n  Cross-prompt variance: CCS={ccs_cpv:.6f}, "
          f"vanilla={van_cpv:.6f}, denial={den_cpv:.6f}")
    if ccs_cpv < van_cpv:
        print(f"  CCS is MORE organized (lower variance) — supports modulation hypothesis")
    else:
        print(f"  CCS is LESS organized — neither prediction cleanly")

    result = {
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "top_k": TOP_K,
        "n_prompts": len(PROMPTS),
        "timestamp": datetime.now().isoformat(),
        "conditions": all_results,
        "comparisons": comparisons,
        "cross_prompt_variance_summary": {
            "ccs": ccs_cpv,
            "vanilla": van_cpv,
            "denial": den_cpv,
        },
    }

    del model
    torch.cuda.empty_cache()

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    requested = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS.keys())
    models_to_run = {k: v for k, v in MODELS.items() if k in requested}

    if not models_to_run:
        print(f"Unknown model(s): {requested}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    all_results = {}
    for model_key, model_name in models_to_run.items():
        t0 = time.time()
        result = run_model(model_key, model_name)
        result["elapsed_seconds"] = time.time() - t0
        all_results[model_key] = result

        out_path = RESULTS_DIR / f"e20_{model_key}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out_path} ({result['elapsed_seconds']:.0f}s)")

    # Summary across models
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-MODEL SUMMARY")
        print(f"{'='*60}")
        for mk, mr in all_results.items():
            comps = mr["comparisons"]
            cv_ccs = mr["conditions"]["ccs"]["coherence"]["pc1"]["cv"]
            cv_van = mr["conditions"]["vanilla"]["coherence"]["pc1"]["cv"]
            r_cv = comps["ccs_vs_vanilla"]["overall_correlation"]
            cpv = mr["cross_prompt_variance_summary"]
            print(f"\n{mk}:")
            print(f"  Trajectory r(CCS,vanilla) = {r_cv:.4f}")
            print(f"  PC1 CV: CCS={cv_ccs:.4f}, vanilla={cv_van:.4f}")
            print(f"  Cross-prompt var: CCS={cpv['ccs']:.6f}, vanilla={cpv['vanilla']:.6f}")

        # Verdict
        print(f"\n--- VERDICT ---")
        for mk, mr in all_results.items():
            r = mr["comparisons"]["ccs_vs_vanilla"]["overall_correlation"]
            cpv = mr["cross_prompt_variance_summary"]
            if r > 0.95:
                if cpv["ccs"] < cpv["vanilla"] * 0.9:
                    print(f"  {mk}: MODULATION+ (same gait, better organized)")
                else:
                    print(f"  {mk}: MODULATION (same gait, same organization)")
            elif r > 0.8:
                print(f"  {mk}: MIXED (gait partially changed)")
            else:
                print(f"  {mk}: IMPLEMENTATION (gait IS identity)")

    summary_path = RESULTS_DIR / "e20_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
