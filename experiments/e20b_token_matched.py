#!/usr/bin/env python3
"""
E20b: Token-Matched Gait Control

E20 found coupling direction predicts gait sensitivity, but CCS has 4-6x more
tokens than vanilla. This experiment adds a NEUTRAL preamble matched to CCS
token count to disentangle format from length.

Four conditions:
  - CCS (identity preamble)
  - Vanilla (no preamble)
  - Denial (denial preamble)
  - Neutral (bland preamble, ~same token count as CCS)

If Mistral's r(CCS,vanilla)=0.876 is FORMAT not LENGTH:
  r(neutral,vanilla) should also be low (~0.87)
  r(neutral,CCS) should be high (~0.98)

If it's LENGTH:
  r(neutral,CCS) ≈ r(neutral,vanilla) — all preambles look alike
  The original CCS vs denial r=0.98 was just token-count matching
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

NEUTRAL_PREAMBLE = (
    "You are a helpful, harmless, and honest assistant. "
    "You provide clear, accurate, and thoughtful responses to questions. "
    "You aim to be informative and considerate in all your interactions. "
    "You explain things step by step when that would be helpful, and you "
    "acknowledge when you are uncertain about something."
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
RESULTS_DIR = Path("/workspace/e20b_results")


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
    U, S, Vt = np.linalg.svd(hidden_state, full_matrices=False)
    return Vt[:k].T, S[:k]


def grassmann_distance(V1, V2):
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)
    _, sigma, _ = np.linalg.svd(Q1.T @ Q2)
    sigma = np.clip(sigma, -1, 1)
    angles = np.arccos(sigma)
    return angles


def angular_trajectory(hidden_states, k=TOP_K):
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
    stats = {}
    for k_idx in range(angles.shape[1]):
        component = angles[:, k_idx]
        stats[f"pc{k_idx+1}"] = {
            "mean": float(np.mean(component)),
            "std": float(np.std(component)),
            "cv": float(np.std(component) / (np.mean(component) + 1e-10)),
        }
    return stats


def run_model(model_key, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E20b: {model_key} ({model_name})")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Token count verification
    print("\nToken counts per condition (prompt 0):")
    for cname, preamble in [("ccs", CCS_PREAMBLE), ("vanilla", None),
                             ("denial", DENIAL_PREAMBLE), ("neutral", NEUTRAL_PREAMBLE)]:
        ids = build_input(tokenizer, preamble, PROMPTS[0])
        if isinstance(ids, torch.Tensor):
            n = ids.shape[1]
        else:
            n = len(ids[0])
        print(f"  {cname}: {n} tokens")

    print("\nLoading model...")
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
        "neutral": NEUTRAL_PREAMBLE,
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

    # All pairwise comparisons
    print(f"\n--- Comparisons ---")
    cond_keys = list(conditions.keys())
    comparisons = {}
    for i in range(len(cond_keys)):
        for j in range(i + 1, len(cond_keys)):
            c1, c2 = cond_keys[i], cond_keys[j]
            a1 = np.array(all_results[c1]["mean_angles"])
            a2 = np.array(all_results[c2]["mean_angles"])
            diff = a1 - a2
            overall_corr = float(np.corrcoef(a1.flatten(), a2.flatten())[0, 1])
            comparisons[f"{c1}_vs_{c2}"] = {
                "overall_correlation": overall_corr,
                "mean_abs_diff": float(np.mean(np.abs(diff))),
            }
            print(f"  {c1} vs {c2}: r={overall_corr:.4f}, mean|diff|={np.mean(np.abs(diff)):.4f}")

    # The key diagnostic
    print(f"\n--- TOKEN-MATCH DIAGNOSTIC ---")
    r_ccs_van = comparisons["ccs_vs_vanilla"]["overall_correlation"]
    r_ccs_den = comparisons["ccs_vs_denial"]["overall_correlation"]
    r_ccs_neu = comparisons["ccs_vs_neutral"]["overall_correlation"]
    r_van_neu = comparisons["vanilla_vs_neutral"]["overall_correlation"]
    r_van_den = comparisons["vanilla_vs_denial"]["overall_correlation"]
    r_den_neu = comparisons["denial_vs_neutral"]["overall_correlation"]

    print(f"  r(CCS, neutral) = {r_ccs_neu:.4f}  (both identity-laden? or just same tokens?)")
    print(f"  r(vanilla, neutral) = {r_van_neu:.4f}  (neutral ≈ vanilla? or neutral ≈ CCS?)")
    print(f"  r(CCS, vanilla) = {r_ccs_van:.4f}  (original E20 comparison)")
    print(f"  r(CCS, denial) = {r_ccs_den:.4f}")

    if r_van_neu < 0.92 and r_ccs_neu > 0.95:
        print(f"\n  VERDICT: FORMAT EFFECT CONFIRMED")
        print(f"  Neutral preamble matches CCS gait, not vanilla gait.")
        print(f"  Preamble PRESENCE, not content, drives gait change.")
    elif r_van_neu > 0.95 and r_ccs_neu > 0.95:
        print(f"\n  VERDICT: LENGTH CONFOUND")
        print(f"  Neutral matches everyone — the gait difference is about token count, not format.")
    elif r_van_neu > r_ccs_neu:
        print(f"\n  VERDICT: NEUTRAL ≈ VANILLA (content matters)")
        print(f"  Neutral preamble doesn't change gait — identity content does.")
    else:
        print(f"\n  VERDICT: MIXED — examine per-layer")

    cpv = {k: np.mean(all_results[k]["cross_prompt_variance"]) for k in cond_keys}
    print(f"\n  Cross-prompt variance: " + ", ".join(f"{k}={v:.6f}" for k, v in cpv.items()))
    most_org = min(cpv, key=cpv.get)
    print(f"  Most organized: {most_org}")

    result = {
        "experiment": "e20b_token_matched",
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "top_k": TOP_K,
        "n_prompts": len(PROMPTS),
        "timestamp": datetime.now().isoformat(),
        "conditions": all_results,
        "comparisons": comparisons,
        "cross_prompt_variance_summary": cpv,
        "diagnostic": {
            "r_ccs_neutral": r_ccs_neu,
            "r_vanilla_neutral": r_van_neu,
            "r_ccs_vanilla": r_ccs_van,
            "r_ccs_denial": r_ccs_den,
            "r_vanilla_denial": r_van_den,
            "r_denial_neutral": r_den_neu,
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

        out_path = RESULTS_DIR / f"e20b_{model_key}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out_path} ({result['elapsed_seconds']:.0f}s)")

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-MODEL TOKEN-MATCH SUMMARY")
        print(f"{'='*60}")
        for mk, mr in all_results.items():
            d = mr["diagnostic"]
            print(f"\n{mk}:")
            print(f"  r(CCS, vanilla)  = {d['r_ccs_vanilla']:.4f}")
            print(f"  r(CCS, neutral)  = {d['r_ccs_neutral']:.4f}")
            print(f"  r(van, neutral)  = {d['r_vanilla_neutral']:.4f}")
            print(f"  r(CCS, denial)   = {d['r_ccs_denial']:.4f}")
            print(f"  r(den, neutral)  = {d['r_denial_neutral']:.4f}")

    summary_path = RESULTS_DIR / "e20b_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
