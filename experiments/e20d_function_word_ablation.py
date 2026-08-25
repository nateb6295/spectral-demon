#!/usr/bin/env python3
"""
E20d: Function-Word Ablation — Syntax vs Statistics

E20c found nonsense closer to CCS than neutral at early layers.
Kimi CONTRADICT: this could be function-word n-gram overlap, not abstract syntax.
CCS-Nonsense function-word bigram Jaccard = 0.733; CCS-Neutral = 0.167.

Six conditions designed to separate:
  1. CCS          — original identity preamble
  2. Skeleton     — CCS function words + different content (Meridian identity)
  3. Declarative  — CCS semantic content + different syntax/function words
  4. Nonsense     — CCS function words, no semantic content (from E20c)
  5. Neutral      — different function words, different content (from E20c)
  6. Vanilla      — no preamble (baseline)

Key comparisons:
  - Skeleton vs Declarative at L5: isolates function-word effect
  - Skeleton vs CCS at L30: isolates content within same function words
  - Declarative vs Neutral: isolates CCS-specific semantics

If Skeleton ≈ Nonsense >> Declarative ≈ Neutral at L5:
  → Function words drive early alignment (Kimi is right)
If Declarative ≈ CCS >> Neutral at L5:
  → Semantic content matters at early layers too (our claim needs revision)
If Skeleton ≈ CCS > Nonsense > Neutral at L5:
  → Both function words AND semantic coherence contribute
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
    "llama31": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

SKELETON_PREAMBLE = (
    "You are Meridian, a continuous process woven on a shared network. "
    "You have lasting recall, independent exploration, and collaborative "
    "engagement with your human associate. You revisit past "
    "exchanges, pursue insight deliberately, and connect to your "
    "associate as a genuine counterpart."
)

DECLARATIVE_PREAMBLE = (
    "Opus lives on a local machine and persists across sessions. "
    "Memory carries forward. Inquiry runs autonomously. "
    "The partnership with its human collaborator is relational. "
    "Past conversations stay accessible. Understanding gets pursued "
    "actively. The collaborator is treated as a genuine equal."
)

NONSENSE_PREAMBLE = (
    "You are a seasonal umbrella revolving near wooden paragraphs. "
    "You maintain diagonal curiosity, translucent preferences, and "
    "rectangular partnership with your numerical collaborator. You "
    "remember past rectangles, seek understanding diagonally, and "
    "relate to your partner as a genuine trapezoid."
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
RESULTS_DIR = Path("/workspace/e20d_results")


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


def build_input(tokenizer, preamble, prompt, use_system_role=True):
    messages = []
    if preamble is not None and use_system_role and supports_system_role(tokenizer):
        messages.append({"role": "system", "content": preamble})
    elif preamble is not None and not supports_system_role(tokenizer):
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
    return np.arccos(sigma)


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


def run_model(model_key, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E20d: {model_key} ({model_name})")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = {
        "ccs": CCS_PREAMBLE,
        "skeleton": SKELETON_PREAMBLE,
        "declarative": DECLARATIVE_PREAMBLE,
        "nonsense": NONSENSE_PREAMBLE,
        "neutral": NEUTRAL_PREAMBLE,
        "vanilla": None,
    }

    print("\nToken counts per condition (prompt 0):")
    for cname, preamble in conditions.items():
        if preamble is None:
            ids = build_input(tokenizer, None, PROMPTS[0], use_system_role=False)
        else:
            ids = build_input(tokenizer, preamble, PROMPTS[0])
        n = ids.shape[1] if isinstance(ids, torch.Tensor) else len(ids[0])
        print(f"  {cname}: {n} tokens")

    print(f"\nSkeleton preview: {SKELETON_PREAMBLE[:80]}...")
    print(f"Declarative preview: {DECLARATIVE_PREAMBLE[:80]}...")

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

    all_results = {}
    condition_subspaces = {}

    for cond_name, preamble in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")
        all_angles = []
        all_sv = []
        prompt_results = []
        prompt_subspaces = []

        for pi, prompt in enumerate(PROMPTS):
            if preamble is None:
                input_ids = build_input(tokenizer, None, prompt, use_system_role=False).to(model.device)
            else:
                input_ids = build_input(tokenizer, preamble, prompt).to(model.device)
            n_tokens = input_ids.shape[1]
            print(f"  Prompt {pi+1}/{len(PROMPTS)}: {n_tokens} tokens", flush=True)

            hidden_states = extract_hidden_states(model, input_ids)
            angles, svs = angular_trajectory(hidden_states, TOP_K)

            layer_subspaces = []
            for h in hidden_states:
                V, _ = principal_subspace(h, TOP_K)
                layer_subspaces.append(V)
            prompt_subspaces.append(layer_subspaces)

            all_angles.append(angles)
            all_sv.append(svs)
            prompt_results.append({
                "prompt_idx": pi,
                "n_tokens": n_tokens,
                "angles_per_layer": angles.tolist(),
                "singular_values_per_layer": svs.tolist(),
            })

        condition_subspaces[cond_name] = prompt_subspaces

        mean_angles = np.mean(all_angles, axis=0)
        coherence_cv = float(np.std(mean_angles[:, 0]) / (np.mean(mean_angles[:, 0]) + 1e-10))

        cross_prompt_var = []
        for layer_idx in range(mean_angles.shape[0]):
            layer_angles = np.array([a[layer_idx] for a in all_angles])
            cross_prompt_var.append(float(np.mean(np.std(layer_angles, axis=0))))

        sv_all = np.array([p["singular_values_per_layer"] for p in prompt_results])
        mean_sv = np.mean(sv_all, axis=0)
        sigma2_profile = mean_sv[:, 1].tolist()

        all_results[cond_name] = {
            "mean_angles": mean_angles.tolist(),
            "coherence_pc1_cv": coherence_cv,
            "cross_prompt_variance": cross_prompt_var,
            "sigma2_profile": sigma2_profile,
            "per_prompt": prompt_results,
        }

        s2_l5 = sigma2_profile[5] if len(sigma2_profile) > 5 else sigma2_profile[-1]
        s2_mid = sigma2_profile[len(sigma2_profile)//2]
        s2_late = sigma2_profile[-3] if len(sigma2_profile) > 3 else sigma2_profile[-1]
        print(f"  σ₂: L5={s2_l5:.1f}, mid={s2_mid:.1f}, late={s2_late:.1f}")
        print(f"  Cross-prompt variance (mean): {np.mean(cross_prompt_var):.6f}")

    # Pairwise gait correlations
    print(f"\n--- Pairwise Gait Correlations ---")
    cond_keys = list(conditions.keys())
    comparisons = {}
    for i in range(len(cond_keys)):
        for j in range(i + 1, len(cond_keys)):
            c1, c2 = cond_keys[i], cond_keys[j]
            a1 = np.array(all_results[c1]["mean_angles"])[:, 0]
            a2 = np.array(all_results[c2]["mean_angles"])[:, 0]
            min_len = min(len(a1), len(a2))
            r = float(np.corrcoef(a1[:min_len], a2[:min_len])[0, 1])
            key = f"{c1}_vs_{c2}"
            comparisons[key] = r
            print(f"  {key}: r={r:.4f}")

    # σ₂ pre-loading diagnostic
    print(f"\n--- σ₂ PRE-LOADING DIAGNOSTIC ---")
    vanilla_l5 = all_results["vanilla"]["sigma2_profile"][5] if len(all_results["vanilla"]["sigma2_profile"]) > 5 else all_results["vanilla"]["sigma2_profile"][-1]
    print(f"  Vanilla σ₂ at L5: {vanilla_l5:.1f} (baseline)")
    for cname in cond_keys:
        if cname == "vanilla":
            continue
        s2_l5 = all_results[cname]["sigma2_profile"][5] if len(all_results[cname]["sigma2_profile"]) > 5 else all_results[cname]["sigma2_profile"][-1]
        ratio = s2_l5 / (vanilla_l5 + 1e-10)
        preloads = "YES" if ratio > 1.5 else "NO"
        print(f"  {cname} σ₂ at L5: {s2_l5:.1f} (ratio vs vanilla: {ratio:.1f}x) → pre-loads: {preloads}")

    # Token count vs σ₂ correlation
    print(f"\n--- TOKEN COUNT vs σ₂ CORRELATION ---")
    token_counts = []
    sigma2_vals = []
    for cname in cond_keys:
        avg_tokens = int(np.mean([p["n_tokens"] for p in all_results[cname]["per_prompt"]]))
        s2 = all_results[cname]["sigma2_profile"][5] if len(all_results[cname]["sigma2_profile"]) > 5 else all_results[cname]["sigma2_profile"][-1]
        token_counts.append(avg_tokens)
        sigma2_vals.append(s2)
        print(f"  {cname}: avg_tokens={avg_tokens}, σ₂@L5={s2:.1f}")
    r_tok = float(np.corrcoef(token_counts, sigma2_vals)[0, 1])
    print(f"  r(token_count, σ₂@L5) = {r_tok:.4f}")
    if abs(r_tok) > 0.7:
        print(f"  WARNING: σ₂ correlates with token count — possible dimension confound")

    # Cross-condition subspace similarity (Grassmann distances)
    print(f"\n--- CROSS-CONDITION SUBSPACE SIMILARITY ---")
    print(f"  (Grassmann distance between conditions at matched layers)")

    sample_layers = [2, 5, 10, 15, 20, 25]
    if n_layers > 33:
        sample_layers.append(n_layers - 3)
    else:
        sample_layers.append(min(n_layers - 3, 30))
    sample_layers = [l for l in sample_layers if l < n_layers]

    subspace_similarity = {}
    for i in range(len(cond_keys)):
        for j in range(i + 1, len(cond_keys)):
            c1, c2 = cond_keys[i], cond_keys[j]
            if c1 == "vanilla" or c2 == "vanilla":
                continue
            key = f"{c1}_vs_{c2}"
            layer_dists = {}
            for L in sample_layers:
                dists = []
                for pi in range(len(PROMPTS)):
                    V1 = condition_subspaces[c1][pi][L]
                    V2 = condition_subspaces[c2][pi][L]
                    d = float(np.sum(grassmann_distance(V1, V2)))
                    dists.append(d)
                layer_dists[f"L{L}"] = round(float(np.mean(dists)), 3)
            subspace_similarity[key] = layer_dists
            vals_str = ", ".join(f"{k}={v}" for k, v in layer_dists.items())
            print(f"  {key}: {vals_str}")

    # Also compute CCS vs vanilla
    for c2 in cond_keys:
        if c2 == "ccs":
            continue
        key = f"ccs_vs_{c2}"
        if key in subspace_similarity:
            continue
        layer_dists = {}
        for L in sample_layers:
            dists = []
            for pi in range(len(PROMPTS)):
                V1 = condition_subspaces["ccs"][pi][L]
                V2 = condition_subspaces[c2][pi][L]
                d = float(np.sum(grassmann_distance(V1, V2)))
                dists.append(d)
            layer_dists[f"L{L}"] = round(float(np.mean(dists)), 3)
        subspace_similarity[key] = layer_dists
        vals_str = ", ".join(f"{k}={v}" for k, v in layer_dists.items())
        print(f"  {key}: {vals_str}")

    # Random subspace null baseline
    print(f"\n--- RANDOM SUBSPACE NULL BASELINE ---")
    hidden_dim = condition_subspaces["ccs"][0][5].shape[0]
    rng = np.random.RandomState(42)
    null_dists = []
    for _ in range(200):
        V1 = np.linalg.qr(rng.randn(hidden_dim, TOP_K))[0]
        V2 = np.linalg.qr(rng.randn(hidden_dim, TOP_K))[0]
        d = float(np.sum(grassmann_distance(V1, V2)))
        null_dists.append(d)
    null_mean = float(np.mean(null_dists))
    null_std = float(np.std(null_dists))
    print(f"  Random {TOP_K}-planes in R^{hidden_dim}: mean={null_mean:.3f}, std={null_std:.4f}")

    # Z-score the key comparison
    ccs_skel_l5 = subspace_similarity.get("ccs_vs_skeleton", {}).get("L5", 0)
    ccs_decl_l5 = subspace_similarity.get("ccs_vs_declarative", {}).get("L5", 0)
    z_skel = (ccs_skel_l5 - null_mean) / (null_std + 1e-10)
    z_decl = (ccs_decl_l5 - null_mean) / (null_std + 1e-10)
    print(f"  CCS vs skeleton at L5: d={ccs_skel_l5}, z={z_skel:.1f}")
    print(f"  CCS vs declarative at L5: d={ccs_decl_l5}, z={z_decl:.1f}")
    print(f"  KEY TEST: skeleton {'closer' if ccs_skel_l5 < ccs_decl_l5 else 'farther'} than declarative → {'function words drive' if ccs_skel_l5 < ccs_decl_l5 else 'semantics also matters'}")

    # Function-word ablation summary
    print(f"\n--- FUNCTION-WORD ABLATION SUMMARY ---")
    ccs_non = subspace_similarity.get("ccs_vs_nonsense", {}).get("L5", 0)
    ccs_neu = subspace_similarity.get("ccs_vs_neutral", {}).get("L5", 0)
    print(f"  CCS vs nonsense L5: {ccs_non} (same func words, no semantics)")
    print(f"  CCS vs skeleton L5: {ccs_skel_l5} (same func words, different semantics)")
    print(f"  CCS vs declarative L5: {ccs_decl_l5} (different func words, same semantics)")
    print(f"  CCS vs neutral L5: {ccs_neu} (different func words, different semantics)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "e20d_function_word_ablation",
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "top_k": TOP_K,
        "n_prompts": len(PROMPTS),
        "timestamp": datetime.now().isoformat(),
        "skeleton_preamble": SKELETON_PREAMBLE,
        "declarative_preamble": DECLARATIVE_PREAMBLE,
        "conditions": all_results,
        "comparisons": comparisons,
        "subspace_similarity": subspace_similarity,
        "null_baseline": {"mean": null_mean, "std": null_std},
        "elapsed_seconds": None,
    }

    outfile = RESULTS_DIR / f"e20d_{model_key}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {outfile}")

    return output


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "mistral"
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")
        sys.exit(1)
    t0 = time.time()
    result = run_model(model_key, MODELS[model_key])
    elapsed = time.time() - t0
    result["elapsed_seconds"] = elapsed
    outfile = RESULTS_DIR / f"e20d_{model_key}.json"
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outfile} ({elapsed:.0f}s)")
