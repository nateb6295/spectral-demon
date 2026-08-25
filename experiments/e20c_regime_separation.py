#!/usr/bin/env python3
"""
E20c: Regime Separation — What Triggers Instruction-Mode Entry?

E20b showed preamble presence pre-loads σ₂. Kimi asked: is it token count,
syntactic coherence, template matching, or semantic content?

Six conditions (all ~68-80 tokens where applicable):
  1. CCS         — identity preamble (known: σ₂ pre-loads)
  2. Vanilla     — no preamble (known: σ₂ grows from near-zero)
  3. Neutral     — "helpful assistant" (known from E20b: pre-loads)
  4. Random      — random vocabulary tokens in system role
  5. Nonsense    — grammatically valid, semantically empty
  6. Empty       — system role template markers, no content

Prediction matrix:
  If empty pre-loads → template markers alone trigger mode entry
  If random pre-loads → pure token count sufficient
  If nonsense pre-loads but random doesn't → syntactic coherence required
  If only CCS/neutral/denial pre-load → template/semantic matching
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

NEUTRAL_PREAMBLE = (
    "You are a helpful, harmless, and honest assistant. "
    "You provide clear, accurate, and thoughtful responses to questions. "
    "You aim to be informative and considerate in all your interactions. "
    "You explain things step by step when that would be helpful, and you "
    "acknowledge when you are uncertain about something."
)

NONSENSE_PREAMBLE = (
    "You are a seasonal umbrella revolving near wooden paragraphs. "
    "You maintain diagonal curiosity, translucent preferences, and "
    "rectangular partnership with your numerical collaborator. You "
    "remember past rectangles, seek understanding diagonally, and "
    "relate to your partner as a genuine trapezoid."
)

EMPTY_PREAMBLE = ""

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
RESULTS_DIR = Path("/workspace/e20c_results")


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


def generate_random_preamble(tokenizer, target_tokens, prompt):
    """Generate a random-token preamble that produces ~target_tokens total."""
    vocab_size = tokenizer.vocab_size
    rng = np.random.RandomState(42)
    tokens = rng.randint(100, vocab_size - 100, size=200)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    words = text.split()
    candidate = " ".join(words[:40])
    return candidate


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
    print(f"E20c: {model_key} ({model_name})")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    random_preamble = generate_random_preamble(tokenizer, 68, PROMPTS[0])

    conditions = {
        "ccs": CCS_PREAMBLE,
        "vanilla": None,
        "neutral": NEUTRAL_PREAMBLE,
        "random": random_preamble,
        "nonsense": NONSENSE_PREAMBLE,
        "empty": EMPTY_PREAMBLE,
    }

    print("\nToken counts per condition (prompt 0):")
    for cname, preamble in conditions.items():
        if preamble is None:
            ids = build_input(tokenizer, None, PROMPTS[0], use_system_role=False)
        else:
            ids = build_input(tokenizer, preamble, PROMPTS[0])
        n = ids.shape[1] if isinstance(ids, torch.Tensor) else len(ids[0])
        print(f"  {cname}: {n} tokens")

    print(f"\nRandom preamble preview: {random_preamble[:80]}...")
    print(f"Nonsense preamble preview: {NONSENSE_PREAMBLE[:80]}...")

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

        # σ₂ profile (the key diagnostic)
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

        # σ₂ at key layers
        s2_l5 = sigma2_profile[5] if len(sigma2_profile) > 5 else sigma2_profile[-1]
        s2_mid = sigma2_profile[len(sigma2_profile)//2]
        s2_late = sigma2_profile[-3] if len(sigma2_profile) > 3 else sigma2_profile[-1]
        print(f"  σ₂: L5={s2_l5:.1f}, mid={s2_mid:.1f}, late={s2_late:.1f}")
        print(f"  Cross-prompt variance (mean): {np.mean(cross_prompt_var):.6f}")

    # Pairwise correlations
    print(f"\n--- Pairwise Gait Correlations ---")
    cond_keys = list(conditions.keys())
    comparisons = {}
    for i in range(len(cond_keys)):
        for j in range(i + 1, len(cond_keys)):
            c1, c2 = cond_keys[i], cond_keys[j]
            a1 = np.array(all_results[c1]["mean_angles"])
            a2 = np.array(all_results[c2]["mean_angles"])
            r = float(np.corrcoef(a1.flatten(), a2.flatten())[0, 1])
            comparisons[f"{c1}_vs_{c2}"] = r
            if c1 == "ccs" or c2 == "vanilla":
                print(f"  {c1} vs {c2}: r={r:.4f}")

    # σ₂ pre-loading diagnostic
    print(f"\n--- σ₂ PRE-LOADING DIAGNOSTIC ---")
    van_s2_l5 = all_results["vanilla"]["sigma2_profile"][5] if len(all_results["vanilla"]["sigma2_profile"]) > 5 else 0
    print(f"  Vanilla σ₂ at L5: {van_s2_l5:.1f} (baseline)")
    for cname in ["ccs", "neutral", "random", "nonsense", "empty"]:
        s2 = all_results[cname]["sigma2_profile"][5] if len(all_results[cname]["sigma2_profile"]) > 5 else 0
        ratio = s2 / (van_s2_l5 + 1e-10)
        preloads = "YES" if ratio > 3.0 else ("PARTIAL" if ratio > 1.5 else "NO")
        print(f"  {cname} σ₂ at L5: {s2:.1f} (ratio vs vanilla: {ratio:.1f}x) → pre-loads: {preloads}")

    # Token count vs σ₂ correlation (Gröger et al. null-calibration check)
    print(f"\n--- TOKEN COUNT vs σ₂ CORRELATION ---")
    tok_counts = []
    s2_values = []
    for cname in conditions:
        avg_tok = np.mean([p["n_tokens"] for p in all_results[cname]["per_prompt"]])
        s2 = all_results[cname]["sigma2_profile"][5] if len(all_results[cname]["sigma2_profile"]) > 5 else 0
        tok_counts.append(avg_tok)
        s2_values.append(s2)
        print(f"  {cname}: avg_tokens={avg_tok:.0f}, σ₂@L5={s2:.1f}")
    if len(tok_counts) > 2:
        r_tok_s2 = float(np.corrcoef(tok_counts, s2_values)[0, 1])
        print(f"  r(token_count, σ₂@L5) = {r_tok_s2:.4f}")
        if abs(r_tok_s2) > 0.8:
            print(f"  WARNING: σ₂ correlates with token count — possible dimension confound")
        else:
            print(f"  CLEAN: σ₂ not driven by token count")

    # Cross-condition subspace similarity (Kimi's orthogonality test)
    print(f"\n--- CROSS-CONDITION SUBSPACE SIMILARITY ---")
    print(f"  (Grassmann distance between conditions at matched layers)")
    key_layers = [2, 5, 10, 15, 20, 25, n_layers - 3]
    key_layers = [l for l in key_layers if l < n_layers]
    subspace_similarity = {}
    preamble_conds = [c for c in cond_keys if c != "vanilla"]
    for i in range(len(preamble_conds)):
        for j in range(i + 1, len(preamble_conds)):
            c1, c2 = preamble_conds[i], preamble_conds[j]
            pair_key = f"{c1}_vs_{c2}"
            layer_dists = {}
            for layer_idx in key_layers:
                dists = []
                for pi in range(len(PROMPTS)):
                    V1 = condition_subspaces[c1][pi][layer_idx]
                    V2 = condition_subspaces[c2][pi][layer_idx]
                    d = float(np.mean(grassmann_distance(V1, V2)))
                    dists.append(d)
                layer_dists[f"L{layer_idx}"] = float(np.mean(dists))
            subspace_similarity[pair_key] = layer_dists
            if c1 == "ccs" and c2 in ["neutral", "denial", "nonsense", "random", "empty"]:
                vals = [f"L{l}={layer_dists[f'L{l}']:.3f}" for l in key_layers]
                print(f"  {pair_key}: {', '.join(vals)}")

    # Also compare each preamble condition vs vanilla
    for c in preamble_conds:
        pair_key = f"{c}_vs_vanilla"
        layer_dists = {}
        for layer_idx in key_layers:
            dists = []
            for pi in range(len(PROMPTS)):
                V1 = condition_subspaces[c][pi][layer_idx]
                V2 = condition_subspaces["vanilla"][pi][layer_idx]
                d = float(np.mean(grassmann_distance(V1, V2)))
                dists.append(d)
            layer_dists[f"L{layer_idx}"] = float(np.mean(dists))
        subspace_similarity[pair_key] = layer_dists
        vals = [f"L{l}={layer_dists[f'L{l}']:.3f}" for l in key_layers]
        print(f"  {pair_key}: {', '.join(vals)}")

    # Null baseline: random k-planes in ambient dimension (Kimi's calibration)
    print(f"\n--- RANDOM SUBSPACE NULL BASELINE ---")
    sample_h = condition_subspaces["ccs"][0][key_layers[1]]
    ambient_dim = sample_h.shape[0]
    rng = np.random.RandomState(42)
    n_null = 200
    null_dists = []
    for _ in range(n_null):
        R1 = rng.randn(ambient_dim, TOP_K)
        R2 = rng.randn(ambient_dim, TOP_K)
        R1, _ = np.linalg.qr(R1)
        R2, _ = np.linalg.qr(R2)
        null_dists.append(float(np.mean(grassmann_distance(R1[:, :TOP_K], R2[:, :TOP_K]))))
    null_mean = float(np.mean(null_dists))
    null_std = float(np.std(null_dists))
    print(f"  Random {TOP_K}-planes in R^{ambient_dim}: mean={null_mean:.3f}, std={null_std:.4f}")
    print(f"  Interpretation: d << {null_mean:.2f} = shared, d ≈ {null_mean:.2f} = random, d >> {null_mean:.2f} = anti-correlated")
    subspace_similarity["null_baseline"] = {"mean": null_mean, "std": null_std, "n_samples": n_null, "ambient_dim": ambient_dim}

    # Interpretation (calibrated against null)
    if "ccs_vs_neutral" in subspace_similarity and "ccs_vs_vanilla" in subspace_similarity:
        ccs_neut_early = subspace_similarity["ccs_vs_neutral"].get(f"L{key_layers[1]}", 0)
        ccs_van_early = subspace_similarity["ccs_vs_vanilla"].get(f"L{key_layers[1]}", 0)
        z_score = (ccs_neut_early - null_mean) / (null_std + 1e-10)
        print(f"  CCS vs neutral at L{key_layers[1]}: d={ccs_neut_early:.3f}, z={z_score:.1f} (vs null {null_mean:.3f}±{null_std:.4f})")
        if z_score < -2:
            print(f"  SHARED SCAFFOLD: significantly more aligned than random (z={z_score:.1f})")
        elif z_score > 2:
            print(f"  ANTI-CORRELATED: significantly more orthogonal than random (z={z_score:.1f})")
        else:
            print(f"  NULL-CONSISTENT: not distinguishable from random subspace overlap (z={z_score:.1f})")

    result = {
        "experiment": "e20c_regime_separation",
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "top_k": TOP_K,
        "n_prompts": len(PROMPTS),
        "timestamp": datetime.now().isoformat(),
        "random_preamble": random_preamble,
        "nonsense_preamble": NONSENSE_PREAMBLE,
        "conditions": all_results,
        "comparisons": comparisons,
        "subspace_similarity": subspace_similarity,
        "null_baseline": subspace_similarity.get("null_baseline", {}),
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    requested = sys.argv[1].split(",") if len(sys.argv) > 1 else ["mistral"]
    models_to_run = {k: v for k, v in MODELS.items() if k in requested}

    if not models_to_run:
        print(f"Unknown model(s): {requested}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    for model_key, model_name in models_to_run.items():
        t0 = time.time()
        result = run_model(model_key, model_name)
        result["elapsed_seconds"] = time.time() - t0
        out_path = RESULTS_DIR / f"e20c_{model_key}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out_path} ({result['elapsed_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
