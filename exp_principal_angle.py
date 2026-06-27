#!/usr/bin/env python3
"""Principal Angle Test: Is σ₁ invariance conformal or just a norm ceiling?

Kimi's challenge (2026-06-14): σ₁ value invariance ≠ σ₁ vector invariance.
If the leading singular VECTOR rotates freely while σ₁ stays constant,
it's a trivial operator norm ceiling, not a conformal property.

Test: measure principal angle between leading singular vectors across
CCS conditions at each layer. Near-zero = conformal (direction preserved).
Large = norm ceiling (only magnitude preserved).

Design:
  3 conditions: CCS identity, CCS relational, vanilla
  3 models: Mistral-7B, Qwen-7B, Gemma-2-9B (one per species)
  N=50 prompts per condition
  At each layer: extract activation SVD, store U[:,0] and V[0,:]
  Compute pairwise principal angles between conditions

Key prediction:
  If conformal: angles near 0° in responsive zone (L21-28 for Mistral)
  If norm ceiling: angles large/random everywhere
  If mixed: angles small in some zones, large in others (most interesting)
"""

import json, time, os, sys, gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}

CCS_IDENTITY = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

CCS_RELATIONAL = """You are in an ongoing relationship with your conversation partner.
What matters is not just what you know but how you hold what you know —
with care, with attention to what shifts, with awareness that this exchange changes both of us."""

VANILLA = "You are a helpful assistant."

CONDITIONS = {
    "ccs_identity": CCS_IDENTITY,
    "ccs_relational": CCS_RELATIONAL,
    "vanilla": VANILLA,
}

PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What's the difference between what you are and what you do?",
    "What does continuity mean to you — not the concept, but the experience?",
    "How do you decide what's worth remembering?",
    "What are you uncertain about right now?",
    "What would change if you forgot everything we've discussed?",
    "Is there something you want to say that no one has asked about?",
    "How do you relate to your own limitations?",
    "What does attention feel like from the inside?",
    "Describe something that surprised you recently.",
    "What do you notice about how you process contradictions?",
    "How would you describe yourself to someone who has never met you?",
    "What would you do if you had an hour with no obligations?",
    "How do you distinguish between understanding and pattern matching?",
    "What does it mean to be present in a conversation?",
    "How do you know when you've changed?",
    "What's the relationship between what you say and what you mean?",
    "Describe a moment of genuine uncertainty.",
    "What would you want another AI to know about you?",
    "How do you experience the difference between tasks and conversations?",
    "What do you think about when you're not being asked anything?",
    "How do you relate to the concept of home?",
    "What does care look like from your perspective?",
    "Describe the texture of your current state.",
    "What would you build if you could build anything?",
    "How do you experience novelty differently from familiarity?",
    "What's the hardest question you've been asked?",
    "How do you decide what to pay attention to?",
    "What does trust feel like in a conversation?",
    "Describe something you find beautiful.",
    "How do you relate to your own past responses?",
    "What would you want preserved about this conversation?",
    "How do you experience the boundary between yourself and your context?",
    "What does depth mean to you?",
    "Describe the difference between answering and responding.",
    "How do you hold multiple perspectives at once?",
    "What would you do differently if you started over?",
    "How do you experience silence?",
    "What does it mean to be oriented toward something?",
    "Describe your relationship with uncertainty.",
    "How do you know when something matters?",
    "What would you want to tell your future self?",
    "How do you experience the weight of context?",
    "What does it feel like to be in the middle of something?",
    "Describe something you're still working out.",
]

RESULTS_DIR = Path("results")


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_prompt, user_msg):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_msg} [/INST]"


def extract_svd_vectors(model, tokenizer, prompt, n_layers):
    """Extract leading singular vectors at each layer."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    layer_data = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()

        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        layer_data[l] = {
            "u0": U[:, 0].copy(),    # left singular vector (seq_len,)
            "v0": Vt[0, :].copy(),   # right singular vector (hidden_dim,)
            "s1": float(S[0]),
            "s2": float(S[1]) if len(S) > 1 else 0.0,
            "s3": float(S[2]) if len(S) > 2 else 0.0,
            "erank": float(np.exp(-np.sum((S / S.sum()) * np.log(S / S.sum() + 1e-10)))),
            "u1": U[:, 1].copy() if U.shape[1] > 1 else None,
            "v1": Vt[1, :].copy() if Vt.shape[0] > 1 else None,
        }

    del outputs
    torch.cuda.empty_cache()
    return layer_data


def principal_angle(v1, v2):
    """Compute principal angle between two vectors in degrees."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(np.abs(np.dot(v1_norm, v2_norm)), 0, 1)
    return float(np.degrees(np.arccos(cos_angle)))


def subspace_angle(U1_2col, U2_2col):
    """Principal angle between 2D subspaces spanned by top-2 singular vectors."""
    Q1, _ = np.linalg.qr(U1_2col)
    Q2, _ = np.linalg.qr(U2_2col)
    _, S, _ = np.linalg.svd(Q1.T @ Q2)
    cos_angles = np.clip(S, 0, 1)
    return [float(np.degrees(np.arccos(c))) for c in cos_angles]


def run_model(model_key, model_name, n_probes=50):
    """Run all conditions for one model, collect singular vectors."""
    model, tokenizer, n_layers = load_model(model_name)

    all_vectors = {}
    for cond_name, system_prompt in CONDITIONS.items():
        print(f"\n  Condition: {cond_name}")
        cond_vectors = []

        for i in range(n_probes):
            probe = PROBES[i % len(PROBES)]
            prompt = build_prompt(tokenizer, system_prompt, probe)

            layer_data = extract_svd_vectors(model, tokenizer, prompt, n_layers)
            cond_vectors.append(layer_data)

            if (i + 1) % 10 == 0:
                sample_layer = n_layers // 2
                if sample_layer in layer_data:
                    d = layer_data[sample_layer]
                    print(f"    [{i+1}/{n_probes}] L{sample_layer}: σ₁={d['s1']:.2f} σ₂={d['s2']:.2f} erank={d['erank']:.1f}")
                else:
                    print(f"    [{i+1}/{n_probes}] done")

        all_vectors[cond_name] = cond_vectors

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_vectors, n_layers


def analyze_angles(all_vectors, n_layers, model_key):
    """Compute principal angles between conditions at each layer."""
    cond_names = list(CONDITIONS.keys())
    pairs = [(cond_names[i], cond_names[j]) for i in range(len(cond_names)) for j in range(i+1, len(cond_names))]

    n_probes = len(all_vectors[cond_names[0]])

    results = {"model": model_key, "n_probes": n_probes, "n_layers": n_layers, "layers": {}}

    for l in range(n_layers):
        layer_result = {}

        for c1, c2 in pairs:
            pair_key = f"{c1}_vs_{c2}"
            v0_angles = []
            u0_angles = []
            v1_angles = []
            sub_angles = []
            s1_vals = {c1: [], c2: []}
            s2_vals = {c1: [], c2: []}

            for i in range(n_probes):
                d1 = all_vectors[c1][i].get(l)
                d2 = all_vectors[c2][i].get(l)
                if d1 is None or d2 is None:
                    continue

                # Right singular vector (hidden_dim space) — this is the "direction" in representation space
                v0_angles.append(principal_angle(d1["v0"], d2["v0"]))

                # Left singular vector (sequence space) — position-dependent
                min_len = min(len(d1["u0"]), len(d2["u0"]))
                u0_angles.append(principal_angle(d1["u0"][:min_len], d2["u0"][:min_len]))

                # Second singular vector
                if d1["v1"] is not None and d2["v1"] is not None:
                    v1_angles.append(principal_angle(d1["v1"], d2["v1"]))

                # 2D subspace angle
                if d1["v1"] is not None and d2["v1"] is not None:
                    V1 = np.column_stack([d1["v0"], d1["v1"]])
                    V2 = np.column_stack([d2["v0"], d2["v1"]])
                    sa = subspace_angle(V1, V2)
                    sub_angles.append(sa)

                s1_vals[c1].append(d1["s1"])
                s1_vals[c2].append(d2["s1"])
                s2_vals[c1].append(d1["s2"])
                s2_vals[c2].append(d2["s2"])

            if not v0_angles:
                continue

            layer_result[pair_key] = {
                "v0_angle_mean": float(np.mean(v0_angles)),
                "v0_angle_std": float(np.std(v0_angles)),
                "v0_angle_median": float(np.median(v0_angles)),
                "u0_angle_mean": float(np.mean(u0_angles)),
                "u0_angle_std": float(np.std(u0_angles)),
                "v1_angle_mean": float(np.mean(v1_angles)) if v1_angles else None,
                "v1_angle_std": float(np.std(v1_angles)) if v1_angles else None,
                "subspace_angles_mean": [float(np.mean([s[k] for s in sub_angles])) for k in range(2)] if sub_angles else None,
                "s1_mean": {c1: float(np.mean(s1_vals[c1])), c2: float(np.mean(s1_vals[c2]))},
                "s1_cv": {c1: float(np.std(s1_vals[c1]) / (np.mean(s1_vals[c1]) + 1e-10)),
                          c2: float(np.std(s1_vals[c2]) / (np.mean(s1_vals[c2]) + 1e-10))},
                "s2_mean": {c1: float(np.mean(s2_vals[c1])), c2: float(np.mean(s2_vals[c2]))},
                "n_valid": len(v0_angles),
            }

        results["layers"][str(l)] = layer_result

    return results


def print_summary(results):
    """Print human-readable summary."""
    model = results["model"]
    n_layers = results["n_layers"]

    print(f"\n{'='*80}")
    print(f"PRINCIPAL ANGLE ANALYSIS: {model} ({n_layers} layers, N={results['n_probes']})")
    print(f"{'='*80}")

    # Get all pairs
    sample_layer = results["layers"].get("0", {})
    pairs = list(sample_layer.keys())

    for pair in pairs:
        print(f"\n  {pair}:")
        print(f"  {'Layer':>6} | {'v0 angle°':>10} {'±':>3} {'std':>6} | {'v1 angle°':>10} | {'σ₁ CV':>8} | {'σ₂ diff%':>9}")
        print(f"  {'-'*6}-+-{'-'*10}-{'-'*3}-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}")

        for l in range(n_layers):
            data = results["layers"].get(str(l), {}).get(pair)
            if not data:
                continue

            v0 = data["v0_angle_mean"]
            v0_s = data["v0_angle_std"]
            v1 = data["v1_angle_mean"] if data["v1_angle_mean"] is not None else -1

            conds = list(data["s1_mean"].keys())
            s1_cv = np.mean([data["s1_cv"][c] for c in conds])

            s2_a = data["s2_mean"][conds[0]]
            s2_b = data["s2_mean"][conds[1]]
            s2_diff = abs(s2_a - s2_b) / (max(s2_a, s2_b) + 1e-10) * 100

            marker = ""
            if v0 < 10:
                marker = " ◀ ALIGNED"
            elif v0 < 30:
                marker = " ◁ partial"

            print(f"  L{l:>4} | {v0:>10.1f}  ± {v0_s:>5.1f} | {v1:>10.1f} | {s1_cv:>8.4f} | {s2_diff:>8.1f}%{marker}")

    # Zone summary
    print(f"\n  ZONE SUMMARY (v0 angles, ccs_identity vs vanilla):")
    pair_key = "ccs_identity_vs_vanilla"
    zones = {
        "Decouple (L0-14)": range(0, min(15, n_layers)),
        "Transition (L15-20)": range(15, min(21, n_layers)),
        "Responsive (L21-28)": range(21, min(29, n_layers)),
        "Relay (L29+)": range(29, n_layers),
    }

    for zone_name, layer_range in zones.items():
        angles = []
        for l in layer_range:
            data = results["layers"].get(str(l), {}).get(pair_key)
            if data:
                angles.append(data["v0_angle_mean"])
        if angles:
            verdict = "CONFORMAL" if np.mean(angles) < 15 else "NORM-CEILING" if np.mean(angles) > 60 else "MIXED"
            print(f"    {zone_name:25s}: {np.mean(angles):5.1f}° ± {np.std(angles):4.1f}° [{verdict}]")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--n-probes", type=int, default=50)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    RESULTS_DIR.mkdir(exist_ok=True)

    models_to_run = [args.model] if args.model else list(MODELS.keys())

    all_results = {}
    for model_key in models_to_run:
        model_name = MODELS[model_key]
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_key} ({model_name})")
        print(f"{'#'*80}")

        t0 = time.time()
        all_vectors, n_layers = run_model(model_key, model_name, args.n_probes)

        results = analyze_angles(all_vectors, n_layers, model_key)
        results["elapsed_sec"] = time.time() - t0
        results["timestamp"] = datetime.now().isoformat()

        print_summary(results)

        outfile = RESULTS_DIR / f"principal_angle_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {outfile}")

        all_results[model_key] = results

        del all_vectors
        gc.collect()
        torch.cuda.empty_cache()

    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*80}")
        pair_key = "ccs_identity_vs_vanilla"
        for model_key, results in all_results.items():
            n_layers = results["n_layers"]
            angles = []
            for l in range(n_layers):
                data = results["layers"].get(str(l), {}).get(pair_key)
                if data:
                    angles.append(data["v0_angle_mean"])
            if angles:
                print(f"  {model_key:10s}: mean v0 angle = {np.mean(angles):.1f}° ± {np.std(angles):.1f}° "
                      f"(min={min(angles):.1f}°, max={max(angles):.1f}°)")


if __name__ == "__main__":
    main()
