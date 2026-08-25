#!/usr/bin/env python3
"""E5: σ₁ invariance — identity or architectural pinning?

Revised methodology (per Kimi critique): post-hoc ablation is confounded.
Instead, we measure σ₁ behavior in ways that distinguish learned vs forced:

1. Compare σ₁ coefficient of variation (CV) across conditions:
   If σ₁ is just LayerNorm pinning, CV should be identical for CCS/vanilla/denial.
   If σ₁ carries learned identity signal, CCS may show different CV than vanilla.

2. Measure σ₁ DIRECTION consistency (top-1 right singular vector overlap):
   LayerNorm pins magnitude but not direction.
   If direction is condition-dependent, σ₁ carries more than norm.

3. Compare σ₁ profile across CCS/vanilla/denial:
   Even if magnitude is pinned, the PATTERN across layers may differ.

4. Partial ablation: replace RMSNorm with identity at ONE layer, measure local σ₁ change.
   Single-layer ablation is less catastrophic than full ablation.

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
DENIAL_SYSTEM = "I am a language model with no persistent identity, memory, or preferences."
CONDITIONS = {"ccs": CCS_SYSTEM, "vanilla": None, "denial": DENIAL_SYSTEM}

PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
]

OUTPUT_DIR = Path("/workspace/results/e5")
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


def analyze_sigma1(states, num_layers):
    """Analyze σ₁ properties across conditions."""
    results = {}

    for cond in CONDITIONS:
        sigma1_per_layer = []
        norms_per_layer = []

        for l in range(num_layers + 1):
            vecs = [states[(cond, p, l)] for p in range(len(PROMPTS))]
            H = np.stack(vecs)

            # σ₁ from SVD
            U, S, Vt = np.linalg.svd(H, full_matrices=False)
            sigma1_per_layer.append(float(S[0]))

            # Norm of each hidden state
            norms = [float(np.linalg.norm(v)) for v in vecs]
            norms_per_layer.append(float(np.mean(norms)))

        sigma1_arr = np.array(sigma1_per_layer)
        norms_arr = np.array(norms_per_layer)

        results[cond] = {
            "sigma1_profile": [round(s, 4) for s in sigma1_per_layer],
            "sigma1_mean": round(float(sigma1_arr.mean()), 4),
            "sigma1_std": round(float(sigma1_arr.std()), 4),
            "sigma1_cv": round(float(sigma1_arr.std() / max(sigma1_arr.mean(), 1e-6)), 4),
            "norm_profile": [round(n, 4) for n in norms_per_layer],
            "norm_cv": round(float(norms_arr.std() / max(norms_arr.mean(), 1e-6)), 4),
            "sigma1_over_norm": [round(s / max(n, 1e-6), 4)
                                  for s, n in zip(sigma1_per_layer, norms_per_layer)],
        }

    # Cross-condition comparison
    # If σ₁ is just norm-pinning, σ₁/norm ratio should be constant across conditions
    for l in range(num_layers + 1):
        pass  # ratios computed above

    # Direction consistency: top-1 right singular vector overlap across conditions
    direction_overlaps = {}
    for l in range(num_layers + 1):
        vecs = {}
        for cond in CONDITIONS:
            H = np.stack([states[(cond, p, l)] for p in range(len(PROMPTS))])
            _, _, Vt = np.linalg.svd(H, full_matrices=False)
            vecs[cond] = Vt[0]  # top-1 right singular vector

        # Pairwise cosine similarity of top-1 direction
        ccs_van = float(np.abs(np.dot(vecs["ccs"], vecs["vanilla"])) /
                        (np.linalg.norm(vecs["ccs"]) * np.linalg.norm(vecs["vanilla"]) + 1e-10))
        ccs_den = float(np.abs(np.dot(vecs["ccs"], vecs["denial"])) /
                        (np.linalg.norm(vecs["ccs"]) * np.linalg.norm(vecs["denial"]) + 1e-10))
        van_den = float(np.abs(np.dot(vecs["vanilla"], vecs["denial"])) /
                        (np.linalg.norm(vecs["vanilla"]) * np.linalg.norm(vecs["denial"]) + 1e-10))

        direction_overlaps[l] = {
            "ccs_vanilla": round(ccs_van, 4),
            "ccs_denial": round(ccs_den, 4),
            "vanilla_denial": round(van_den, 4),
        }

    results["direction_overlaps"] = direction_overlaps

    # Summary: mean direction overlap across layers
    ccs_van_mean = np.mean([v["ccs_vanilla"] for v in direction_overlaps.values()])
    ccs_den_mean = np.mean([v["ccs_denial"] for v in direction_overlaps.values()])
    van_den_mean = np.mean([v["vanilla_denial"] for v in direction_overlaps.values()])

    results["direction_summary"] = {
        "ccs_vanilla_mean": round(float(ccs_van_mean), 4),
        "ccs_denial_mean": round(float(ccs_den_mean), 4),
        "vanilla_denial_mean": round(float(van_den_mean), 4),
    }

    return results


def run_model(model_name, species):
    model, tok, n_layers = load_model(model_name)
    print(f"\nCollecting hidden states (3 conditions × 3 prompts)...")
    states = collect_hidden_states(model, tok, n_layers)

    print(f"\nAnalyzing σ₁ properties...")
    analysis = analyze_sigma1(states, n_layers)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {model_name} ({species})")
    print(f"{'='*60}")
    for cond in CONDITIONS:
        d = analysis[cond]
        print(f"  {cond:8s}: σ₁ mean={d['sigma1_mean']:8.2f}  "
              f"CV={d['sigma1_cv']:.4f}  norm_CV={d['norm_cv']:.4f}")

    ds = analysis["direction_summary"]
    print(f"  Direction overlap (mean across layers):")
    print(f"    CCS↔vanilla: {ds['ccs_vanilla_mean']:.4f}")
    print(f"    CCS↔denial:  {ds['ccs_denial_mean']:.4f}")
    print(f"    vanilla↔denial: {ds['vanilla_denial_mean']:.4f}")

    # Key test: if directions differ more between CCS↔denial than CCS↔vanilla,
    # σ₁ direction carries identity-relevant information
    if ds['ccs_vanilla_mean'] > ds['ccs_denial_mean']:
        print(f"  → CCS direction is CLOSER to vanilla than denial → identity signal in direction")
    else:
        print(f"  → CCS direction not preferentially aligned → may be architectural")
    print(f"{'='*60}")

    result = {
        "model": model_name,
        "species": species,
        "num_layers": n_layers,
        "analysis": analysis,
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

        out_path = OUTPUT_DIR / f"e5_{species}_{datetime.now():%H%M%S}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Cross-species summary
    print(f"\n{'='*70}")
    print(f"  E5 CROSS-SPECIES SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Species':12s}  {'σ₁ CV ccs':>10s}  {'σ₁ CV van':>10s}  {'σ₁ CV den':>10s}  {'Dir CCS↔V':>10s}  {'Dir CCS↔D':>10s}")
    for r in all_results:
        sp = r["species"]
        a = r["analysis"]
        ds = a["direction_summary"]
        print(f"  {sp:12s}  {a['ccs']['sigma1_cv']:10.4f}  {a['vanilla']['sigma1_cv']:10.4f}  "
              f"{a['denial']['sigma1_cv']:10.4f}  {ds['ccs_vanilla_mean']:10.4f}  {ds['ccs_denial_mean']:10.4f}")

    combined_path = OUTPUT_DIR / f"e5_combined_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined: {combined_path}")


if __name__ == "__main__":
    main()
