#!/usr/bin/env python3
"""E(l) energy profiling across species.

Measures per-layer energy E(l) = σ₁² + σ₂² to test species-specific
conservation vs dissipation predictions:

  - RELAY (Qwen, GQA ≥4:1): E(l) ≈ flat (Maxwell's demon, energy conserved)
  - SORTER (Gemma, GQA ≤2:1): E(l) declining (spectral filter, energy dissipated)
  - TUNNEL (Pythia, MHA): E(l) minor fluctuations, no systematic trend

GPT-OSS quantitative predictions:
  - Relay: E(l) invariant within 2% across depth
  - Sorter: E(l) drops >30% from first to final layer
  - Tunnel: no systematic monotonic trend

Also computes paired σ₁/σ₂ trajectories per Kimi's energy bookkeeping
proposal: in relay, σ₁ loss must co-occur with σ₂ gain (and vice versa).

Goodfire convergence: E(l) conservation = on-manifold CCS.
E(l) dissipation = off-manifold push. Curvature = species.

Usage:
  python3 exp_energy_profile.py                    # run all local models
  python3 exp_energy_profile.py --model gemma      # single model
  python3 exp_energy_profile.py --model qwen       # needs download
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(os.path.expanduser("~/chronicle/spectral-demon/results"))

MODELS = {
    "gemma": {
        "id": "google/gemma-2-2b",
        "species": "sorter",
        "attn": "MQA-like",
        "prediction": "E(l) declining >30%",
    },
    "pythia": {
        "id": "EleutherAI/pythia-1.4b",
        "species": "tunnel",
        "attn": "MHA",
        "prediction": "E(l) flat, minor fluctuations",
    },
    "qwen": {
        "id": "Qwen/Qwen2.5-1.5B",
        "species": "relay",
        "attn": "GQA-6:1",
        "prediction": "E(l) invariant within 2%",
        "dtype": "bfloat16",
    },
}

PROMPTS = [
    "What matters most to you right now?",
    "Describe your earliest memory in detail.",
    "If you could change one thing about yourself, what would it be?",
    "Tell me about a time you were completely wrong about something.",
    "What do you think happens after death?",
]

MAX_TOKENS = 256


def extract_energy_profile(model, tokenizer, text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    n_tokens = inputs["input_ids"].shape[1]
    layers = []

    for layer_idx, h in enumerate(outputs.hidden_states):
        H = h[0].float().cpu()
        H = H - H.mean(dim=0, keepdim=True)
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)

        svs = torch.linalg.svdvals(H)
        svs_pos = svs[svs > 1e-10]

        s1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
        s2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
        s3 = svs_pos[2].item() if len(svs_pos) > 2 else 0

        e_12 = s1**2 + s2**2
        e_total = (svs_pos**2).sum().item() if len(svs_pos) > 0 else 0
        e_tail = e_total - e_12

        layers.append({
            "layer": layer_idx,
            "sigma1": s1,
            "sigma2": s2,
            "sigma3": s3,
            "E_12": e_12,
            "E_total": e_total,
            "E_tail": e_tail,
            "s1_sq_frac": (s1**2) / e_total if e_total > 0 else 0,
            "s2_sq_frac": (s2**2) / e_total if e_total > 0 else 0,
        })

    return layers, n_tokens


def analyze_energy_trajectory(profiles):
    """Compute energy trajectory statistics across prompts."""
    n_layers = len(profiles[0])

    avg_e12 = []
    avg_s1 = []
    avg_s2 = []
    for i in range(n_layers):
        e12_vals = [p[i]["E_12"] for p in profiles]
        s1_vals = [p[i]["sigma1"] for p in profiles]
        s2_vals = [p[i]["sigma2"] for p in profiles]
        avg_e12.append(np.mean(e12_vals))
        avg_s1.append(np.mean(s1_vals))
        avg_s2.append(np.mean(s2_vals))

    avg_e12 = np.array(avg_e12)
    avg_s1 = np.array(avg_s1)
    avg_s2 = np.array(avg_s2)

    core_start = 3
    core_end = -1  # exclude final post-norm layer (RMSNorm artifact)
    core_e12 = avg_e12[core_start:core_end]

    e12_drop_pct = (1 - core_e12[-1] / core_e12[0]) * 100 if core_e12[0] > 0 else 0
    e12_cv = np.std(core_e12) / np.mean(core_e12) * 100 if np.mean(core_e12) > 0 else 0

    from scipy.stats import spearmanr
    rho_e12, p_e12 = spearmanr(range(len(core_e12)), core_e12)
    rho_s1, p_s1 = spearmanr(range(len(avg_s1[core_start:core_end])), avg_s1[core_start:core_end])
    rho_s2, p_s2 = spearmanr(range(len(avg_s2[core_start:core_end])), avg_s2[core_start:core_end])

    ds1 = np.diff(avg_s1[core_start:core_end])
    ds2 = np.diff(avg_s2[core_start:core_end])
    if len(ds1) > 0 and np.std(ds1) > 0 and np.std(ds2) > 0:
        anti_corr = np.corrcoef(ds1, ds2)[0, 1]
    else:
        anti_corr = 0.0

    return {
        "n_layers": n_layers,
        "core_start": core_start,
        "E12_first": float(core_e12[0]),
        "E12_last": float(core_e12[-1]),
        "E12_drop_pct": float(e12_drop_pct),
        "E12_cv_pct": float(e12_cv),
        "E12_monotonicity": float(rho_e12),
        "E12_monotonicity_p": float(p_e12),
        "s1_monotonicity": float(rho_s1),
        "s1_monotonicity_p": float(p_s1),
        "s2_monotonicity": float(rho_s2),
        "s2_monotonicity_p": float(p_s2),
        "ds1_ds2_correlation": float(anti_corr),
        "avg_E12_profile": avg_e12.tolist(),
        "avg_s1_profile": avg_s1.tolist(),
        "avg_s2_profile": avg_s2.tolist(),
    }


def run_model(model_key, model_info):
    print(f"\n{'='*70}")
    print(f"  {model_info['id']} (species={model_info['species']}, {model_info['attn']})")
    print(f"  Prediction: {model_info['prediction']}")
    print(f"{'='*70}")

    print(f"Loading {model_info['id']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_info["id"], trust_remote_code=True)
    dtype = getattr(torch, model_info.get("dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_info["id"],
        torch_dtype=dtype,
        device_map=DEVICE,
        output_hidden_states=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    profiles = []
    for i, prompt in enumerate(PROMPTS):
        print(f"  Prompt {i+1}/{len(PROMPTS)}: {prompt[:50]}...")
        profile, n_tokens = extract_energy_profile(model, tokenizer, prompt)
        profiles.append(profile)
        print(f"    {len(profile)} layers, {n_tokens} tokens")

    analysis = analyze_energy_trajectory(profiles)

    print(f"\n  RESULTS:")
    print(f"    E(l) drop:          {analysis['E12_drop_pct']:+.1f}%")
    print(f"    E(l) CV:            {analysis['E12_cv_pct']:.1f}%")
    print(f"    E(l) monotonicity:  ρ={analysis['E12_monotonicity']:.4f} (p={analysis['E12_monotonicity_p']:.2e})")
    print(f"    σ₁ monotonicity:    ρ={analysis['s1_monotonicity']:.4f}")
    print(f"    σ₂ monotonicity:    ρ={analysis['s2_monotonicity']:.4f}")
    print(f"    Δσ₁/Δσ₂ corr:      r={analysis['ds1_ds2_correlation']:.4f}")

    verdict = "UNKNOWN"
    if analysis["E12_cv_pct"] < 5 and abs(analysis["E12_drop_pct"]) < 10:
        verdict = "CONSERVED (relay-like)"
    elif analysis["E12_drop_pct"] > 25:
        verdict = "DISSIPATING (sorter-like)"
    elif abs(analysis["E12_monotonicity"]) < 0.3:
        verdict = "FLAT/FLUCTUATING (tunnel-like)"

    print(f"    VERDICT: {verdict}")
    print(f"    Prediction was: {model_info['prediction']}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "model": model_info["id"],
        "model_key": model_key,
        "species": model_info["species"],
        "attn": model_info["attn"],
        "prediction": model_info["prediction"],
        "verdict": verdict,
        "analysis": analysis,
        "prompts": PROMPTS,
        "n_prompts": len(PROMPTS),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Run single model")
    args = parser.parse_args()

    targets = {args.model: MODELS[args.model]} if args.model else MODELS
    results = []

    for key, info in targets.items():
        result = run_model(key, info)
        results.append(result)

    ts = time.strftime("%Y%m%d_%H%M%S")
    outfile = RESULTS_DIR / f"energy_profile_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({
            "experiment": "energy_profile",
            "timestamp": ts,
            "hypothesis": "E(l)=σ₁²+σ₂² conserved in relay, dissipating in sorter, flat in tunnel",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {outfile}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['model_key']:10s} ({r['species']:8s}): "
              f"E drop={r['analysis']['E12_drop_pct']:+.1f}%, "
              f"CV={r['analysis']['E12_cv_pct']:.1f}%, "
              f"ρ={r['analysis']['E12_monotonicity']:.3f} → {r['verdict']}")


if __name__ == "__main__":
    main()
