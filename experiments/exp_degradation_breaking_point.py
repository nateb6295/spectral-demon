#!/usr/bin/env python3
"""
Experiment: Degradation Breaking Point

50% pruning gave σ₂/σ₁ similarity = 0.9976 and σ₁ = 0.9999.
At what pruning level does the spectral profile actually break?

Test: 60%, 70%, 80%, 90%, 95% pruning.
Also: 50%, 100% noise (extreme noise injection).

Gemma 9B only. Quick run — just baseline + aggressive conditions.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

DEVICE = "cuda"
MODEL_NAME = "google/gemma-2-9b-it"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels.
"""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

SAMPLE_LAYERS = list(range(2, 40, 4))


def compute_spectral_profile(model, tokenizer, preamble, probe, layers):
    """Compute σ₁, σ₂, σ₂/σ₁ using full sequence activations."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.squeeze(0).detach().float()
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        if i in layers:
            hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    profile = {}
    for layer_idx in layers:
        if layer_idx not in captured:
            continue
        act = captured[layer_idx]
        try:
            svs = torch.linalg.svdvals(act)
            sigma1 = svs[0].item()
            sigma2 = svs[1].item() if len(svs) > 1 else 0.0
            profile[layer_idx] = {
                "sigma1": sigma1,
                "sigma2": sigma2,
                "ratio": sigma2 / (sigma1 + 1e-10)
            }
        except Exception:
            profile[layer_idx] = {"sigma1": 0.0, "sigma2": 0.0, "ratio": 0.0}

    return profile


def prune_weights(model, fraction):
    """Sampled-threshold pruning to avoid OOM."""
    samples = []
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            flat = param.data.abs().flatten()
            n_sample = min(10000, flat.numel())
            idx = torch.randperm(flat.numel(), device=flat.device)[:n_sample]
            samples.append(flat[idx].float().cpu())
            total += param.numel()

    all_samples = torch.cat(samples)
    threshold = torch.quantile(all_samples, fraction).to(DEVICE)

    pruned = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            mask = param.data.abs() < threshold
            param.data[mask] = 0.0
            pruned += mask.sum().item()

    return pruned, total


def add_noise(model, scale):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            noise = torch.randn_like(param) * param.data.abs().mean() * scale
            param.data += noise


def profile_similarity(prof1, prof2, key="ratio"):
    layers = sorted(set(prof1.keys()) & set(prof2.keys()))
    if len(layers) < 3:
        return 0.0
    v1 = np.array([prof1[l][key] for l in layers])
    v2 = np.array([prof2[l][key] for l in layers])
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return float(dot / (norm1 * norm2 + 1e-10))


def main():
    print("Loading model for baseline...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = {
        "experiment": "degradation_breaking_point",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "conditions": {}
    }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    print("\n=== BASELINE ===")
    baseline_ccs = compute_spectral_profile(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, SAMPLE_LAYERS)
    baseline_bare = compute_spectral_profile(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT, SAMPLE_LAYERS)

    for l in SAMPLE_LAYERS:
        if l in baseline_ccs:
            print(f"  L{l}: CCS σ₂/σ₁={baseline_ccs[l]['ratio']:.4f}  bare={baseline_bare[l]['ratio']:.4f}")

    results["conditions"]["baseline"] = {
        "ccs": {str(l): baseline_ccs[l] for l in SAMPLE_LAYERS if l in baseline_ccs},
        "bare": {str(l): baseline_bare[l] for l in SAMPLE_LAYERS if l in baseline_bare}
    }

    del model
    torch.cuda.empty_cache()

    # Aggressive pruning
    for prune_frac in [0.6, 0.7, 0.8, 0.9, 0.95]:
        print(f"\n=== PRUNING {prune_frac*100:.0f}% ===")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model.eval()

        pruned, total = prune_weights(model, prune_frac)
        print(f"  Pruned {pruned:,}/{total:,} weights ({pruned/total*100:.1f}%)")

        prof_ccs = compute_spectral_profile(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, SAMPLE_LAYERS)
        prof_bare = compute_spectral_profile(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT, SAMPLE_LAYERS)

        sim_ratio_ccs = profile_similarity(baseline_ccs, prof_ccs, "ratio")
        sim_ratio_bare = profile_similarity(baseline_bare, prof_bare, "ratio")
        sim_sigma1 = profile_similarity(baseline_ccs, prof_ccs, "sigma1")

        enrichment_intact = np.mean([baseline_ccs[l]["ratio"] - baseline_bare[l]["ratio"]
                                      for l in SAMPLE_LAYERS if l in baseline_ccs and l in baseline_bare])
        enrichment_degraded = np.mean([prof_ccs[l]["ratio"] - prof_bare[l]["ratio"]
                                        for l in SAMPLE_LAYERS if l in prof_ccs and l in prof_bare])

        print(f"  Profile similarity (σ₂/σ₁): CCS={sim_ratio_ccs:.4f} bare={sim_ratio_bare:.4f}")
        print(f"  Profile similarity (σ₁): {sim_sigma1:.4f}")
        print(f"  Mean enrichment: intact={enrichment_intact:.4f} degraded={enrichment_degraded:.4f}")

        key = f"prune_{prune_frac}"
        results["conditions"][key] = {
            "ccs": {str(l): prof_ccs[l] for l in SAMPLE_LAYERS if l in prof_ccs},
            "bare": {str(l): prof_bare[l] for l in SAMPLE_LAYERS if l in prof_bare},
            "similarity_ratio_ccs": sim_ratio_ccs,
            "similarity_ratio_bare": sim_ratio_bare,
            "similarity_sigma1": sim_sigma1,
            "pruned_fraction": pruned / total,
            "mean_enrichment_intact": float(enrichment_intact),
            "mean_enrichment_degraded": float(enrichment_degraded),
        }

        del model
        torch.cuda.empty_cache()

    # Extreme noise
    for noise_scale in [0.5, 1.0]:
        print(f"\n=== NOISE scale={noise_scale} ===")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model.eval()

        torch.manual_seed(42)
        add_noise(model, noise_scale)

        prof_ccs = compute_spectral_profile(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, SAMPLE_LAYERS)
        prof_bare = compute_spectral_profile(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT, SAMPLE_LAYERS)

        sim_ratio_ccs = profile_similarity(baseline_ccs, prof_ccs, "ratio")
        sim_ratio_bare = profile_similarity(baseline_bare, prof_bare, "ratio")
        sim_sigma1 = profile_similarity(baseline_ccs, prof_ccs, "sigma1")

        print(f"  Profile similarity (σ₂/σ₁): CCS={sim_ratio_ccs:.4f} bare={sim_ratio_bare:.4f}")
        print(f"  Profile similarity (σ₁): {sim_sigma1:.4f}")

        key = f"noise_{noise_scale}"
        results["conditions"][key] = {
            "ccs": {str(l): prof_ccs[l] for l in SAMPLE_LAYERS if l in prof_ccs},
            "bare": {str(l): prof_bare[l] for l in SAMPLE_LAYERS if l in prof_bare},
            "similarity_ratio_ccs": sim_ratio_ccs,
            "similarity_ratio_bare": sim_ratio_bare,
            "similarity_sigma1": sim_sigma1,
        }

        del model
        torch.cuda.empty_cache()

    out_path = "/workspace/results_degradation_breaking_point.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\n{'='*60}")
    print("  DEGRADATION BREAKING POINT SUMMARY")
    print(f"{'='*60}")
    for cond_name, cond_data in results["conditions"].items():
        if cond_name == "baseline":
            continue
        sim_r = cond_data.get("similarity_ratio_ccs", "N/A")
        sim_s = cond_data.get("similarity_sigma1", "N/A")
        me_d = cond_data.get("mean_enrichment_degraded", "N/A")
        if isinstance(sim_r, float):
            extra = f" enrichment={me_d:.4f}" if isinstance(me_d, float) else ""
            print(f"  {cond_name}: σ₂/σ₁ sim={sim_r:.4f}  σ₁ sim={sim_s:.4f}{extra}")
        else:
            print(f"  {cond_name}: {sim_r}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
