#!/usr/bin/env python3
"""Exp: Scale Vector Spectrum — GQA vs MHA

Hypothesis: The γ (scale vector) spectrum in RMSNorm layers shows structural
differences between GQA and MHA models that correlate with witness sensitivity.

Specifically:
- GQA models (Mistral) share KV across heads, creating unified preconditioning
  of the shared attention space
- MHA models (LLaMA-1) have per-head KV, fragmenting preconditioning
- The γ spectrum at tunnel layers (L17) should show more concentration/hierarchy
  in GQA models, correlating with σ₂ channel availability

Method: Extract RMSNorm γ weights from safetensors (CPU only, no GPU needed).
Compare spectrum properties: entropy, Gini coefficient, top-k concentration,
and the ratio of γ variance between Q/K/V branches.

Wang et al. (2605.26895) show γ is expressively redundant but creates
self-amplifying preconditioners P = γ²I + wwᵀ. We test whether this
preconditioning structure differs between architectures that show vs don't
show witness sensitivity.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

HF_HOME = os.environ.get("HF_HOME", "/mnt/hdd/huggingface")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_gamma_vectors(model_path):
    """Extract all RMSNorm/LayerNorm gamma vectors from safetensors."""
    from safetensors import safe_open
    import torch

    gammas = {}
    safetensor_files = sorted(Path(model_path).glob("*.safetensors"))

    if not safetensor_files:
        print(f"No safetensors files in {model_path}")
        return gammas

    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt") as f:
            for key in f.keys():
                if "norm" in key.lower() and "weight" in key.lower():
                    tensor = f.get_tensor(key)
                    gammas[key] = tensor.float().numpy()

    return gammas


def spectrum_stats(gamma):
    """Compute spectrum statistics for a gamma vector."""
    g = np.abs(gamma.flatten())
    g_sorted = np.sort(g)[::-1]

    # Normalize for distribution analysis
    g_norm = g / (g.sum() + 1e-10)

    # Entropy (how uniform is the distribution)
    g_pos = g_norm[g_norm > 0]
    entropy = -np.sum(g_pos * np.log(g_pos))
    max_entropy = np.log(len(g))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Gini coefficient (inequality measure)
    n = len(g_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * g_sorted) / (n * np.sum(g_sorted))) - (n + 1) / n

    # Top-k concentration
    total = g.sum()
    top10_frac = g_sorted[:max(1, len(g)//100*10)].sum() / total if total > 0 else 0
    top1_frac = g_sorted[:max(1, len(g)//100)].sum() / total if total > 0 else 0

    # Variance and CV
    cv = np.std(g) / (np.mean(g) + 1e-10)

    # Spectral gap (ratio of largest to second-largest)
    spectral_gap = g_sorted[0] / (g_sorted[1] + 1e-10) if len(g_sorted) > 1 else 0

    return {
        "mean": float(np.mean(g)),
        "std": float(np.std(g)),
        "cv": float(cv),
        "min": float(np.min(g)),
        "max": float(np.max(g)),
        "range": float(np.max(g) - np.min(g)),
        "normalized_entropy": float(normalized_entropy),
        "gini": float(gini),
        "top10pct_concentration": float(top10_frac),
        "top1pct_concentration": float(top1_frac),
        "spectral_gap": float(spectral_gap),
        "dim": int(len(g)),
    }


def classify_norm_layer(key, model_type):
    """Classify a norm layer as input-norm or output-norm per Wang et al."""
    key_lower = key.lower()

    if "input_layernorm" in key_lower or "pre_norm" in key_lower:
        return "input_norm"  # Pre-Norm before attention
    elif "post_attention" in key_lower or "post_norm" in key_lower:
        return "input_norm"  # Pre-Norm before FFN (still input-side)
    elif "q_norm" in key_lower or "k_norm" in key_lower:
        return "output_norm"  # Q/K-Norm (output-side)
    elif "final" in key_lower or "model.norm" in key_lower:
        return "output_norm"  # Final norm
    else:
        return "input_norm"  # Default to input-norm for standard Pre-Norm


def extract_layer_number(key):
    """Extract layer number from parameter key."""
    import re
    match = re.search(r'layers\.(\d+)', key)
    if match:
        return int(match.group(1))
    return -1


def analyze_model(model_path, model_name, model_type, num_layers):
    """Full analysis of a model's scale vectors."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name} ({model_type})")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    gammas = load_gamma_vectors(model_path)

    if not gammas:
        print("  No gamma vectors found!")
        return None

    print(f"  Found {len(gammas)} norm weight vectors")

    results = {
        "model": model_name,
        "type": model_type,
        "num_layers": num_layers,
        "layers": {},
        "summary": {},
    }

    # Per-layer analysis
    layer_stats = {}
    for key, gamma in sorted(gammas.items()):
        layer_num = extract_layer_number(key)
        norm_type = classify_norm_layer(key, model_type)
        stats = spectrum_stats(gamma)
        stats["key"] = key
        stats["norm_type"] = norm_type
        stats["layer"] = layer_num

        if layer_num not in layer_stats:
            layer_stats[layer_num] = []
        layer_stats[layer_num].append(stats)

        # Print key layers
        if layer_num in [0, 1, num_layers//4, num_layers//2,
                         int(num_layers*0.53), int(num_layers*0.75),
                         num_layers-2, num_layers-1, -1]:
            print(f"  L{layer_num:2d} {norm_type:12s} | "
                  f"cv={stats['cv']:.4f} gini={stats['gini']:.4f} "
                  f"entropy={stats['normalized_entropy']:.4f} "
                  f"range={stats['range']:.4f} | {key}")

    results["layers"] = {
        str(k): v for k, v in layer_stats.items()
    }

    # Aggregate statistics by depth zone
    tunnel_layers = list(range(num_layers//4, int(num_layers*0.6)))
    relay_layers = list(range(int(num_layers*0.8), num_layers))
    early_layers = list(range(0, num_layers//4))

    for zone_name, zone_layers in [("early", early_layers),
                                     ("tunnel", tunnel_layers),
                                     ("relay", relay_layers)]:
        zone_cvs = []
        zone_ginis = []
        zone_entropies = []
        zone_ranges = []

        for layer_num in zone_layers:
            if layer_num in layer_stats:
                for s in layer_stats[layer_num]:
                    if s["norm_type"] == "input_norm":
                        zone_cvs.append(s["cv"])
                        zone_ginis.append(s["gini"])
                        zone_entropies.append(s["normalized_entropy"])
                        zone_ranges.append(s["range"])

        if zone_cvs:
            results["summary"][zone_name] = {
                "mean_cv": float(np.mean(zone_cvs)),
                "mean_gini": float(np.mean(zone_ginis)),
                "mean_entropy": float(np.mean(zone_entropies)),
                "mean_range": float(np.mean(zone_ranges)),
                "n_layers": len(zone_layers),
            }
            print(f"\n  {zone_name.upper():6s} zone (L{zone_layers[0]}-L{zone_layers[-1]}): "
                  f"cv={np.mean(zone_cvs):.4f} gini={np.mean(zone_ginis):.4f} "
                  f"entropy={np.mean(zone_entropies):.4f}")

    return results


def main():
    # Model paths — check what we have locally
    models = []

    hub_path = Path(HF_HOME) / "hub"

    def find_model(pattern, name, arch_type, num_layers):
        paths = list(hub_path.glob(pattern))
        if not paths:
            return None
        snapshot = paths[0] / "snapshots"
        if not snapshot.exists():
            return None
        snaps = list(snapshot.iterdir())
        if not snaps:
            return None
        sf = list(snaps[0].glob("*.safetensors"))
        if not sf:
            return None
        return {"path": str(snaps[0]), "name": name, "type": arch_type, "num_layers": num_layers}

    # GQA models
    m = find_model("models--google--gemma-2-2b*", "Gemma-2-2B", "GQA", 26)
    if m: models.append(m)

    m = find_model("models--Qwen--Qwen2.5-3B*", "Qwen2.5-3B", "GQA", 36)
    if m: models.append(m)

    m = find_model("models--Qwen--Qwen2.5-7B*", "Qwen2.5-7B", "GQA", 28)
    if m: models.append(m)

    # MHA models
    m = find_model("models--EleutherAI--pythia-6.9b*", "Pythia-6.9B", "MHA", 32)
    if m: models.append(m)

    m = find_model("models--openai-community--gpt2-large*", "GPT2-Large", "MHA", 36)
    if m: models.append(m)

    m = find_model("models--gpt2", "GPT2", "MHA", 12)
    if m: models.append(m)

    if not models:
        print("No models found in HF_HOME. Checking available:")
        if hub_path.exists():
            for d in sorted(hub_path.iterdir()):
                if d.name.startswith("models--"):
                    print(f"  {d.name}")
        return

    print(f"Found {len(models)} models to analyze")

    all_results = {}
    for model_info in models:
        try:
            result = analyze_model(
                model_info["path"],
                model_info["name"],
                model_info["type"],
                model_info["num_layers"],
            )
            if result:
                all_results[model_info["name"]] = result
        except Exception as e:
            print(f"  ERROR analyzing {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-model comparison
    if len(all_results) >= 2:
        print(f"\n{'='*60}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*60}")

        for zone in ["early", "tunnel", "relay"]:
            print(f"\n  {zone.upper()} zone:")
            for name, result in all_results.items():
                if zone in result.get("summary", {}):
                    s = result["summary"][zone]
                    arch = result["type"]
                    print(f"    {name:20s} ({arch:3s}): "
                          f"cv={s['mean_cv']:.4f} gini={s['mean_gini']:.4f} "
                          f"entropy={s['mean_entropy']:.4f} range={s['mean_range']:.4f}")

        # Test: do GQA models show different γ spectrum properties at tunnel?
        gqa_tunnel_cvs = []
        mha_tunnel_cvs = []
        for name, result in all_results.items():
            if "tunnel" in result.get("summary", {}):
                cv = result["summary"]["tunnel"]["mean_cv"]
                if result["type"] == "GQA":
                    gqa_tunnel_cvs.append(cv)
                else:
                    mha_tunnel_cvs.append(cv)

        if gqa_tunnel_cvs and mha_tunnel_cvs:
            gqa_mean = np.mean(gqa_tunnel_cvs)
            mha_mean = np.mean(mha_tunnel_cvs)
            print(f"\n  GQA tunnel mean CV: {gqa_mean:.4f}")
            print(f"  MHA tunnel mean CV: {mha_mean:.4f}")
            print(f"  Ratio (GQA/MHA): {gqa_mean/mha_mean:.3f}")
            if gqa_mean > mha_mean:
                print("  → GQA shows MORE heterogeneous γ spectrum at tunnel")
            else:
                print("  → MHA shows MORE heterogeneous γ spectrum at tunnel")

    # Save results
    out_path = RESULTS_DIR / "exp_scale_vectors.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
