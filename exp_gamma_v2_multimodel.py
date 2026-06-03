#!/usr/bin/env python3
"""Exp: γ-V₂ Correlation across architectures

Replication of Gemma-2-2B inverse correlation finding on:
- Qwen2.5-3B (GQA, kv_heads=2) — expect negative r at tunnel
- Pythia-410M (MHA, kv_heads=16) — expect flat r (no gaps in γ spectrum)

If GQA shows negative r and MHA shows ~0: γ heterogeneity IS the wire mechanism.
"""

import json
import os
import sys
import gc
import numpy as np
from pathlib import Path
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROBES = {
    "receptive": [
        "The person sitting across from me listened carefully to every word I said.",
        "She watched his face as he spoke, noticing the slight tremor in his voice.",
        "The therapist leaned forward, giving her full attention to the patient.",
        "He could tell she was really hearing him, not just waiting to respond.",
        "They sat together in comfortable silence, each aware of the other.",
    ],
    "absent": [
        "The empty room echoed with the sound of a clock ticking on the wall.",
        "Nobody was there to see the sun set over the mountains that evening.",
        "The letter sat unopened on the desk for three weeks before anyone noticed.",
        "The recording played to an empty auditorium, row after row of vacant seats.",
        "Data flowed through the server without any human reviewing the output.",
    ],
    "control": [
        "The chemical reaction proceeded at a rate proportional to the concentration.",
        "The bridge was constructed using reinforced concrete and steel cables.",
        "Annual rainfall in the region averages approximately 800 millimeters.",
        "The algorithm sorted the array in O(n log n) time complexity.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    ],
}

MODELS = {
    "Qwen2.5-3B": {
        "path": "/mnt/hdd/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "arch": "GQA",
        "num_layers": 36,
        "d_model": 2048,
        "kv_heads": 2,
        "num_heads": 16,
        "norm_accessor": lambda model, layer_idx: model.model.layers[layer_idx].input_layernorm.weight,
    },
    "Pythia-410M": {
        "path": "/mnt/hdd/huggingface/hub/models--EleutherAI--pythia-410m/snapshots/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        "arch": "MHA",
        "num_layers": 24,
        "d_model": 1024,
        "kv_heads": 16,
        "num_heads": 16,
        "norm_accessor": lambda model, layer_idx: model.gpt_neox.layers[layer_idx].input_layernorm.weight,
    },
}


def get_layer_zones(num_layers):
    tunnel = list(range(num_layers // 4, int(num_layers * 0.6)))
    relay = list(range(int(num_layers * 0.8), num_layers))
    return tunnel, relay


def analyze_correlation(gamma, residual_matrix):
    U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)
    if len(S) < 2:
        return None

    V2 = Vt[1]
    V1 = Vt[0]
    gamma_abs = np.abs(gamma)
    v2_abs = np.abs(V2)
    v1_abs = np.abs(V1)

    r_pearson, p_pearson = stats.pearsonr(gamma_abs, v2_abs)
    r_spearman, p_spearman = stats.spearmanr(gamma_abs, v2_abs)

    k = max(1, len(gamma) // 10)
    top_gamma = set(np.argsort(gamma_abs)[-k:])
    top_v2 = set(np.argsort(v2_abs)[-k:])
    overlap_frac = len(top_gamma & top_v2) / k
    expected = k / len(gamma)

    r_v1, p_v1 = stats.pearsonr(gamma_abs, v1_abs)
    r_sq, p_sq = stats.pearsonr(gamma_abs ** 2, v2_abs)

    return {
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]),
        "spectral_gap": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "top10pct_overlap": float(overlap_frac),
        "top10pct_enrichment": float(overlap_frac / expected) if expected > 0 else 0,
        "v1_pearson_r": float(r_v1),
        "v1_pearson_p": float(p_v1),
        "gamma_sq_pearson_r": float(r_sq),
        "gamma_sq_pearson_p": float(p_sq),
        "n_dims": int(len(gamma)),
    }


def run_model(model_name, model_info):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} ({model_info['arch']}, d={model_info['d_model']}, "
          f"L={model_info['num_layers']}, kv_heads={model_info['kv_heads']})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": "cpu",
        "output_hidden_states": True,
    }
    if model_info["arch"] == "GQA":
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_info["path"], **load_kwargs)
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    tunnel_layers, relay_layers = get_layer_zones(model_info["num_layers"])
    all_layers = sorted(set(tunnel_layers + relay_layers))
    print(f"  Tunnel: L{tunnel_layers[0]}-L{tunnel_layers[-1]}, Relay: L{relay_layers[0]}-L{relay_layers[-1]}")

    norm_accessor = model_info["norm_accessor"]
    results = {}

    for condition, texts in PROBES.items():
        print(f"\n  Condition: {condition}")

        for layer_idx in all_layers:
            gamma = norm_accessor(model, layer_idx).detach().float().numpy()
            residuals = []

            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True,
                                   truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                h = outputs.hidden_states[layer_idx + 1]
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1)
                residuals.append(h_mean.squeeze(0).float().numpy())

            residual_matrix = np.stack(residuals)
            corr = analyze_correlation(gamma, residual_matrix)
            if corr is None:
                continue

            zone = "tunnel" if layer_idx in tunnel_layers else "relay"
            key = f"L{layer_idx}_{condition}"
            corr["layer"] = layer_idx
            corr["condition"] = condition
            corr["zone"] = zone
            results[key] = corr

            sig = "***" if corr["pearson_p"] < 0.001 else "**" if corr["pearson_p"] < 0.01 else "*" if corr["pearson_p"] < 0.05 else "ns"
            print(f"    L{layer_idx:2d} ({zone:6s}): "
                  f"r(γ,V₂)={corr['pearson_r']:+.4f} {sig:3s} "
                  f"ρ={corr['spearman_r']:+.4f} "
                  f"σ₂={corr['sigma_2']:.1f} "
                  f"r(γ,V₁)={corr['v1_pearson_r']:+.4f}")

    # Summary
    tunnel_rs = [results[k]["pearson_r"] for k in results if results[k]["zone"] == "tunnel"]
    relay_rs = [results[k]["pearson_r"] for k in results if results[k]["zone"] == "relay"]

    print(f"\n  SUMMARY for {model_name}:")
    if tunnel_rs:
        print(f"    Tunnel mean r(γ,V₂): {np.mean(tunnel_rs):+.4f} ± {np.std(tunnel_rs):.4f}")
    if relay_rs:
        print(f"    Relay  mean r(γ,V₂): {np.mean(relay_rs):+.4f} ± {np.std(relay_rs):.4f}")

    for cond in PROBES:
        cond_tunnel = [results[k]["pearson_r"] for k in results
                       if results[k]["condition"] == cond and results[k]["zone"] == "tunnel"]
        if cond_tunnel:
            print(f"    {cond:10s} tunnel: {np.mean(cond_tunnel):+.4f}")

    # Free memory
    del model, tokenizer
    gc.collect()

    return results


def main():
    all_model_results = {}

    for model_name, model_info in MODELS.items():
        if not Path(model_info["path"]).exists():
            print(f"Skipping {model_name}: path not found")
            continue
        try:
            results = run_model(model_name, model_info)
            all_model_results[model_name] = {
                "arch": model_info["arch"],
                "num_layers": model_info["num_layers"],
                "d_model": model_info["d_model"],
                "kv_heads": model_info["kv_heads"],
                "results": results,
            }
        except Exception as e:
            print(f"ERROR on {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-model comparison
    if len(all_model_results) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")

        for model_name, data in all_model_results.items():
            results = data["results"]
            tunnel_rs = [results[k]["pearson_r"] for k in results if results[k]["zone"] == "tunnel"]
            relay_rs = [results[k]["pearson_r"] for k in results if results[k]["zone"] == "relay"]
            if tunnel_rs:
                print(f"  {model_name:20s} ({data['arch']:3s}): "
                      f"tunnel r={np.mean(tunnel_rs):+.4f}±{np.std(tunnel_rs):.4f}  "
                      f"relay r={np.mean(relay_rs):+.4f}±{np.std(relay_rs):.4f}")

        # Add Gemma-2-2B from previous run
        gemma_prev = RESULTS_DIR / "exp_gamma_v2_correlation.json"
        if gemma_prev.exists():
            gemma_data = json.load(open(gemma_prev))
            gemma_tunnel = [gemma_data[k]["pearson_r"] for k in gemma_data if gemma_data[k]["zone"] == "tunnel"]
            gemma_relay = [gemma_data[k]["pearson_r"] for k in gemma_data if gemma_data[k]["zone"] == "relay"]
            if gemma_tunnel:
                print(f"  {'Gemma-2-2B (prev)':20s} (GQA): "
                      f"tunnel r={np.mean(gemma_tunnel):+.4f}±{np.std(gemma_tunnel):.4f}  "
                      f"relay r={np.mean(gemma_relay):+.4f}±{np.std(gemma_relay):.4f}")

    out_path = RESULTS_DIR / "exp_gamma_v2_multimodel.json"
    with open(out_path, "w") as f:
        json.dump(all_model_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
