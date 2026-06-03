#!/usr/bin/env python3
"""Exp 6: γ Ablation — Causal Test of Wire Mechanism

Correlation ≠ causation. We've shown γ-V₂ anti-correlation at tunnel layers
across 3 models. Now test: does γ CAUSE V₂ routing, or are they both
downstream of some other structural feature?

Method:
1. Normal forward pass → measure r(γ, V₂) at L16 (peak layer)
2. Set γ → 1.0 (uniform) → re-run → measure r(original_γ, V₂_ablated)
3. Set γ → shuffled (same spectrum, random channel assignment) → re-run

If uniform γ kills the correlation: γ is causally routing V₂.
If shuffled γ kills it: the specific channel assignments matter, not just the spectrum.
If neither kills it: V₂ structure comes from upstream, γ is epiphenomenal.
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

MODEL_PATH = "/mnt/hdd/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

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

ABLATION_LAYERS = [14, 15, 16, 17, 18]
N_SHUFFLE_TRIALS = 5


def compute_v2_stats(gamma_original, residual_matrix):
    U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)
    if len(S) < 2:
        return None
    V2 = Vt[1]
    V1 = Vt[0]
    gamma_abs = np.abs(gamma_original)
    v2_abs = np.abs(V2)
    v1_abs = np.abs(V1)

    r_v2, p_v2 = stats.pearsonr(gamma_abs, v2_abs)
    r_v1, p_v1 = stats.pearsonr(gamma_abs, v1_abs)
    rho_v2, _ = stats.spearmanr(gamma_abs, v2_abs)

    k = max(1, len(gamma_original) // 10)
    top_gamma = set(np.argsort(gamma_abs)[-k:])
    top_v2 = set(np.argsort(v2_abs)[-k:])
    overlap = len(top_gamma & top_v2) / k

    return {
        "r_v2": float(r_v2), "p_v2": float(p_v2),
        "rho_v2": float(rho_v2),
        "r_v1": float(r_v1), "p_v1": float(p_v1),
        "sigma_1": float(S[0]), "sigma_2": float(S[1]),
        "gap": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
        "top10_overlap": float(overlap),
        "v2_vector": V2.tolist(),
    }


def run_forward(model, tokenizer, texts, target_layers):
    import torch
    all_hidden = {l: [] for l in target_layers}
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        for layer_idx in target_layers:
            h = outputs.hidden_states[layer_idx + 1]
            h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1)
            all_hidden[layer_idx].append(h_mean.squeeze(0).float().numpy())
        del outputs
    return {l: np.stack(vecs) for l, vecs in all_hidden.items()}


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading Qwen2.5-3B for γ ablation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu",
        output_hidden_states=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Store original γ for each layer
    original_gammas = {}
    for layer_idx in ABLATION_LAYERS:
        original_gammas[layer_idx] = model.model.layers[layer_idx].input_layernorm.weight.data.clone()

    results = {}

    for ablation_layer in ABLATION_LAYERS:
        print(f"\n{'='*70}")
        print(f"ABLATING L{ablation_layer}")
        print(f"{'='*70}")

        original_gamma_np = original_gammas[ablation_layer].float().numpy()

        for condition, texts in PROBES.items():
            print(f"\n  Condition: {condition}")

            # 1. Normal (no ablation)
            model.model.layers[ablation_layer].input_layernorm.weight.data = original_gammas[ablation_layer].clone()
            hidden_normal = run_forward(model, tokenizer, texts, [ablation_layer])
            stats_normal = compute_v2_stats(original_gamma_np, hidden_normal[ablation_layer])
            print(f"    Normal:   r(γ,V₂)={stats_normal['r_v2']:+.4f}  σ₂={stats_normal['sigma_2']:.1f}  gap={stats_normal['gap']:.1f}")

            # 2. Uniform (γ → 1.0)
            model.model.layers[ablation_layer].input_layernorm.weight.data = torch.ones_like(original_gammas[ablation_layer])
            hidden_uniform = run_forward(model, tokenizer, texts, [ablation_layer])
            stats_uniform = compute_v2_stats(original_gamma_np, hidden_uniform[ablation_layer])
            print(f"    Uniform:  r(γ,V₂)={stats_uniform['r_v2']:+.4f}  σ₂={stats_uniform['sigma_2']:.1f}  gap={stats_uniform['gap']:.1f}")

            # 3. Shuffled (multiple trials)
            shuffle_rs = []
            for trial in range(N_SHUFFLE_TRIALS):
                shuffled = original_gammas[ablation_layer].clone()
                perm = torch.randperm(shuffled.shape[0])
                shuffled = shuffled[perm]
                model.model.layers[ablation_layer].input_layernorm.weight.data = shuffled
                hidden_shuf = run_forward(model, tokenizer, texts, [ablation_layer])
                stats_shuf = compute_v2_stats(original_gamma_np, hidden_shuf[ablation_layer])
                shuffle_rs.append(stats_shuf['r_v2'])

            mean_shuf = np.mean(shuffle_rs)
            std_shuf = np.std(shuffle_rs)
            print(f"    Shuffled: r(γ,V₂)={mean_shuf:+.4f} ± {std_shuf:.4f}  (n={N_SHUFFLE_TRIALS})")

            # V₂ similarity between normal and ablated
            v2_normal = np.array(stats_normal['v2_vector'])
            v2_uniform = np.array(stats_uniform['v2_vector'])
            v2_cosine = np.dot(v2_normal, v2_uniform) / (np.linalg.norm(v2_normal) * np.linalg.norm(v2_uniform) + 1e-10)
            print(f"    V₂ cosine(normal, uniform): {v2_cosine:+.4f}")

            delta_r = stats_normal['r_v2'] - stats_uniform['r_v2']
            pct_change = (delta_r / abs(stats_normal['r_v2'])) * 100 if stats_normal['r_v2'] != 0 else 0
            print(f"    Δr = {delta_r:+.4f} ({pct_change:+.1f}%)")

            key = f"L{ablation_layer}_{condition}"
            results[key] = {
                "layer": ablation_layer,
                "condition": condition,
                "normal_r": stats_normal['r_v2'],
                "uniform_r": stats_uniform['r_v2'],
                "shuffle_mean_r": float(mean_shuf),
                "shuffle_std_r": float(std_shuf),
                "delta_r": float(delta_r),
                "pct_change": float(pct_change),
                "v2_cosine": float(v2_cosine),
                "normal_sigma2": stats_normal['sigma_2'],
                "uniform_sigma2": stats_uniform['sigma_2'],
                "normal_gap": stats_normal['gap'],
                "uniform_gap": stats_uniform['gap'],
            }

            # Restore original weights
            model.model.layers[ablation_layer].input_layernorm.weight.data = original_gammas[ablation_layer].clone()

    # Summary
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")

    for layer_idx in ABLATION_LAYERS:
        layer_results = {k: v for k, v in results.items() if v['layer'] == layer_idx}
        if not layer_results:
            continue
        normal_rs = [v['normal_r'] for v in layer_results.values()]
        uniform_rs = [v['uniform_r'] for v in layer_results.values()]
        shuffle_rs = [v['shuffle_mean_r'] for v in layer_results.values()]
        pcts = [v['pct_change'] for v in layer_results.values()]
        cosines = [v['v2_cosine'] for v in layer_results.values()]

        print(f"  L{layer_idx}: normal r={np.mean(normal_rs):+.4f}  "
              f"uniform r={np.mean(uniform_rs):+.4f}  "
              f"shuffled r={np.mean(shuffle_rs):+.4f}  "
              f"Δ%={np.mean(pcts):+.1f}%  "
              f"V₂cos={np.mean(cosines):+.4f}")

    # Interpretation
    mean_normal = np.mean([v['normal_r'] for v in results.values()])
    mean_uniform = np.mean([v['uniform_r'] for v in results.values()])
    mean_shuffle = np.mean([v['shuffle_mean_r'] for v in results.values()])
    mean_pct = np.mean([v['pct_change'] for v in results.values()])

    print(f"\n  Overall: normal={mean_normal:+.4f}  uniform={mean_uniform:+.4f}  "
          f"shuffled={mean_shuffle:+.4f}  mean_Δ%={mean_pct:+.1f}%")

    if abs(mean_pct) > 50:
        print("  → γ is CAUSAL: flattening it substantially reduces V₂ organization along γ channels")
    elif abs(mean_pct) > 20:
        print("  → γ is PARTIALLY causal: local γ contributes but upstream structure also matters")
    else:
        print("  → γ is NOT locally causal: V₂ structure comes primarily from upstream layers")

    if abs(mean_shuffle) < abs(mean_normal) * 0.5:
        print("  → Channel-specific assignment matters (shuffling disrupts as much as flattening)")
    elif abs(mean_shuffle) > abs(mean_normal) * 0.8:
        print("  → Spectrum shape matters more than channel assignment (shuffling preserves correlation)")

    out_path = RESULTS_DIR / "exp_gamma_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
