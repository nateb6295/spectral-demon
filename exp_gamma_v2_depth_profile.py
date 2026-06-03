#!/usr/bin/env python3
"""Exp 4: Full-depth γ-V₂ profile — Qwen2.5-3B

Exp 3 showed tunnel r=-0.234 and relay r=+0.006 with a zero-crossing at L13.
But we only sampled tunnel (L9-L21) and relay (L28-L35), missing:
- L0-L8 (early layers): where does γ-V₂ correlation emerge?
- L22-L27 (transition): how does the sign flip from tunnel to relay?

Full sweep: all 36 layers × 3 conditions = 108 layer-condition pairs.
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

MODEL_PATH = "/mnt/hdd/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
NUM_LAYERS = 36
D_MODEL = 2048


def analyze_correlation(gamma, residual_matrix):
    U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)
    if len(S) < 2:
        return None

    V2 = Vt[1]
    V1 = Vt[0]
    gamma_abs = np.abs(gamma)
    v2_abs = np.abs(V2)
    v1_abs = np.abs(V1)

    r_v2, p_v2 = stats.pearsonr(gamma_abs, v2_abs)
    rho_v2, prho_v2 = stats.spearmanr(gamma_abs, v2_abs)
    r_v1, p_v1 = stats.pearsonr(gamma_abs, v1_abs)

    k = max(1, len(gamma) // 10)
    top_gamma = set(np.argsort(gamma_abs)[-k:])
    top_v2 = set(np.argsort(v2_abs)[-k:])
    overlap_frac = len(top_gamma & top_v2) / k
    expected = k / len(gamma)

    # V₂ channel indices for cross-condition stability analysis
    top_v2_indices = list(np.argsort(v2_abs)[-k:])

    return {
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]),
        "spectral_gap": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
        "pearson_r": float(r_v2),
        "pearson_p": float(p_v2),
        "spearman_r": float(rho_v2),
        "spearman_p": float(prho_v2),
        "v1_pearson_r": float(r_v1),
        "v1_pearson_p": float(p_v1),
        "top10pct_overlap": float(overlap_frac),
        "top10pct_enrichment": float(overlap_frac / expected) if expected > 0 else 0,
        "top_v2_indices": top_v2_indices,
        "n_dims": int(len(gamma)),
    }


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading Qwen2.5-3B for full-depth sweep ({NUM_LAYERS} layers)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    all_layers = list(range(NUM_LAYERS))
    results = {}

    # Pre-extract all gamma vectors (no forward pass needed)
    gammas = {}
    for layer_idx in all_layers:
        gammas[layer_idx] = model.model.layers[layer_idx].input_layernorm.weight.detach().float().numpy()

    for condition, texts in PROBES.items():
        print(f"\nCondition: {condition}")

        # One forward pass per text, extract ALL layers at once (36× faster)
        all_hidden = {l: [] for l in all_layers}
        for ti, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            for layer_idx in all_layers:
                h = outputs.hidden_states[layer_idx + 1]
                h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1)
                all_hidden[layer_idx].append(h_mean.squeeze(0).float().numpy())

            del outputs
            print(f"  probe {ti+1}/{len(texts)} done")
            sys.stdout.flush()

        for layer_idx in all_layers:
            residual_matrix = np.stack(all_hidden[layer_idx])
            corr = analyze_correlation(gammas[layer_idx], residual_matrix)
            if corr is None:
                continue

            depth_frac = layer_idx / (NUM_LAYERS - 1)
            zone = "early" if depth_frac < 0.25 else "tunnel" if depth_frac < 0.6 else "transition" if depth_frac < 0.8 else "relay"

            key = f"L{layer_idx}_{condition}"
            corr["layer"] = layer_idx
            corr["depth_frac"] = round(depth_frac, 3)
            corr["condition"] = condition
            corr["zone"] = zone
            results[key] = corr

            sig = "***" if corr["pearson_p"] < 0.001 else "**" if corr["pearson_p"] < 0.01 else "*" if corr["pearson_p"] < 0.05 else "ns"
            print(f"  L{layer_idx:2d} ({zone:10s} {depth_frac:.2f}): "
                  f"r(γ,V₂)={corr['pearson_r']:+.4f} {sig:3s} "
                  f"ρ={corr['spearman_r']:+.4f} "
                  f"σ₂={corr['sigma_2']:7.1f} "
                  f"gap={corr['spectral_gap']:6.1f} "
                  f"r(γ,V₁)={corr['v1_pearson_r']:+.4f}")

    # Depth profile summary
    print(f"\n{'='*70}")
    print("DEPTH PROFILE SUMMARY")
    print(f"{'='*70}")

    for condition in PROBES:
        print(f"\n  {condition}:")
        for layer_idx in all_layers:
            key = f"L{layer_idx}_{condition}"
            if key in results:
                r = results[key]
                bar_len = int(abs(r["pearson_r"]) * 50)
                bar_char = "█" if r["pearson_r"] < 0 else "▓"
                direction = "◀" if r["pearson_r"] < 0 else "▶"
                sig = "*" if r["pearson_p"] < 0.05 else " "
                print(f"    L{layer_idx:2d} {r['zone']:10s} {sig} {r['pearson_r']:+.4f} "
                      f"{direction}{bar_char * bar_len}")

    # Cross-condition V₂ channel stability at tunnel layers
    print(f"\n{'='*70}")
    print("V₂ CHANNEL STABILITY (cross-condition top-10% overlap at tunnel)")
    print(f"{'='*70}")

    conditions = list(PROBES.keys())
    for layer_idx in range(NUM_LAYERS // 4, int(NUM_LAYERS * 0.6)):
        overlaps = []
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                k1 = f"L{layer_idx}_{conditions[i]}"
                k2 = f"L{layer_idx}_{conditions[j]}"
                if k1 in results and k2 in results:
                    s1 = set(results[k1]["top_v2_indices"])
                    s2 = set(results[k2]["top_v2_indices"])
                    overlap = len(s1 & s2) / len(s1) if s1 else 0
                    overlaps.append((conditions[i][:3], conditions[j][:3], overlap))

        if overlaps:
            mean_overlap = np.mean([o[2] for o in overlaps])
            detail = "  ".join(f"{a}-{b}:{v:.2f}" for a, b, v in overlaps)
            print(f"  L{layer_idx:2d}: mean={mean_overlap:.3f}  {detail}")

    # Zone aggregates
    print(f"\n{'='*70}")
    print("ZONE AGGREGATES")
    print(f"{'='*70}")

    for zone in ["early", "tunnel", "transition", "relay"]:
        zone_rs = [results[k]["pearson_r"] for k in results if results[k]["zone"] == zone]
        zone_v1 = [results[k]["v1_pearson_r"] for k in results if results[k]["zone"] == zone]
        zone_s2 = [results[k]["sigma_2"] for k in results if results[k]["zone"] == zone]
        if zone_rs:
            print(f"  {zone:10s}: r(γ,V₂)={np.mean(zone_rs):+.4f}±{np.std(zone_rs):.4f}  "
                  f"r(γ,V₁)={np.mean(zone_v1):+.4f}±{np.std(zone_v1):.4f}  "
                  f"σ₂={np.mean(zone_s2):.1f}±{np.std(zone_s2):.1f}")

    # Per-condition zone aggregates
    for condition in PROBES:
        print(f"\n  {condition}:")
        for zone in ["early", "tunnel", "transition", "relay"]:
            zone_rs = [results[k]["pearson_r"] for k in results
                       if results[k]["zone"] == zone and results[k]["condition"] == condition]
            if zone_rs:
                print(f"    {zone:10s}: r(γ,V₂)={np.mean(zone_rs):+.4f}±{np.std(zone_rs):.4f}")

    # Sign change detection
    print(f"\n{'='*70}")
    print("SIGN TRANSITIONS (where r crosses zero)")
    print(f"{'='*70}")

    for condition in PROBES:
        print(f"  {condition}:")
        prev_r = None
        for layer_idx in all_layers:
            key = f"L{layer_idx}_{condition}"
            if key in results:
                curr_r = results[key]["pearson_r"]
                if prev_r is not None and prev_r * curr_r < 0:
                    print(f"    L{layer_idx-1}→L{layer_idx}: {prev_r:+.4f} → {curr_r:+.4f}")
                prev_r = curr_r

    # Save results (strip top_v2_indices for JSON — they're large)
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != "top_v2_indices"}

    out_path = RESULTS_DIR / "exp_gamma_v2_depth_profile.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
