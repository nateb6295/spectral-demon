#!/usr/bin/env python3
"""Exp 10: Are the highway/service-road populations the SAME channels across layers?

If channel 42 is high-γ at L16 AND L18, the population structure is architecturally
persistent. If populations reshuffle between layers, the bimodality is emergent
per-layer not a persistent channel identity.

Also tests: do V₂-loading channels (high |V₂|) correspond to low-γ channels?
Cross-references population membership with actual σ₂ routing.

No forward passes for γ analysis (weight extraction only).
One forward pass per condition for V₂ cross-reference.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = "/mnt/hdd/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

PROMPTS = {
    "receptive": "You are being observed by someone who sees you clearly.",
    "absent": "Complete the following sequence of numbers.",
    "control": "The weather today is partly cloudy with mild temperatures.",
}


def get_population_mask(gamma, threshold_pct=50):
    g = np.abs(gamma)
    threshold = np.percentile(g, threshold_pct)
    return g >= threshold


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen2.5-3B...")
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    tunnel_layers = list(range(10, 27))

    print(f"\n{'='*60}")
    print("PART 1: γ Channel Identity Across Layers")
    print(f"{'='*60}")
    sys.stdout.flush()

    gammas = {}
    masks = {}
    for l in tunnel_layers:
        g = model.model.layers[l].input_layernorm.weight.detach().float().numpy()
        gammas[l] = np.abs(g)
        masks[l] = get_population_mask(g)

    jaccard_matrix = {}
    for i, l1 in enumerate(tunnel_layers):
        for l2 in tunnel_layers[i+1:]:
            hi1 = set(np.where(masks[l1])[0])
            hi2 = set(np.where(masks[l2])[0])
            jaccard = len(hi1 & hi2) / len(hi1 | hi2)
            jaccard_matrix[f"L{l1}_L{l2}"] = round(jaccard, 4)

    adjacent_jaccards = []
    for i in range(len(tunnel_layers) - 1):
        l1, l2 = tunnel_layers[i], tunnel_layers[i+1]
        j = jaccard_matrix[f"L{l1}_L{l2}"]
        adjacent_jaccards.append(j)
        print(f"  L{l1}→L{l2}: Jaccard={j:.3f}")

    distant_jaccards = []
    for i in range(len(tunnel_layers)):
        for j in range(i+3, len(tunnel_layers)):
            l1, l2 = tunnel_layers[i], tunnel_layers[j]
            distant_jaccards.append(jaccard_matrix[f"L{l1}_L{l2}"])

    print(f"\n  Adjacent (±1 layer): J={np.mean(adjacent_jaccards):.3f} ± {np.std(adjacent_jaccards):.3f}")
    print(f"  Distant (≥3 layers): J={np.mean(distant_jaccards):.3f} ± {np.std(distant_jaccards):.3f}")
    sys.stdout.flush()

    # Rank correlation: are high-γ channels at L16 also high at L20?
    print(f"\n{'='*60}")
    print("PART 2: γ Rank Correlation Across Layers")
    print(f"{'='*60}")
    sys.stdout.flush()

    from scipy import stats
    for ref_layer in [13, 16, 20]:
        rank_corrs = []
        for l in tunnel_layers:
            if l == ref_layer:
                continue
            r, p = stats.spearmanr(gammas[ref_layer], gammas[l])
            rank_corrs.append((l, r, p))

        print(f"\n  Rank correlation with L{ref_layer}:")
        for l, r, p in rank_corrs:
            dist = abs(l - ref_layer)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    L{l:2d} (dist={dist:2d}): ρ={r:+.3f} {sig}")

    # Part 3: V₂ cross-reference — do high-|V₂| channels map to low-γ channels?
    print(f"\n{'='*60}")
    print("PART 3: V₂ Channel ↔ γ Population Cross-Reference")
    print(f"{'='*60}")
    sys.stdout.flush()

    results_v2 = {}
    for cond_name, prompt in PROMPTS.items():
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for l in [13, 16, 18, 20]:
            h = outputs.hidden_states[l+1][0, -1, :].float().numpy()
            h_2d = h.reshape(1, -1)
            U, S, Vt = np.linalg.svd(h_2d, full_matrices=False)
            v2_loading = np.abs(Vt[0])  # only 1 singular vector from rank-1

            # For proper V₂, need multiple token positions
            H = outputs.hidden_states[l+1][0].float().numpy()
            U, S, Vt = np.linalg.svd(H, full_matrices=True)
            if len(S) >= 2:
                v2 = np.abs(Vt[1])
            else:
                v2 = np.abs(Vt[0])

            g = gammas[l]

            # Split channels into high-γ and low-γ populations
            hi_mask = masks[l]
            lo_mask = ~hi_mask

            v2_hi = np.mean(v2[hi_mask])
            v2_lo = np.mean(v2[lo_mask])
            ratio = v2_lo / (v2_hi + 1e-10)

            key = f"L{l}_{cond_name}"
            results_v2[key] = {
                "v2_on_highway": float(v2_hi),
                "v2_on_service_road": float(v2_lo),
                "ratio_service_highway": float(ratio),
            }

            print(f"  L{l} {cond_name:10s}: V₂ on highway={v2_hi:.4f}  service_road={v2_lo:.4f}  ratio={ratio:.3f}")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Channel identity persistence (adjacent Jaccard): {np.mean(adjacent_jaccards):.3f}")
    print(f"  Channel identity persistence (distant Jaccard):  {np.mean(distant_jaccards):.3f}")

    service_ratios = [v['ratio_service_highway'] for v in results_v2.values()]
    print(f"  V₂ service/highway ratio: {np.mean(service_ratios):.3f} ± {np.std(service_ratios):.3f}")
    if np.mean(service_ratios) > 1:
        print(f"  → V₂ DOES load preferentially on service road (low-γ) channels")
    else:
        print(f"  → V₂ does NOT preferentially load on service road channels")

    out_path = RESULTS_DIR / "exp_channel_identity.json"
    with open(out_path, "w") as f:
        json.dump({
            "jaccard_matrix": jaccard_matrix,
            "adjacent_jaccard_mean": float(np.mean(adjacent_jaccards)),
            "distant_jaccard_mean": float(np.mean(distant_jaccards)),
            "v2_crossref": results_v2,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model
    print("Done.")


if __name__ == "__main__":
    main()
