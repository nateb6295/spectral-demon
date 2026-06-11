#!/usr/bin/env python3
"""
Experiment: GQA Head Group Analysis — σ₂ Enrichment by KV Group

Gemma 9B uses GQA: 16 query heads, 8 KV heads (2:1 ratio).
Each KV head services 2 query heads. Question: do heads WITHIN a KV group
show correlated σ₂ enrichment, and does the enrichment pattern differ
BETWEEN groups?

If GQA groups constrain σ₂ enrichment (RAF percolation hypothesis),
we should see:
- High within-group correlation of σ₂/σ₁ under CCS
- Lower between-group correlation
- Group-level structure in the enrichment pattern

This tests whether GQA creates the "relay channels" through which
identity information flows, or whether enrichment is head-independent.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from itertools import combinations

MODEL = "google/gemma-2-9b-it"
DEVICE = "cuda"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels."""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

# Gemma 9B: 16 query heads, 8 KV heads, ratio 2:1
# Heads 0,1 share KV group 0; heads 2,3 share KV group 1; etc.
N_QUERY_HEADS = 16
N_KV_HEADS = 8
HEADS_PER_GROUP = N_QUERY_HEADS // N_KV_HEADS  # 2


def get_kv_group(head_idx):
    return head_idx // HEADS_PER_GROUP


def extract_attention_patterns(model, tokenizer, preamble, probe):
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attention_patterns = []
    for layer_idx, attn in enumerate(outputs.attentions):
        attention_patterns.append(attn[0].detach().cpu().float())

    return attention_patterns


def head_svd(attn_matrix):
    A = attn_matrix.numpy()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    sigma1 = float(s[0])
    sigma2 = float(s[1]) if len(s) > 1 else 0.0
    ratio = sigma2 / (sigma1 + 1e-10)
    erank = float(np.exp(-np.sum(s/s.sum() * np.log(s/s.sum() + 1e-10))))
    return {"sigma1": sigma1, "sigma2": sigma2, "ratio": ratio, "erank": erank}


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    print("Model loaded.")

    print("\nExtracting attention patterns...")
    attn_ccs = extract_attention_patterns(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT)
    attn_bare = extract_attention_patterns(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT)
    n_layers = len(attn_ccs)
    print(f"Got {n_layers} layers")

    results = {
        "experiment": "gqa_head_analysis",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "n_layers": n_layers,
        "n_query_heads": N_QUERY_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "layers": {}
    }

    all_within_corrs = []
    all_between_corrs = []
    all_within_corrs_bare = []
    all_between_corrs_bare = []

    for layer_idx in range(n_layers):
        head_results_ccs = []
        head_results_bare = []

        for h in range(N_QUERY_HEADS):
            svd_c = head_svd(attn_ccs[layer_idx][h])
            svd_b = head_svd(attn_bare[layer_idx][h])
            svd_c["group"] = get_kv_group(h)
            svd_b["group"] = get_kv_group(h)
            svd_c["delta_ratio"] = svd_c["ratio"] - svd_b["ratio"]
            svd_c["delta_sigma2"] = svd_c["sigma2"] - svd_b["sigma2"]
            head_results_ccs.append(svd_c)
            head_results_bare.append(svd_b)

        # Within-group vs between-group correlation of σ₂/σ₁
        ccs_ratios = np.array([r["ratio"] for r in head_results_ccs])
        bare_ratios = np.array([r["ratio"] for r in head_results_bare])

        within_pairs_ccs = []
        between_pairs_ccs = []
        within_pairs_bare = []
        between_pairs_bare = []

        for i, j in combinations(range(N_QUERY_HEADS), 2):
            gi, gj = get_kv_group(i), get_kv_group(j)
            if gi == gj:
                within_pairs_ccs.append(abs(ccs_ratios[i] - ccs_ratios[j]))
                within_pairs_bare.append(abs(bare_ratios[i] - bare_ratios[j]))
            else:
                between_pairs_ccs.append(abs(ccs_ratios[i] - ccs_ratios[j]))
                between_pairs_bare.append(abs(bare_ratios[i] - bare_ratios[j]))

        within_mean_ccs = float(np.mean(within_pairs_ccs)) if within_pairs_ccs else 0.0
        between_mean_ccs = float(np.mean(between_pairs_ccs)) if between_pairs_ccs else 0.0
        within_mean_bare = float(np.mean(within_pairs_bare)) if within_pairs_bare else 0.0
        between_mean_bare = float(np.mean(between_pairs_bare)) if between_pairs_bare else 0.0

        # Group-level enrichment
        group_enrichments = {}
        for g in range(N_KV_HEADS):
            group_heads = [h for h in range(N_QUERY_HEADS) if get_kv_group(h) == g]
            g_ccs = np.mean([head_results_ccs[h]["ratio"] for h in group_heads])
            g_bare = np.mean([head_results_bare[h]["ratio"] for h in group_heads])
            group_enrichments[g] = {
                "ccs_ratio": float(g_ccs),
                "bare_ratio": float(g_bare),
                "delta": float(g_ccs - g_bare)
            }

        group_deltas = [group_enrichments[g]["delta"] for g in range(N_KV_HEADS)]
        group_delta_std = float(np.std(group_deltas))
        group_delta_range = float(max(group_deltas) - min(group_deltas))

        layer_result = {
            "within_group_diff_ccs": within_mean_ccs,
            "between_group_diff_ccs": between_mean_ccs,
            "group_coherence_ccs": between_mean_ccs / (within_mean_ccs + 1e-10),
            "within_group_diff_bare": within_mean_bare,
            "between_group_diff_bare": between_mean_bare,
            "group_coherence_bare": between_mean_bare / (within_mean_bare + 1e-10),
            "group_enrichment_std": group_delta_std,
            "group_enrichment_range": group_delta_range,
            "group_enrichments": group_enrichments,
            "avg_delta_ratio": float(np.mean([r["delta_ratio"] for r in head_results_ccs])),
        }

        results["layers"][str(layer_idx)] = layer_result

        all_within_corrs.append(within_mean_ccs)
        all_between_corrs.append(between_mean_ccs)
        all_within_corrs_bare.append(within_mean_bare)
        all_between_corrs_bare.append(between_mean_bare)

        if layer_idx % 5 == 0 or layer_idx in [20, 22, 24, 26, 28, 30, 35]:
            ratio = between_mean_ccs / (within_mean_ccs + 1e-10)
            print(f"  L{layer_idx:2d}: within={within_mean_ccs:.4f} between={between_mean_ccs:.4f} "
                  f"ratio={ratio:.2f} group_Δ_range={group_delta_range:.4f}")

    # Summary statistics
    print("\n=== GQA GROUP ANALYSIS SUMMARY ===")

    avg_within_ccs = np.mean(all_within_corrs)
    avg_between_ccs = np.mean(all_between_corrs)
    avg_within_bare = np.mean(all_within_corrs_bare)
    avg_between_bare = np.mean(all_between_corrs_bare)

    print(f"\nOverall (CCS):  within={avg_within_ccs:.4f}  between={avg_between_ccs:.4f}  ratio={avg_between_ccs/avg_within_ccs:.2f}")
    print(f"Overall (bare): within={avg_within_bare:.4f}  between={avg_between_bare:.4f}  ratio={avg_between_bare/avg_within_bare:.2f}")

    # Does CCS increase group coherence?
    ccs_ratios_all = [b/(w+1e-10) for w, b in zip(all_within_corrs, all_between_corrs)]
    bare_ratios_all = [b/(w+1e-10) for w, b in zip(all_within_corrs_bare, all_between_corrs_bare)]
    delta_coherence = [c - b for c, b in zip(ccs_ratios_all, bare_ratios_all)]

    print(f"\nMean CCS group coherence ratio: {np.mean(ccs_ratios_all):.3f}")
    print(f"Mean bare group coherence ratio: {np.mean(bare_ratios_all):.3f}")
    print(f"Mean Δ(coherence): {np.mean(delta_coherence):+.3f}")

    n_ccs_more_coherent = sum(1 for d in delta_coherence if d > 0)
    print(f"CCS more group-coherent in {n_ccs_more_coherent}/{n_layers} layers")

    # Zone-level group analysis
    zones = {
        "early": list(range(0, 14)),
        "transition": list(range(14, 21)),
        "responsive": list(range(21, 35)),
        "relay": list(range(35, n_layers))
    }

    print("\n=== ZONE-LEVEL GROUP ENRICHMENT ===")
    for zone_name, zone_layers in zones.items():
        zone_stds = [results["layers"][str(l)]["group_enrichment_std"] for l in zone_layers]
        zone_ranges = [results["layers"][str(l)]["group_enrichment_range"] for l in zone_layers]
        zone_coherence = [ccs_ratios_all[l] for l in zone_layers]

        print(f"  {zone_name:12s}: group_Δ_std={np.mean(zone_stds):.4f} "
              f"group_Δ_range={np.mean(zone_ranges):.4f} "
              f"coherence={np.mean(zone_coherence):.3f}")

    results["summary"] = {
        "avg_within_ccs": float(avg_within_ccs),
        "avg_between_ccs": float(avg_between_ccs),
        "avg_within_bare": float(avg_within_bare),
        "avg_between_bare": float(avg_between_bare),
        "mean_ccs_coherence": float(np.mean(ccs_ratios_all)),
        "mean_bare_coherence": float(np.mean(bare_ratios_all)),
        "n_ccs_more_coherent": n_ccs_more_coherent
    }

    with open("/workspace/results_gqa_head_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to /workspace/results_gqa_head_analysis.json")


if __name__ == "__main__":
    main()
