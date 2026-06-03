#!/usr/bin/env python3
"""Exp 5: L13 Structural Node Anatomy

The full-depth profile shows L13 is a zero-crossing in γ-V₂ correlation
across ALL conditions (r ≈ 0, p > 0.05) at exactly 37% depth in Qwen2.5-3B.
L12 is negative, L14 is negative — L13 is a pulse, not a transition.

Questions:
1. Is L13's γ spectrum different from neighbors? (Different CV, distribution shape?)
2. Does V₂ at L13 have unusual structure? (Lower σ₂? Different spectral gap?)
3. Is the zero correlation because γ and V₂ are both random, or because
   V₂ restructures at this layer?
4. Does L13 correspond to a known architectural feature (attention head count change,
   or norm weight anomaly)?

Method: Compare L12, L13, L14 in detail — γ spectrum properties, V₂ structure,
per-head attention pattern analysis.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = Path(__file__).parent / "results"
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

FOCUS_LAYERS = list(range(10, 18))  # L10-L17 to bracket the node


def gamma_spectrum_analysis(gamma):
    g = np.abs(gamma)
    g_sorted = np.sort(g)[::-1]
    g_norm = g / (g.sum() + 1e-10)
    g_pos = g_norm[g_norm > 0]
    entropy = -np.sum(g_pos * np.log(g_pos))
    max_entropy = np.log(len(g))

    n = len(g_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * g_sorted) / (n * np.sum(g_sorted))) - (n + 1) / n

    return {
        "mean": float(np.mean(g)),
        "std": float(np.std(g)),
        "cv": float(np.std(g) / (np.mean(g) + 1e-10)),
        "min": float(np.min(g)),
        "max": float(np.max(g)),
        "range": float(np.max(g) - np.min(g)),
        "skewness": float(stats.skew(g)),
        "kurtosis": float(stats.kurtosis(g)),
        "normalized_entropy": float(entropy / max_entropy),
        "gini": float(gini),
        "near_zero_frac": float(np.mean(g < 0.1)),
        "high_frac": float(np.mean(g > np.mean(g) + 2 * np.std(g))),
    }


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading Qwen2.5-3B for L13 anatomy...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu",
        output_hidden_states=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # 1. γ spectrum comparison
    print(f"\n{'='*70}")
    print("γ SPECTRUM COMPARISON (L10-L17)")
    print(f"{'='*70}")

    gammas = {}
    for layer_idx in FOCUS_LAYERS:
        gamma = model.model.layers[layer_idx].input_layernorm.weight.detach().float().numpy()
        gammas[layer_idx] = gamma
        sp = gamma_spectrum_analysis(gamma)
        marker = " <<<" if layer_idx == 13 else ""
        print(f"  L{layer_idx:2d}: cv={sp['cv']:.4f} gini={sp['gini']:.4f} "
              f"entropy={sp['normalized_entropy']:.4f} "
              f"skew={sp['skewness']:+.3f} kurt={sp['kurtosis']:+.3f} "
              f"range={sp['range']:.4f} near0={sp['near_zero_frac']:.3f}{marker}")

    # 2. γ pairwise similarity between adjacent layers
    print(f"\n{'='*70}")
    print("γ SIMILARITY BETWEEN ADJACENT LAYERS")
    print(f"{'='*70}")

    for i in range(len(FOCUS_LAYERS) - 1):
        l1, l2 = FOCUS_LAYERS[i], FOCUS_LAYERS[i + 1]
        g1 = np.abs(gammas[l1])
        g2 = np.abs(gammas[l2])
        r, p = stats.pearsonr(g1, g2)
        cosine = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
        marker = " <<<" if l2 == 13 or l1 == 13 else ""
        print(f"  L{l1}→L{l2}: pearson_r={r:+.4f} (p={p:.1e}) cosine={cosine:.4f}{marker}")

    # 3. V₂ structure at node layers
    print(f"\n{'='*70}")
    print("V₂ STRUCTURE AROUND L13 NODE")
    print(f"{'='*70}")

    for condition, texts in PROBES.items():
        print(f"\n  Condition: {condition}")

        all_hidden = {l: [] for l in FOCUS_LAYERS}
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            for layer_idx in FOCUS_LAYERS:
                h = outputs.hidden_states[layer_idx + 1]
                h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1)
                all_hidden[layer_idx].append(h_mean.squeeze(0).float().numpy())
            del outputs

        for layer_idx in FOCUS_LAYERS:
            residual_matrix = np.stack(all_hidden[layer_idx])
            U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)
            V2 = Vt[1]
            V1 = Vt[0]
            gamma = gammas[layer_idx]
            gamma_abs = np.abs(gamma)
            v2_abs = np.abs(V2)
            v1_abs = np.abs(V1)

            r_v2, p_v2 = stats.pearsonr(gamma_abs, v2_abs)
            r_v1, p_v1 = stats.pearsonr(gamma_abs, v1_abs)

            # V₂ concentration: how spread is V₂ across dimensions?
            v2_entropy = -np.sum(v2_abs[v2_abs > 0] / v2_abs.sum() *
                                 np.log(v2_abs[v2_abs > 0] / v2_abs.sum()))
            v2_max_entropy = np.log(len(V2))
            v2_norm_entropy = v2_entropy / v2_max_entropy

            # Pairwise V₂ stability (do same dims carry V₂ as in L12/L14?)
            if layer_idx > FOCUS_LAYERS[0]:
                prev_V2 = np.abs(Vt_prev[1])
                k = max(1, len(V2) // 10)
                curr_top = set(np.argsort(v2_abs)[-k:])
                prev_top = set(np.argsort(prev_V2)[-k:])
                stability = len(curr_top & prev_top) / k
                stab_str = f" V₂_stab={stability:.3f}"
            else:
                stab_str = ""

            Vt_prev = Vt

            marker = " <<<" if layer_idx == 13 else ""
            print(f"    L{layer_idx:2d}: r(γ,V₂)={r_v2:+.4f} r(γ,V₁)={r_v1:+.4f} "
                  f"σ₁={S[0]:.0f} σ₂={S[1]:.1f} gap={S[0]/S[1]:.1f} "
                  f"V₂_ent={v2_norm_entropy:.4f}{stab_str}{marker}")

    # 4. Check if L13 has unusual attention weight norms
    print(f"\n{'='*70}")
    print("ATTENTION WEIGHT NORMS (L10-L17)")
    print(f"{'='*70}")

    for layer_idx in FOCUS_LAYERS:
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        q_norm = attn.q_proj.weight.norm().item()
        k_norm = attn.k_proj.weight.norm().item()
        v_norm = attn.v_proj.weight.norm().item()
        o_norm = attn.o_proj.weight.norm().item()
        mlp_up = layer.mlp.up_proj.weight.norm().item()
        mlp_gate = layer.mlp.gate_proj.weight.norm().item()
        mlp_down = layer.mlp.down_proj.weight.norm().item()
        marker = " <<<" if layer_idx == 13 else ""
        print(f"  L{layer_idx:2d}: Q={q_norm:.2f} K={k_norm:.2f} V={v_norm:.2f} O={o_norm:.2f} "
              f"| gate={mlp_gate:.2f} up={mlp_up:.2f} down={mlp_down:.2f}{marker}")

    # 5. Post-attention vs post-FFN layernorm comparison
    print(f"\n{'='*70}")
    print("POST-ATTENTION LAYERNORM γ (L10-L17)")
    print(f"{'='*70}")

    for layer_idx in FOCUS_LAYERS:
        layer = model.model.layers[layer_idx]
        post_norm_gamma = layer.post_attention_layernorm.weight.detach().float().numpy()
        sp = gamma_spectrum_analysis(post_norm_gamma)
        input_gamma = gammas[layer_idx]
        r_cross, p_cross = stats.pearsonr(np.abs(input_gamma), np.abs(post_norm_gamma))
        marker = " <<<" if layer_idx == 13 else ""
        print(f"  L{layer_idx:2d}: cv={sp['cv']:.4f} gini={sp['gini']:.4f} "
              f"r(input_γ, post_γ)={r_cross:+.4f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
