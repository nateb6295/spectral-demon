#!/usr/bin/env python3
"""Exp 7: Wire Propagation — Does γ Ablation at One Layer Affect Downstream V₂?

Exp 6 tests whether γ causes V₂ routing at the SAME layer.
This tests whether ablating γ at one layer disrupts V₂ at DOWNSTREAM layers.

If ablating L14's γ changes V₂ at L20: the wire propagates (cumulative routing).
If it only affects V₂ at L14: each layer routes independently (local refresh).

Method:
1. Normal forward pass → record V₂ at all tunnel+relay layers
2. Ablate γ → 1.0 at ONE source layer → re-run → record V₂ at all downstream layers
3. Compare: V₂ cosine similarity (normal vs ablated) at each downstream layer
4. Repeat for different source layers (L14, L16, L18, L20)

The decay curve of the perturbation tells us the wire's memory length:
- Immediate recovery (1-2 layers): wire refreshes locally
- Slow decay (5-10 layers): wire has medium memory
- Persistent disruption: wire state is cumulative
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

SOURCE_LAYERS = [13, 14, 16, 18, 20]
MEASURE_LAYERS = list(range(10, 32))
NUM_LAYERS = 36


def compute_v2(residual_matrix):
    U, S, Vt = np.linalg.svd(residual_matrix, full_matrices=False)
    if len(S) < 2:
        return None, None, None
    return Vt[1], S[1], S[0] / S[1]


def run_forward_all_layers(model, tokenizer, texts, target_layers):
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

    print("Loading Qwen2.5-3B for wire propagation test...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu",
        output_hidden_states=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    original_gammas = {}
    for layer_idx in SOURCE_LAYERS:
        original_gammas[layer_idx] = model.model.layers[layer_idx].input_layernorm.weight.data.clone()

    gamma_np = {}
    for layer_idx in MEASURE_LAYERS:
        gamma_np[layer_idx] = model.model.layers[layer_idx].input_layernorm.weight.detach().float().numpy()

    results = {}

    for condition, texts in PROBES.items():
        print(f"\n{'='*70}")
        print(f"CONDITION: {condition}")
        print(f"{'='*70}")

        # 1. Normal baseline — V₂ at all measurement layers
        print("  Running normal baseline...")
        hidden_normal = run_forward_all_layers(model, tokenizer, texts, MEASURE_LAYERS)
        normal_v2 = {}
        normal_stats = {}
        for l in MEASURE_LAYERS:
            v2, sigma2, gap = compute_v2(hidden_normal[l])
            if v2 is not None:
                normal_v2[l] = v2
                g = np.abs(gamma_np[l])
                r_gv2, p_gv2 = stats.pearsonr(g, np.abs(v2))
                normal_stats[l] = {
                    "sigma2": float(sigma2), "gap": float(gap),
                    "r_gamma_v2": float(r_gv2), "p_gamma_v2": float(p_gv2),
                }

        # 2. Ablate each source layer, measure downstream V₂
        for src in SOURCE_LAYERS:
            print(f"\n  Ablating L{src} → uniform...")
            model.model.layers[src].input_layernorm.weight.data = torch.ones_like(original_gammas[src])

            hidden_ablated = run_forward_all_layers(model, tokenizer, texts, MEASURE_LAYERS)

            for l in MEASURE_LAYERS:
                v2_abl, sigma2_abl, gap_abl = compute_v2(hidden_ablated[l])
                if v2_abl is None or l not in normal_v2:
                    continue

                v2_norm = normal_v2[l]
                cosine = float(np.dot(v2_norm, v2_abl) / (
                    np.linalg.norm(v2_norm) * np.linalg.norm(v2_abl) + 1e-10))
                cosine_abs = float(np.dot(np.abs(v2_norm), np.abs(v2_abl)) / (
                    np.linalg.norm(np.abs(v2_norm)) * np.linalg.norm(np.abs(v2_abl)) + 1e-10))

                g = np.abs(gamma_np[l])
                r_gv2_abl, p_gv2_abl = stats.pearsonr(g, np.abs(v2_abl))

                delta_layers = l - src
                key = f"src{src}_meas{l}_{condition}"
                results[key] = {
                    "source_layer": src,
                    "measure_layer": l,
                    "delta_layers": delta_layers,
                    "condition": condition,
                    "v2_cosine": cosine,
                    "v2_cosine_abs": cosine_abs,
                    "normal_r": normal_stats[l]["r_gamma_v2"],
                    "ablated_r": float(r_gv2_abl),
                    "delta_r": float(r_gv2_abl - normal_stats[l]["r_gamma_v2"]),
                    "normal_sigma2": normal_stats[l]["sigma2"],
                    "ablated_sigma2": float(sigma2_abl),
                    "normal_gap": normal_stats[l]["gap"],
                    "ablated_gap": float(gap_abl),
                }

                if l in [src, src+1, src+2, src+4, src+8, 30]:
                    tag = "SAME" if l == src else f"+{delta_layers}"
                    print(f"    L{l:2d} ({tag:>5s}): cos={cosine:+.4f}  "
                          f"r(γ,V₂): {normal_stats[l]['r_gamma_v2']:+.4f}→{r_gv2_abl:+.4f}  "
                          f"σ₂: {normal_stats[l]['sigma2']:.1f}→{sigma2_abl:.1f}")

            # Restore
            model.model.layers[src].input_layernorm.weight.data = original_gammas[src].clone()

    # Summary: propagation decay curves
    print(f"\n{'='*70}")
    print("PROPAGATION DECAY (mean |Δr| by distance from ablation)")
    print(f"{'='*70}")

    for src in SOURCE_LAYERS:
        deltas_by_dist = {}
        for k, v in results.items():
            if v["source_layer"] == src and v["delta_layers"] >= 0:
                d = v["delta_layers"]
                deltas_by_dist.setdefault(d, []).append(abs(v["delta_r"]))

        if deltas_by_dist:
            print(f"\n  Ablating L{src}:")
            for d in sorted(deltas_by_dist.keys())[:12]:
                mean_dr = np.mean(deltas_by_dist[d])
                bar = "█" * int(mean_dr * 100)
                print(f"    +{d:2d}: |Δr|={mean_dr:.4f} {bar}")

    # V₂ stability: how quickly does V₂ recover its normal shape?
    print(f"\n{'='*70}")
    print("V₂ RECOVERY (mean |cosine| by distance from ablation)")
    print(f"{'='*70}")

    for src in SOURCE_LAYERS:
        cos_by_dist = {}
        for k, v in results.items():
            if v["source_layer"] == src and v["delta_layers"] >= 0:
                d = v["delta_layers"]
                cos_by_dist.setdefault(d, []).append(abs(v["v2_cosine"]))

        if cos_by_dist:
            print(f"\n  Ablating L{src}:")
            for d in sorted(cos_by_dist.keys())[:12]:
                mean_cos = np.mean(cos_by_dist[d])
                bar = "▓" * int(mean_cos * 50)
                print(f"    +{d:2d}: |cos|={mean_cos:.4f} {bar}")

    # Half-life estimation
    print(f"\n{'='*70}")
    print("WIRE MEMORY HALF-LIFE")
    print(f"{'='*70}")

    for src in SOURCE_LAYERS:
        cos_by_dist = {}
        for k, v in results.items():
            if v["source_layer"] == src and v["delta_layers"] > 0:
                d = v["delta_layers"]
                cos_by_dist.setdefault(d, []).append(abs(v["v2_cosine"]))

        if cos_by_dist and 0 in {v["delta_layers"] for k, v in results.items()
                                   if v["source_layer"] == src}:
            initial_disruption = 1.0 - np.mean([abs(v["v2_cosine"])
                for k, v in results.items()
                if v["source_layer"] == src and v["delta_layers"] == 0])

            half_target = initial_disruption / 2
            for d in sorted(cos_by_dist.keys()):
                recovery = 1.0 - (1.0 - np.mean(cos_by_dist[d]))
                disruption_remaining = 1.0 - recovery
                if disruption_remaining <= half_target:
                    print(f"  L{src}: half-life ≈ {d} layers (initial disruption={initial_disruption:.4f})")
                    break
            else:
                print(f"  L{src}: half-life > {max(cos_by_dist.keys())} layers (persistent)")

    out_path = RESULTS_DIR / "exp_wire_propagation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
