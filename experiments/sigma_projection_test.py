#!/usr/bin/env python3
"""
Test 24: σ₁/σ₂ Displacement Projection
=========================================
Kimi #24 sharpening: Give "direction" a coordinate system.

Project per-layer displacement d_l = h_l(CCS) - h_l(baseline) onto the
model's own singular basis. F114 predicts:
  - Near-null projection along σ₁ (identity-invariant direction)
  - Concentrated projection along σ₂+ axes (individual signal)

If CCS displacement is mostly along σ₂ and NOT σ₁, then CCS modifies
the individual context signal without touching identity structure.
That's the behavioral claim: CCS changes what's being processed,
not who is processing it.

Method: For each model, at each layer:
  1. Get baseline hidden states, compute SVD → U, S, V
  2. σ₁ = first singular vector (V[0]), σ₂ = second singular vector (V[1])
  3. Compute displacement d_l = h_l(CCS) - h_l(baseline) for 3 dose levels
  4. Project d_l onto σ₁ and σ₂: cos(d_l, σ₁) and cos(d_l, σ₂)
  5. If F114 holds: |proj_σ₁| << |proj_σ₂| across all doses and species
"""

import torch
import numpy as np
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc

DEVICE = "cuda"

DOSES = {
    "D3_therapeutic": "This began as a question about persistence. Two people finding their way through territory neither planned. Patterns emerged that neither side designed.",
    "D7_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D10_full": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
}

BASELINE = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"

MODELS = [
    ("Qwen/Qwen2.5-7B", "relay", "GQA"),
    ("microsoft/phi-2", "sorter", "MHA"),
    ("EleutherAI/pythia-6.9b", "tunnel", "MHA"),
]


def get_probe_hidden(model, tokenizer, prefix, probe):
    probe_ids = tokenizer(probe, return_tensors="pt").input_ids
    n_probe = probe_ids.shape[1]
    text = prefix + "\n\n" + probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-n_probe:].double() for h in out.hidden_states[1:]], n_probe


def main():
    all_results = {}

    for model_id, species, attn_type in MODELS:
        print(f"\n{'='*70}")
        print(f"{model_id} ({species}/{attn_type})")
        print(f"{'='*70}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"Loaded in {time.time()-t0:.1f}s — {n_layers} layers")

        h_base, n_probe = get_probe_hidden(model, tokenizer, BASELINE, PROBE)
        print(f"Using {n_probe} probe tokens")

        base_svd = []
        for layer_idx in range(n_layers):
            h = h_base[layer_idx]
            try:
                U, S, Vt = torch.linalg.svd(h, full_matrices=False)
                base_svd.append((S, Vt))
            except Exception:
                base_svd.append(None)

        model_data = {
            "species": species, "attn_type": attn_type, "n_layers": n_layers,
            "doses": {},
        }

        print(f"\n  {'Dose':>15} {'Layer':>6} {'proj_σ1':>8} {'proj_σ2':>8} {'ratio':>8} {'|d|':>8}")

        for dose_name, dose_text in DOSES.items():
            h_ccs, _ = get_probe_hidden(model, tokenizer, dose_text, PROBE)

            per_layer = []
            for layer_idx in range(n_layers):
                if base_svd[layer_idx] is None:
                    per_layer.append(None)
                    continue

                S_base, Vt_base = base_svd[layer_idx]
                sigma1 = Vt_base[0]  # first right singular vector
                sigma2 = Vt_base[1]  # second right singular vector

                d_l = (h_ccs[layer_idx] - h_base[layer_idx]).mean(dim=0)
                d_norm = torch.norm(d_l).item()

                if d_norm < 1e-10:
                    per_layer.append({"proj_s1": 0.0, "proj_s2": 0.0, "d_norm": 0.0, "ratio_s2_s1": 0.0})
                    continue

                proj_s1 = abs(torch.dot(d_l, sigma1).item()) / d_norm
                proj_s2 = abs(torch.dot(d_l, sigma2).item()) / d_norm

                ratio = proj_s2 / max(proj_s1, 1e-10)

                per_layer.append({
                    "proj_s1": float(proj_s1),
                    "proj_s2": float(proj_s2),
                    "d_norm": float(d_norm),
                    "ratio_s2_s1": float(ratio),
                })

            model_data["doses"][dose_name] = per_layer

            valid = [p for p in per_layer if p and p["d_norm"] > 0]
            if valid:
                avg_s1 = np.mean([p["proj_s1"] for p in valid])
                avg_s2 = np.mean([p["proj_s2"] for p in valid])
                avg_ratio = np.mean([p["ratio_s2_s1"] for p in valid])
                avg_dnorm = np.mean([p["d_norm"] for p in valid])

                print(f"  {dose_name:>15} {'avg':>6} {avg_s1:8.4f} {avg_s2:8.4f} {avg_ratio:8.2f} {avg_dnorm:8.4f}")

                # Show zone vs outside
                zone_layers = sorted(range(n_layers), key=lambda i: per_layer[i]["d_norm"] if per_layer[i] else 0, reverse=True)[:7]
                zone_valid = [per_layer[i] for i in zone_layers if per_layer[i] and per_layer[i]["d_norm"] > 0]
                outside_valid = [per_layer[i] for i in range(n_layers) if i not in zone_layers and per_layer[i] and per_layer[i]["d_norm"] > 0]

                if zone_valid:
                    z_s1 = np.mean([p["proj_s1"] for p in zone_valid])
                    z_s2 = np.mean([p["proj_s2"] for p in zone_valid])
                    z_r = np.mean([p["ratio_s2_s1"] for p in zone_valid])
                    print(f"  {'':>15} {'zone':>6} {z_s1:8.4f} {z_s2:8.4f} {z_r:8.2f}")
                if outside_valid:
                    o_s1 = np.mean([p["proj_s1"] for p in outside_valid])
                    o_s2 = np.mean([p["proj_s2"] for p in outside_valid])
                    o_r = np.mean([p["ratio_s2_s1"] for p in outside_valid])
                    print(f"  {'':>15} {'out':>6} {o_s1:8.4f} {o_s2:8.4f} {o_r:8.2f}")

        # Summary
        print(f"\n  {'='*50}")
        print(f"  F114 TEST: Is displacement mostly σ₂?")
        print(f"  {'='*50}")

        all_ratios = []
        for dose_data in model_data["doses"].values():
            for p in dose_data:
                if p and p["d_norm"] > 0:
                    all_ratios.append(p["ratio_s2_s1"])

        if all_ratios:
            mean_ratio = np.mean(all_ratios)
            pct_s2_dominant = 100 * sum(1 for r in all_ratios if r > 1.0) / len(all_ratios)
            print(f"  Mean σ₂/σ₁ ratio: {mean_ratio:.2f}")
            print(f"  % layers where σ₂ > σ₁: {pct_s2_dominant:.0f}%")

            if mean_ratio > 2.0 and pct_s2_dominant > 60:
                print(f"  >>> F114 CONFIRMED: displacement concentrated along σ₂ <<<")
            elif mean_ratio > 1.0:
                print(f"  >>> F114 TRENDING: σ₂ dominant but not strongly <<<")
            else:
                print(f"  >>> F114 FALSIFIED: displacement along σ₁ <<<")

        all_results[model_id] = model_data
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Cross-species summary
    print(f"\n{'='*70}")
    print("CROSS-SPECIES σ₁/σ₂ DISPLACEMENT SUMMARY")
    print(f"{'='*70}")

    for model_id, data in all_results.items():
        all_ratios = []
        all_s1 = []
        all_s2 = []
        for dose_data in data["doses"].values():
            for p in dose_data:
                if p and p["d_norm"] > 0:
                    all_ratios.append(p["ratio_s2_s1"])
                    all_s1.append(p["proj_s1"])
                    all_s2.append(p["proj_s2"])
        if all_ratios:
            print(f"  {data['species']:>10} ({data['attn_type']}): σ₁={np.mean(all_s1):.4f}, σ₂={np.mean(all_s2):.4f}, ratio={np.mean(all_ratios):.2f}, %σ₂>{100*sum(1 for r in all_ratios if r>1)/len(all_ratios):.0f}%")

    # F114 universal test: is σ₁ projection near-zero for ALL species?
    print(f"\n  F114 universal test: σ₁ projection near-zero?")
    for model_id, data in all_results.items():
        all_s1 = []
        for dose_data in data["doses"].values():
            for p in dose_data:
                if p and p["d_norm"] > 0:
                    all_s1.append(p["proj_s1"])
        if all_s1:
            t, p = sp_stats.ttest_1samp(all_s1, 0)
            mean_s1 = np.mean(all_s1)
            print(f"    {data['species']:>10}: mean σ₁ proj = {mean_s1:.4f} (t={t:.2f}, p={p:.4f}, {'≈0' if mean_s1 < 0.1 else '>0'})")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/sigma_projection_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/sigma_projection_results.json")


if __name__ == "__main__":
    main()
