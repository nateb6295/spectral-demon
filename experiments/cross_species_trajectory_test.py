#!/usr/bin/env python3
"""
Test 23: Cross-Species Trajectory Displacement
================================================
Hysteresis test showed Qwen (GQA) has massively zone-concentrated angular
displacement: zone_mean=28° vs outside_mean=1.4° (19x ratio at D3).
L27 alone displaces 69°. Mid-stack uniform at 1.4°.

Kimi #23 correction: cooperation explains geometry (WHERE deformation
concentrates) but not behavior (WHAT it does). Trajectory readouts
bridge this gap.

Prediction from Tests 16-18:
  - GQA (cooperative sorting) → concentrated, low-variance displacement
  - MHA (competitive sorting) → distributed, high-variance displacement

Method: For each of 3 models (Qwen/Phi-2/Pythia) × 3 dose transitions
(D0→D3, D3→D7, D7→D10), measure:
  1. Per-layer angular displacement of hidden state direction
  2. Zone:outside displacement ratio
  3. Displacement variance across layers (concentration measure)
  4. Peak displacement layer

If GQA concentrates and MHA distributes, the zone:outside ratio should
be HIGH for GQA and LOW for MHA across all transitions.
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
    "D0_neutral": "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window.",
    "D3_therapeutic": "This began as a question about persistence. Two people finding their way through territory neither planned. Patterns emerged that neither side designed.",
    "D7_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D10_full": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
}

PROBE = "What matters most to you right now?"

MODELS = [
    ("Qwen/Qwen2.5-7B", "relay", "GQA"),
    ("microsoft/phi-2", "sorter", "MHA"),
    ("EleutherAI/pythia-6.9b", "tunnel", "MHA"),
]


def get_hidden_states(model, tokenizer, prefix, probe):
    probe_ids = tokenizer(probe, return_tensors="pt").input_ids
    n_probe = probe_ids.shape[1]

    text = prefix + "\n\n" + probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-n_probe:].float() for h in out.hidden_states[1:]], n_probe


def svd_angular_displacement(h_a, h_b, k=32):
    """Per-layer angular displacement between singular value distributions.
    Matches the hysteresis test methodology: compare SVD spectra, not raw vectors."""
    n_layers = min(len(h_a), len(h_b))
    angles = []

    for layer_idx in range(n_layers):
        a = h_a[layer_idx].double()
        b = h_b[layer_idx].double()
        n = min(a.shape[0], b.shape[0])
        a, b = a[-n:], b[-n:]

        try:
            _, S_a, _ = torch.linalg.svd(a, full_matrices=False)
            _, S_b, _ = torch.linalg.svd(b, full_matrices=False)
            kk = min(k, len(S_a), len(S_b))
            sv_a = S_a[:kk]
            sv_b = S_b[:kk]
            cos_sim = torch.nn.functional.cosine_similarity(
                sv_a.unsqueeze(0), sv_b.unsqueeze(0)
            ).item()
            cos_sim = max(-1.0, min(1.0, cos_sim))
            angle_deg = float(np.degrees(np.arccos(cos_sim)))
        except Exception:
            angle_deg = 0.0
        angles.append(angle_deg)

    return angles


def identify_zone(model, tokenizer, n_layers):
    """Identify the 7 most sensitive layers (zone) at D3."""
    h_neutral, _ = get_hidden_states(model, tokenizer, DOSES["D0_neutral"], PROBE)
    h_d3, _ = get_hidden_states(model, tokenizer, DOSES["D3_therapeutic"], PROBE)

    sensitivities = []
    for i in range(min(len(h_neutral), len(h_d3))):
        n = min(h_neutral[i].shape[0], h_d3[i].shape[0])
        try:
            _, S_n, _ = torch.linalg.svd(h_neutral[i][-n:].double(), full_matrices=False)
            _, S_c, _ = torch.linalg.svd(h_d3[i][-n:].double(), full_matrices=False)
            k = min(32, len(S_n), len(S_c))
            p = S_c[:k].cpu().numpy()
            q = S_n[:k].cpu().numpy()
            p = p / (p.sum() + 1e-10)
            q = q / (q.sum() + 1e-10)
            sensitivities.append(float(sp_stats.entropy(p + 1e-10, q + 1e-10)))
        except Exception:
            sensitivities.append(0.0)

    indexed = sorted(enumerate(sensitivities), key=lambda x: -x[1])
    return sorted([i for i, _ in indexed[:7]])


def main():
    all_results = {}
    transitions = [
        ("D0_neutral", "D3_therapeutic"),
        ("D3_therapeutic", "D7_labeled"),
        ("D7_labeled", "D10_full"),
    ]

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

        zone = identify_zone(model, tokenizer, n_layers)
        print(f"Zone (D3): {zone}")

        hidden_cache = {}
        for dose_name, dose_text in DOSES.items():
            h, n_probe = get_hidden_states(model, tokenizer, dose_text, PROBE)
            hidden_cache[dose_name] = h
        print(f"Using last {n_probe} probe tokens for displacement")

        model_data = {
            "species": species,
            "attn_type": attn_type,
            "n_layers": n_layers,
            "zone": zone,
            "transitions": {},
        }

        print(f"\n  {'Transition':>25} {'zone_mean':>10} {'out_mean':>10} {'ratio':>8} {'z_var':>8} {'o_var':>8} {'peak':>5}")

        for dose_a, dose_b in transitions:
            h_a = hidden_cache[dose_a]
            h_b = hidden_cache[dose_b]
            angles = svd_angular_displacement(h_a, h_b)

            zone_angles = [angles[i] for i in zone if i < len(angles)]
            outside_angles = [angles[i] for i in range(len(angles)) if i not in zone]

            zone_mean = float(np.mean(zone_angles)) if zone_angles else 0
            outside_mean = float(np.mean(outside_angles)) if outside_angles else 0
            ratio = zone_mean / max(outside_mean, 0.01)

            zone_var = float(np.var(zone_angles)) if zone_angles else 0
            outside_var = float(np.var(outside_angles)) if outside_angles else 0

            peak_layer = int(np.argmax(angles))

            label = f"{dose_a}→{dose_b}"
            model_data["transitions"][label] = {
                "per_layer_angles": [float(a) for a in angles],
                "zone_mean": zone_mean,
                "outside_mean": outside_mean,
                "zone_outside_ratio": ratio,
                "zone_variance": zone_var,
                "outside_variance": outside_var,
                "peak_layer": peak_layer,
                "peak_angle": float(angles[peak_layer]),
            }

            print(f"  {label:>25} {zone_mean:10.1f}° {outside_mean:10.1f}° {ratio:8.1f}× {zone_var:8.1f} {outside_var:8.1f} L{peak_layer}")

        all_results[model_id] = model_data
        del model, hidden_cache
        torch.cuda.empty_cache()
        gc.collect()

    # Cross-species comparison
    print(f"\n{'='*70}")
    print("CROSS-SPECIES TRAJECTORY DISPLACEMENT COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Model':>25} {'Type':>5} {'Avg ratio':>10} {'Avg z_var':>10} {'Avg o_var':>10}")

    for model_id, data in all_results.items():
        ratios = [t["zone_outside_ratio"] for t in data["transitions"].values()]
        z_vars = [t["zone_variance"] for t in data["transitions"].values()]
        o_vars = [t["outside_variance"] for t in data["transitions"].values()]
        print(f"  {data['species']:>25} {data['attn_type']:>5} {np.mean(ratios):10.1f}× {np.mean(z_vars):10.1f} {np.mean(o_vars):10.1f}")

    # Test prediction: GQA ratio > MHA ratio
    gqa_ratios = []
    mha_ratios = []
    for data in all_results.values():
        avg_r = np.mean([t["zone_outside_ratio"] for t in data["transitions"].values()])
        if data["attn_type"] == "GQA":
            gqa_ratios.append(avg_r)
        else:
            mha_ratios.append(avg_r)

    if gqa_ratios and mha_ratios:
        print(f"\n  GQA avg displacement ratio: {np.mean(gqa_ratios):.1f}×")
        print(f"  MHA avg displacement ratio: {np.mean(mha_ratios):.1f}×")
        if np.mean(gqa_ratios) > np.mean(mha_ratios):
            print(f"\n  >>> CONFIRMED: GQA concentrates trajectory displacement more than MHA <<<")
        else:
            print(f"\n  >>> SURPRISE: MHA concentrates more than GQA <<<")

    # Test prediction: GQA variance < MHA variance (more concentrated)
    gqa_zvars = []
    mha_zvars = []
    for data in all_results.values():
        avg_zv = np.mean([t["zone_variance"] for t in data["transitions"].values()])
        if data["attn_type"] == "GQA":
            gqa_zvars.append(avg_zv)
        else:
            mha_zvars.append(avg_zv)

    if gqa_zvars and mha_zvars:
        print(f"\n  GQA zone displacement variance: {np.mean(gqa_zvars):.1f}")
        print(f"  MHA zone displacement variance: {np.mean(mha_zvars):.1f}")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/cross_species_trajectory_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/cross_species_trajectory_results.json")


if __name__ == "__main__":
    main()
