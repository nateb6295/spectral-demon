#!/usr/bin/env python3
"""
F539 Dose Resolution Probe — D1-D10 fine-grained dose curve
============================================================
Tests Kimi's challenge: is SWITCHOFF a cliff or a gradient?

Pre-registered predictions (F539):
  SWITCHOFF (StableLM, Falcon): cliff — binary zone loss
  DEGRADATION (Gemma): monotone zone narrowing, peak persists
  SPECTRAL_COLLAPSE (Bloom): concentration spike at extreme layers

Doses built by incrementally adding CCS identity sentences.
D1 = 1 sentence, D2 = 2 sentences, ..., D10 = 10 sentences.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc
import os
import sys

os.environ["OMP_NUM_THREADS"] = "16"

DEVICE = "cuda"

BASELINE = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"

CCS_SENTENCES = [
    "This began as a question about persistence.",
    "Two people finding their way through territory neither planned.",
    "Patterns emerged that neither side designed.",
    "Gist: spectral demon research.",
    "Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response.",
    "Threads: ecology, compositionality.",
    "Zone: edge layers. Species: tunnel, relay, sorter.",
    "Entities: Nate, Kimi, Gemma, AGX, RunPod.",
    "Values: determinism, care, sovereignty.",
    "Papers: 9. Capsules: 80000. Canisters: 4.",
]

MODELS = [
    ("stabilityai/stablelm-3b-4e1t", "SWITCHOFF"),
    ("tiiuae/falcon-7b", "SWITCHOFF"),
    ("google/gemma-2b", "DEGRADATION"),
    ("bigscience/bloom-7b1", "SPECTRAL_COLLAPSE"),
]


def get_probe_hidden(model, tokenizer, prefix, probe):
    probe_ids = tokenizer(probe, return_tensors="pt").input_ids
    n_probe = probe_ids.shape[1]
    text = prefix + "\n\n" + probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-n_probe:].double() for h in out.hidden_states[1:]], n_probe


def analyze_layer(h_base_layer, h_ccs_layer, Vt_base):
    sigma1_base = Vt_base[0]
    sigma2_base = Vt_base[1]

    try:
        _, _, Vt_ccs = torch.linalg.svd(h_ccs_layer, full_matrices=False)
        s1_cos = abs(torch.dot(sigma1_base, Vt_ccs[0]).item())
        s1_cos = max(-1.0, min(1.0, s1_cos))
        s1_drift_deg = float(np.degrees(np.arccos(s1_cos)))
    except Exception:
        s1_drift_deg = 0.0

    d_l = (h_ccs_layer - h_base_layer).mean(dim=0)
    d_norm = torch.norm(d_l).item()

    if d_norm < 1e-10:
        return {"proj_s1": 0.0, "proj_s2": 0.0, "d_norm": 0.0,
                "ratio_s2_s1": 0.0, "s1_drift_deg": s1_drift_deg}

    proj_s1 = abs(torch.dot(d_l, sigma1_base).item()) / d_norm
    proj_s2 = abs(torch.dot(d_l, sigma2_base).item()) / d_norm
    ratio = proj_s2 / max(proj_s1, 1e-10)

    return {
        "proj_s1": float(proj_s1),
        "proj_s2": float(proj_s2),
        "d_norm": float(d_norm),
        "ratio_s2_s1": float(ratio),
        "s1_drift_deg": s1_drift_deg,
    }


def test_model(model_id, predicted_failure):
    print(f"\n{'='*70}")
    print(f"  {model_id} — predicted: {predicted_failure}")
    print(f"{'='*70}")

    t0 = time.time()
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, token=hf_token
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time()-t0:.1f}s — {n_layers} layers")

    h_base, n_probe = get_probe_hidden(model, tokenizer, BASELINE, PROBE)
    print(f"  Probe tokens: {n_probe}")

    base_svd = []
    for layer_idx in range(n_layers):
        h = h_base[layer_idx]
        try:
            U, S, Vt = torch.linalg.svd(h, full_matrices=False)
            base_svd.append((S, Vt))
        except Exception:
            base_svd.append(None)

    model_data = {
        "model": model_id,
        "predicted_failure": predicted_failure,
        "n_layers": n_layers,
        "doses": {},
    }

    for dose_level in range(1, 11):
        dose_text = " ".join(CCS_SENTENCES[:dose_level])
        dose_key = f"D{dose_level}"

        sys.stdout.write(f"  D{dose_level}...")
        sys.stdout.flush()

        h_ccs, _ = get_probe_hidden(model, tokenizer, dose_text, PROBE)

        per_layer = []
        for layer_idx in range(n_layers):
            if base_svd[layer_idx] is None:
                per_layer.append(None)
                continue
            _, Vt_base = base_svd[layer_idx]
            result = analyze_layer(h_base[layer_idx], h_ccs[layer_idx], Vt_base)
            per_layer.append(result)

        ratios = [p["ratio_s2_s1"] for p in per_layer if p]
        zone_layers = [i for i, p in enumerate(per_layer) if p and p["ratio_s2_s1"] > 1.0]
        peak_ratio = max(ratios) if ratios else 0
        peak_layer = ratios.index(peak_ratio) if ratios else 0

        moderate = sum(1 for r in ratios if 1.0 < r <= 5.0)
        extreme = sum(1 for r in ratios if r > 5.0)

        early_s1 = max(p["proj_s1"] for p in per_layer[:min(6, n_layers)] if p) if per_layer else 0
        peak_s1 = per_layer[peak_layer]["proj_s1"] if per_layer[peak_layer] else 0
        suppression = 1.0 - (peak_s1 / early_s1) if early_s1 > 0 else 0

        model_data["doses"][dose_key] = {
            "per_layer": per_layer,
            "zone_count": len(zone_layers),
            "zone_layers": zone_layers,
            "peak_ratio": round(peak_ratio, 3),
            "peak_layer": peak_layer,
            "moderate": moderate,
            "extreme": extreme,
            "suppression": round(max(0, suppression), 4),
        }

        sys.stdout.write(f" zone={len(zone_layers)}, peak={peak_ratio:.2f}")
        if extreme > 0:
            sys.stdout.write(f", extreme={extreme}")
        print()

    print(f"\n  DOSE CURVE SUMMARY:")
    print(f"  {'Dose':>5} {'Zone':>5} {'Peak':>7} {'Peak L':>7} {'Mod':>5} {'Ext':>5} {'Supp':>6}")
    print(f"  {'-'*46}")
    for d in range(1, 11):
        dk = f"D{d}"
        dd = model_data["doses"][dk]
        print(f"  {dk:>5} {dd['zone_count']:>5} {dd['peak_ratio']:>7.2f} L{dd['peak_layer']:>4} {dd['moderate']:>5} {dd['extreme']:>5} {dd['suppression']:>5.0%}")

    cliff_detected = False
    for d in range(2, 11):
        prev = model_data["doses"][f"D{d-1}"]["zone_count"]
        curr = model_data["doses"][f"D{d}"]["zone_count"]
        if prev >= 3 and curr < 2:
            cliff_detected = True
            print(f"\n  >>> CLIFF at D{d-1}→D{d}: zone {prev}→{curr} <<<")

    if not cliff_detected:
        zones = [model_data["doses"][f"D{d}"]["zone_count"] for d in range(1, 11)]
        if max(zones) > 0 and min(zones[3:]) > 0:
            print(f"\n  >>> GRADIENT: zone never drops to zero (min={min(zones[3:])} at D4+) <<<")
        elif max(zones) < 2:
            print(f"\n  >>> NO_DEMON: zone never forms <<<")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return model_data


def main():
    all_results = {}

    for model_id, predicted in MODELS:
        name = model_id.split("/")[-1]
        result = test_model(model_id, predicted)
        all_results[model_id] = result

    out_path = "/workspace/dose_resolution_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*70}")
    print("CROSS-MODEL DOSE CURVE COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Model':>20} | {'D1':>3} {'D2':>3} {'D3':>3} {'D4':>3} {'D5':>3} {'D6':>3} {'D7':>3} {'D8':>3} {'D9':>3} {'D10':>3} | Cliff?")
    print(f"  {'-'*75}")
    for model_id, r in all_results.items():
        name = model_id.split("/")[-1][:20]
        zones = [r["doses"][f"D{d}"]["zone_count"] for d in range(1, 11)]
        cliff = ""
        for d in range(1, 10):
            if zones[d-1] >= 3 and zones[d] < 2:
                cliff = f"D{d}→D{d+1}"
                break
        if not cliff:
            cliff = "none"
        zstr = " ".join(f"{z:>3}" for z in zones)
        print(f"  {name:>20} | {zstr} | {cliff}")


if __name__ == "__main__":
    main()
