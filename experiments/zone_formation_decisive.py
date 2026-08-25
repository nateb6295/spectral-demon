#!/usr/bin/env python3
"""
Tests 26-30: σ₂ Zone Formation — Decisive Architecture Tests
==============================================================
Tests the RoPE × sequential hypothesis:
  σ₂ zone formation requires BOTH rotary position encoding (≥40%)
  AND sequential residual computation (MLP sees attention output).

Known results (Tests 24-25):
  Qwen 2.5 7B  : RoPE 100% + seq → ZONE (σ₂/σ₁ = 7.73)
  Phi-2 2.7B   : RoPE 40%  + seq → ZONE (σ₂/σ₁ = 5.36)
  Pythia 6.9B  : RoPE 25%  + par → none (σ₂/σ₁ = 0.54)
  GPT-2 XL     : learned   + seq → none (σ₂/σ₁ = 0.24)
  OPT 6.7B     : learned   + seq → none (σ₂/σ₁ = 0.54)

Predictions:
  Test 26: Mistral 7B  (RoPE 100%, seq, GQA-4)  → ZONE
  Test 27: Gemma 2B    (RoPE 100%, seq, MQA)     → ZONE
  Test 28: StableLM 3B (RoPE 25%,  seq, MHA)     → ??? (decisive)
  Test 29: Falcon 7B   (RoPE, parallel, MHA)      → none
  Test 30: Bloom 7B    (ALiBi, seq, MHA)          → none
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc

DEVICE = "cuda"

BASELINE = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"

DOSES = {
    "D3_therapeutic": "This began as a question about persistence. Two people finding their way through territory neither planned. Patterns emerged that neither side designed.",
    "D7_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D10_full": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
}

MODELS = [
    ("mistralai/Mistral-7B-v0.1", "RoPE 100%", "sequential", "GQA-4", "ZONE"),
    ("google/gemma-2b", "RoPE 100%", "sequential", "MQA", "ZONE"),
    ("stabilityai/stablelm-3b-4e1t", "RoPE 25%", "sequential", "MHA", "???"),
    ("tiiuae/falcon-7b", "RoPE", "parallel", "MHA", "none"),
    ("bigscience/bloom-7b1", "ALiBi", "sequential", "MHA", "none"),
]


def get_probe_hidden(model, tokenizer, prefix, probe):
    probe_ids = tokenizer(probe, return_tensors="pt").input_ids
    n_probe = probe_ids.shape[1]
    text = prefix + "\n\n" + probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-n_probe:].double() for h in out.hidden_states[1:]], n_probe


def test_model(model_id, rope_desc, comp_desc, attn_desc, prediction):
    print(f"\n{'='*70}")
    print(f"  {model_id}")
    print(f"  PosEnc: {rope_desc} | Comp: {comp_desc} | Attn: {attn_desc}")
    print(f"  Prediction: {prediction}")
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
    print(f"  Loaded in {time.time()-t0:.1f}s — {n_layers} layers")

    h_base, n_probe = get_probe_hidden(model, tokenizer, BASELINE, PROBE)
    print(f"  Using {n_probe} probe tokens")

    base_svd = []
    for layer_idx in range(n_layers):
        h = h_base[layer_idx]
        try:
            U, S, Vt = torch.linalg.svd(h, full_matrices=False)
            base_svd.append((S, Vt))
        except Exception:
            base_svd.append(None)

    model_data = {
        "model": model_id, "rope": rope_desc, "comp": comp_desc,
        "attn": attn_desc, "prediction": prediction,
        "n_layers": n_layers, "doses": {},
    }

    for dose_name, dose_text in DOSES.items():
        h_ccs, _ = get_probe_hidden(model, tokenizer, dose_text, PROBE)

        per_layer = []
        for layer_idx in range(n_layers):
            if base_svd[layer_idx] is None:
                per_layer.append(None)
                continue

            S_base, Vt_base = base_svd[layer_idx]
            sigma1_base = Vt_base[0]
            sigma2_base = Vt_base[1]

            # σ₁ DRIFT: SVD the CCS hidden states and compare σ₁ directions
            h_ccs_layer = h_ccs[layer_idx]
            try:
                _, _, Vt_ccs = torch.linalg.svd(h_ccs_layer, full_matrices=False)
                sigma1_ccs = Vt_ccs[0]
                s1_cos = abs(torch.dot(sigma1_base, sigma1_ccs).item())
                s1_cos = max(-1.0, min(1.0, s1_cos))
                s1_drift_deg = float(np.degrees(np.arccos(s1_cos)))
            except Exception:
                s1_drift_deg = 0.0

            d_l = (h_ccs_layer - h_base[layer_idx]).mean(dim=0)
            d_norm = torch.norm(d_l).item()

            if d_norm < 1e-10:
                per_layer.append({"proj_s1": 0.0, "proj_s2": 0.0, "d_norm": 0.0,
                                  "ratio_s2_s1": 0.0, "s1_drift_deg": s1_drift_deg})
                continue

            proj_s1 = abs(torch.dot(d_l, sigma1_base).item()) / d_norm
            proj_s2 = abs(torch.dot(d_l, sigma2_base).item()) / d_norm
            ratio = proj_s2 / max(proj_s1, 1e-10)

            per_layer.append({
                "proj_s1": float(proj_s1),
                "proj_s2": float(proj_s2),
                "d_norm": float(d_norm),
                "ratio_s2_s1": float(ratio),
                "s1_drift_deg": s1_drift_deg,
            })

        model_data["doses"][dose_name] = per_layer

    # Analysis: D3 per-layer profile with drift
    d3 = model_data["doses"]["D3_therapeutic"]
    print(f"\n  D3 per-layer profile:")
    print(f"  {'L':>3}  {'proj_σ1':>8} {'proj_σ2':>8} {'ratio':>7} {'σ1_drift°':>9} {'d_norm':>8}")

    s2_dom = 0
    for i, p in enumerate(d3):
        if p and p.get("d_norm", 0) > 0:
            r = p["ratio_s2_s1"]
            drift = p.get("s1_drift_deg", 0)
            marker = " ***" if r > 2.0 else (" <<" if r < 0.5 else "")
            if drift > 5.0:
                marker += " DRIFT"
            if r > 1.0:
                s2_dom += 1
            print(f"  {i:3d}  {p['proj_s1']:8.4f} {p['proj_s2']:8.4f} {r:7.2f} {drift:9.2f} {p['d_norm']:8.3f}{marker}")

    # Late-half metrics
    late = d3[n_layers // 2:]
    late_ratios = [p["ratio_s2_s1"] for p in late if p and p.get("d_norm", 0) > 0]
    late_drifts = [p.get("s1_drift_deg", 0) for p in late if p and p.get("d_norm", 0) > 0]
    mean_late = np.mean(late_ratios) if late_ratios else 0
    mean_drift = np.mean(late_drifts) if late_drifts else 0
    max_drift = max(late_drifts) if late_drifts else 0
    late_s2 = sum(1 for r in late_ratios if r > 1.0)

    print(f"\n  σ₂ dominant: {s2_dom}/{n_layers} total, {late_s2}/{len(late_ratios)} late-half")
    print(f"  Late-half mean σ₂/σ₁: {mean_late:.2f}")
    print(f"  Late-half σ₁ drift: mean={mean_drift:.2f}°, max={max_drift:.2f}°")

    # σ₁ drift across all doses
    print(f"\n  σ₁ DRIFT across doses (mean late-half):")
    for dose_name in DOSES:
        dd = model_data["doses"][dose_name]
        late_d = dd[n_layers // 2:]
        drifts = [p.get("s1_drift_deg", 0) for p in late_d if p]
        md = np.mean(drifts) if drifts else 0
        mx = max(drifts) if drifts else 0
        print(f"    {dose_name:>15}: mean={md:.2f}°, max={mx:.2f}°")

    zone = "YES" if mean_late > 1.5 else ("weak" if mean_late > 0.8 else "NO")
    correct = "CONFIRMED" if zone == prediction else ("DECISIVE" if prediction == "???" else "FALSIFIED")
    print(f"\n  >>> ZONE: {zone} (predicted: {prediction}) → {correct} <<<")

    model_data["result"] = {
        "zone": zone, "late_mean_ratio": float(mean_late),
        "s2_dominant_total": s2_dom, "s2_dominant_late": late_s2,
        "verdict": correct,
    }

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return model_data


def main():
    all_results = {}

    for model_id, rope, comp, attn, pred in MODELS:
        try:
            data = test_model(model_id, rope, comp, attn, pred)
            all_results[model_id] = data
        except Exception as e:
            print(f"\n  ERROR on {model_id}: {e}")
            all_results[model_id] = {"error": str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — Zone Formation Decisive Tests")
    print(f"{'='*70}")
    print(f"{'Model':>25} | {'RoPE':>10} | {'Comp':>10} | {'Late σ₂/σ₁':>10} | {'Zone':>5} | {'Verdict'}")
    print("-" * 85)

    for model_id, data in all_results.items():
        if "error" in data:
            short = model_id.split("/")[-1]
            print(f"{short:>25} | {'ERROR':>10} | {'':>10} | {'':>10} | {'':>5} | {data['error'][:30]}")
        else:
            short = model_id.split("/")[-1]
            r = data["result"]
            print(f"{short:>25} | {data['rope']:>10} | {data['comp']:>10} | {r['late_mean_ratio']:>10.2f} | {r['zone']:>5} | {r['verdict']}")

    # Hypothesis test
    print(f"\n  HYPOTHESIS: σ₂ zone ⟺ (RoPE ≥ 40%) ∧ (sequential residual)")
    confirmed = sum(1 for d in all_results.values() if "result" in d and d["result"]["verdict"] == "CONFIRMED")
    falsified = sum(1 for d in all_results.values() if "result" in d and d["result"]["verdict"] == "FALSIFIED")
    decisive = [d for d in all_results.values() if "result" in d and d["result"]["verdict"] == "DECISIVE"]
    print(f"  Confirmed: {confirmed}/5, Falsified: {falsified}/5")
    if decisive:
        for d in decisive:
            print(f"  DECISIVE ({d['model']}): zone={d['result']['zone']} → {'sequential sufficient at 25% RoPE' if d['result']['zone'] == 'YES' else '25% RoPE insufficient even with sequential'}")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/zone_formation_decisive_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/zone_formation_decisive_results.json")


if __name__ == "__main__":
    main()
