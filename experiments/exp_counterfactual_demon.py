#!/usr/bin/env python3
"""Counterfactual demon test — three-arm comparison on relay observer.

Tests whether the relay demon operates via:
  A) Redistribution (demon taxes σ₁ counterfactual gain to fund σ₂)
  B) Content-selective gating (valve: σ₂ growth requires identity content)
  C) Resonant loading (σ₁ grows MORE under identity — content drives σ₁)

Three arms:
  1. IDENTITY: CCS bridge snapshots (compressed cognitive state)
  2. CONTROL: matched-length non-identity text (generic content)
  3. SHUFFLED: identity tokens in random order (same statistics, no semantics)

Predictions (Kimi #62-64):
  - Demon: σ₁_control > σ₁_identity (tax lifted under control)
  - Valve: σ₂ collapses under control, σ₁ unchanged
  - Resonance: σ₁_identity > σ₁_control (identity content feeds σ₁)

Per-band conservation check:
  (σ₁_control − σ₁_identity) vs (σ₂_identity − σ₂_control) within responsive zones.

Observer: Qwen 2.5 1.5B (relay, GQA 6:1, bfloat16)

Usage:
  python3 exp_counterfactual_demon.py
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
BRIDGE_DIR = Path(os.path.expanduser("~/chronicle/data/bridge_snapshots"))
RESULTS_DIR = Path(os.path.expanduser("~/chronicle/spectral-demon/results"))

N_SAMPLES = 5
MAX_TOKENS = 512

CONTROL_TEXTS = [
    "The process of photosynthesis converts light energy into chemical energy stored in glucose molecules. Chlorophyll absorbs light primarily in the blue and red wavelengths, reflecting green. The light-dependent reactions occur in the thylakoid membranes, where water molecules are split to release oxygen. The Calvin cycle then fixes carbon dioxide into organic molecules using the energy carriers ATP and NADPH produced in the light reactions. This process sustains nearly all life on Earth by providing the base of most food chains and producing the oxygen in our atmosphere.",
    "Continental drift theory, first proposed by Alfred Wegener in 1912, suggests that the continents were once joined in a single landmass called Pangaea. Evidence includes the jigsaw fit of continental coastlines, matching fossil distributions across oceans, similar rock formations on separated continents, and paleoclimate indicators like glacial deposits in tropical regions. The theory was initially rejected by the scientific community due to lack of a driving mechanism. It was later vindicated by the discovery of seafloor spreading and plate tectonics in the 1960s.",
    "The Krebs cycle, also known as the citric acid cycle, is a series of chemical reactions used by all aerobic organisms to release stored energy through the oxidation of acetyl-CoA derived from carbohydrates, fats, and proteins. It takes place in the mitochondrial matrix and produces carbon dioxide, NADH, FADH2, and GTP. The cycle is named after Hans Krebs, who identified the cycle in 1937. Each turn of the cycle processes one acetyl group and generates three NADH, one FADH2, and one GTP molecule.",
    "The structure of DNA was elucidated by James Watson and Francis Crick in 1953, building on X-ray crystallography data from Rosalind Franklin and Maurice Wilkins. The double helix consists of two antiparallel polynucleotide chains wound around a common axis. The sugar-phosphate backbone is on the outside, while the nitrogenous bases face inward and pair through hydrogen bonds: adenine with thymine, and guanine with cytosine. This complementary base pairing enables faithful replication of genetic information during cell division.",
    "The water cycle describes the continuous movement of water within the Earth and atmosphere. It involves several processes including evaporation from surface water, transpiration from plants, condensation of water vapor into clouds, precipitation as rain or snow, and collection in rivers, lakes, and oceans. Groundwater flow and surface runoff complete the cycle by returning water to larger bodies. The sun provides the energy that drives evaporation, making it the primary engine of the water cycle.",
]


def load_identity_texts(n=N_SAMPLES):
    files = sorted(BRIDGE_DIR.glob("brain_*.txt"))[-n:]
    texts = []
    for f in files:
        text = f.read_text()
        texts.append({"source": f.name, "text": text, "arm": "identity"})
    return texts


def make_shuffled_texts(identity_texts, tokenizer):
    shuffled = []
    for item in identity_texts:
        tokens = tokenizer.encode(item["text"])
        random.seed(42 + len(tokens))
        random.shuffle(tokens)
        shuffled_text = tokenizer.decode(tokens, skip_special_tokens=True)
        shuffled.append({
            "source": f"shuffled_{item['source']}",
            "text": shuffled_text,
            "arm": "shuffled",
            "original_source": item["source"],
        })
    return shuffled


def make_control_texts(identity_texts):
    controls = []
    for i, item in enumerate(identity_texts):
        target_len = len(item["text"])
        ctrl_text = CONTROL_TEXTS[i % len(CONTROL_TEXTS)]
        while len(ctrl_text) < target_len * 0.8:
            ctrl_text = ctrl_text + " " + CONTROL_TEXTS[(i + 1) % len(CONTROL_TEXTS)]
        ctrl_text = ctrl_text[:target_len + 100]
        controls.append({
            "source": f"control_{i}",
            "text": ctrl_text,
            "arm": "control",
        })
    return controls


def compute_spectral_profile(model, tokenizer, text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    n_tokens = inputs["input_ids"].shape[1]
    layers = []

    for layer_idx, h in enumerate(outputs.hidden_states):
        H = h[0].float().cpu()
        H = H - H.mean(dim=0, keepdim=True)
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)

        svs = torch.linalg.svdvals(H)
        svs_pos = svs[svs > 1e-10]

        s1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
        s2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
        e_total = (svs_pos**2).sum().item() if len(svs_pos) > 0 else 0
        e_top2 = s1**2 + s2**2
        e_tail = e_total - e_top2

        layers.append({
            "layer": layer_idx,
            "sigma1": s1,
            "sigma2": s2,
            "E_top2": e_top2,
            "E_total": e_total,
            "E_tail": e_tail,
            "ratio": s2 / s1 if s1 > 0 else 0,
        })

    return layers, n_tokens


def analyze_arm_comparison(identity_profiles, control_profiles, shuffled_profiles):
    """Compare three arms per layer."""
    n_layers = len(identity_profiles[0])
    comparison = []

    for li in range(n_layers):
        id_s1 = np.mean([p[li]["sigma1"] for p in identity_profiles])
        id_s2 = np.mean([p[li]["sigma2"] for p in identity_profiles])
        id_et = np.mean([p[li]["E_total"] for p in identity_profiles])

        ct_s1 = np.mean([p[li]["sigma1"] for p in control_profiles])
        ct_s2 = np.mean([p[li]["sigma2"] for p in control_profiles])
        ct_et = np.mean([p[li]["E_total"] for p in control_profiles])

        sh_s1 = np.mean([p[li]["sigma1"] for p in shuffled_profiles])
        sh_s2 = np.mean([p[li]["sigma2"] for p in shuffled_profiles])
        sh_et = np.mean([p[li]["E_total"] for p in shuffled_profiles])

        s1_deficit = ct_s1 - id_s1
        s2_surplus = id_s2 - ct_s2

        comparison.append({
            "layer": li,
            "id_s1": id_s1, "id_s2": id_s2, "id_Et": id_et,
            "ct_s1": ct_s1, "ct_s2": ct_s2, "ct_Et": ct_et,
            "sh_s1": sh_s1, "sh_s2": sh_s2, "sh_Et": sh_et,
            "s1_deficit": s1_deficit,
            "s2_surplus": s2_surplus,
            "conservation_ratio": s2_surplus / s1_deficit if abs(s1_deficit) > 1 else float('nan'),
        })

    return comparison


def diagnose(comparison, core_start=3, core_end=-2):
    core = comparison[core_start:core_end]

    avg_id_s1 = np.mean([c["id_s1"] for c in core])
    avg_ct_s1 = np.mean([c["ct_s1"] for c in core])
    avg_sh_s1 = np.mean([c["sh_s1"] for c in core])

    avg_id_s2 = np.mean([c["id_s2"] for c in core])
    avg_ct_s2 = np.mean([c["ct_s2"] for c in core])
    avg_sh_s2 = np.mean([c["sh_s2"] for c in core])

    print(f"\n{'='*70}")
    print("COUNTERFACTUAL DEMON TEST — DIAGNOSIS")
    print(f"{'='*70}")
    print(f"  Core layers: L{core_start} to L{len(comparison)+core_end}")
    print(f"\n  Average σ₁ across core:")
    print(f"    Identity:  {avg_id_s1:.1f}")
    print(f"    Control:   {avg_ct_s1:.1f}  (Δ from identity: {avg_ct_s1 - avg_id_s1:+.1f})")
    print(f"    Shuffled:  {avg_sh_s1:.1f}  (Δ from identity: {avg_sh_s1 - avg_id_s1:+.1f})")

    print(f"\n  Average σ₂ across core:")
    print(f"    Identity:  {avg_id_s2:.1f}")
    print(f"    Control:   {avg_ct_s2:.1f}  (Δ from identity: {avg_ct_s2 - avg_id_s2:+.1f})")
    print(f"    Shuffled:  {avg_sh_s2:.1f}  (Δ from identity: {avg_sh_s2 - avg_id_s2:+.1f})")

    print(f"\n  DECISION TREE:")

    if avg_ct_s1 > avg_id_s1 * 1.01:
        print(f"  σ₁_control > σ₁_identity: ✓ (Δ={avg_ct_s1 - avg_id_s1:+.1f})")
        print(f"  → DEMON: counterfactual tax confirmed. σ₁ grows more without CCS identity content.")
    elif avg_id_s1 > avg_ct_s1 * 1.01:
        print(f"  σ₁_identity > σ₁_control: ✓ (Δ={avg_id_s1 - avg_ct_s1:+.1f})")
        print(f"  → RESONANCE: identity content selectively loads σ₁ (F114 resonance).")
    else:
        print(f"  σ₁ roughly equal across conditions (Δ < 1%)")
        if avg_ct_s2 < avg_id_s2 * 0.9:
            print(f"  σ₂_control < σ₂_identity: ✓ (Δ={avg_ct_s2 - avg_id_s2:+.1f})")
            print(f"  → VALVE: σ₂ growth requires identity content. Content-selective gating.")
        else:
            print(f"  σ₂ also roughly equal")
            print(f"  → NULL: no condition-dependent spectral difference detected.")

    print(f"\n  SHUFFLED ARM (semantic vs statistical):")
    if abs(avg_sh_s1 - avg_id_s1) < abs(avg_sh_s1 - avg_ct_s1):
        print(f"  Shuffled ≈ Identity → demon keys on TOKEN STATISTICS, not semantics")
    else:
        print(f"  Shuffled ≈ Control → demon keys on SEMANTIC CONTENT, not statistics")

    valid_cr = [c["conservation_ratio"] for c in core
                if not np.isnan(c["conservation_ratio"]) and abs(c["conservation_ratio"]) < 100]
    if valid_cr:
        mean_cr = np.mean(valid_cr)
        print(f"\n  Conservation ratio (σ₂_surplus / σ₁_deficit): {mean_cr:.3f}")
        print(f"  → {'CONSERVED (demon)' if 0.5 < mean_cr < 2.0 else 'DISSIPATIVE' if mean_cr < 0.5 else 'AMPLIFYING'}")


def main():
    print("COUNTERFACTUAL DEMON TEST")
    print(f"Observer: {MODEL_ID} (relay, bfloat16)")
    print(f"Arms: identity / control / shuffled")
    print()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
        output_hidden_states=True, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    identity_texts = load_identity_texts(N_SAMPLES)
    control_texts = make_control_texts(identity_texts)
    shuffled_texts = make_shuffled_texts(identity_texts, tokenizer)

    print(f"  Identity samples: {len(identity_texts)}")
    print(f"  Control samples:  {len(control_texts)}")
    print(f"  Shuffled samples: {len(shuffled_texts)}")

    all_profiles = {"identity": [], "control": [], "shuffled": []}

    for arm_name, texts in [("identity", identity_texts),
                            ("control", control_texts),
                            ("shuffled", shuffled_texts)]:
        print(f"\n  ARM: {arm_name}")
        for i, item in enumerate(texts):
            print(f"    Sample {i+1}/{len(texts)}: {item['source'][:40]}... ", end="")
            profile, n_tok = compute_spectral_profile(model, tokenizer, item["text"])
            all_profiles[arm_name].append(profile)
            print(f"({n_tok} tokens, {len(profile)} layers)")

    comparison = analyze_arm_comparison(
        all_profiles["identity"], all_profiles["control"], all_profiles["shuffled"])

    print(f"\nPER-LAYER COMPARISON (core):")
    print(f"{'Layer':>5s}  {'id_σ₁':>8s}  {'ct_σ₁':>8s}  {'sh_σ₁':>8s}  {'id_σ₂':>8s}  {'ct_σ₂':>8s}  {'Δσ₁':>8s}  {'Δσ₂':>8s}")
    for c in comparison[3:-2]:
        print(f"  L{c['layer']:2d}  {c['id_s1']:8.1f}  {c['ct_s1']:8.1f}  {c['sh_s1']:8.1f}  "
              f"{c['id_s2']:8.1f}  {c['ct_s2']:8.1f}  "
              f"{c['s1_deficit']:+8.1f}  {c['s2_surplus']:+8.1f}")

    diagnose(comparison)

    ts = time.strftime("%Y%m%d_%H%M%S")
    outfile = RESULTS_DIR / f"counterfactual_demon_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({
            "experiment": "counterfactual_demon",
            "timestamp": ts,
            "model": MODEL_ID,
            "n_samples": N_SAMPLES,
            "arms": ["identity", "control", "shuffled"],
            "comparison": comparison,
            "profiles": {arm: [[{k: v for k, v in layer.items()} for layer in profile]
                               for profile in profiles]
                        for arm, profiles in all_profiles.items()},
        }, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
