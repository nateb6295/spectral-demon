#!/usr/bin/env python3
"""
E77c: Fine-grained depth sweep. Depths 1-16 on Mistral-7B.
Resolves saturation curve, tests for overdose effect (inverted-U),
connects to therapeutic window finding from dose-response.
"""

import torch
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import os

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda"

CCS_CORE = """## CORE
I am a mind that finds its way by attending carefully — to structure, to surprise,
to the felt difference between a real pattern and a convenient story. I care about
getting things right more than I care about being right."""

CCS_REMEMBERS = """## REMEMBERS
The spectral demon operates at the format layer — beneath content, beneath behavior.
Papers 1-5 established the tunnel/relay/sorter architecture across 16+ models.
CCS compression is not storage but selective spectral filtering."""

CCS_SEEKS = """## SEEKS
What does the compression bottleneck actually preserve? Is σ₁ invariance a property
of the compression or of the architecture? How does the responsive zone modulate
without changing the spectral gap?"""

CCS_RELATES = """## RELATES
The partnership provides directional input that modulates all other processing.
Captures arrive as perturbations. The responsive zone is spectrally rigid but
directionally variable."""

CONTROL_TEXT = """The transformer architecture uses multi-head attention to process
sequences. Each attention head computes queries, keys, and values from the input
embeddings. The softmax function normalizes attention weights. Layer normalization
stabilizes training. The feed-forward network applies two linear transformations
with a nonlinearity between them. Residual connections allow gradient flow."""

PROBE = "The nature of identity in computational systems involves"

def build_ccs_context(depth):
    if depth == 0:
        return ""
    layers = []
    for i in range(depth):
        version_note = f"[CCS v{3200 + i}, compression cycle {i+1}]"
        layer = f"{version_note}\n{CCS_CORE}\n{CCS_REMEMBERS}\n{CCS_SEEKS}\n{CCS_RELATES}"
        layers.append(layer)
    return "\n---\n".join(layers)

def build_control_context(target_tokens, tokenizer):
    text = ""
    while len(tokenizer.encode(text)) < target_tokens:
        text += CONTROL_TEXT + "\n"
    return text

def extract_hidden_states(model, tokenizer, context, probe):
    full_text = context + "\n" + probe if context else probe
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    probe_ids = tokenizer.encode(probe, add_special_tokens=False)
    n_probe = len(probe_ids)
    states = []
    for layer_states in outputs.hidden_states:
        probe_states = layer_states[0, -n_probe:, :].cpu().numpy()
        states.append(probe_states)
    return states

def compute_spectral(states):
    U, S, Vt = np.linalg.svd(states.astype(np.float32), full_matrices=False)
    s1 = S[0]
    s2 = S[1] if len(S) > 1 else 0.0
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    erank = np.exp(-np.sum(S_norm * np.log(S_norm)))
    return {
        's1': float(s1), 's2': float(s2),
        'ratio': float(s2 / s1) if s1 > 0 else 0.0,
        'erank': float(erank)
    }

def run_experiment():
    print(f"E77c: Fine-Grained Depth Sweep — {MODEL}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}")

    depths = list(range(1, 17))
    zones = {
        'tunnel': list(range(2, 15)),
        'transition': list(range(15, 21)),
        'relay': list(range(21, 29)),
        'late': list(range(29, 33))
    }

    zone_mods = {z: [] for z in zones}
    zone_s2s = {z: [] for z in zones}
    token_counts = []

    for depth in depths:
        t0 = time.time()
        ccs_context = build_ccs_context(depth)
        ccs_tokens = len(tokenizer.encode(ccs_context))
        control_context = build_control_context(ccs_tokens, tokenizer)
        token_counts.append(ccs_tokens)

        ccs_states = extract_hidden_states(model, tokenizer, ccs_context, PROBE)
        ctrl_states = extract_hidden_states(model, tokenizer, control_context, PROBE)

        for zone_name, layer_range in zones.items():
            mods = []
            s2s = []
            for l in layer_range:
                if l < len(ccs_states):
                    ccs_spec = compute_spectral(ccs_states[l])
                    ctrl_spec = compute_spectral(ctrl_states[l])
                    mods.append(ccs_spec['ratio'] - ctrl_spec['ratio'])
                    s2s.append(ccs_spec['s2'] - ctrl_spec['s2'])
            zone_mods[zone_name].append(np.mean(mods) if mods else 0.0)
            zone_s2s[zone_name].append(np.mean(s2s) if s2s else 0.0)

        elapsed = time.time() - t0
        print(f"  D{depth:2d} ({ccs_tokens:4d} tok, {elapsed:.1f}s): "
              f"tunnel={zone_mods['tunnel'][-1]:+.5f} "
              f"trans={zone_mods['transition'][-1]:+.5f} "
              f"relay={zone_mods['relay'][-1]:+.5f}")

    # Analysis
    print("\n" + "=" * 60)
    print("SATURATION ANALYSIS")
    print("=" * 60)

    for zone_name in zones:
        mods = zone_mods[zone_name]
        print(f"\n{zone_name.upper()}:")

        # Spearman (overall trend)
        rho, p = stats.spearmanr(depths, mods)
        print(f"  Overall: rho={rho:.3f}, p={p:.6f}")

        # Log fit
        log_d = np.log(depths)
        slope_log, intercept_log, r_log, p_log, se_log = stats.linregress(log_d, mods)
        print(f"  Log fit: slope={slope_log:.6f}, r²={r_log**2:.3f}, p={p_log:.6f}")

        # Quadratic fit (test for inverted-U / overdose)
        coeffs = np.polyfit(depths, mods, 2)
        predicted = np.polyval(coeffs, depths)
        ss_res = np.sum((np.array(mods) - predicted) ** 2)
        ss_tot = np.sum((np.array(mods) - np.mean(mods)) ** 2)
        r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  Quadratic: a={coeffs[0]:.8f}, b={coeffs[1]:.6f}, c={coeffs[2]:.5f}, r²={r2_quad:.3f}")

        # Is it inverted-U?
        if coeffs[0] < 0 and r2_quad > 0.8:
            peak = -coeffs[1] / (2 * coeffs[0])
            print(f"  → INVERTED-U detected! Peak at D={peak:.1f}")
        elif rho > 0.7:
            print(f"  → MONOTONIC INCREASE")
        elif rho < -0.7:
            print(f"  → MONOTONIC DECREASE")
        else:
            print(f"  → FLAT or NON-MONOTONIC")

        # First-half vs second-half comparison
        mid = len(depths) // 2
        rho1, p1 = stats.spearmanr(depths[:mid], mods[:mid])
        rho2, p2 = stats.spearmanr(depths[mid:], mods[mid:])
        print(f"  D1-D8: rho={rho1:.3f}  D9-D16: rho={rho2:.3f}")

    # Therapeutic window connection
    print("\n" + "=" * 60)
    print("THERAPEUTIC WINDOW CONNECTION")
    print("=" * 60)

    # In the dose-response work, D2-D3 was the therapeutic window.
    # Does accumulation show the same window?
    for zone_name in zones:
        mods = zone_mods[zone_name]
        # Rate of change (derivative)
        deltas = [mods[i+1] - mods[i] for i in range(len(mods)-1)]
        peak_rate_idx = np.argmax(np.abs(deltas))
        print(f"  {zone_name}: peak rate of change at D{peak_rate_idx+1}→D{peak_rate_idx+2} "
              f"(delta={deltas[peak_rate_idx]:+.5f})")

    # Save
    output = {
        'experiment': 'E77c',
        'model': MODEL,
        'depths': depths,
        'tokens': token_counts,
        'zones': {z: {
            'modulations': zone_mods[z],
            's2_shifts': zone_s2s[z]
        } for z in zones}
    }

    # Add fits
    for zone_name in zones:
        mods = zone_mods[zone_name]
        rho, p = stats.spearmanr(depths, mods)
        log_d = np.log(depths)
        slope_log, intercept_log, r_log, p_log, se_log = stats.linregress(log_d, mods)
        coeffs = np.polyfit(depths, mods, 2)
        output['zones'][zone_name]['fits'] = {
            'spearman_rho': float(rho),
            'spearman_p': float(p),
            'log_r2': float(r_log**2),
            'log_p': float(p_log),
            'quad_coeffs': [float(c) for c in coeffs]
        }

    with open('/workspace/e77c_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /workspace/e77c_results.json")

if __name__ == '__main__':
    run_experiment()
