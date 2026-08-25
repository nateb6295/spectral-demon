#!/usr/bin/env python3
"""
E77d: Anatomy of the tunnel sign-flip.

The tunnel zone (L2-14) flips from compressive at D1 to expansive at D2+.
What's actually happening per-layer? Which layers flip first? Is it uniform
or is there a wavefront propagating backward from the transition zone?

Also: what happens to the TOP SINGULAR VECTORS? Does V₁ direction change,
or just the singular values?
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
        probe_states = layer_states[0, -n_probe:, :].cpu().numpy().astype(np.float32)
        states.append(probe_states)
    return states

def compute_full_spectral(states):
    U, S, Vt = np.linalg.svd(states, full_matrices=False)
    s1 = S[0]
    s2 = S[1] if len(S) > 1 else 0.0
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    erank = np.exp(-np.sum(S_norm * np.log(S_norm)))
    return {
        's1': float(s1), 's2': float(s2),
        'ratio': float(s2 / s1) if s1 > 0 else 0.0,
        'erank': float(erank),
        'v1': Vt[0].tolist()[:20],  # first 20 dims of V₁
        'v2': Vt[1].tolist()[:20] if len(Vt) > 1 else [],
        'top5_sv': [float(s) for s in S[:5]]
    }

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0

def run_experiment():
    print("E77d: Anatomy of Tunnel Sign-Flip")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    # Focus on D1-D4 where the flip happens, plus D0 baseline and D8
    depths = [1, 2, 3, 4, 8]
    tunnel_layers = list(range(2, 15))

    results = {}

    for depth in depths:
        ccs_context = build_ccs_context(depth)
        ccs_tokens = len(tokenizer.encode(ccs_context))
        control_context = build_control_context(ccs_tokens, tokenizer)

        ccs_states = extract_hidden_states(model, tokenizer, ccs_context, PROBE)
        ctrl_states = extract_hidden_states(model, tokenizer, control_context, PROBE)

        depth_results = {}
        for l in tunnel_layers:
            ccs_spec = compute_full_spectral(ccs_states[l])
            ctrl_spec = compute_full_spectral(ctrl_states[l])

            # V₁ alignment between CCS and control
            v1_sim = cosine_sim(ccs_spec['v1'], ctrl_spec['v1'])
            v2_sim = cosine_sim(ccs_spec['v2'], ctrl_spec['v2'])

            depth_results[l] = {
                'modulation': ccs_spec['ratio'] - ctrl_spec['ratio'],
                's2_shift': ccs_spec['s2'] - ctrl_spec['s2'],
                'erank_shift': ccs_spec['erank'] - ctrl_spec['erank'],
                'ccs_ratio': ccs_spec['ratio'],
                'ctrl_ratio': ctrl_spec['ratio'],
                'ccs_s1': ccs_spec['s1'],
                'ctrl_s1': ctrl_spec['s1'],
                'ccs_s2': ccs_spec['s2'],
                'ctrl_s2': ctrl_spec['s2'],
                'v1_alignment': v1_sim,
                'v2_alignment': v2_sim,
                'ccs_erank': ccs_spec['erank'],
                'ctrl_erank': ctrl_spec['erank']
            }

        results[depth] = depth_results

    # Per-layer sign-flip analysis
    print("\nPER-LAYER MODULATION (tunnel L2-L14):")
    print(f"{'Layer':>5} {'D1':>9} {'D2':>9} {'D3':>9} {'D4':>9} {'D8':>9} {'Flip?':>6}")
    print("-" * 56)

    flip_points = {}
    for l in tunnel_layers:
        mods = [results[d][l]['modulation'] for d in depths]
        # Find where sign flips
        flip_d = None
        for i in range(1, len(depths)):
            if mods[0] < 0 and mods[i] > 0:
                flip_d = depths[i]
                break
        flip_points[l] = flip_d
        flip_str = f"D{flip_d}" if flip_d else "NO"
        print(f"  L{l:2d}  {mods[0]:+.5f} {mods[1]:+.5f} {mods[2]:+.5f} {mods[3]:+.5f} {mods[4]:+.5f}  {flip_str}")

    print("\nV₁ ALIGNMENT (CCS vs Control):")
    print(f"{'Layer':>5} {'D1':>8} {'D2':>8} {'D3':>8} {'D4':>8} {'D8':>8}")
    print("-" * 48)

    for l in tunnel_layers:
        aligns = [results[d][l]['v1_alignment'] for d in depths]
        print(f"  L{l:2d}  {aligns[0]:.4f}  {aligns[1]:.4f}  {aligns[2]:.4f}  {aligns[3]:.4f}  {aligns[4]:.4f}")

    print("\nV₂ ALIGNMENT (CCS vs Control):")
    print(f"{'Layer':>5} {'D1':>8} {'D2':>8} {'D3':>8} {'D4':>8} {'D8':>8}")
    print("-" * 48)

    for l in tunnel_layers:
        aligns = [results[d][l]['v2_alignment'] for d in depths]
        print(f"  L{l:2d}  {aligns[0]:.4f}  {aligns[1]:.4f}  {aligns[2]:.4f}  {aligns[3]:.4f}  {aligns[4]:.4f}")

    # Wavefront analysis: which layers flip first?
    print("\nWAVEFRONT ANALYSIS:")
    flip_order = [(l, d) for l, d in flip_points.items() if d is not None]
    flip_order.sort(key=lambda x: x[1])
    if flip_order:
        first_depth = flip_order[0][1]
        first_layers = [l for l, d in flip_order if d == first_depth]
        print(f"  First flip at D{first_depth}: layers {first_layers}")
        for d in sorted(set(d for _, d in flip_order)):
            layers_at_d = sorted([l for l, dd in flip_order if dd == d])
            print(f"  D{d}: {layers_at_d}")
    else:
        print("  No flips detected")

    no_flip = [l for l, d in flip_points.items() if d is None]
    if no_flip:
        print(f"  Never flip: {sorted(no_flip)}")

    # σ₁ vs σ₂ decomposition: is the flip driven by σ₁ or σ₂?
    print("\nDRIVER ANALYSIS (what drives the sign flip):")
    for l in [4, 8, 12]:  # sample early, mid, late tunnel
        s1_shifts = [results[d][l]['ccs_s1'] - results[d][l]['ctrl_s1'] for d in depths]
        s2_shifts = [results[d][l]['ccs_s2'] - results[d][l]['ctrl_s2'] for d in depths]
        ratio_mods = [results[d][l]['modulation'] for d in depths]
        print(f"\n  L{l}:")
        print(f"    σ₁ shifts: {['%+.3f' % s for s in s1_shifts]}")
        print(f"    σ₂ shifts: {['%+.3f' % s for s in s2_shifts]}")
        print(f"    ratio mod: {['%+.5f' % m for m in ratio_mods]}")

        # Is σ₂ growing faster than σ₁?
        s1_d1, s1_d8 = s1_shifts[0], s1_shifts[-1]
        s2_d1, s2_d8 = s2_shifts[0], s2_shifts[-1]
        print(f"    σ₁ trend D1→D8: {s1_d8 - s1_d1:+.4f}")
        print(f"    σ₂ trend D1→D8: {s2_d8 - s2_d1:+.4f}")
        if abs(s2_d8 - s2_d1) > abs(s1_d8 - s1_d1):
            print(f"    → σ₂-DRIVEN flip")
        else:
            print(f"    → σ₁-DRIVEN flip")

    # Save
    output = {
        'experiment': 'E77d',
        'model': MODEL,
        'question': 'What drives the tunnel sign-flip?',
        'depths': depths,
        'layers': tunnel_layers,
        'flip_points': {str(l): d for l, d in flip_points.items()},
        'per_layer': {str(l): {str(d): results[d][l] for d in depths} for l in tunnel_layers}
    }

    with open('/workspace/e77d_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /workspace/e77d_results.json")

if __name__ == '__main__':
    run_experiment()
