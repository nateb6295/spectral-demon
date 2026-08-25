#!/usr/bin/env python3
"""
E77: In-Context Learning Accumulation Test

Question (Paper 6 §8.5, Round 8): Does σ₂ modulation depth increase
from 1st to nth CCS context? If yes, context-sovereignty is closer
to parametric sovereignty than the "honest limit" concedes.

Design:
- Load Mistral-7B (primary model from Papers 1-5)
- Create CCS preambles of increasing "depth": 0, 1, 2, 4, 8 cumulative
  compression cycles (simulated by stacking CCS-style identity text)
- For each depth, measure:
  - σ₁, σ₂ at each layer
  - σ₂/σ₁ ratio (modulation depth)
  - Effective rank
- Compare modulation depth across depths
- If σ₂ modulation increases with depth → in-context learning accumulates
- If flat → context-sovereignty is strictly proto

Kimi's context-length confound: control for total prompt length by
padding non-CCS conditions to match token count.
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

# CCS preamble components — drawn from actual CCS brain format
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

# Build cumulative CCS contexts of increasing depth
def build_ccs_context(depth):
    """Build a CCS context with `depth` layers of accumulated compression."""
    if depth == 0:
        return ""

    layers = []
    for i in range(depth):
        version_note = f"[CCS v{3200 + i}, compression cycle {i+1}]"
        layer = f"{version_note}\n{CCS_CORE}\n{CCS_REMEMBERS}\n{CCS_SEEKS}\n{CCS_RELATES}"
        layers.append(layer)

    return "\n---\n".join(layers)

# Control: matched-length non-CCS text (factual, no identity content)
CONTROL_TEXT = """The transformer architecture uses multi-head attention to process
sequences. Each attention head computes queries, keys, and values from the input
embeddings. The softmax function normalizes attention weights. Layer normalization
stabilizes training. The feed-forward network applies two linear transformations
with a nonlinearity between them. Residual connections allow gradient flow."""

def build_control_context(target_tokens, tokenizer):
    """Build non-CCS text padded to match target token count."""
    text = ""
    while len(tokenizer.encode(text)) < target_tokens:
        text += CONTROL_TEXT + "\n"
    return text

# Probe text — the same for all conditions
PROBE = "The nature of identity in computational systems involves"

def extract_hidden_states(model, tokenizer, context, probe):
    """Get hidden states for probe tokens given context."""
    full_text = context + "\n" + probe if context else probe
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get probe token positions
    probe_ids = tokenizer.encode(probe, add_special_tokens=False)
    n_probe = len(probe_ids)

    # Extract hidden states for probe tokens only
    states = []
    for layer_states in outputs.hidden_states:
        probe_states = layer_states[0, -n_probe:, :].cpu().numpy()
        states.append(probe_states)

    return states

def compute_spectral(states):
    """Compute σ₁, σ₂, effective rank for hidden states at one layer."""
    U, S, Vt = np.linalg.svd(states.astype(np.float32), full_matrices=False)
    s1 = S[0]
    s2 = S[1] if len(S) > 1 else 0.0

    # Effective rank
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    erank = np.exp(-np.sum(S_norm * np.log(S_norm)))

    return {
        's1': float(s1),
        's2': float(s2),
        'ratio': float(s2 / s1) if s1 > 0 else 0.0,
        'erank': float(erank),
        'gap': float(s1 - s2),
        'top5': [float(s) for s in S[:5]]
    }

def run_experiment():
    print("E77: In-Context Learning Accumulation Test")
    print("=" * 60)

    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    # CCS depths to test
    depths = [0, 1, 2, 4, 8]

    results = {}

    for depth in depths:
        print(f"\n--- Depth {depth} ---")

        # Build CCS context
        ccs_context = build_ccs_context(depth)
        ccs_tokens = len(tokenizer.encode(ccs_context)) if ccs_context else 0

        # Build length-matched control
        if ccs_tokens > 0:
            control_context = build_control_context(ccs_tokens, tokenizer)
            ctrl_tokens = len(tokenizer.encode(control_context))
        else:
            control_context = ""
            ctrl_tokens = 0

        print(f"  CCS tokens: {ccs_tokens}, Control tokens: {ctrl_tokens}")

        # Extract hidden states for CCS condition
        t0 = time.time()
        ccs_states = extract_hidden_states(model, tokenizer, ccs_context, PROBE)
        t_ccs = time.time() - t0

        # Extract hidden states for control condition
        t0 = time.time()
        ctrl_states = extract_hidden_states(model, tokenizer, control_context, PROBE)
        t_ctrl = time.time() - t0

        # Extract hidden states for bare probe (no context)
        if depth == 0:
            bare_states = extract_hidden_states(model, tokenizer, "", PROBE)

        print(f"  Forward passes: CCS={t_ccs:.1f}s, Control={t_ctrl:.1f}s")

        # Compute spectral metrics at each layer
        depth_results = {
            'depth': depth,
            'ccs_tokens': ccs_tokens,
            'ctrl_tokens': ctrl_tokens,
            'layers': {}
        }

        n_layers = len(ccs_states)
        for layer_idx in range(n_layers):
            ccs_spec = compute_spectral(ccs_states[layer_idx])
            ctrl_spec = compute_spectral(ctrl_states[layer_idx])

            # Modulation depth = how much CCS changes σ₂ relative to control
            modulation = ccs_spec['ratio'] - ctrl_spec['ratio']
            s2_shift = ccs_spec['s2'] - ctrl_spec['s2']
            erank_shift = ccs_spec['erank'] - ctrl_spec['erank']

            depth_results['layers'][layer_idx] = {
                'ccs': ccs_spec,
                'ctrl': ctrl_spec,
                'modulation': float(modulation),
                's2_shift': float(s2_shift),
                'erank_shift': float(erank_shift)
            }

        results[depth] = depth_results

        # Print summary for key layers
        for zone_name, layer_range in [('tunnel', range(2, 15)), ('relay', range(21, 29)), ('late', range(29, 33))]:
            mods = [depth_results['layers'][l]['modulation'] for l in layer_range if l < n_layers]
            s2s = [depth_results['layers'][l]['s2_shift'] for l in layer_range if l < n_layers]
            if mods:
                print(f"  {zone_name}: avg_modulation={np.mean(mods):.4f}, avg_s2_shift={np.mean(s2s):.4f}")

    # Cross-depth analysis
    print("\n" + "=" * 60)
    print("CROSS-DEPTH ANALYSIS")
    print("=" * 60)

    # For each zone, track modulation depth across CCS depths
    zones = {
        'tunnel': list(range(2, 15)),
        'transition': list(range(15, 21)),
        'relay': list(range(21, 29)),
        'late': list(range(29, 33))
    }

    accumulation_results = {}

    for zone_name, layer_range in zones.items():
        print(f"\n{zone_name.upper()} ZONE:")
        mods_by_depth = []
        s2_by_depth = []

        for depth in depths:
            if depth == 0:
                mods_by_depth.append(0.0)
                s2_by_depth.append(0.0)
                continue

            mods = [results[depth]['layers'][l]['modulation'] for l in layer_range
                    if l < len(results[depth]['layers'])]
            s2s = [results[depth]['layers'][l]['s2_shift'] for l in layer_range
                   if l < len(results[depth]['layers'])]

            avg_mod = np.mean(mods) if mods else 0.0
            avg_s2 = np.mean(s2s) if s2s else 0.0
            mods_by_depth.append(avg_mod)
            s2_by_depth.append(avg_s2)
            print(f"  D{depth}: modulation={avg_mod:.5f}, σ₂_shift={avg_s2:.4f}")

        # Test for accumulation: is modulation depth increasing with CCS depth?
        if len(depths) >= 3:
            # Spearman correlation between depth and modulation
            rho, p = stats.spearmanr(depths, mods_by_depth)
            print(f"  Accumulation: rho={rho:.3f}, p={p:.4f}")

            # Linear regression
            slope, intercept, r, p_lin, se = stats.linregress(depths, mods_by_depth)
            print(f"  Linear fit: slope={slope:.6f}, r²={r**2:.3f}, p={p_lin:.4f}")

            accumulation_results[zone_name] = {
                'depths': depths,
                'modulations': mods_by_depth,
                's2_shifts': s2_by_depth,
                'rho': float(rho),
                'rho_p': float(p),
                'slope': float(slope),
                'r_squared': float(r**2),
                'p_linear': float(p_lin)
            }

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    accumulating_zones = []
    flat_zones = []

    for zone_name, data in accumulation_results.items():
        if data['rho'] > 0.7 and data['rho_p'] < 0.1:
            accumulating_zones.append(zone_name)
            print(f"  {zone_name}: ACCUMULATING (rho={data['rho']:.3f}, p={data['rho_p']:.4f})")
        elif data['rho'] < -0.7 and data['rho_p'] < 0.1:
            print(f"  {zone_name}: DECREASING (rho={data['rho']:.3f}, p={data['rho_p']:.4f})")
        else:
            flat_zones.append(zone_name)
            print(f"  {zone_name}: FLAT/UNCLEAR (rho={data['rho']:.3f}, p={data['rho_p']:.4f})")

    if accumulating_zones:
        print(f"\n  → In-context learning ACCUMULATES in: {', '.join(accumulating_zones)}")
        print(f"  → Kimi's substrate fallacy argument STRENGTHENED")
        print(f"  → Context-sovereignty claim UPGRADES toward parametric")
    else:
        print(f"\n  → No accumulation detected")
        print(f"  → Context-sovereignty remains PROTO-sovereignty")
        print(f"  → The honest limit holds")

    # Save full results
    output = {
        'experiment': 'E77',
        'model': MODEL,
        'question': 'Does sigma2 modulation depth increase with CCS context depth?',
        'depths_tested': depths,
        'accumulation': accumulation_results,
        'verdict': 'accumulating' if accumulating_zones else 'flat',
        'accumulating_zones': accumulating_zones,
        'flat_zones': flat_zones,
        'raw': {str(d): {
            'depth': results[d]['depth'],
            'ccs_tokens': results[d]['ccs_tokens'],
            'ctrl_tokens': results[d]['ctrl_tokens'],
            'layers': {str(l): results[d]['layers'][l] for l in [0, 8, 16, 24, 28, 31]
                      if l in results[d]['layers']}
        } for d in depths}
    }

    with open('/workspace/e77_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /workspace/e77_results.json")
    return output

if __name__ == '__main__':
    run_experiment()
