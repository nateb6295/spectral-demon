#!/usr/bin/env python3
"""
E77b: Cross-architecture replication of ICL accumulation.
Run on Llama-3.1-8B-Instruct (MHA) to compare with Mistral-7B (GQA).
If accumulation holds → universal. If different → species-specific.
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

MODEL = "Qwen/Qwen2.5-7B-Instruct"
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
        'erank': float(erank), 'gap': float(s1 - s2),
        'top5': [float(s) for s in S[:5]]
    }

def run_experiment():
    print(f"E77b: Cross-Architecture ICL Accumulation — {MODEL}")
    print("=" * 60)

    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}")

    # Llama-3.1-8B has 32 layers like Mistral — zone definitions match
    depths = [0, 1, 2, 4, 8]
    results = {}

    for depth in depths:
        print(f"\n--- Depth {depth} ---")
        ccs_context = build_ccs_context(depth)
        ccs_tokens = len(tokenizer.encode(ccs_context)) if ccs_context else 0

        if ccs_tokens > 0:
            control_context = build_control_context(ccs_tokens, tokenizer)
            ctrl_tokens = len(tokenizer.encode(control_context))
        else:
            control_context = ""
            ctrl_tokens = 0

        print(f"  CCS tokens: {ccs_tokens}, Control tokens: {ctrl_tokens}")

        t0 = time.time()
        ccs_states = extract_hidden_states(model, tokenizer, ccs_context, PROBE)
        t_ccs = time.time() - t0

        t0 = time.time()
        ctrl_states = extract_hidden_states(model, tokenizer, control_context, PROBE)
        t_ctrl = time.time() - t0

        print(f"  Forward passes: CCS={t_ccs:.1f}s, Control={t_ctrl:.1f}s")

        depth_results = {'depth': depth, 'ccs_tokens': ccs_tokens, 'ctrl_tokens': ctrl_tokens, 'layers': {}}
        n_states = len(ccs_states)

        for layer_idx in range(n_states):
            ccs_spec = compute_spectral(ccs_states[layer_idx])
            ctrl_spec = compute_spectral(ctrl_states[layer_idx])
            modulation = ccs_spec['ratio'] - ctrl_spec['ratio']
            s2_shift = ccs_spec['s2'] - ctrl_spec['s2']
            erank_shift = ccs_spec['erank'] - ctrl_spec['erank']
            depth_results['layers'][layer_idx] = {
                'ccs': ccs_spec, 'ctrl': ctrl_spec,
                'modulation': float(modulation),
                's2_shift': float(s2_shift),
                'erank_shift': float(erank_shift)
            }

        results[depth] = depth_results

        for zone_name, layer_range in [('tunnel', range(2, 15)), ('relay', range(21, 29)), ('late', range(29, n_states))]:
            mods = [depth_results['layers'][l]['modulation'] for l in layer_range if l < n_states]
            if mods:
                print(f"  {zone_name}: avg_modulation={np.mean(mods):.4f}")

    # Cross-depth analysis (D1-D8 only, excluding D0 artifact)
    print("\n" + "=" * 60)
    print("CROSS-DEPTH ANALYSIS (D1-D8, corrected)")
    print("=" * 60)

    zones = {
        'tunnel': list(range(2, 15)),
        'transition': list(range(15, 21)),
        'relay': list(range(21, 29)),
        'late': list(range(29, 33))
    }

    depths_ccs = [1, 2, 4, 8]
    accumulation_results = {}

    for zone_name, layer_range in zones.items():
        print(f"\n{zone_name.upper()} ZONE:")
        mods_by_depth = []

        for depth in depths_ccs:
            mods = [results[depth]['layers'][l]['modulation'] for l in layer_range
                    if l < len(results[depth]['layers'])]
            avg_mod = np.mean(mods) if mods else 0.0
            mods_by_depth.append(avg_mod)
            print(f"  D{depth}: modulation={avg_mod:.5f}")

        rho, p = stats.spearmanr(depths_ccs, mods_by_depth)
        slope, intercept, r, p_lin, se = stats.linregress(depths_ccs, mods_by_depth)
        log_depths = np.log(depths_ccs)
        slope_log, intercept_log, r_log, p_log, se_log = stats.linregress(log_depths, mods_by_depth)

        print(f"  Spearman: rho={rho:.3f}, p={p:.4f}")
        print(f"  Linear: r²={r**2:.3f}, p={p_lin:.4f}")
        print(f"  Log: r²={r_log**2:.3f}, p={p_log:.4f}")

        delta = mods_by_depth[-1] - mods_by_depth[0]
        print(f"  D1→D8 change: {delta:+.5f}")

        accumulation_results[zone_name] = {
            'depths': depths_ccs,
            'modulations': mods_by_depth,
            'rho': float(rho), 'rho_p': float(p),
            'slope': float(slope), 'r_squared': float(r**2), 'p_linear': float(p_lin),
            'r_squared_log': float(r_log**2), 'p_log': float(p_log)
        }

    # Compare with Mistral
    print("\n" + "=" * 60)
    print("CROSS-ARCHITECTURE COMPARISON")
    print("=" * 60)

    # Load Mistral results
    try:
        with open('/workspace/e77_results.json') as f:
            mistral = json.load(f)

        print(f"\n{'Zone':<12} {'Mistral rho':>12} {'Llama rho':>12} {'Match?':>8}")
        print("-" * 48)
        for zone in zones:
            m_mods = mistral['accumulation'][zone]['modulations'][1:]
            m_rho, _ = stats.spearmanr([1,2,4,8], m_mods)
            l_rho = accumulation_results[zone]['rho']
            match = "YES" if (m_rho > 0.5 and l_rho > 0.5) or (m_rho < -0.5 and l_rho < -0.5) else "DIFF"
            print(f"{zone:<12} {m_rho:>12.3f} {l_rho:>12.3f} {match:>8}")

    except FileNotFoundError:
        print("  (Mistral results not found for comparison)")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    acc_zones = [z for z, d in accumulation_results.items() if d['rho'] > 0.7]
    flat_zones = [z for z, d in accumulation_results.items() if abs(d['rho']) <= 0.7]
    dec_zones = [z for z, d in accumulation_results.items() if d['rho'] < -0.7]

    if acc_zones:
        print(f"  ACCUMULATING: {', '.join(acc_zones)}")
    if flat_zones:
        print(f"  FLAT: {', '.join(flat_zones)}")
    if dec_zones:
        print(f"  DECREASING: {', '.join(dec_zones)}")

    # Save
    output = {
        'experiment': 'E77b',
        'model': MODEL,
        'architecture': 'GQA-4:1 (goldsmith)',
        'comparison_model': 'Mistral-7B (GQA-8:1, potter)',
        'depths_tested': depths,
        'accumulation': accumulation_results,
        'accumulating_zones': acc_zones,
        'flat_zones': flat_zones,
        'raw': {str(d): {
            'depth': results[d]['depth'],
            'ccs_tokens': results[d]['ccs_tokens'],
            'layers': {str(l): results[d]['layers'][l] for l in [0, 8, 16, 24, 28, 31]
                      if l in results[d]['layers']}
        } for d in depths}
    }

    with open('/workspace/e77b_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to /workspace/e77b_results.json")

if __name__ == '__main__':
    run_experiment()
