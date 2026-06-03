#!/usr/bin/env python3
"""Analyze InternLM per-layer spectral entropy results and compare with Mistral."""
import json
import sys
import os
import numpy as np

def analyze(path, plot=True):
    with open(path) as f:
        data = json.load(f)

    # Handle both possible data formats
    if isinstance(data, dict) and any(k.startswith('layer') for k in data.keys()):
        layers = sorted([k for k in data.keys() if k.startswith('layer')],
                       key=lambda x: int(x.replace('layer_', '')))
        get_S = lambda layer, cond: np.mean(data[layer].get(cond, {}).get('S',
                    data[layer].get(cond, {}).get('spectral_entropy', [0])))
        get_s2 = lambda layer, cond: np.mean(data[layer].get(cond, {}).get('sigma2',
                    data[layer].get(cond, {}).get('sigma_2', [0])))
        get_layer_num = lambda k: int(k.replace('layer_', ''))
    elif isinstance(data, dict) and 'raw' in data:
        from collections import defaultdict
        layers_data = defaultdict(lambda: defaultdict(list))
        for r in data['raw']:
            layers_data[r['layer']][r['condition']].append(r)
        layers = sorted(layers_data.keys())
        get_S = lambda layer, cond: np.mean([r['spectral_entropy'] for r in layers_data[layer].get(cond, [{'spectral_entropy': 0}])])
        get_s2 = lambda layer, cond: np.mean([r.get('sigma_2', r.get('sigma2', 0)) for r in layers_data[layer].get(cond, [{}])])
        get_layer_num = lambda k: k
    else:
        print(f"Unknown data format. Top keys: {list(data.keys())[:10]}")
        return

    print(f"InternLM 2.5 7B Chat — Per-Layer Spectral Analysis")
    print(f"{'Layer':<10} {'S_recv':>8} {'S_abs':>8} {'dS':>8} {'sig2_r':>8} {'sig2_a':>8}")
    print("-" * 55)

    layer_nums = []
    dS_vals = []
    S_r_vals = []
    S_a_vals = []

    for layer in layers:
        S_r = get_S(layer, 'receptive')
        S_a = get_S(layer, 'absent')
        dS = S_r - S_a
        sig2_r = get_s2(layer, 'receptive')
        sig2_a = get_s2(layer, 'absent')

        lnum = get_layer_num(layer)
        layer_nums.append(lnum)
        dS_vals.append(dS)
        S_r_vals.append(S_r)
        S_a_vals.append(S_a)

        marker = " ***" if abs(dS) > 0.02 else ""
        print(f"L{lnum:<9} {S_r:>8.3f} {S_a:>8.3f} {dS:>+8.4f} {sig2_r:>8.1f} {sig2_a:>8.1f}{marker}")

    dS_arr = np.array(dS_vals)
    peak_idx = np.argmax(np.abs(dS_arr))
    peak_layer = layer_nums[peak_idx]

    n = len(layer_nums)
    tunnel_slice = slice(2, min(21, n))
    relay_slice = slice(min(24, n-4), n)

    print(f"\n--- Summary ---")
    print(f"Peak |dS| at layer {peak_layer}: {dS_arr[peak_idx]:+.4f}")
    print(f"Sign at peak: {'ENRICHMENT (GQA confirmed)' if dS_arr[peak_idx] > 0 else 'DEPLETION'}")
    print(f"Mean dS (all layers): {dS_arr.mean():+.4f}")
    print(f"Mean dS (tunnel L2-L20): {dS_arr[tunnel_slice].mean():+.4f}")
    print(f"Mean dS (relay L24+): {dS_arr[relay_slice].mean():+.4f}")

    # Passage distance
    S_r_arr = np.array(S_r_vals)
    S_a_arr = np.array(S_a_vals)
    d_r = abs(S_r_arr.max() - S_r_arr[0])
    d_a = abs(S_a_arr.max() - S_a_arr[0])
    relay_r = layer_nums[np.argmax(S_r_arr)]
    relay_a = layer_nums[np.argmax(S_a_arr)]

    print(f"\nPassage distance (receptive): {d_r:.3f} (L0 -> L{relay_r})")
    print(f"Passage distance (absent):    {d_a:.3f} (L0 -> L{relay_a})")
    print(f"Delta d: {d_r - d_a:+.4f}")

    # Comparison plot
    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            mistral_path = os.path.join(os.path.dirname(path), 'mistral_perlayer_profile.json')
            has_mistral = os.path.exists(mistral_path)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Top: S profiles
            ax1.plot(layer_nums, S_r_vals, 'o-', color='#2196F3', label='InternLM receptive', markersize=4)
            ax1.plot(layer_nums, S_a_vals, 's-', color='#FF5722', label='InternLM absent', markersize=4)

            if has_mistral:
                with open(mistral_path) as f:
                    mistral = json.load(f)
                m_layers = sorted(mistral['layers'].keys(), key=int)
                m_S_r = [mistral['layers'][l]['S_receptive'] for l in m_layers]
                m_S_a = [mistral['layers'][l]['S_absent'] for l in m_layers]
                m_nums = [int(l) for l in m_layers]
                ax1.plot(m_nums, m_S_r, 'o--', color='#2196F3', alpha=0.4, label='Mistral receptive', markersize=3)
                ax1.plot(m_nums, m_S_a, 's--', color='#FF5722', alpha=0.4, label='Mistral absent', markersize=3)

            ax1.set_ylabel('Spectral Entropy (S)')
            ax1.set_title('Per-Layer Spectral Entropy: InternLM 2.5 7B vs Mistral 7B')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.2)

            # Bottom: ΔS profiles
            ax2.plot(layer_nums, dS_vals, 'o-', color='#4CAF50', label='InternLM ΔS', markersize=4)
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

            if has_mistral:
                m_dS = [mistral['layers'][l]['delta_S'] for l in m_layers]
                ax2.plot(m_nums, m_dS, 'o--', color='#4CAF50', alpha=0.4, label='Mistral ΔS', markersize=3)

            ax2.set_xlabel('Layer')
            ax2.set_ylabel('ΔS (receptive − absent)')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.2)

            plt.tight_layout()
            plot_path = os.path.join(os.path.dirname(path), 'internlm_vs_mistral_perlayer.png')
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            print(f"\nComparison plot saved: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"\nPlot generation failed: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/internlm_results.json"
    analyze(path)
