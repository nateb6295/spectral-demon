#!/usr/bin/env python3
"""Quick analysis: What distribution does γ follow at tunnel layers?

If power-law: preferential attachment / rich-get-richer (Wang's self-amplification)
If log-normal: multiplicative random process
If bimodal: two distinct channel populations (high-γ highway / low-γ service road)

Also: does the shape change from early → tunnel → relay?
"""

import json
import os
import numpy as np
from pathlib import Path
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "16"

MODEL_PATH = "/mnt/hdd/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fit_distributions(data):
    data_pos = np.abs(data)
    data_pos = data_pos[data_pos > 0]

    results = {}

    mu_ln, sigma_ln = np.mean(np.log(data_pos)), np.std(np.log(data_pos))
    ks_ln, p_ln = stats.kstest(data_pos, 'lognorm', args=(sigma_ln, 0, np.exp(mu_ln)))
    results['lognormal'] = {'ks': float(ks_ln), 'p': float(p_ln),
                             'mu': float(mu_ln), 'sigma': float(sigma_ln)}

    mu_n, std_n = np.mean(data_pos), np.std(data_pos)
    ks_n, p_n = stats.kstest(data_pos, 'norm', args=(mu_n, std_n))
    results['normal'] = {'ks': float(ks_n), 'p': float(p_n)}

    log_data = np.log(data_pos)
    log_sorted = np.sort(log_data)[::-1]
    n = len(log_sorted)
    log_rank = np.log(np.arange(1, n + 1))
    slope, intercept, r, p, se = stats.linregress(log_rank[:n//2], log_sorted[:n//2])
    results['power_law_fit'] = {'alpha': float(-slope), 'r_squared': float(r**2),
                                 'p': float(p)}

    dip_stat = hartigan_dip(data_pos)
    results['bimodality'] = {'dip_statistic': float(dip_stat),
                              'ashman_D': float(ashman_d(data_pos))}

    q25, q50, q75 = np.percentile(data_pos, [25, 50, 75])
    results['shape'] = {
        'skewness': float(stats.skew(data_pos)),
        'kurtosis': float(stats.kurtosis(data_pos)),
        'q25': float(q25), 'median': float(q50), 'q75': float(q75),
        'iqr': float(q75 - q25),
        'range': float(np.max(data_pos) - np.min(data_pos)),
    }

    return results


def hartigan_dip(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    diffs = sorted_data[1:] - sorted_data[:-1]
    mid = n // 2
    gap1 = np.max(diffs[:mid]) if mid > 0 else 0
    gap2 = np.max(diffs[mid:]) if mid < len(diffs) else 0
    return float(max(gap1, gap2) / (sorted_data[-1] - sorted_data[0] + 1e-10))


def ashman_d(data):
    sorted_d = np.sort(data)
    n = len(sorted_d)
    mid = n // 2
    g1, g2 = sorted_d[:mid], sorted_d[mid:]
    mu1, mu2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1), np.std(g2)
    return float(np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1**2 + s2**2 + 1e-10))


def main():
    import torch
    from transformers import AutoModelForCausalLM

    print("Loading Qwen2.5-3B for γ spectrum analysis...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager",
    )

    results = {}
    sample_layers = [0, 2, 5, 8, 10, 13, 14, 16, 18, 20, 22, 25, 27, 30, 33, 35]

    for layer_idx in sample_layers:
        gamma = model.model.layers[layer_idx].input_layernorm.weight.detach().float().numpy()
        fits = fit_distributions(gamma)

        depth_frac = layer_idx / 35
        zone = "early" if depth_frac < 0.25 else "tunnel" if depth_frac < 0.6 else "transition" if depth_frac < 0.8 else "relay"

        best = min(['lognormal', 'normal'], key=lambda k: fits[k]['ks'])

        print(f"L{layer_idx:2d} ({zone:10s}): "
              f"best={best:10s} "
              f"skew={fits['shape']['skewness']:+.3f} "
              f"kurt={fits['shape']['kurtosis']:+.3f} "
              f"dip={fits['bimodality']['dip_statistic']:.4f} "
              f"ashman_D={fits['bimodality']['ashman_D']:.3f} "
              f"pl_α={fits['power_law_fit']['alpha']:.3f} "
              f"pl_R²={fits['power_law_fit']['r_squared']:.3f}")

        results[f"L{layer_idx}"] = {
            "layer": layer_idx, "zone": zone, "depth_frac": round(depth_frac, 3),
            **fits
        }

    # Cross-layer γ spectrum evolution
    print(f"\n{'='*60}")
    print("γ SPECTRUM EVOLUTION")
    print(f"{'='*60}")

    for zone in ["early", "tunnel", "transition", "relay"]:
        zone_results = {k: v for k, v in results.items() if v["zone"] == zone}
        if zone_results:
            skews = [v["shape"]["skewness"] for v in zone_results.values()]
            kurts = [v["shape"]["kurtosis"] for v in zone_results.values()]
            dips = [v["bimodality"]["dip_statistic"] for v in zone_results.values()]
            print(f"  {zone:10s}: skew={np.mean(skews):+.3f}±{np.std(skews):.3f}  "
                  f"kurt={np.mean(kurts):+.3f}±{np.std(kurts):.3f}  "
                  f"dip={np.mean(dips):.4f}±{np.std(dips):.4f}")

    out_path = RESULTS_DIR / "gamma_spectrum_shape.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model
    print("Done.")


if __name__ == "__main__":
    main()
