#!/usr/bin/env python3
"""Exp 9: Does γ bimodality appear only in GQA models?

If Pythia (MHA) has unimodal γ at all layers while Qwen/Gemma (GQA) have
bimodal γ at tunnel layers, that closes the loop on WHY MHA can't sustain
the wire: no two-population channel structure = no service road.

No forward passes needed — just weight extraction + distribution fitting.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "qwen2.5-3b": {
        "path": "/mnt/hdd/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "attention": "GQA",
        "n_layers": 36,
        "gamma_path": "model.layers.{}.input_layernorm.weight",
    },
    "gemma-2-2b": {
        "path": "/mnt/hdd/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        "attention": "GQA",
        "n_layers": 26,
        "gamma_path": "model.layers.{}.input_layernorm.weight",
    },
    "pythia-410m": {
        "path": "/mnt/hdd/huggingface/hub/models--EleutherAI--pythia-410m/snapshots/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        "attention": "MHA",
        "n_layers": 24,
        "gamma_path": "gpt_neox.layers.{}.input_layernorm.weight",
    },
}


def ashman_d(data):
    sorted_d = np.sort(data)
    n = len(sorted_d)
    mid = n // 2
    g1, g2 = sorted_d[:mid], sorted_d[mid:]
    mu1, mu2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1), np.std(g2)
    return float(np.sqrt(2) * abs(mu1 - mu2) / np.sqrt(s1**2 + s2**2 + 1e-10))


def analyze_model(name, config):
    import torch
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"Loading {name} ({config['attention']})...")
    print(f"{'='*60}")
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        config['path'], torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager", local_files_only=True,
    )

    n_layers = config['n_layers']
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)

    results = {}
    for l in sample_layers:
        gamma_path = config['gamma_path'].format(l)
        parts = gamma_path.split('.')
        obj = model
        for p in parts:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        gamma = obj.detach().float().numpy()

        depth_frac = l / (n_layers - 1)
        zone = "early" if depth_frac < 0.25 else "tunnel" if depth_frac < 0.6 else "transition" if depth_frac < 0.8 else "relay"

        gamma_pos = np.abs(gamma)
        gamma_pos = gamma_pos[gamma_pos > 0]

        cv = float(np.std(gamma_pos) / (np.mean(gamma_pos) + 1e-10))
        d = ashman_d(gamma_pos)
        skew = float(stats.skew(gamma_pos))
        kurt = float(stats.kurtosis(gamma_pos))

        print(f"  L{l:2d} ({zone:10s}): D={d:.3f}  CV={cv:.3f}  skew={skew:+.3f}  kurt={kurt:+.3f}")
        sys.stdout.flush()

        results[f"L{l}"] = {
            "layer": l, "zone": zone, "depth_frac": round(depth_frac, 3),
            "ashman_D": d, "cv": cv, "skewness": skew, "kurtosis": kurt,
            "dim": len(gamma),
        }

    del model
    return results


def main():
    all_results = {}

    for name, config in MODELS.items():
        all_results[name] = analyze_model(name, config)

    print(f"\n{'='*60}")
    print("CROSS-MODEL BIMODALITY COMPARISON")
    print(f"{'='*60}")

    for name in MODELS:
        attn = MODELS[name]['attention']
        zone_ds = {}
        for k, v in all_results[name].items():
            zone = v['zone']
            if zone not in zone_ds:
                zone_ds[zone] = []
            zone_ds[zone].append(v['ashman_D'])

        print(f"\n  {name} ({attn}):")
        for zone in ['early', 'tunnel', 'transition', 'relay']:
            if zone in zone_ds:
                ds = zone_ds[zone]
                bimodal_count = sum(1 for d in ds if d > 2.0)
                print(f"    {zone:10s}: D={np.mean(ds):.3f} ± {np.std(ds):.3f}  "
                      f"bimodal={bimodal_count}/{len(ds)}")

    tunnel_ds = {}
    for name in MODELS:
        attn = MODELS[name]['attention']
        ds = [v['ashman_D'] for v in all_results[name].values() if v['zone'] == 'tunnel']
        tunnel_ds[name] = np.mean(ds) if ds else 0
        print(f"\n  {name} ({attn}) tunnel mean D = {tunnel_ds[name]:.3f}")

    gqa_d = np.mean([v for k, v in tunnel_ds.items() if MODELS[k]['attention'] == 'GQA'])
    mha_d = np.mean([v for k, v in tunnel_ds.items() if MODELS[k]['attention'] == 'MHA'])
    print(f"\n  GQA mean tunnel D = {gqa_d:.3f}")
    print(f"  MHA mean tunnel D = {mha_d:.3f}")
    if mha_d > 0:
        print(f"  Ratio = {gqa_d/mha_d:.2f}×")

    out_path = RESULTS_DIR / "exp_gamma_bimodality_crossmodel.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
