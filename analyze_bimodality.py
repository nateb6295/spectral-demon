#!/usr/bin/env python3
"""Quick analysis: Is Pythia's D≈2 meaningful or trivial?

Compare absolute separation between populations, not just Ashman D.
"""

import json
import os
import numpy as np
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path(__file__).parent / "results"

MODEL_PATHS = {
    "qwen2.5-3b": "/mnt/hdd/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "pythia-410m": "/mnt/hdd/huggingface/hub/models--EleutherAI--pythia-410m/snapshots/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
}

GAMMA_PATHS = {
    "qwen2.5-3b": "model.layers.{}.input_layernorm.weight",
    "pythia-410m": "gpt_neox.layers.{}.input_layernorm.weight",
}

TUNNEL_LAYERS = {
    "qwen2.5-3b": [16, 18, 20],
    "pythia-410m": [6, 8, 10],
}


def get_gamma(model, path_template, layer):
    parts = path_template.format(layer).split('.')
    obj = model
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj.detach().float().numpy()


def population_analysis(gamma):
    g = np.abs(gamma)
    g = np.sort(g)
    n = len(g)
    mid = n // 2
    lo, hi = g[:mid], g[mid:]

    return {
        "lo_mean": float(np.mean(lo)),
        "hi_mean": float(np.mean(hi)),
        "lo_std": float(np.std(lo)),
        "hi_std": float(np.std(hi)),
        "abs_separation": float(np.mean(hi) - np.mean(lo)),
        "rel_separation": float((np.mean(hi) - np.mean(lo)) / (np.mean(g) + 1e-10)),
        "range": float(np.max(g) - np.min(g)),
        "global_mean": float(np.mean(g)),
        "global_std": float(np.std(g)),
        "cv": float(np.std(g) / (np.mean(g) + 1e-10)),
        "max_min_ratio": float(np.max(g) / (np.min(g) + 1e-10)),
    }


def main():
    import torch
    from transformers import AutoModelForCausalLM

    for name in ["qwen2.5-3b", "pythia-410m"]:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[name], torch_dtype=torch.float32, device_map="cpu",
            attn_implementation="eager", local_files_only=True,
        )

        for l in TUNNEL_LAYERS[name]:
            gamma = get_gamma(model, GAMMA_PATHS[name], l)
            pa = population_analysis(gamma)

            print(f"\n  L{l}:")
            print(f"    Global: mean={pa['global_mean']:.4f}  std={pa['global_std']:.4f}  CV={pa['cv']:.3f}")
            print(f"    Range: {pa['range']:.4f}  max/min={pa['max_min_ratio']:.2f}×")
            print(f"    Lo pop: {pa['lo_mean']:.4f} ± {pa['lo_std']:.4f}")
            print(f"    Hi pop: {pa['hi_mean']:.4f} ± {pa['hi_std']:.4f}")
            print(f"    Absolute separation: {pa['abs_separation']:.4f}")
            print(f"    Relative separation: {pa['rel_separation']:.3f} ({pa['rel_separation']*100:.1f}%)")

        del model
        import gc; gc.collect()


if __name__ == "__main__":
    main()
