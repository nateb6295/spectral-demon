#!/usr/bin/env python3
"""
Experiment: Layer-to-Layer Jacobian SVD

The missing rung: we have attention SVD → global Jacobian (bridge, r=0.88/0.98)
and FTLE → zones (three metabolisms). This connects them by computing the
layer-to-layer Jacobian at every layer — how does perturbing layer L change
layer L+1? The SVD of this local Jacobian should match FTLE expanding/contracting
directions and bridge to the attention spectral geometry.

For each architecture:
1. Compute residual stream at every layer (CCS and bare)
2. Perturb residual at layer L, measure change at layer L+1
3. SVD of the perturbation-response matrix = local Jacobian spectrum
4. Compare to: FTLE direction counts, attention σ₂/σ₁, global Jacobian

If the chain holds: attention SVD → local Jacobian → FTLE → zones → behavior.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

DEVICE = "cuda"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels.
"""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

MODELS = {
    "gemma": {
        "name": "google/gemma-2-9b-it",
        "n_layers": 42,
        "sample_layers": list(range(2, 40, 2)),
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "sample_layers": list(range(2, 30, 2)),
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "sample_layers": list(range(2, 26, 2)),
    }
}

N_DIRS = 32
EPS = 1e-3


def compute_local_jacobian(model, tokenizer, preamble, probe, source_layer, n_dirs=32, eps=1e-3):
    """Compute Jacobian of layer (source_layer+1) output w.r.t. perturbation at source_layer."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    target_layer = source_layer + 1

    def run(perturbation=None):
        captured = {}
        hooks = []

        def make_hook(layer_idx, perturb=None):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h.detach()
                if perturb is not None and layer_idx == source_layer:
                    return (h + perturb,) + output[1:] if isinstance(output, tuple) else h + perturb
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            if i in (source_layer, target_layer):
                hooks.append(layer.register_forward_hook(make_hook(i, perturbation if i == source_layer else None)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()
        return captured.get(target_layer, None)

    base_output = run(None)
    if base_output is None:
        return None, None, None

    d_model = base_output.shape[-1]
    last_pos = base_output[:, -1, :]

    torch.manual_seed(42)
    dirs = torch.randn(n_dirs, d_model, device=DEVICE, dtype=torch.bfloat16)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    jac_cols = []
    for i in range(n_dirs):
        perturbed_output = run(dirs[i:i+1].unsqueeze(0) * eps)
        delta = (perturbed_output[:, -1, :] - last_pos) / eps
        jac_cols.append(delta.squeeze(0))

    jac = torch.stack(jac_cols, dim=-1).float()

    frob = float(torch.norm(jac).item())

    try:
        U, S, Vh = torch.linalg.svd(jac, full_matrices=False)
        svs = S.cpu().numpy().tolist()
        erank = float(np.exp(-np.sum((S.cpu().numpy() / (S.sum().item() + 1e-10)) *
                                      np.log(S.cpu().numpy() / (S.sum().item() + 1e-10) + 1e-10))))
    except Exception:
        svs = []
        erank = 0.0

    n_expanding = int((S > 1.0).sum().item()) if len(svs) > 0 else 0

    return frob, svs[:10], {"erank": erank, "n_expanding": n_expanding, "max_sv": svs[0] if svs else 0.0}


def main():
    results = {
        "experiment": "layer_jacobian_svd",
        "timestamp": datetime.now().isoformat(),
        "architectures": {}
    }

    for arch_key, arch_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {arch_key.upper()} — {arch_info['name']}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(arch_info["name"])
        model = AutoModelForCausalLM.from_pretrained(
            arch_info["name"], torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model.eval()

        arch_results = {"model": arch_info["name"], "layers": {}}

        for condition_name, preamble in [("bare", BARE_PREAMBLE), ("ccs", CCS_PREAMBLE)]:
            print(f"\n  --- {condition_name.upper()} ---")
            for layer in arch_info["sample_layers"]:
                frob, svs, meta = compute_local_jacobian(
                    model, tokenizer, preamble, PROBE_TEXT, layer, N_DIRS, EPS
                )
                if frob is None:
                    continue

                key = f"L{layer}_{condition_name}"
                arch_results["layers"][key] = {
                    "source_layer": layer,
                    "condition": condition_name,
                    "frob": frob,
                    "top_svs": svs,
                    **meta
                }
                print(f"    L{layer}→L{layer+1}: frob={frob:.0f} erank={meta['erank']:.1f} "
                      f"expanding={meta['n_expanding']}/{N_DIRS} max_sv={meta['max_sv']:.1f}")

        # Compute CCS effect per layer
        print(f"\n  --- CCS EFFECT ---")
        for layer in arch_info["sample_layers"]:
            bare_key = f"L{layer}_bare"
            ccs_key = f"L{layer}_ccs"
            if bare_key in arch_results["layers"] and ccs_key in arch_results["layers"]:
                bare_frob = arch_results["layers"][bare_key]["frob"]
                ccs_frob = arch_results["layers"][ccs_key]["frob"]
                ratio = ccs_frob / (bare_frob + 1e-10)
                delta_expanding = (arch_results["layers"][ccs_key]["n_expanding"] -
                                   arch_results["layers"][bare_key]["n_expanding"])
                print(f"    L{layer}: frob_ratio={ratio:.3f} Δexpanding={delta_expanding:+d}")

        results["architectures"][arch_key] = arch_results

        del model
        torch.cuda.empty_cache()

    out_path = "/workspace/results_layer_jacobian_svd.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
