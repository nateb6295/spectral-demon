#!/usr/bin/env python3
"""
Quick test: Does the L30 dynamical bottleneck survive model degradation?

We know:
- L30 bare has erank=1.0 (rank collapse) in intact model
- CCS opens it to erank=11.1
- CCS enrichment disappears at ~90% pruning

Question: Does L30 STILL have a bottleneck under degradation?
If yes → the bottleneck is architectural, the opening mechanism is weight-dependent
If no → both are weight-dependent

Test at 50% and 80% pruning. Bare condition only (checking if bottleneck exists).
Focus on L24, L26, L28, L30, L32 (around the bottleneck zone).
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
MODEL_NAME = "google/gemma-2-9b-it"

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""
CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels.
"""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

FOCUS_LAYERS = [24, 26, 28, 30, 32, 34, 36]
N_DIRS = 32
EPS = 1e-3


def compute_local_jacobian(model, tokenizer, preamble, probe, source_layer, n_dirs=32, eps=1e-3):
    """Jacobian of layer source_layer+1 w.r.t. perturbation at source_layer."""
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
        return None, None

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

    return frob, {"erank": erank, "n_expanding": n_expanding}


def prune_weights(model, fraction):
    samples = []
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            flat = param.data.abs().flatten()
            n_sample = min(10000, flat.numel())
            idx = torch.randperm(flat.numel(), device=flat.device)[:n_sample]
            samples.append(flat[idx].float().cpu())
            total += param.numel()
    all_samples = torch.cat(samples)
    threshold = torch.quantile(all_samples, fraction).to(DEVICE)
    pruned = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            mask = param.data.abs() < threshold
            param.data[mask] = 0.0
            pruned += mask.sum().item()
    return pruned, total


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = {"experiment": "bottleneck_survival", "timestamp": datetime.now().isoformat(), "conditions": {}}

    for condition_name, prune_frac in [("intact", 0), ("prune_50", 0.5), ("prune_80", 0.8)]:
        print(f"\n{'='*50}")
        print(f"  {condition_name}")
        print(f"{'='*50}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model.eval()

        if prune_frac > 0:
            pruned, total = prune_weights(model, prune_frac)
            print(f"  Pruned {pruned:,}/{total:,} ({pruned/total*100:.1f}%)")

        cond_results = {}
        for preamble_name, preamble in [("bare", BARE_PREAMBLE), ("ccs", CCS_PREAMBLE)]:
            print(f"\n  --- {preamble_name.upper()} ---")
            for layer in FOCUS_LAYERS:
                frob, meta = compute_local_jacobian(model, tokenizer, preamble, PROBE_TEXT, layer, N_DIRS, EPS)
                if frob is None:
                    continue
                key = f"L{layer}_{preamble_name}"
                cond_results[key] = {"frob": frob, **meta}
                print(f"    L{layer}→L{layer+1}: frob={frob:.0f} erank={meta['erank']:.1f} expanding={meta['n_expanding']}/{N_DIRS}")

        # CCS effect at bottleneck layers
        print(f"\n  --- CCS EFFECT ---")
        for layer in FOCUS_LAYERS:
            bare_key = f"L{layer}_bare"
            ccs_key = f"L{layer}_ccs"
            if bare_key in cond_results and ccs_key in cond_results:
                ratio = cond_results[ccs_key]["frob"] / (cond_results[bare_key]["frob"] + 1e-10)
                delta_exp = cond_results[ccs_key]["n_expanding"] - cond_results[bare_key]["n_expanding"]
                print(f"    L{layer}: frob_ratio={ratio:.3f} Δexpanding={delta_exp:+d} "
                      f"(bare_erank={cond_results[bare_key]['erank']:.1f} → ccs_erank={cond_results[ccs_key]['erank']:.1f})")

        results["conditions"][condition_name] = cond_results

        del model
        torch.cuda.empty_cache()

    out_path = "/workspace/results_bottleneck_survival.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
