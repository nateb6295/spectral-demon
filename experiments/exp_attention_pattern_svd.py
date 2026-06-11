#!/usr/bin/env python3
"""
Experiment: Attention Pattern SVD vs Activation SVD

Reconcile the evening's weaker bridge correlations with the original r=0.88.
The original experiments may have computed SVD on attention PATTERNS
(softmax output, shape: n_heads × seq_len × seq_len per head) rather than
on activation outputs (seq_len × d_model).

This experiment computes BOTH measurements at every layer and correlates
each with Jacobian Frobenius norm. Whichever gives r≈0.88 is the correct
measurement for the paper.

Three architectures: Gemma 9B, Mistral 7B, Qwen 7B.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from scipy import stats
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
        "layers": list(range(4, 40, 4)),
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "layers": list(range(4, 30, 4)),
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "layers": list(range(4, 26, 4)),
    }
}

N_DIRS = 32
EPS = 1e-3


def compute_dual_spectral(model, tokenizer, preamble, probe, layers):
    """Compute BOTH attention-pattern SVD and activation SVD at each layer."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    activation_captured = {}
    attn_weights_captured = {}
    hooks = []

    def make_layer_hook(layer_idx):
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            activation_captured[layer_idx] = h.squeeze(0).detach().float()
        return hook_fn

    def make_attn_hook(layer_idx):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn_weights_captured[layer_idx] = output[1].detach().float()
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        if i in layers:
            hooks.append(layer.register_forward_hook(make_layer_hook(i)))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    results = {}
    for layer_idx in layers:
        entry = {}

        # Activation SVD (seq_len × d_model)
        if layer_idx in activation_captured:
            act = activation_captured[layer_idx]
            try:
                svs = torch.linalg.svdvals(act)
                entry["act_sigma1"] = svs[0].item()
                entry["act_sigma2"] = svs[1].item() if len(svs) > 1 else 0.0
                entry["act_ratio"] = entry["act_sigma2"] / (entry["act_sigma1"] + 1e-10)
            except Exception:
                entry["act_sigma1"] = 0.0
                entry["act_sigma2"] = 0.0
                entry["act_ratio"] = 0.0

        # Attention pattern SVD
        if layer_idx in attn_weights_captured:
            attn = attn_weights_captured[layer_idx]
            # Shape: (batch, n_heads, seq_q, seq_k)
            attn = attn.squeeze(0)  # (n_heads, seq_q, seq_k)
            n_heads, seq_q, seq_k = attn.shape

            # Method 1: SVD of (n_heads, seq_q × seq_k) — heads as samples
            flat = attn.reshape(n_heads, seq_q * seq_k)
            try:
                svs = torch.linalg.svdvals(flat)
                entry["attn_flat_sigma1"] = svs[0].item()
                entry["attn_flat_sigma2"] = svs[1].item() if len(svs) > 1 else 0.0
                entry["attn_flat_ratio"] = entry["attn_flat_sigma2"] / (entry["attn_flat_sigma1"] + 1e-10)
            except Exception:
                entry["attn_flat_sigma1"] = 0.0
                entry["attn_flat_ratio"] = 0.0

            # Method 2: Average per-head SVD (seq_q × seq_k per head)
            head_ratios = []
            for h in range(n_heads):
                try:
                    svs_h = torch.linalg.svdvals(attn[h])
                    hr = (svs_h[1] / (svs_h[0] + 1e-10)).item() if len(svs_h) > 1 else 0.0
                    head_ratios.append(hr)
                except Exception:
                    head_ratios.append(0.0)
            entry["attn_head_mean_ratio"] = float(np.mean(head_ratios))
            entry["attn_head_std_ratio"] = float(np.std(head_ratios))

        results[layer_idx] = entry

    return results


def compute_jacobian_frob(model, tokenizer, preamble, probe, layer_idx, n_dirs=32, eps=1e-3):
    """Finite-difference Jacobian Frobenius norm."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    def run(perturbation=None):
        captured = {}
        hooks = []

        def make_hook(li, perturb=None):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if li == layer_idx:
                    captured["base"] = h.detach()
                if perturb is not None and li == layer_idx:
                    return (h + perturb,) + output[1:] if isinstance(output, tuple) else h + perturb
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(make_hook(i, perturbation if i == layer_idx else None)))

        with torch.no_grad():
            out = model(**inputs)

        for h in hooks:
            h.remove()
        return out.logits[:, -1, :].detach(), captured.get("base")

    base_logits, base_h = run()
    if base_h is None:
        return None

    d_model = base_h.shape[-1]
    torch.manual_seed(42)
    dirs = torch.randn(n_dirs, d_model, device=DEVICE, dtype=torch.bfloat16)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    jac_cols = []
    for i in range(n_dirs):
        p_logits, _ = run(dirs[i:i+1].unsqueeze(0) * eps)
        jac_cols.append(((p_logits - base_logits) / eps).squeeze(0))

    jac = torch.stack(jac_cols, dim=-1)
    return float(torch.norm(jac).item())


def run_architecture(arch_key, arch_info):
    print(f"\n{'='*60}")
    print(f"  {arch_key.upper()} — {arch_info['name']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(arch_info["name"])
    model = AutoModelForCausalLM.from_pretrained(
        arch_info["name"], torch_dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="eager"
    )
    model.eval()

    layers = arch_info["layers"]
    results = {"model": arch_info["name"], "layers": {}}

    # Phase 1: Dual spectral measurements
    print(f"\n  Phase 1: Dual spectral measurements (CCS + bare)")
    ccs_spectral = compute_dual_spectral(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layers)
    n_attn = sum(1 for l in layers if 'attn_flat_ratio' in ccs_spectral.get(l, {}))
    print(f"    Attention weights captured at {n_attn}/{len(layers)} layers")
    if n_attn == 0:
        print(f"    WARNING: No attention weights captured. Check attn_implementation.")
    bare_spectral = compute_dual_spectral(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT, layers)

    for l in layers:
        c = ccs_spectral.get(l, {})
        b = bare_spectral.get(l, {})
        print(f"    L{l}: act_ratio CCS={c.get('act_ratio',0):.4f} bare={b.get('act_ratio',0):.4f} | "
              f"attn_flat CCS={c.get('attn_flat_ratio',0):.4f} bare={b.get('attn_flat_ratio',0):.4f} | "
              f"attn_head CCS={c.get('attn_head_mean_ratio',0):.4f} bare={b.get('attn_head_mean_ratio',0):.4f}")

    # Phase 2: Jacobian
    print(f"\n  Phase 2: Jacobian Frobenius norm")
    j_frobs = {}
    for l in layers:
        j = compute_jacobian_frob(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, l)
        j_frobs[l] = j
        print(f"    L{l}: J_frob = {j:.0f}")

    # Merge
    for l in layers:
        l_str = str(l)
        results["layers"][l_str] = {
            "ccs": ccs_spectral.get(l, {}),
            "bare": bare_spectral.get(l, {}),
            "j_frob": j_frobs.get(l),
        }

    # Correlations: which spectral measurement best predicts Jacobian?
    print(f"\n  Phase 3: Bridge correlations")
    valid = [l for l in layers if l in ccs_spectral and l in j_frobs and j_frobs[l] is not None]

    if len(valid) > 3:
        frobs = [j_frobs[l] for l in valid]

        for metric_name, metric_key in [
            ("act_ratio", "act_ratio"),
            ("attn_flat_ratio", "attn_flat_ratio"),
            ("attn_head_mean_ratio", "attn_head_mean_ratio"),
        ]:
            vals = [ccs_spectral[l].get(metric_key, 0) for l in valid]
            if any(v != 0 for v in vals):
                r, p = stats.pearsonr(vals, frobs)
                print(f"    r({metric_name}, J_frob) = {r:.3f} (p={p:.4f})")
                results[f"bridge_{metric_name}"] = {"r": float(r), "p": float(p)}

            # Also enrichment version
            enrichments = [ccs_spectral[l].get(metric_key, 0) - bare_spectral[l].get(metric_key, 0)
                           for l in valid]
            if any(e != 0 for e in enrichments):
                r_e, p_e = stats.pearsonr(enrichments, frobs)
                print(f"    r(Δ{metric_name}, J_frob) = {r_e:.3f} (p={p_e:.4f})")
                results[f"bridge_delta_{metric_name}"] = {"r": float(r_e), "p": float(p_e)}

    del model
    torch.cuda.empty_cache()
    return results


def main():
    results = {
        "experiment": "attention_pattern_svd",
        "timestamp": datetime.now().isoformat(),
        "architectures": {}
    }

    for arch_key, arch_info in MODELS.items():
        arch_results = run_architecture(arch_key, arch_info)
        results["architectures"][arch_key] = arch_results

    out_path = "/workspace/results_attention_pattern_svd.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\n{'='*60}")
    print(f"  DUAL SPECTRAL BRIDGE — SUMMARY")
    print(f"{'='*60}")
    for arch_key, arch_data in results["architectures"].items():
        print(f"\n  {arch_key.upper()}:")
        for k, v in arch_data.items():
            if k.startswith("bridge_") and isinstance(v, dict):
                print(f"    {k}: r={v['r']:.3f} (p={v['p']:.4f})")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
