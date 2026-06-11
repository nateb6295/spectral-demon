#!/usr/bin/env python3
"""
Experiment: Attention Head Knockout × Bridge Correlation

Causal test of the spectral-dynamic bridge. If σ₂/σ₁ predicts J_frob (r=0.88),
then knocking out the heads responsible for σ₂ enrichment should break the bridge.

For each architecture:
1. Identify the top-5 heads by CCS σ₂/σ₁ enrichment (from existing head-level data)
2. Zero-ablate those heads and re-measure J_frob at every layer
3. Compare bridge correlation (ablated vs intact)

If bridge breaks under head knockout → the specific attention heads carrying
spectral enrichment ARE the mechanism connecting geometry to dynamics.
If bridge survives → the correlation is driven by something else (MLPs, residual).
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
        "n_heads": 16,
        "n_kv_heads": 8,
        "sample_layers": list(range(4, 40, 4)),
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "sample_layers": list(range(4, 30, 4)),
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "n_heads": 28,
        "n_kv_heads": 4,
        "sample_layers": list(range(4, 26, 4)),
    }
}

N_DIRS = 32
EPS = 1e-3


def compute_per_head_enrichment(model, tokenizer, layer_idx, n_heads, head_dim):
    """Measure σ₂/σ₁ enrichment per head at a given layer."""
    enrichments = []

    for condition_name, preamble in [("bare", BARE_PREAMBLE), ("ccs", CCS_PREAMBLE)]:
        text = preamble + PROBE_TEXT
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        captured = {}
        def make_hook(name):
            def hook_fn(module, inp, output):
                captured[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
            return hook_fn

        hook = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
            make_hook(condition_name)
        )
        with torch.no_grad():
            model(**inputs)
        hook.remove()

    bare_out = captured.get("bare")
    ccs_out = captured.get("ccs")
    if bare_out is None or ccs_out is None:
        return []

    head_enrichments = []
    for h in range(n_heads):
        start = h * head_dim
        end = start + head_dim

        bare_slice = bare_out[:, -1, start:end].float()
        ccs_slice = ccs_out[:, -1, start:end].float()

        try:
            bare_svs = torch.linalg.svdvals(bare_slice)
            ccs_svs = torch.linalg.svdvals(ccs_slice)
            bare_ratio = (bare_svs[1] / (bare_svs[0] + 1e-10)).item()
            ccs_ratio = (ccs_svs[1] / (ccs_svs[0] + 1e-10)).item()
            head_enrichments.append({
                "head": h,
                "bare_ratio": bare_ratio,
                "ccs_ratio": ccs_ratio,
                "enrichment": ccs_ratio - bare_ratio
            })
        except Exception:
            head_enrichments.append({"head": h, "enrichment": 0.0})

    return head_enrichments


def compute_jacobian_frob(model, tokenizer, preamble, probe, layer_idx,
                          ablate_heads=None, n_dirs=32, eps=1e-3):
    """Compute Jacobian Frobenius norm, optionally ablating specific heads."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    def run(perturbation=None):
        captured = {}
        hooks = []

        def make_hook(name, perturb=None):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[name] = h.detach()
                if perturb is not None and name == f"layer_{layer_idx}":
                    return (h + perturb,) + output[1:] if isinstance(output, tuple) else h + perturb
            return hook_fn

        def make_ablation_hook(layer_i, heads_to_ablate, head_dim):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                for head_idx in heads_to_ablate.get(layer_i, []):
                    start = head_idx * head_dim
                    end = start + head_dim
                    h[:, :, start:end] = 0.0
                return (h,) + output[1:] if isinstance(output, tuple) else h
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(
                make_hook(f"layer_{i}", perturbation if i == layer_idx else None)
            ))
            if ablate_heads and i in ablate_heads:
                hooks.append(layer.self_attn.o_proj.register_forward_hook(
                    make_ablation_hook(i, ablate_heads, model.config.hidden_size // model.config.num_attention_heads)
                ))

        with torch.no_grad():
            out = model(**inputs)

        for h in hooks:
            h.remove()
        return out.logits[:, -1, :].detach(), captured.get(f"layer_{layer_idx}")

    base_logits, base_residual = run(None)
    if base_residual is None:
        return None

    d_model = base_residual.shape[-1]

    torch.manual_seed(42)
    dirs = torch.randn(n_dirs, d_model, device=DEVICE, dtype=torch.bfloat16)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    jac_cols = []
    for i in range(n_dirs):
        p_logits, _ = run(dirs[i:i+1].unsqueeze(0) * eps)
        jac_cols.append(((p_logits - base_logits) / eps).squeeze(0))

    jac = torch.stack(jac_cols, dim=-1)
    return float(torch.norm(jac).item())


def main():
    results = {
        "experiment": "head_knockout_bridge",
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

        head_dim = model.config.hidden_size // arch_info["n_heads"]
        arch_results = {"model": arch_info["name"], "conditions": {}}

        # Phase 1: Find top enrichment heads per layer
        print(f"\n  Phase 1: Per-head enrichment scan")
        top_heads_per_layer = {}
        for layer in arch_info["sample_layers"]:
            enrichments = compute_per_head_enrichment(
                model, tokenizer, layer, arch_info["n_heads"], head_dim
            )
            sorted_heads = sorted(enrichments, key=lambda x: abs(x.get("enrichment", 0)), reverse=True)
            top5 = [h["head"] for h in sorted_heads[:5]]
            top_heads_per_layer[layer] = top5
            print(f"    L{layer}: top-5 enrichment heads = {top5}")

        # Phase 2: Intact bridge measurement
        print(f"\n  Phase 2: Intact Jacobian")
        intact_frobs = {}
        for layer in arch_info["sample_layers"]:
            frob = compute_jacobian_frob(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layer)
            intact_frobs[layer] = frob
            print(f"    L{layer}: J_frob = {frob:.0f}")

        # Phase 3: Ablated bridge measurement
        print(f"\n  Phase 3: Top-5 head knockout Jacobian")
        ablated_frobs = {}
        for layer in arch_info["sample_layers"]:
            ablate_map = {layer: top_heads_per_layer[layer]}
            frob = compute_jacobian_frob(
                model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layer,
                ablate_heads=ablate_map
            )
            ablated_frobs[layer] = frob
            ratio = frob / (intact_frobs[layer] + 1e-10)
            print(f"    L{layer}: J_frob = {frob:.0f} (ratio to intact: {ratio:.3f})")

        arch_results["conditions"]["intact"] = {str(l): intact_frobs[l] for l in arch_info["sample_layers"]}
        arch_results["conditions"]["ablated"] = {str(l): ablated_frobs[l] for l in arch_info["sample_layers"]}
        arch_results["top_heads"] = {str(l): top_heads_per_layer[l] for l in arch_info["sample_layers"]}

        # Summary
        intact_vals = [intact_frobs[l] for l in arch_info["sample_layers"]]
        ablated_vals = [ablated_frobs[l] for l in arch_info["sample_layers"]]
        if len(intact_vals) > 2 and len(ablated_vals) > 2:
            intact_corr = np.corrcoef(list(range(len(intact_vals))), intact_vals)[0, 1]
            ablated_corr = np.corrcoef(list(range(len(ablated_vals))), ablated_vals)[0, 1]
            print(f"\n  Layer-order correlation: intact={intact_corr:.3f}, ablated={ablated_corr:.3f}")

        results["architectures"][arch_key] = arch_results

        del model
        torch.cuda.empty_cache()

    out_path = "/workspace/results_head_knockout_bridge.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
