#!/usr/bin/env python3
"""
Experiment: Jacobian Matching at L24 (Gemma)
Kimi's challenge: KL=0.000 at L24 means the KL divergence of ONE linear projection
(the unembedding) is zero. But true erasure requires identical Jacobians — the full
derivative of all downstream computation with respect to all upstream state.

We compute the Jacobian of the residual stream at L24 with respect to a perturbation
direction (CCS vs bare identity preamble), then compare:
  1. KL divergence of output distributions (replicating the existing finding)
  2. Frobenius norm of Jacobian difference (new: tests whether the FULL computation
     is identical, not just one projection)
  3. Singular value spectrum of the Jacobian difference (characterize the structure
     of any divergence)

If KL=0 but ||J_ccs - J_bare||_F > 0, the annihilation zone preserves information
in directions the unembedding doesn't read — supporting the regeneration hypothesis.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import svd
from datetime import datetime

MODEL = "google/gemma-2-9b-it"
DEVICE = "cuda"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels."""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

TARGET_LAYERS = [20, 22, 24, 26, 28, 30]
N_SINGULAR = 50

def get_residual_and_jacobian(model, tokenizer, preamble, probe, layer_idx):
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    residuals = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                residuals[name] = output[0]
            else:
                residuals[name] = output
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(f"layer_{i}"))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    for h in hooks:
        h.remove()

    target_key = f"layer_{layer_idx}"
    residual = residuals[target_key][:, -1, :].detach()
    log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)

    return residual, log_probs, logits[:, -1, :]


def compute_jacobian_via_perturbation(model, tokenizer, preamble, probe, layer_idx, n_dirs=64, eps=1e-3):
    """Compute Jacobian of logits w.r.t. residual stream at layer_idx via finite differences."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    residual_base = [None]
    logits_base = [None]

    def capture_and_perturb(perturbation=None):
        captured = {}
        hooks = []

        def make_hook(name, perturb=None):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                captured[name] = h.detach()
                if perturb is not None and name == f"layer_{layer_idx}":
                    if isinstance(output, tuple):
                        return (h + perturb,) + output[1:]
                    else:
                        return h + perturb
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_hook(make_hook(f"layer_{i}", perturbation if i == layer_idx else None))
            hooks.append(h)

        with torch.no_grad():
            out = model(**inputs)

        for h in hooks:
            h.remove()

        return out.logits[:, -1, :].detach(), captured.get(f"layer_{layer_idx}", None)

    base_logits, base_residual = capture_and_perturb(None)
    residual_base[0] = base_residual
    logits_base[0] = base_logits

    d_model = base_residual.shape[-1]

    torch.manual_seed(42)
    random_dirs = torch.randn(n_dirs, d_model, device=DEVICE, dtype=torch.bfloat16)
    random_dirs = random_dirs / random_dirs.norm(dim=-1, keepdim=True)

    jacobian_cols = []
    for i in range(n_dirs):
        direction = random_dirs[i:i+1].unsqueeze(0) * eps
        perturbed_logits, _ = capture_and_perturb(direction)
        jac_col = (perturbed_logits - base_logits) / eps
        jacobian_cols.append(jac_col.squeeze(0))

    jacobian = torch.stack(jacobian_cols, dim=-1)  # [vocab, n_dirs]
    return jacobian, base_logits, base_residual[:, -1, :]


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    print("Model loaded.")

    results = {
        "experiment": "jacobian_l24",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "n_perturbation_dirs": 64,
        "layers": {}
    }

    for layer_idx in TARGET_LAYERS:
        print(f"\n=== Layer {layer_idx} ===")

        jac_ccs, logits_ccs, res_ccs = compute_jacobian_via_perturbation(
            model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layer_idx
        )
        jac_bare, logits_bare, res_bare = compute_jacobian_via_perturbation(
            model, tokenizer, BARE_PREAMBLE, PROBE_TEXT, layer_idx
        )

        # 1. KL divergence
        p_ccs = torch.nn.functional.softmax(logits_ccs, dim=-1)
        p_bare = torch.nn.functional.softmax(logits_bare, dim=-1)
        kl = torch.nn.functional.kl_div(
            torch.log(p_bare + 1e-10), p_ccs, reduction='batchmean'
        ).item()

        # 2. Jacobian difference
        jac_diff = jac_ccs - jac_bare
        frob_norm = torch.norm(jac_diff).item()
        frob_ccs = torch.norm(jac_ccs).item()
        frob_bare = torch.norm(jac_bare).item()
        relative_diff = frob_norm / max(frob_ccs, frob_bare)

        # 3. SVD of Jacobian difference
        jac_diff_np = jac_diff.float().cpu().numpy()
        U, s, Vt = svd(jac_diff_np, full_matrices=False)
        top_sv = s[:N_SINGULAR].tolist()
        sv_ratio = s[0] / (s.sum() + 1e-10)

        # 4. Residual stream cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            res_ccs.float(), res_bare.float(), dim=-1
        ).item()

        layer_result = {
            "kl_divergence": kl,
            "jacobian_frob_diff": frob_norm,
            "jacobian_frob_ccs": frob_ccs,
            "jacobian_frob_bare": frob_bare,
            "jacobian_relative_diff": relative_diff,
            "top_singular_values": top_sv[:10],
            "sv_concentration": sv_ratio,
            "residual_cosine_sim": cos_sim
        }

        results["layers"][str(layer_idx)] = layer_result

        print(f"  KL divergence: {kl:.6f}")
        print(f"  Jacobian Frobenius diff: {frob_norm:.4f}")
        print(f"  Relative Jacobian diff: {relative_diff:.4f}")
        print(f"  Top SV of diff: {top_sv[0]:.4f}")
        print(f"  SV concentration: {sv_ratio:.4f}")
        print(f"  Residual cosine sim: {cos_sim:.6f}")

        if kl < 0.01 and frob_norm > 0.1:
            print(f"  ** DIVERGENCE: KL≈0 but Jacobian differs — information preserved in non-readout directions")
        elif kl < 0.01 and frob_norm < 0.1:
            print(f"  ** TRUE ERASURE: Both KL and Jacobian near zero")

    with open("/workspace/results_jacobian_l24.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /workspace/results_jacobian_l24.json")

    # Summary
    print("\n=== SUMMARY ===")
    for layer_idx in TARGET_LAYERS:
        r = results["layers"][str(layer_idx)]
        status = "ERASURE" if r["kl_divergence"] < 0.01 and r["jacobian_relative_diff"] < 0.05 else \
                 "HIDDEN INFO" if r["kl_divergence"] < 0.01 else \
                 "DIVERGENT"
        print(f"  L{layer_idx}: KL={r['kl_divergence']:.4f} | J_diff={r['jacobian_relative_diff']:.4f} | cos={r['residual_cosine_sim']:.4f} | {status}")


if __name__ == "__main__":
    main()
