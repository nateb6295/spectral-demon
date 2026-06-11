#!/usr/bin/env python3
"""
Experiment: Spectral Bridge — Cross-Architecture (Qwen 2.5 7B)

Qwen uses GQA (28 query heads, 4 KV heads, 7:1 ratio).
Prediction: should replicate GEMMA pattern (absolute richness predicts),
NOT Mistral pattern (delta predicts). If true, confirms GQA/MHA as the
mechanism determinant for spectral-dynamic coupling.

Also tests whether higher GQA ratio (7:1 vs Gemma 2:1) strengthens
the group coherence effect.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from itertools import combinations

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels."""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

JACOBIAN_LAYERS = [4, 8, 12, 16, 20, 24, 27]
N_KV_HEADS = 4
N_QUERY_HEADS = 28
HEADS_PER_GROUP = N_QUERY_HEADS // N_KV_HEADS  # 7


def get_kv_group(head_idx):
    return head_idx // HEADS_PER_GROUP


def extract_attention_patterns(model, tokenizer, preamble, probe):
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    return [attn[0].detach().cpu().float() for attn in outputs.attentions], outputs.logits[:, -1, :].detach()


def analyze_attention_svd(attn_pattern):
    n_heads = attn_pattern.shape[0]
    results = []
    for h in range(n_heads):
        A = attn_pattern[h].numpy()
        try:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            sigma1, sigma2 = float(s[0]), float(s[1]) if len(s) > 1 else 0.0
            ratio = sigma2 / (sigma1 + 1e-10)
            erank = float(np.exp(-np.sum(s/s.sum() * np.log(s/s.sum() + 1e-10))))
            results.append({"sigma1": sigma1, "sigma2": sigma2, "ratio": ratio, "erank": erank})
        except:
            results.append({"sigma1": 0, "sigma2": 0, "ratio": 0, "erank": 0})
    return results


def compute_jacobian(model, tokenizer, preamble, probe, layer_idx, n_dirs=64, eps=1e-3):
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    def run_with_perturbation(perturbation=None):
        captured = {}
        hooks = []
        def make_hook(name, perturb=None):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[name] = h.detach()
                if perturb is not None and name == f"layer_{layer_idx}":
                    return (h + perturb,) + output[1:] if isinstance(output, tuple) else h + perturb
            return hook_fn
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}", perturbation if i == layer_idx else None)))
        with torch.no_grad():
            out = model(**inputs)
        for h in hooks:
            h.remove()
        return out.logits[:, -1, :].detach(), captured.get(f"layer_{layer_idx}")

    base_logits, base_residual = run_with_perturbation(None)
    d_model = base_residual.shape[-1]
    torch.manual_seed(42)
    dirs = torch.randn(n_dirs, d_model, device=DEVICE, dtype=torch.bfloat16)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    jac_cols = []
    for i in range(n_dirs):
        p_logits, _ = run_with_perturbation(dirs[i:i+1].unsqueeze(0) * eps)
        jac_cols.append(((p_logits - base_logits) / eps).squeeze(0))

    return torch.stack(jac_cols, dim=-1), base_logits, base_residual[:, -1, :]


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Model loaded. {n_layers} layers, {N_QUERY_HEADS} query heads, {N_KV_HEADS} KV heads ({HEADS_PER_GROUP}:1)")

    attn_ccs, logits_ccs = extract_attention_patterns(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT)
    attn_bare, logits_bare = extract_attention_patterns(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT)

    results = {
        "experiment": "spectral_bridge_qwen",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "n_layers": n_layers,
        "gqa_ratio": f"{HEADS_PER_GROUP}:1",
        "attention_svd": {},
        "jacobian": {},
        "gqa_groups": {}
    }

    # Part 1: Attention SVD + GQA group analysis
    print("\n=== ATTENTION SVD + GQA GROUPS ===")
    for layer_idx in range(n_layers):
        heads_ccs = analyze_attention_svd(attn_ccs[layer_idx])
        heads_bare = analyze_attention_svd(attn_bare[layer_idx])

        avg_ccs = {k: float(np.mean([h[k] for h in heads_ccs])) for k in ["sigma1", "sigma2", "ratio", "erank"]}
        avg_bare = {k: float(np.mean([h[k] for h in heads_bare])) for k in ["sigma1", "sigma2", "ratio", "erank"]}

        results["attention_svd"][str(layer_idx)] = {
            "ccs": avg_ccs, "bare": avg_bare,
            "delta_ratio": avg_ccs["ratio"] - avg_bare["ratio"],
            "delta_erank": avg_ccs["erank"] - avg_bare["erank"]
        }

        # GQA group coherence
        ccs_ratios = np.array([h["ratio"] for h in heads_ccs])
        within_diffs, between_diffs = [], []
        for i, j in combinations(range(N_QUERY_HEADS), 2):
            d = abs(ccs_ratios[i] - ccs_ratios[j])
            (within_diffs if get_kv_group(i) == get_kv_group(j) else between_diffs).append(d)

        within_mean = float(np.mean(within_diffs)) if within_diffs else 0.0
        between_mean = float(np.mean(between_diffs)) if between_diffs else 0.0

        # Per-group enrichment
        group_deltas = []
        for g in range(N_KV_HEADS):
            gh = [h for h in range(N_QUERY_HEADS) if get_kv_group(h) == g]
            gc = np.mean([heads_ccs[h]["ratio"] for h in gh])
            gb = np.mean([heads_bare[h]["ratio"] for h in gh])
            group_deltas.append(float(gc - gb))

        results["gqa_groups"][str(layer_idx)] = {
            "within_diff": within_mean,
            "between_diff": between_mean,
            "coherence": between_mean / (within_mean + 1e-10),
            "group_delta_std": float(np.std(group_deltas)),
            "group_delta_range": float(max(group_deltas) - min(group_deltas))
        }

        if layer_idx % 4 == 0 or layer_idx in JACOBIAN_LAYERS:
            coh = between_mean / (within_mean + 1e-10)
            print(f"  L{layer_idx:2d}: σ₂/σ₁={avg_ccs['ratio']:.4f}/{avg_bare['ratio']:.4f} "
                  f"Δ={avg_ccs['ratio']-avg_bare['ratio']:+.4f} "
                  f"GQA_coh={coh:.2f}")

    # Part 2: Jacobian
    print("\n=== JACOBIAN ===")
    for layer_idx in JACOBIAN_LAYERS:
        print(f"  Computing L{layer_idx}...")
        jac_ccs, _, res_ccs = compute_jacobian(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layer_idx)
        jac_bare, _, res_bare = compute_jacobian(model, tokenizer, BARE_PREAMBLE, PROBE_TEXT, layer_idx)

        jac_diff = jac_ccs - jac_bare
        frob = float(torch.norm(jac_diff).item())
        rel = frob / max(float(torch.norm(jac_ccs).item()), float(torch.norm(jac_bare).item()))
        cos = float(torch.nn.functional.cosine_similarity(res_ccs.float(), res_bare.float(), dim=-1).item())

        results["jacobian"][str(layer_idx)] = {"frob_diff": frob, "relative_diff": rel, "cosine_sim": cos}
        print(f"  L{layer_idx}: J_frob={frob:.0f} rel={rel:.4f} cos={cos:.4f}")

    # Part 3: Bridge correlations
    print("\n=== SPECTRAL-DYNAMIC BRIDGE (Qwen) ===")
    jac_frobs = [results["jacobian"][str(l)]["frob_diff"] for l in JACOBIAN_LAYERS]
    ccs_ratios_jl = [results["attention_svd"][str(l)]["ccs"]["ratio"] for l in JACOBIAN_LAYERS]
    delta_ratios_jl = [results["attention_svd"][str(l)]["delta_ratio"] for l in JACOBIAN_LAYERS]
    delta_eranks_jl = [results["attention_svd"][str(l)]["delta_erank"] for l in JACOBIAN_LAYERS]

    def pr(x, y):
        x, y = np.array(x), np.array(y)
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-10 and np.std(y) > 1e-10 else 0.0

    r1 = pr(ccs_ratios_jl, jac_frobs)
    r2 = pr(delta_ratios_jl, jac_frobs)
    r3 = pr(delta_eranks_jl, jac_frobs)

    print(f"  r(CCS σ₂/σ₁, J_frob) = {r1:+.4f}  (Gemma +0.88, Mistral -0.01)")
    print(f"  r(Δ(σ₂/σ₁), J_frob)  = {r2:+.4f}  (Gemma -0.04, Mistral +0.98)")
    print(f"  r(Δerank, J_frob)     = {r3:+.4f}  (Gemma +0.84, Mistral -0.39)")

    results["bridge_correlations"] = {"r_ccs_ratio": r1, "r_delta_ratio": r2, "r_delta_erank": r3}

    # Part 4: GQA zone analysis
    print("\n=== GQA GROUP COHERENCE BY ZONE ===")
    zones = {"early": list(range(0, 8)), "transition": list(range(8, 14)),
             "responsive": list(range(14, 24)), "relay": list(range(24, n_layers))}

    for zn, zl in zones.items():
        cohs = [results["gqa_groups"][str(l)]["coherence"] for l in zl]
        stds = [results["gqa_groups"][str(l)]["group_delta_std"] for l in zl]
        print(f"  {zn:12s}: coherence={np.mean(cohs):.3f} group_Δ_std={np.mean(stds):.4f}")

    # Prediction check
    print("\n=== PREDICTION CHECK ===")
    if abs(r1) > abs(r2):
        print("  ✓ CONFIRMED: Qwen (GQA) follows Gemma pattern — absolute richness predicts")
    else:
        print("  ✗ FALSIFIED: Qwen follows Mistral pattern — delta predicts")

    with open("/workspace/results_spectral_bridge_qwen.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
