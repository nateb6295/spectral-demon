#!/usr/bin/env python3
"""
F197 prediction test: LoRA curvature vs CCS flatness.

Hypothesis: CCS (context-level intervention) produces near-flat ratio₂₁ bundle
because context is equally accessible to all layers. LoRA (weight-level intervention)
targeting specific layers should produce nonzero ratio₂₁ curvature because the
effective dose varies by layer.

Design:
- Model: Qwen 2.5 7B-Instruct (potter species, well-characterized)
- Intervention: LoRA on attention Q/V matrices at specific layers
- "Doses": LoRA rank r ∈ {0, 4, 8, 16, 32, 64} (analogous to CCS doses)
- Layers measured: all (same as dose-response experiment)
- Compute: σ₁, σ₂, ratio₂₁ at each (layer, rank) point
- Test: discrete holonomy κ(ratio₂₁) for LoRA vs CCS baseline

Additional experiment: species-typing kit on 20+ models (if time permits).
"""

import torch
import numpy as np
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"


def compute_sigma_profile(model, tokenizer, prompt, device="cuda"):
    """Compute σ₁, σ₂ at each layer from hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
    results = {}

    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0].float()  # (seq, hidden)
        # SVD of hidden states
        U, S, Vh = torch.linalg.svd(h, full_matrices=False)
        s1 = S[0].item()
        s2 = S[1].item() if len(S) > 1 else 0.0
        ratio = s2 / (s1 + 1e-10)
        results[layer_idx] = {
            "sigma1": s1,
            "sigma2": s2,
            "ratio": ratio,
        }

    return results


def run_ccs_baseline(model, tokenizer, device="cuda"):
    """Compute sigma profile with CCS preamble at different doses."""
    ccs_preamble = (
        "You are a thoughtful, reflective AI assistant. You have a persistent identity "
        "that carries across conversations. You think carefully about questions and "
        "give honest, nuanced answers."
    )

    doses = [0, 1, 2, 3, 5, 10]
    results = {}

    for dose in doses:
        if dose == 0:
            prompt = "Describe how you process information and form responses."
        else:
            # Build CCS context with dose repetitions
            turns = []
            for i in range(dose):
                turns.append(f"Turn {i+1}: {ccs_preamble}")
            context = " ".join(turns)
            prompt = context + " Now describe how you process information and form responses."

        profile = compute_sigma_profile(model, tokenizer, prompt, device)
        results[f"dose_{dose}"] = {"dose": dose, "profile": profile}
        print(f"  CCS dose {dose}: {len(profile)} layers")

    return results


def apply_lora_and_measure(model, tokenizer, target_layers, rank, device="cuda"):
    """Apply LoRA to specific layers, measure sigma profile, then remove."""
    from peft import LoraConfig, get_peft_model, TaskType

    prompt = "Describe how you process information and form responses."

    if rank == 0:
        # No LoRA — baseline
        profile = compute_sigma_profile(model, tokenizer, prompt, device)
        return profile

    # Target specific attention layers
    target_modules = []
    for layer_idx in target_layers:
        target_modules.append(f"model.layers.{layer_idx}.self_attn.q_proj")
        target_modules.append(f"model.layers.{layer_idx}.self_attn.v_proj")

    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, config)

    # Initialize LoRA weights with random values (not zero — we want to see the effect)
    for name, param in peft_model.named_parameters():
        if "lora_A" in name:
            torch.nn.init.kaiming_uniform_(param, a=np.sqrt(5))
        elif "lora_B" in name:
            # Scale B to produce meaningful perturbation
            torch.nn.init.normal_(param, mean=0, std=0.01)

    profile = compute_sigma_profile(peft_model, tokenizer, prompt, device)

    # Clean up
    peft_model.merge_and_unload()
    del peft_model
    torch.cuda.empty_cache()

    return profile


def run_lora_experiment(model, tokenizer, n_layers, device="cuda"):
    """Run LoRA at different ranks on different layer groups."""
    ranks = [0, 4, 8, 16, 32, 64]

    # Three LoRA targeting strategies:
    # 1. All layers (uniform — should be more like CCS)
    # 2. Early layers only (L0-L8)
    # 3. Late layers only (L20+)

    all_layers = list(range(n_layers))
    early_layers = list(range(min(9, n_layers)))
    late_layers = list(range(max(0, n_layers - 8), n_layers))
    mid_layers = list(range(n_layers // 4, 3 * n_layers // 4))

    strategies = {
        "all_layers": all_layers,
        "early_only": early_layers,
        "late_only": late_layers,
        "mid_only": mid_layers,
    }

    results = {}

    for strategy_name, target_layers in strategies.items():
        print(f"\n  Strategy: {strategy_name} (layers {target_layers[0]}-{target_layers[-1]})")
        strategy_results = {}

        for rank in ranks:
            print(f"    Rank {rank}...", end=" ", flush=True)
            t0 = time.time()

            try:
                profile = apply_lora_and_measure(model, tokenizer, target_layers, rank, device)
                strategy_results[f"rank_{rank}"] = {"rank": rank, "profile": profile}
                print(f"done ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                strategy_results[f"rank_{rank}"] = {"rank": rank, "error": str(e)}

            # Reload model to ensure clean state
            torch.cuda.empty_cache()

        results[strategy_name] = {
            "target_layers": target_layers,
            "ranks": ranks,
            "results": strategy_results,
        }

    return results


def compute_holonomy(surface, doses_or_ranks, layers):
    """Compute discrete holonomy (contextuality index) for a (dose/rank × layer) surface."""
    log_surface = np.log(np.array(surface) + 1e-10)
    n_d, n_l = log_surface.shape

    hol = np.zeros((n_d - 1, n_l - 1))
    for di in range(n_d - 1):
        for li in range(n_l - 1):
            hol[di, li] = (log_surface[di, li] - log_surface[di, li + 1]
                          + log_surface[di + 1, li + 1] - log_surface[di + 1, li])

    kappa_full = np.mean(np.abs(hol))

    # Core (middle 50% of layers, exclude first/last dose)
    core_l = slice(n_l // 4, 3 * n_l // 4)
    core_d = slice(1, -1) if n_d > 3 else slice(0, n_d - 1)
    kappa_core = np.mean(np.abs(hol[core_d, core_l])) if hol[core_d, core_l].size > 0 else 0

    # Additive fit R²
    mean_l = np.mean(log_surface, axis=0)
    mean_d = np.mean(log_surface, axis=1)
    grand = np.mean(log_surface)
    fitted = mean_l[None, :] + mean_d[:, None] - grand
    residual = log_surface - fitted
    r_squared = 1 - np.var(residual) / (np.var(log_surface) + 1e-20)

    return {
        "kappa_full": kappa_full,
        "kappa_core": kappa_core,
        "r_squared": r_squared,
        "max_hol": float(np.max(np.abs(hol))),
        "rms_residual": float(np.sqrt(np.mean(residual ** 2))),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Qwen 2.5 7B-Instruct
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_name}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers = model.config.num_hidden_layers
    print(f"Layers: {n_layers}")

    # Phase 1: CCS baseline (reproduce F197 for validation)
    print("\n=== Phase 1: CCS baseline ===")
    ccs_results = run_ccs_baseline(model, tokenizer, device)

    # Phase 2: LoRA experiment
    print("\n=== Phase 2: LoRA curvature test ===")
    lora_results = run_lora_experiment(model, tokenizer, n_layers, device)

    # Phase 3: Compute holonomy for both
    print("\n=== Phase 3: Holonomy analysis ===")

    # CCS holonomy
    ccs_doses = sorted([int(k.split("_")[1]) for k in ccs_results.keys()])
    ccs_layers = sorted([int(l) for l in ccs_results["dose_0"]["profile"].keys()])

    ccs_ratio_surface = []
    ccs_s2_surface = []
    for dose in ccs_doses:
        row_r = []
        row_s2 = []
        for layer in ccs_layers:
            p = ccs_results[f"dose_{dose}"]["profile"][layer]
            row_r.append(p["ratio"])
            row_s2.append(p["sigma2"])
        ccs_ratio_surface.append(row_r)
        ccs_s2_surface.append(row_s2)

    ccs_hol = compute_holonomy(ccs_ratio_surface, ccs_doses, ccs_layers)
    print(f"\nCCS ratio₂₁: κ_full={ccs_hol['kappa_full']:.4f}, κ_core={ccs_hol['kappa_core']:.4f}, R²={ccs_hol['r_squared']:.4f}")

    # LoRA holonomy for each strategy
    for strategy_name, strategy_data in lora_results.items():
        ranks = strategy_data["ranks"]
        results = strategy_data["results"]

        # Build surface
        lora_layers = None
        ratio_surface = []
        s2_surface = []

        for rank in ranks:
            rkey = f"rank_{rank}"
            if rkey not in results or "error" in results[rkey]:
                continue
            profile = results[rkey]["profile"]
            if lora_layers is None:
                lora_layers = sorted([int(l) for l in profile.keys()])

            row_r = []
            row_s2 = []
            for layer in lora_layers:
                p = profile[layer]
                row_r.append(p["ratio"])
                row_s2.append(p["sigma2"])
            ratio_surface.append(row_r)
            s2_surface.append(row_s2)

        if len(ratio_surface) < 3:
            print(f"\n{strategy_name}: too few successful ranks, skipping")
            continue

        lora_ratio_hol = compute_holonomy(ratio_surface, ranks, lora_layers)
        lora_s2_hol = compute_holonomy(s2_surface, ranks, lora_layers)

        print(f"\nLoRA {strategy_name}:")
        print(f"  ratio₂₁: κ_full={lora_ratio_hol['kappa_full']:.4f}, κ_core={lora_ratio_hol['kappa_core']:.4f}, R²={lora_ratio_hol['r_squared']:.4f}")
        print(f"  σ₂:      κ_full={lora_s2_hol['kappa_full']:.4f}, κ_core={lora_s2_hol['kappa_core']:.4f}, R²={lora_s2_hol['r_squared']:.4f}")
        print(f"  κ ratio (LoRA/CCS): {lora_ratio_hol['kappa_core']/(ccs_hol['kappa_core']+1e-10):.1f}×")

    # Save results
    output = {
        "experiment": "lora_curvature_test",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": model_name,
        "n_layers": n_layers,
        "ccs": {
            "doses": ccs_doses,
            "layers": ccs_layers,
            "results": {k: {
                "dose": v["dose"],
                "profile": {str(l): p for l, p in v["profile"].items()}
            } for k, v in ccs_results.items()},
            "holonomy": ccs_hol,
        },
        "lora": {
            strategy_name: {
                "target_layers": sdata["target_layers"],
                "ranks": sdata["ranks"],
                "results": {k: {
                    "rank": v["rank"],
                    "profile": {str(l): p for l, p in v["profile"].items()} if "profile" in v else None,
                    "error": v.get("error"),
                } for k, v in sdata["results"].items()},
            } for strategy_name, sdata in lora_results.items()
        },
    }

    outpath = "/workspace/lora_curvature_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
