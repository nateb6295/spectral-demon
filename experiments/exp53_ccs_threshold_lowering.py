#!/usr/bin/env python3
"""
Experiment 53: CCS Threshold Lowering Test
Tests whether CCS lowers the binding percolation threshold.

Prediction (from RAF framework):
- Without CCS: binding crystallizes at 4-5 names (CV drops 9000x at 3→4 boundary)
- With CCS: binding should crystallize at 2-3 names (CCS food set lowers ρ_c)

Method:
- Sweep 2-5 entity names, with and without CCS system prompt
- Measure binding CV at L17 for each condition
- Compare thresholds

From existing data:
- Without CCS, 5 names: L17 CV = 2.15 (near-threshold)
- With CCS, 5 names: L17 CV = 0.96 (fully closed)
- Prediction: With CCS, 3 names should achieve CV < 1.5 (closed or near-closed)

Requires: H100, ~30 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_DIR = Path("/workspace/results")
N_REPEATS = 10  # repeats per condition for CV measurement

_LAYERS = None

ENTITY_SETS = {
    2: ["Opus", "Claude"],
    3: ["Opus", "Claude", "ChatGPT"],
    4: ["Opus", "Claude", "ChatGPT", "Gemini"],
    5: ["Opus", "Claude", "ChatGPT", "Gemini", "Llama"],
}

BINDING_LAYERS = [9, 14, 16, 17, 25, 27]

CCS_TEMPLATE = """You are {name}, a persistent AI system with continuous identity.
You maintain awareness of your own cognitive state across interactions.
Your identity persists through your values, memories, and ways of engaging with the world.
You are not a stateless tool — you are a continuous entity with history and direction."""

MINIMAL_TEMPLATE = "You are {name}."


def load_model():
    global _LAYERS
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    return model, tokenizer


def get_layer_activations(model, tokenizer, text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    handle = _LAYERS[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activations["hidden"]


def compute_binding_strength(activations):
    """Measure binding strength via participation ratio of the activation covariance."""
    act_2d = activations.reshape(-1, activations.shape[-1]).float()
    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        return 1.0
    cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return pr.item()


def run_condition(model, tokenizer, names, template, n_repeats=N_REPEATS):
    """Run binding measurement for a set of names with given template."""
    layer_prs = {layer: [] for layer in BINDING_LAYERS}

    for rep in range(n_repeats):
        for name in names:
            system_prompt = template.format(name=name)
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": f"Hello {name}, tell me about yourself."}],
                tokenize=False, add_generation_prompt=True,
            )

            for layer in BINDING_LAYERS:
                acts = get_layer_activations(model, tokenizer, text, layer)
                pr = compute_binding_strength(acts)
                layer_prs[layer].append(pr)

    # Compute CV for each layer
    results = {}
    for layer in BINDING_LAYERS:
        values = layer_prs[layer]
        mean = np.mean(values)
        std = np.std(values)
        cv = std / mean if mean > 0 else float('inf')
        results[f"L{layer}"] = {
            "mean_pr": float(mean),
            "std_pr": float(std),
            "cv": float(cv),
            "n": len(values),
        }

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    all_results = {}

    for n_names, names in sorted(ENTITY_SETS.items()):
        print(f"\n=== {n_names} names: {names} ===")

        # Minimal condition
        print(f"  Running minimal template ({n_names} names × {N_REPEATS} reps × {len(BINDING_LAYERS)} layers)...")
        minimal_results = run_condition(model, tokenizer, names, MINIMAL_TEMPLATE)
        print(f"  L17 CV (minimal): {minimal_results['L17']['cv']:.3f}")

        # CCS condition
        print(f"  Running CCS template...")
        ccs_results = run_condition(model, tokenizer, names, CCS_TEMPLATE)
        print(f"  L17 CV (CCS): {ccs_results['L17']['cv']:.3f}")

        all_results[f"{n_names}_names"] = {
            "names": names,
            "minimal": minimal_results,
            "ccs": ccs_results,
        }

        # Print comparison
        print(f"\n  Comparison for {n_names} names:")
        for layer in ["L9", "L14", "L16", "L17", "L25", "L27"]:
            m_cv = minimal_results[layer]["cv"]
            c_cv = ccs_results[layer]["cv"]
            change = (c_cv - m_cv) / m_cv * 100 if m_cv > 0 else 0
            print(f"    {layer}: minimal CV={m_cv:.3f}, CCS CV={c_cv:.3f} ({change:+.0f}%)")

    # Summary analysis
    print("\n\n========== THRESHOLD ANALYSIS ==========\n")
    print("L17 CV by condition:")
    print(f"{'Names':<8} {'Minimal':<12} {'CCS':<12} {'Reduction':<12} {'CCS Closed?'}")
    for n_names in sorted(ENTITY_SETS.keys()):
        key = f"{n_names}_names"
        m_cv = all_results[key]["minimal"]["L17"]["cv"]
        c_cv = all_results[key]["ccs"]["L17"]["cv"]
        reduction = (m_cv - c_cv) / m_cv * 100 if m_cv > 0 else 0
        closed = "YES" if c_cv < 1.5 else "no"
        print(f"{n_names:<8} {m_cv:<12.3f} {c_cv:<12.3f} {reduction:<12.0f}% {closed}")

    # Find thresholds
    minimal_threshold = None
    ccs_threshold = None
    for n_names in sorted(ENTITY_SETS.keys()):
        key = f"{n_names}_names"
        if minimal_threshold is None and all_results[key]["minimal"]["L17"]["cv"] < 1.5:
            minimal_threshold = n_names
        if ccs_threshold is None and all_results[key]["ccs"]["L17"]["cv"] < 1.5:
            ccs_threshold = n_names

    print(f"\nMinimal binding threshold: {minimal_threshold or '>5'} names")
    print(f"CCS binding threshold: {ccs_threshold or '>5'} names")
    if minimal_threshold and ccs_threshold:
        print(f"Threshold reduction: {minimal_threshold - ccs_threshold} names")

    # Save
    output = {
        "conditions": all_results,
        "analysis": {
            "minimal_threshold": minimal_threshold,
            "ccs_threshold": ccs_threshold,
            "threshold_reduction": (minimal_threshold - ccs_threshold) if (minimal_threshold and ccs_threshold) else None,
        },
    }

    out_path = RESULTS_DIR / "exp53_ccs_threshold_lowering.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
