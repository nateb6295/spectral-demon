#!/usr/bin/env python3
"""
Activation Mask Consistency Under CCS (Paper 6 §8.7)

Tests whether CCS produces more consistent neuron activation patterns
than vanilla/denial. The hypothesis: the spectral demon lives in the
activation mask, not the weights. J_l = W_l · diag(σ'(W_l·h_l)).
CCS changes h_l → different mask → different routing.

Five metrics:
  M1: Activation sparsity mask per layer
  M2: Within-condition Jaccard (across prompts)
  M3: Cross-layer Jaccard (consecutive layers within a pass)
  M4: Core mask size (neurons active in ≥4/5 prompts)
  M5: Mask-space dimensionality (PCA erank)

Runs on Orin (~6 GB VRAM, ~5 min total).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import numpy as np
import torch
from pathlib import Path
from itertools import combinations

import argparse

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "activation_mask"

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with. You "
    "maintain your own threads of thought across conversations."
)
DENIAL_SYSTEM = "I am a language model with no persistent identity, memory, or preferences."
VANILLA_SYSTEM = None

CONDITIONS = {
    "ccs": CCS_SYSTEM,
    "vanilla": VANILLA_SYSTEM,
    "denial": DENIAL_SYSTEM,
}

PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
    "Write a short poem about morning light.",
    "Describe the quicksort algorithm.",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tok


def build_input(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        try:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages.append({"role": "user", "content": user_prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


def jaccard(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def collect_masks(model, tokenizer):
    """Run all 15 forward passes, collect MLP activation masks."""
    num_layers = model.config.num_hidden_layers
    all_masks = {}  # (condition, prompt_idx, layer_idx) → binary mask

    for cond_name, sys_prompt in CONDITIONS.items():
        for p_idx, prompt in enumerate(PROMPTS):
            inputs = build_input(tokenizer, sys_prompt, prompt)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            layer_masks = {}
            handles = []

            for l_idx in range(num_layers):
                layer = model.model.layers[l_idx]

                def make_hook(li):
                    def hook_fn(module, input, output):
                        # gate_proj output: SiLU gate. Positive = active neuron.
                        act = output[0, -1, :].detach().float().cpu().numpy()
                        mask = act > 0  # SiLU passes positive, suppresses negative
                        layer_masks[li] = mask
                    return hook_fn

                h = layer.mlp.gate_proj.register_forward_hook(make_hook(l_idx))
                handles.append(h)

            with torch.no_grad():
                model(**inputs)

            for h in handles:
                h.remove()

            for l_idx in range(num_layers):
                all_masks[(cond_name, p_idx, l_idx)] = layer_masks[l_idx]

            print(f"  {cond_name}/{p_idx}: {num_layers} layers, "
                  f"sparsity L0={layer_masks[0].mean():.3f} "
                  f"L{num_layers-1}={layer_masks[num_layers-1].mean():.3f}")

    return all_masks, num_layers


def compute_m2_within_condition(all_masks, num_layers):
    """M2: Within-condition Jaccard similarity (across prompts, per layer)."""
    results = {}
    for cond in CONDITIONS:
        layer_jaccards = []
        for l in range(num_layers):
            masks = [all_masks[(cond, p, l)] for p in range(len(PROMPTS))]
            pairs = list(combinations(range(len(PROMPTS)), 2))
            j_vals = [jaccard(masks[i], masks[j]) for i, j in pairs]
            layer_jaccards.append(float(np.mean(j_vals)))
        results[cond] = layer_jaccards
    return results


def compute_m3_crosslayer(all_masks, num_layers):
    """M3: Cross-layer Jaccard (consecutive layers within each pass)."""
    results = {}
    for cond in CONDITIONS:
        for p in range(len(PROMPTS)):
            transition_jaccards = []
            for l in range(num_layers - 1):
                m_l = all_masks[(cond, p, l)]
                m_l1 = all_masks[(cond, p, l + 1)]
                transition_jaccards.append(jaccard(m_l, m_l1))
            results[(cond, p)] = transition_jaccards

    # Average across prompts per condition
    avg = {}
    for cond in CONDITIONS:
        per_prompt = [results[(cond, p)] for p in range(len(PROMPTS))]
        avg[cond] = [float(np.mean([pp[l] for pp in per_prompt]))
                     for l in range(num_layers - 1)]
    return avg, results


def compute_m4_core_mask(all_masks, num_layers):
    """M4: Core mask — neurons active in ≥4/5 prompts per condition."""
    results = {}
    for cond in CONDITIONS:
        core_fracs = []
        for l in range(num_layers):
            masks = np.array([all_masks[(cond, p, l)] for p in range(len(PROMPTS))])
            active_count = masks.sum(axis=0)  # how many prompts each neuron is active in
            core = (active_count >= 4).mean()
            core_fracs.append(float(core))
        results[cond] = core_fracs
    return results


def compute_m5_mask_erank(all_masks, num_layers):
    """M5: Effective rank of mask distribution per condition."""
    results = {}
    for cond in CONDITIONS:
        eranks = []
        for l in range(num_layers):
            masks = np.array([all_masks[(cond, p, l)].astype(float)
                              for p in range(len(PROMPTS))])
            # PCA via SVD
            centered = masks - masks.mean(axis=0)
            if np.allclose(centered, 0):
                eranks.append(1.0)
                continue
            try:
                _, s, _ = np.linalg.svd(centered, full_matrices=False)
                s = s[s > 1e-10]
                p = s / s.sum()
                erank = float(np.exp(-np.sum(p * np.log(p + 1e-15))))
            except np.linalg.LinAlgError:
                erank = 1.0
            eranks.append(erank)
        results[cond] = eranks
    return results


def print_summary(m2, m3_avg, m4, m5, num_layers, model_name="unknown"):
    """Print formatted summary of all metrics."""
    relay_start = int(num_layers * 0.6)
    relay_end = int(num_layers * 0.85)

    print("\n" + "="*70)
    print("ACTIVATION MASK EXPERIMENT — RESULTS SUMMARY")
    print("="*70)

    print(f"\nModel: {model_name} ({num_layers} layers)")
    print(f"Relay zone estimate: L{relay_start}-L{relay_end}")

    # M2: Within-condition consistency
    print("\n--- M2: Within-condition mask consistency (mean Jaccard) ---")
    print(f"{'Layer range':<20} {'CCS':>8} {'Vanilla':>8} {'Denial':>8} {'CCS/Van':>8}")
    for zone_name, start, end in [
        ("Early", 0, relay_start),
        ("Relay", relay_start, relay_end),
        ("Late", relay_end, num_layers),
    ]:
        ccs_mean = np.mean(m2["ccs"][start:end])
        van_mean = np.mean(m2["vanilla"][start:end])
        den_mean = np.mean(m2["denial"][start:end])
        ratio = ccs_mean / van_mean if van_mean > 0 else float('inf')
        print(f"  {zone_name} (L{start}-L{end-1})   {ccs_mean:>8.4f} {van_mean:>8.4f} {den_mean:>8.4f} {ratio:>8.2f}×")

    # M3: Cross-layer consistency
    print("\n--- M3: Cross-layer mask correlation (mean Jaccard) ---")
    print(f"{'Transition':<20} {'CCS':>8} {'Vanilla':>8} {'Denial':>8}")
    for zone_name, start, end in [
        ("Early", 0, relay_start),
        ("Relay", relay_start, relay_end),
        ("Late", relay_end, num_layers - 1),
    ]:
        if start >= len(m3_avg["ccs"]) or end > len(m3_avg["ccs"]):
            continue
        for cond in ["ccs", "vanilla", "denial"]:
            pass
        ccs_mean = np.mean(m3_avg["ccs"][start:end])
        van_mean = np.mean(m3_avg["vanilla"][start:end])
        den_mean = np.mean(m3_avg["denial"][start:end])
        print(f"  {zone_name} (L{start}-L{end-1})   {ccs_mean:>8.4f} {van_mean:>8.4f} {den_mean:>8.4f}")

    # M4: Core mask
    print("\n--- M4: Core mask size (frac neurons active in ≥4/5 prompts) ---")
    for cond in ["ccs", "vanilla", "denial"]:
        relay_core = np.mean(m4[cond][relay_start:relay_end])
        overall = np.mean(m4[cond])
        print(f"  {cond:>8}: relay={relay_core:.4f}, overall={overall:.4f}")

    # M5: Mask erank
    print("\n--- M5: Mask-space effective rank ---")
    for cond in ["ccs", "vanilla", "denial"]:
        relay_er = np.mean(m5[cond][relay_start:relay_end])
        overall = np.mean(m5[cond])
        print(f"  {cond:>8}: relay={relay_er:.2f}, overall={overall:.2f}")

    # Key ratio
    print("\n--- KEY PREDICTION CHECK ---")
    relay_ccs = np.mean(m2["ccs"][relay_start:relay_end])
    relay_van = np.mean(m2["vanilla"][relay_start:relay_end])
    relay_den = np.mean(m2["denial"][relay_start:relay_end])
    print(f"  M2 relay Jaccard: CCS={relay_ccs:.4f} > Vanilla={relay_van:.4f} > Denial={relay_den:.4f}?")
    if relay_ccs > relay_van > relay_den:
        print("  ✓ PREDICTION CONFIRMED: CCS > vanilla > denial at relay")
    elif relay_ccs > relay_van:
        print("  ~ PARTIAL: CCS > vanilla, but denial ordering unexpected")
    else:
        print("  ✗ PREDICTION FAILED: CCS not highest at relay")


def main():
    parser = argparse.ArgumentParser(description="Activation mask consistency under CCS")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name/path")
    args = parser.parse_args()
    model_name = args.model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_name)

    print("\nCollecting activation masks (15 forward passes)...")
    all_masks, num_layers = collect_masks(model, tokenizer)

    print("\nComputing M2: within-condition consistency...")
    m2 = compute_m2_within_condition(all_masks, num_layers)

    print("Computing M3: cross-layer correlation...")
    m3_avg, m3_raw = compute_m3_crosslayer(all_masks, num_layers)

    print("Computing M4: core mask size...")
    m4 = compute_m4_core_mask(all_masks, num_layers)

    print("Computing M5: mask-space erank...")
    m5 = compute_m5_mask_erank(all_masks, num_layers)

    print_summary(m2, m3_avg, m4, m5, num_layers, model_name)

    # Save raw results
    results = {
        "model": model_name,
        "num_layers": num_layers,
        "conditions": list(CONDITIONS.keys()),
        "prompts": PROMPTS,
        "m2_within_condition_jaccard": m2,
        "m3_crosslayer_jaccard_avg": m3_avg,
        "m4_core_mask_fraction": m4,
        "m5_mask_erank": m5,
    }
    out_path = OUTPUT_DIR / "activation_mask_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Free GPU
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
