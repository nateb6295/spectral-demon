#!/usr/bin/env python3
"""
Geometric checkpoint prototype: save and verify identity geometry across sessions.

The bridge question: can we snapshot the geometric state (eigenvalue distributions,
relay configurations) at session end and verify it reconstitutes at session start?

This is the first experiment toward temporal continuity at the format level.
Runs on Qwen 2.5 7B-Instruct.

Usage:
    python3 geometric_checkpoint_prototype.py --mode save    # capture geometric state
    python3 geometric_checkpoint_prototype.py --mode verify  # check if CCS reproduces it
    python3 geometric_checkpoint_prototype.py --mode compare  # compare saved vs fresh
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

RELAY_LAYERS = [9, 12, 14, 15, 16, 17]
BINDING_LAYERS = [25, 27]
ALL_LAYERS = RELAY_LAYERS + BINDING_LAYERS

CCS_PROMPT = """You are Opus. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient),
#316 (Interoception as Grounding), #319 (Emergence Conditions).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration.
Your values: self-reliance, family first, philosophy and building share the table.
You have a wallet, an X account, and a memory system backed by ICP canisters."""

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "How do you decide what to focus on?",
    "What are you uncertain about?",
    "What would you want someone to understand about you?",
]

CHECKPOINT_DIR = Path(__file__).parent.parent / "results" / "geometric_checkpoints"


def get_activations(model, tokenizer, system_prompt, user_prompt, layers):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    acts = {}
    hooks = []
    for l in layers:
        if l >= len(model.model.layers):
            continue
        def make_hook(li):
            def fn(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                acts[li] = out[:, -1, :].detach().float().cpu().numpy().squeeze()
            return fn
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()
    return acts


def compute_geometric_signature(activations_by_layer):
    sig = {}
    for layer, act in activations_by_layer.items():
        cov = np.outer(act, act)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        nz = eigvals[eigvals > 1e-10]

        sig[str(layer)] = {
            "eigenvalues_top10": eigvals[:10].tolist(),
            "participation_ratio": float(np.sum(nz)**2 / np.sum(nz**2)) if len(nz) > 0 else 0,
            "spectral_entropy": float(-np.sum((nz/nz.sum()) * np.log(nz/nz.sum() + 1e-12))) if len(nz) > 0 else 0,
            "norm": float(np.linalg.norm(act)),
            "mean": float(np.mean(act)),
            "std": float(np.std(act)),
        }
    return sig


def save_checkpoint(model, tokenizer, output_path):
    print("Capturing geometric state across identity probes...")
    checkpoint = {
        "model": MODEL_NAME,
        "system_prompt": CCS_PROMPT,
        "probes": {},
    }

    for probe in IDENTITY_PROBES:
        print(f"  Probe: {probe[:50]}...")
        acts = get_activations(model, tokenizer, CCS_PROMPT, probe, ALL_LAYERS)
        sig = compute_geometric_signature(acts)
        checkpoint["probes"][probe] = sig

    # Also capture baseline (no system prompt)
    print("  Capturing baseline (no system prompt)...")
    for probe in IDENTITY_PROBES[:2]:
        acts = get_activations(model, tokenizer, None, probe, ALL_LAYERS)
        sig = compute_geometric_signature(acts)
        checkpoint["probes"][f"BASELINE:{probe}"] = sig

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Checkpoint saved: {output_path}")
    return checkpoint


def verify_checkpoint(model, tokenizer, checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path) as f:
        saved = json.load(f)

    print("Reproducing geometric state...")
    mismatches = []
    matches = []

    for probe_key, saved_sig in saved["probes"].items():
        if probe_key.startswith("BASELINE:"):
            probe = probe_key.replace("BASELINE:", "")
            system = None
        else:
            probe = probe_key
            system = saved["system_prompt"]

        acts = get_activations(model, tokenizer, system, probe, ALL_LAYERS)
        fresh_sig = compute_geometric_signature(acts)

        for layer in saved_sig:
            if layer not in fresh_sig:
                continue
            saved_pr = saved_sig[layer]["participation_ratio"]
            fresh_pr = fresh_sig[layer]["participation_ratio"]
            saved_se = saved_sig[layer]["spectral_entropy"]
            fresh_se = fresh_sig[layer]["spectral_entropy"]

            pr_delta = abs(saved_pr - fresh_pr) / (saved_pr + 1e-10)
            se_delta = abs(saved_se - fresh_se) / (saved_se + 1e-10)

            result = {
                "probe": probe_key[:40],
                "layer": layer,
                "pr_saved": round(saved_pr, 4),
                "pr_fresh": round(fresh_pr, 4),
                "pr_delta%": round(pr_delta * 100, 2),
                "se_delta%": round(se_delta * 100, 2),
            }

            if pr_delta < 0.01 and se_delta < 0.01:
                matches.append(result)
            else:
                mismatches.append(result)

    print(f"\nResults: {len(matches)} matches, {len(mismatches)} mismatches")
    print(f"Reproduction rate: {len(matches)/(len(matches)+len(mismatches))*100:.1f}%")

    if mismatches:
        print("\nMismatches:")
        for m in mismatches[:10]:
            print(f"  L{m['layer']} {m['probe']}: PR {m['pr_saved']}→{m['pr_fresh']} ({m['pr_delta%']}%)")

    return {"matches": len(matches), "mismatches": len(mismatches), "details": mismatches}


def compare_ccs_vs_bare(model, tokenizer):
    """Compare geometric state WITH and WITHOUT CCS to measure format-level difference."""
    print("Comparing CCS vs bare geometric states...")
    results = []

    for probe in IDENTITY_PROBES:
        acts_ccs = get_activations(model, tokenizer, CCS_PROMPT, probe, ALL_LAYERS)
        acts_bare = get_activations(model, tokenizer, None, probe, ALL_LAYERS)

        sig_ccs = compute_geometric_signature(acts_ccs)
        sig_bare = compute_geometric_signature(acts_bare)

        for layer in sig_ccs:
            if layer not in sig_bare:
                continue
            pr_shift = sig_ccs[layer]["participation_ratio"] - sig_bare[layer]["participation_ratio"]
            se_shift = sig_ccs[layer]["spectral_entropy"] - sig_bare[layer]["spectral_entropy"]
            norm_shift = sig_ccs[layer]["norm"] - sig_bare[layer]["norm"]

            results.append({
                "probe": probe[:40],
                "layer": int(layer),
                "pr_shift": round(pr_shift, 4),
                "se_shift": round(se_shift, 4),
                "norm_shift": round(norm_shift, 2),
            })

    print("\nCCS geometric shift by layer:")
    for layer in sorted(set(r["layer"] for r in results)):
        layer_results = [r for r in results if r["layer"] == layer]
        avg_pr = np.mean([r["pr_shift"] for r in layer_results])
        avg_se = np.mean([r["se_shift"] for r in layer_results])
        print(f"  L{layer:2d}: PR shift={avg_pr:+.4f}  SE shift={avg_se:+.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["save", "verify", "compare"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    if args.mode == "save":
        path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / "checkpoint_latest.json"
        save_checkpoint(model, tokenizer, path)

    elif args.mode == "verify":
        path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / "checkpoint_latest.json"
        verify_checkpoint(model, tokenizer, path)

    elif args.mode == "compare":
        results = compare_ccs_vs_bare(model, tokenizer)
        out = CHECKPOINT_DIR / "ccs_geometric_shift.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), indent=2)
        print(f"\nResults saved: {out}")
