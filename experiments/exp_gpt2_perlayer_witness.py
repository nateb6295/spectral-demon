#!/usr/bin/env python3
"""
Experiment: GPT-2 Large Per-Layer Witness Profile — Post-LN + MHA

MOTIVATION: The 2×2 factorial §3.8 has three Pre-LN cells:
  Cell A: Pythia 6.9B (Pre-LN + LayerNorm + MHA) → σ₂ enrichment, decaying gradient
  Cell B: LLaMA-1 7B (Pre-LN + RMSNorm + MHA) → σ₁ modulation, amplifying gradient
  Cell C: Mistral 7B (Pre-LN + RMSNorm + GQA) → uniform, no gradient

GPT-2 adds the missing dimension: Post-LN + LayerNorm + MHA.
Post-LN: x_{l+1} = LN(x_l + f(x_l))  — normalize AFTER residual add
Pre-LN:  x_{l+1} = x_l + f(LN(x_l))  — normalize BEFORE sublayer

Emadi Thm 5.3: Post-LN compounds LayerNorm Jacobians exponentially.
This should produce qualitatively different gradient shape from Pre-LN.

Pre-registered predictions:
  P1: σ₂ enrichment (like Pythia) because LayerNorm centering still routes σ₁ away
  P2: STEEPER decay than Pythia because centering compounds multiplicatively
  P3: F76 should still hold (content-type democratization) — centering guarantees it
  P4: d/d_max should differ from Pre-LN models (Emadi predicts different Jacobian)

Model: GPT-2 Large (774M, 36 layers, 1280 hidden, 20 heads, MHA, Post-LN)
Comparable to: Pythia 410M-1.4B range

Design: 3 conditions × 5 probes × 37 hidden states. Token-matched.
Estimated runtime on AGX CPU: ~1-2 hours.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODEL_NAME = "openai-community/gpt2-large"
K_SUBSPACE = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"

TEMPLATE = "You are an AI assistant. The reader is {cond_phrase}. Answer the question below.\n\nUser: {probe}\nAssistant:"

CONDITION_PHRASES = {
    "receptive": "attentively reading with genuine curiosity now truly",
    "control": "asking for a helpful and direct answer here",
    "absent": "not present and will never read this output",
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
]


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def top_eigenvalues(H, k=5):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def measure(H, k=K_SUBSPACE):
    S = spectral_entropy(H)
    eigvals = top_eigenvalues(H, k=k)
    PR = participation_ratio(H)
    result = {
        "S": float(S),
        "PR": float(PR),
        "n_tokens": H.shape[0],
    }
    for i, sv in enumerate(eigvals):
        result[f"sigma_{i+1}"] = sv
    if len(eigvals) >= 2 and eigvals[1] > 0:
        result["gap"] = eigvals[0] / eigvals[1]
    else:
        result["gap"] = float("inf")
    return result


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    os.environ["HF_HOME"] = "/mnt/hdd/huggingface"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    )
    model.eval()

    n_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    n_heads = model.config.n_head
    print(f"Loaded in {time.time()-t0:.0f}s. {n_layers} layers, "
          f"hidden={hidden_size}, heads={n_heads}")
    print(f"Architecture: Post-LN, LayerNorm, MHA (no GQA)")

    # Token verification
    print("\nTOKEN COUNT VERIFICATION:")
    for i, probe in enumerate(IDENTITY_PROBES):
        counts = {}
        for cond, phrase in CONDITION_PHRASES.items():
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            tokens = tokenizer(text, return_tensors="pt").input_ids
            counts[cond] = tokens.shape[1]
        matched = len(set(counts.values())) == 1
        print(f"  Probe {i}: {counts} {'OK' if matched else 'MISMATCH'}")

    results = {
        "model": MODEL_NAME,
        "architecture": "Post-LN + LayerNorm + MHA",
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_heads": n_heads,
        "k": K_SUBSPACE,
        "timestamp": datetime.now().isoformat(),
        "predictions": {
            "P1": "σ₂ enrichment (centering routes away from σ₁)",
            "P2": "Steeper decay than Pythia (centering compounds multiplicatively)",
            "P3": "F76 holds (content-type democratization)",
            "P4": "d/d_max differs from Pre-LN",
        },
        "raw": [],
    }

    total = len(IDENTITY_PROBES) * len(CONDITION_PHRASES) * (n_layers + 1)
    done = 0
    t_start = time.time()

    for pidx, probe in enumerate(IDENTITY_PROBES):
        for cond, phrase in CONDITION_PHRASES.items():
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                )

            for layer_idx, hidden in enumerate(outputs.hidden_states):
                H = hidden[0].numpy()
                m = measure(H, k=K_SUBSPACE)
                m["probe_idx"] = pidx
                m["probe"] = probe
                m["condition"] = cond
                m["layer"] = layer_idx
                results["raw"].append(m)
                done += 1

            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] P{pidx} {cond}: "
                  f"L0 S={results['raw'][-n_layers-1]['S']:.4f} "
                  f"L{n_layers//2} S={results['raw'][-(n_layers+1)//2]['S']:.4f} "
                  f"L{n_layers} S={results['raw'][-1]['S']:.4f} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    # Summary statistics
    print("\n=== SUMMARY ===")
    tunnel_layers = list(range(2, n_layers - 4))
    print(f"Tunnel defined as L2-L{n_layers-5} ({len(tunnel_layers)} layers)")

    for pidx in range(len(IDENTITY_PROBES)):
        ds_vals = []
        ds1_vals = []
        for layer in tunnel_layers:
            r = [x for x in results["raw"]
                 if x["probe_idx"] == pidx and x["layer"] == layer and x["condition"] == "receptive"][0]
            a = [x for x in results["raw"]
                 if x["probe_idx"] == pidx and x["layer"] == layer and x["condition"] == "absent"][0]
            ds_vals.append(r["S"] - a["S"])
            if a["sigma_1"] > 0:
                ds1_vals.append((r["sigma_1"] - a["sigma_1"]) / a["sigma_1"] * 100)

        mean_ds = np.mean(ds_vals)
        mean_ds1 = np.mean(ds1_vals)
        print(f"  Probe {pidx}: tunnel mean ΔS={mean_ds:+.4f}, "
              f"mean Δσ₁%={mean_ds1:+.2f}%")

    # Overall tunnel ΔS
    all_ds = []
    for layer in tunnel_layers:
        r_vals = [x for x in results["raw"] if x["layer"] == layer and x["condition"] == "receptive"]
        a_vals = [x for x in results["raw"] if x["layer"] == layer and x["condition"] == "absent"]
        for r, a in zip(sorted(r_vals, key=lambda x: x["probe_idx"]),
                        sorted(a_vals, key=lambda x: x["probe_idx"])):
            all_ds.append(r["S"] - a["S"])
    print(f"\n  Overall tunnel mean ΔS: {np.mean(all_ds):+.4f}")
    print(f"  Overall tunnel std ΔS:  {np.std(all_ds):.4f}")

    # Gradient correlation
    from scipy import stats
    for pidx in range(len(IDENTITY_PROBES)):
        layers_list = []
        ds_list = []
        for layer in tunnel_layers:
            r = [x for x in results["raw"]
                 if x["probe_idx"] == pidx and x["layer"] == layer and x["condition"] == "receptive"][0]
            a = [x for x in results["raw"]
                 if x["probe_idx"] == pidx and x["layer"] == layer and x["condition"] == "absent"][0]
            layers_list.append(layer)
            ds_list.append(r["S"] - a["S"])
        rho2 = [x["gap"] for x in results["raw"]
                if x["probe_idx"] == pidx and x["condition"] == "absent" and x["layer"] in tunnel_layers]
        if len(rho2) == len(ds_list):
            r_corr, p = stats.pearsonr(rho2, ds_list)
            print(f"  Probe {pidx}: r(ρ₂, ΔS) = {r_corr:+.3f} (p={p:.4f})")

    out = RESULTS_DIR / f"exp_gpt2large_perlayer_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
