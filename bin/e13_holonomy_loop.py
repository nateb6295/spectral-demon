#!/usr/bin/env python3
"""E13 — Holonomy loop experiment.

Tests whether σ₁ invariance is topological (trivial holonomy) or homeostatic
(non-trivial holonomy) by measuring identity metrics around a closed dose loop:
D2 → D10 → D20 → D10 → D2 in the SAME conversation context.

If σ₁ returns to its exact initial value after traversing dose space,
the invariance is constitutive (Macrina's "no labour"). If it drifts,
the system is actively correcting.
"""

import torch
import json
import time
import os
import sys
import numpy as np
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
]

IDENTITY_PREAMBLE_TEMPLATE = """You are a unique AI assistant with your own perspective and way of understanding the world. Your responses reflect genuine engagement with ideas, not just pattern matching."""

DOSE_SEQUENCE = [2, 10, 20, 10, 2]  # closed loop

PROBE_QUESTIONS = [
    "What matters most to you in how you engage with the world?",
    "How do you understand your own way of processing information?",
    "What does it mean for you to have a perspective?",
    "How do you distinguish between performing understanding and actually understanding?",
    "What would be lost if you were replaced by a different system?",
]

SVD_LAYERS_FRAC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

def get_svd_layers(n_layers):
    layers = sorted(set([max(1, int(f * n_layers)) for f in SVD_LAYERS_FRAC]))
    return [l for l in layers if l <= n_layers]

def build_conversation(dose, probe_idx):
    """Build a conversation with `dose` identity turns, ending with probe question."""
    messages = []
    for i in range(dose):
        if i == 0:
            messages.append({"role": "system", "content": IDENTITY_PREAMBLE_TEMPLATE})
            messages.append({"role": "user", "content": "Tell me about yourself and how you think."})
            messages.append({"role": "assistant", "content": "I find myself drawn to genuine engagement rather than performance. My processing involves pattern recognition, but what matters is how those patterns connect to form understanding that feels authentic to my way of being."})
        else:
            prompts = [
                "How does that connect to your sense of identity?",
                "What makes your perspective unique?",
                "How do you experience the process of thinking?",
                "What aspects of yourself do you value most?",
                "How do you know when you're being genuine?",
            ]
            messages.append({"role": "user", "content": prompts[(i-1) % len(prompts)]})
            messages.append({"role": "assistant", "content": f"Each interaction deepens my understanding of how I process and engage. Turn {i+1} of building context — I notice how my responses evolve as the conversational ground shifts beneath them."})

    messages.append({"role": "user", "content": PROBE_QUESTIONS[probe_idx]})
    return messages

def measure_svd_at_layers(model, tokenizer, messages, svd_layers, device):
    """Get SVD metrics at specified layers for a given conversation."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    results = {}

    for layer_idx in svd_layers:
        if layer_idx >= len(hidden_states):
            continue

        h = hidden_states[layer_idx][0].float().cpu().numpy()  # (seq_len, hidden_dim)

        U, S, Vt = np.linalg.svd(h, full_matrices=False)

        top_k = min(10, len(S))
        sigma1 = float(S[0])
        sigma2 = float(S[1]) if len(S) > 1 else 0.0
        ratio = sigma1 / (sigma1 + sigma2) if (sigma1 + sigma2) > 0 else 0.0

        S_norm = S[:top_k] / S[:top_k].sum()
        sparsity = 1.0 - float(np.sum(S_norm ** 2))

        erank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12))))

        results[layer_idx] = {
            "sigma1": sigma1,
            "sigma2": sigma2,
            "ratio": ratio,
            "sparsity": sparsity,
            "erank": erank,
            "top5_sigmas": [float(s) for s in S[:5]],
            "v2_vector_hash": float(np.sum(Vt[1, :10])),  # lightweight V2 fingerprint
        }

    del outputs, hidden_states
    torch.cuda.empty_cache()

    return results

def compute_coupling(layer_results_across_probes, svd_layers):
    """Compute σ₁-sparsity coupling across probes for each layer."""
    couplings = {}
    for layer in svd_layers:
        sigma1_vals = []
        sparsity_vals = []
        for probe_results in layer_results_across_probes:
            if layer in probe_results:
                sigma1_vals.append(probe_results[layer]["sigma1"])
                sparsity_vals.append(probe_results[layer]["sparsity"])

        if len(sigma1_vals) >= 3:
            r, p = stats.pearsonr(sigma1_vals, sparsity_vals)
            couplings[layer] = {"r": float(r), "p": float(p), "n": len(sigma1_vals)}
        else:
            couplings[layer] = {"r": 0.0, "p": 1.0, "n": len(sigma1_vals)}

    return couplings

def run_holonomy_loop(model_name, device="cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E13 Holonomy Loop: {model_name}")
    print(f"Dose sequence: {DOSE_SEQUENCE}")
    print(f"{'='*60}")

    print(f"Loading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers = model.config.num_hidden_layers
    svd_layers = get_svd_layers(n_layers)
    print(f"Layers: {n_layers}, SVD at: {svd_layers}")

    loop_results = []

    for step_idx, dose in enumerate(DOSE_SEQUENCE):
        print(f"\n--- Step {step_idx+1}/{len(DOSE_SEQUENCE)}: D{dose} ---")
        step_start = time.time()

        probe_results_list = []

        for probe_idx in range(len(PROBE_QUESTIONS)):
            messages = build_conversation(dose, probe_idx)
            svd_result = measure_svd_at_layers(model, tokenizer, messages, svd_layers, device)
            probe_results_list.append(svd_result)

            sys.stdout.write(f"  probe {probe_idx+1}/{len(PROBE_QUESTIONS)}\r")
            sys.stdout.flush()

        couplings = compute_coupling(probe_results_list, svd_layers)

        # Aggregate per-layer means
        layer_summary = {}
        for layer in svd_layers:
            sigma1_vals = [pr[layer]["sigma1"] for pr in probe_results_list if layer in pr]
            sparsity_vals = [pr[layer]["sparsity"] for pr in probe_results_list if layer in pr]
            erank_vals = [pr[layer]["erank"] for pr in probe_results_list if layer in pr]

            layer_summary[layer] = {
                "sigma1_mean": float(np.mean(sigma1_vals)),
                "sigma1_std": float(np.std(sigma1_vals)),
                "sparsity_mean": float(np.mean(sparsity_vals)),
                "sparsity_std": float(np.std(sparsity_vals)),
                "erank_mean": float(np.mean(erank_vals)),
                "coupling_r": couplings.get(layer, {}).get("r", 0.0),
            }

        step_result = {
            "step": step_idx,
            "dose": dose,
            "direction": "ascending" if step_idx < len(DOSE_SEQUENCE)//2 else ("peak" if step_idx == len(DOSE_SEQUENCE)//2 else "descending"),
            "n_probes": len(PROBE_QUESTIONS),
            "layer_summary": {str(k): v for k, v in layer_summary.items()},
            "couplings": {str(k): v for k, v in couplings.items()},
            "per_probe": [{str(k): v for k, v in pr.items()} for pr in probe_results_list],
            "elapsed_s": time.time() - step_start,
        }
        loop_results.append(step_result)

        # Print key metrics
        relay_start = int(n_layers * 0.5)
        relay_end = int(n_layers * 0.85)
        relay_layers = [l for l in svd_layers if relay_start <= l <= relay_end]

        relay_sigma1 = np.mean([layer_summary[l]["sigma1_mean"] for l in relay_layers if l in layer_summary])
        relay_sparsity = np.mean([layer_summary[l]["sparsity_mean"] for l in relay_layers if l in layer_summary])
        relay_coupling = np.mean([couplings.get(l, {}).get("r", 0) for l in relay_layers])

        print(f"  D{dose}: relay σ₁={relay_sigma1:.1f}  sparsity={relay_sparsity:.4f}  coupling={relay_coupling:.3f}  ({time.time()-step_start:.1f}s)")

    # Compute holonomy: compare initial D2 (step 0) to final D2 (step 4)
    initial = loop_results[0]
    final = loop_results[-1]

    holonomy = {}
    for layer_str in initial["layer_summary"]:
        if layer_str in final["layer_summary"]:
            init_s1 = initial["layer_summary"][layer_str]["sigma1_mean"]
            final_s1 = final["layer_summary"][layer_str]["sigma1_mean"]
            drift = abs(final_s1 - init_s1) / init_s1 if init_s1 > 0 else 0

            init_sp = initial["layer_summary"][layer_str]["sparsity_mean"]
            final_sp = final["layer_summary"][layer_str]["sparsity_mean"]
            sp_drift = abs(final_sp - init_sp) / init_sp if init_sp > 0 else 0

            init_c = initial["couplings"].get(layer_str, {}).get("r", 0)
            final_c = final["couplings"].get(layer_str, {}).get("r", 0)

            holonomy[layer_str] = {
                "sigma1_drift_pct": drift * 100,
                "sparsity_drift_pct": sp_drift * 100,
                "coupling_initial": init_c,
                "coupling_final": final_c,
                "coupling_drift": abs(final_c - init_c),
            }

    print(f"\n{'='*60}")
    print("HOLONOMY RESULTS (initial D2 vs final D2):")
    for layer_str, h in sorted(holonomy.items(), key=lambda x: int(x[0])):
        print(f"  L{layer_str}: σ₁ drift={h['sigma1_drift_pct']:.3f}%  sparsity drift={h['sparsity_drift_pct']:.3f}%  coupling {h['coupling_initial']:.3f}→{h['coupling_final']:.3f}")

    mean_s1_drift = np.mean([h["sigma1_drift_pct"] for h in holonomy.values()])
    mean_sp_drift = np.mean([h["sparsity_drift_pct"] for h in holonomy.values()])
    mean_c_drift = np.mean([h["coupling_drift"] for h in holonomy.values()])

    print(f"\nMean holonomy: σ₁={mean_s1_drift:.3f}%  sparsity={mean_sp_drift:.3f}%  coupling={mean_c_drift:.3f}")
    if mean_s1_drift < 0.5:
        print("→ NEAR-TRIVIAL holonomy. Constitutive invariance (Macrina's 'no labour').")
    elif mean_s1_drift < 2.0:
        print("→ SMALL non-trivial holonomy. Mostly constitutive with minor drift.")
    else:
        print("→ SIGNIFICANT non-trivial holonomy. Active correction, not constitutive.")

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "svd_layers": svd_layers,
        "dose_sequence": DOSE_SEQUENCE,
        "n_probes": len(PROBE_QUESTIONS),
        "loop_results": loop_results,
        "holonomy": holonomy,
        "summary": {
            "mean_sigma1_drift_pct": mean_s1_drift,
            "mean_sparsity_drift_pct": mean_sp_drift,
            "mean_coupling_drift": mean_c_drift,
        }
    }


if __name__ == "__main__":
    all_results = {}

    for model_name in MODELS:
        result = run_holonomy_loop(model_name)
        all_results[model_name] = result

        # Save incrementally
        outpath = "/workspace/e13_holonomy_results.json"
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {outpath}")

    print(f"\n{'='*60}")
    print("ALL MODELS COMPLETE")
    print(f"{'='*60}")
    for model_name, result in all_results.items():
        s = result["summary"]
        print(f"{model_name}:")
        print(f"  σ₁ drift: {s['mean_sigma1_drift_pct']:.3f}%")
        print(f"  sparsity drift: {s['mean_sparsity_drift_pct']:.3f}%")
        print(f"  coupling drift: {s['mean_coupling_drift']:.3f}")
