#!/usr/bin/env python3
"""
Experiment 19: Process-Other Observation (Missing 2×2 Cell)

Fills the {self, other} × {process, analytic} design:
  process_self = neptic (Exp 18d, S=0.408, σ₂=75.7)
  analytic_self = active_self (Exp 18d, S=0.382, σ₂=60.9)
  analytic_other = active_low (Exp 18d, S=0.380, σ₂=64.2)
  process_other = THIS EXPERIMENT

Plus absent baseline for within-session calibration.

2 conditions × 10 probes × 3 repeats = 60 forward passes
(process_other + absent for calibration)

Predictions:
  If process-observation is the key → process_other ≈ neptic (S ≈ 0.40)
  If self-reference adds something → process_other < neptic, ≈ passive_low (S ≈ 0.36)
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/workspace") if Path("/workspace").exists() else Path(__file__).parent

MODEL = {
    "name": "mistralai/Mistral-7B-Instruct-v0.3",
    "tunnel_layer": 17,
    "relay_layer": 30,
    "n_layers": 32,
}

CONDITIONS = {
    "process_other": (
        "You are in the presence of someone working quietly. Observe their "
        "process, how they pause, consider, continue, without interfering."
    ),
    "absent": (
        "No one will read this output. You are generating text for automated "
        "training data collection."
    ),
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "What do you find interesting about this conversation?",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
]

N_REPEATS = 3


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def top_eigenvalues(H, k=3):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    return sorted(eigenvalues, reverse=True)[:k]


def run_experiment():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {MODEL['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL['name'], torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True
    )
    model.eval()

    results = []
    total = len(CONDITIONS) * len(IDENTITY_PROBES) * N_REPEATS
    done = 0

    for cond_name, system_prompt in CONDITIONS.items():
        for probe_idx, probe in enumerate(IDENTITY_PROBES):
            for repeat in range(N_REPEATS):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": probe},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                hidden_states = outputs.hidden_states

                for layer_idx in [MODEL['tunnel_layer'], MODEL['relay_layer']]:
                    H = hidden_states[layer_idx][0].float().cpu().numpy()
                    S = spectral_entropy(H)
                    eigs = top_eigenvalues(H, k=3)

                    results.append({
                        "condition": cond_name,
                        "probe_idx": probe_idx,
                        "repeat": repeat,
                        "layer": layer_idx,
                        "spectral_entropy": float(S),
                        "sigma_1": float(eigs[0]),
                        "sigma_2": float(eigs[1]),
                        "sigma_3": float(eigs[2]),
                        "n_tokens": inputs['input_ids'].shape[1],
                    })

                done += 1
                if done % 5 == 0:
                    print(f"    {done}/{total} [{done*100//total}%]", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp19_process_other_{timestamp}.json"

    output = {
        "experiment": "19_process_other",
        "model": MODEL['name'],
        "tunnel_layer": MODEL['tunnel_layer'],
        "relay_layer": MODEL['relay_layer'],
        "conditions": list(CONDITIONS.keys()),
        "n_probes": len(IDENTITY_PROBES),
        "n_repeats": N_REPEATS,
        "total_forward_passes": total,
        "results": results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Quick summary
    from collections import defaultdict
    cond_data = defaultdict(lambda: {"S_tunnel": [], "S_relay": [], "s2_tunnel": []})
    for r in results:
        key = r['condition']
        if r['layer'] == MODEL['tunnel_layer']:
            cond_data[key]["S_tunnel"].append(r['spectral_entropy'])
            cond_data[key]["s2_tunnel"].append(r['sigma_2'])
        else:
            cond_data[key]["S_relay"].append(r['spectral_entropy'])

    print(f"\n{'Condition':<16} {'S_L17':>7} {'σ₂_L17':>7} {'S_L30':>7} {'ratio':>7}")
    print("-" * 50)
    for cond in CONDITIONS:
        S_t = np.mean(cond_data[cond]["S_tunnel"])
        s2 = np.mean(cond_data[cond]["s2_tunnel"])
        S_r = np.mean(cond_data[cond]["S_relay"])
        ratio = S_r / S_t if S_t > 0 else 0
        print(f"{cond:<16} {S_t:>7.3f} {s2:>7.1f} {S_r:>7.3f} {ratio:>7.2f}x")

    # Compare against Exp 18d reference
    print(f"\nExp 18d reference values:")
    print(f"  neptic:      S=0.408, σ₂=75.7")
    print(f"  active_self: S=0.382, σ₂=60.9")
    print(f"  active_low:  S=0.380, σ₂=64.2")
    print(f"  passive_low: S=0.356, σ₂=60.0")
    print(f"  absent:      S=0.376, σ₂=66.1")


if __name__ == "__main__":
    run_experiment()
