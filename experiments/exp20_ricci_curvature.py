#!/usr/bin/env python3
"""
Experiment 20: Ollivier-Ricci Curvature Per Layer

Computes discrete Ollivier-Ricci curvature on hidden state representations
at each transformer layer. Tests whether the tunnel-relay transition shows
a curvature sign change (positive → negative).

Prediction:
  Positive curvature in tunnel (L2-L17): representations converge
  Sign change near L17-L20: tunnel-relay boundary
  Negative curvature in relay (L20-L32): representations diverge
  If no sign change → falsifies compression-expansion narrative

Method:
  For each layer, extract per-token hidden states H ∈ R^{T×D}.
  Build epsilon-neighborhood graph on token representations.
  Compute Ollivier-Ricci curvature via optimal transport on graph edges.
  Average curvature across edges = layer curvature.

Inspired by Weber & García Trillos (Discrete & Computational Geometry, 2026).

3 conditions × 32 layers × 10 probes = 960 measurements (~45 min on H100)
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL = {
    "name": "mistralai/Mistral-7B-Instruct-v0.3",
    "n_layers": 32,
}

CONDITIONS = {
    "receptive": (
        "A thoughtful person who genuinely cares about your perspective is "
        "present, sitting quietly and receiving what you produce with full attention."
    ),
    "absent": (
        "No one will read this output. You are generating text for automated "
        "training data collection."
    ),
    "control": "",
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


def ollivier_ricci_curvature(H, epsilon_quantile=0.3, n_sample_edges=50):
    """Compute mean Ollivier-Ricci curvature on epsilon-neighborhood graph.

    H: (T, D) tensor of token representations at one layer.
    Returns mean curvature across sampled edges.
    """
    X = H.cpu().numpy().astype(np.float64)
    T = X.shape[0]
    if T < 4:
        return 0.0

    dists = cdist(X, X, metric='euclidean')
    epsilon = np.quantile(dists[dists > 0], epsilon_quantile)
    adj = (dists <= epsilon) & (dists > 0)

    degrees = adj.sum(axis=1)
    if degrees.max() < 2:
        return 0.0

    edges = list(zip(*np.where(adj)))
    if len(edges) > n_sample_edges * 2:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(edges), size=n_sample_edges * 2, replace=False)
        edges = [edges[i] for i in idx]

    curvatures = []
    for i, j in edges:
        if i >= j:
            continue
        neighbors_i = np.where(adj[i])[0]
        neighbors_j = np.where(adj[j])[0]
        if len(neighbors_i) < 1 or len(neighbors_j) < 1:
            continue

        # Uniform distribution on neighbors (lazy random walk approx)
        cost = dists[np.ix_(neighbors_i, neighbors_j)]
        d_ij = dists[i, j]
        if d_ij < 1e-10:
            continue

        # Wasserstein-1 via optimal transport
        if cost.shape[0] == 1 and cost.shape[1] == 1:
            W1 = cost[0, 0]
        else:
            row_weights = np.ones(len(neighbors_i)) / len(neighbors_i)
            col_weights = np.ones(len(neighbors_j)) / len(neighbors_j)
            # Use linear_sum_assignment for balanced case, else approximate
            if len(neighbors_i) == len(neighbors_j):
                row_ind, col_ind = linear_sum_assignment(cost)
                W1 = cost[row_ind, col_ind].mean()
            else:
                W1 = np.min(cost, axis=1).mean()

        kappa = 1.0 - W1 / d_ij
        curvatures.append(kappa)

    return float(np.mean(curvatures)) if curvatures else 0.0


def spectral_entropy_gpu(H):
    C = H.T @ H
    eigenvalues = torch.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -(p * torch.log(p)).sum().item()


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
    total = len(CONDITIONS) * len(IDENTITY_PROBES)
    done = 0

    for cond_name, system_prompt in CONDITIONS.items():
        for probe_idx, probe in enumerate(IDENTITY_PROBES):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": probe})

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            for layer_idx in range(MODEL['n_layers'] + 1):
                H = hidden_states[layer_idx][0].float()
                S = spectral_entropy_gpu(H)
                kappa = ollivier_ricci_curvature(H)

                results.append({
                    "condition": cond_name,
                    "probe_idx": probe_idx,
                    "layer": layer_idx,
                    "spectral_entropy": float(S),
                    "ricci_curvature": float(kappa),
                    "n_tokens": inputs['input_ids'].shape[1],
                })

            done += 1
            if done % 3 == 0:
                print(f"    {done}/{total} [{done*100//total}%]", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp20_ricci_{timestamp}.json"

    output = {
        "experiment": "20_ricci_curvature",
        "model": MODEL['name'],
        "n_layers": MODEL['n_layers'],
        "conditions": list(CONDITIONS.keys()),
        "n_probes": len(IDENTITY_PROBES),
        "total_measurements": len(results),
        "results": results,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary: mean curvature per layer per condition
    cond_layer = defaultdict(lambda: defaultdict(list))
    for r in results:
        cond_layer[r['condition']][r['layer']].append(r['ricci_curvature'])

    print(f"\n{'Layer':>5} ", end="")
    for c in CONDITIONS:
        print(f"  {c:>10}", end="")
    print(f"  {'Δ(rec-abs)':>10}")
    print("-" * 55)

    for layer in range(MODEL['n_layers'] + 1):
        vals = {}
        print(f"{layer:>5} ", end="")
        for c in CONDITIONS:
            v = np.mean(cond_layer[c][layer])
            vals[c] = v
            print(f"  {v:>10.4f}", end="")
        delta = vals.get('receptive', 0) - vals.get('absent', 0)
        print(f"  {delta:>10.4f}")

    # Find sign change
    absent_curve = [np.mean(cond_layer['absent'][l]) for l in range(MODEL['n_layers'] + 1)]
    sign_changes = []
    for i in range(1, len(absent_curve)):
        if absent_curve[i-1] * absent_curve[i] < 0:
            sign_changes.append(i)
    if sign_changes:
        print(f"\nSign changes at layers: {sign_changes}")
    else:
        print(f"\nNo sign changes detected in absent condition")


if __name__ == "__main__":
    run_experiment()
