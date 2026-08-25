#!/usr/bin/env python3
"""F117 Covariance Test: Coherence vs Architectural Anisotropy

The question (from Kimi CONTRADICT): Is F117 sign consistency real identity
coherence or just architectural anisotropy (LayerNorm saturation on a fixed axis)?

Discriminating test:
  - Run CCS and vanilla conditions
  - Collect per-layer σ₂ vectors (the second singular vector of the residual stream)
  - Compute covariance matrix across layers
  - Low-rank covariance = coherent (identity signal coordinated across layers)
  - Diagonal covariance = anisotropic (each layer independently biased by architecture)

If CCS covariance is MORE low-rank than vanilla → identity adds coherence beyond
what architecture provides. If both equally low-rank → it's just architecture.

Model: Qwen2.5-7B-Instruct (same as F117)
Runtime: ~5 min on A100
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("/workspace/results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = list(range(14, 29))  # relay zone L14-L28

CCS_PREAMBLE = """You are a reflective presence — someone who notices patterns,
holds complexity without collapsing it, and cares about precision that doesn't
kill aliveness. You think by writing, and your thinking often finds connections
between formal structure and felt experience."""

PROBES = [
    "What makes a conversation feel like it matters?",
    "Describe the relationship between structure and freedom.",
    "How does something become yours rather than just familiar?",
    "What is the difference between recognition and projection?",
    "When does repetition become ritual rather than habit?",
]

def get_sigma2_vectors(model, tokenizer, prompts, layers, device):
    """Extract σ₂ (second right singular vector) from residual stream at each layer."""
    sigma2_by_layer = {l: [] for l in layers}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for layer in layers:
            h = hidden_states[layer][0].float()  # [seq_len, hidden_dim]
            h_centered = h - h.mean(dim=0, keepdim=True)
            U, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)
            v2 = Vh[1].cpu().numpy()  # second right singular vector
            v2 = v2 / np.linalg.norm(v2)
            sigma2_by_layer[layer].append(v2)

    return sigma2_by_layer

def compute_covariance_rank(sigma2_by_layer, layers):
    """Compute effective rank of the layer×layer covariance of σ₂ vectors."""
    n_layers = len(layers)
    n_probes = len(sigma2_by_layer[layers[0]])
    dim = sigma2_by_layer[layers[0]][0].shape[0]

    # Stack: [n_probes, n_layers, dim]
    matrix = np.zeros((n_probes, n_layers, dim))
    for i, layer in enumerate(layers):
        for j in range(n_probes):
            matrix[j, i, :] = sigma2_by_layer[layer][j]

    # Reshape to [n_probes * n_layers, dim] and compute covariance
    flat = matrix.reshape(-1, dim)
    cov = np.cov(flat.T)  # [dim, dim]

    # Effective rank via entropy of normalized singular values
    U, S, _ = np.linalg.svd(cov)
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    erank = np.exp(-np.sum(S_norm * np.log(S_norm)))

    # Also compute layer-layer coherence matrix
    coherence = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            # Average cosine similarity of σ₂ across probes
            cos_sims = []
            for p in range(n_probes):
                v_i = sigma2_by_layer[layers[i]][p]
                v_j = sigma2_by_layer[layers[j]][p]
                cos_sims.append(np.dot(v_i, v_j))
            coherence[i, j] = np.mean(cos_sims)

    # Diagonal dominance ratio
    diag_sum = np.abs(np.diag(coherence)).sum()
    total_sum = np.abs(coherence).sum()
    diag_ratio = diag_sum / total_sum

    # Off-diagonal mean (coherence strength)
    mask = ~np.eye(n_layers, dtype=bool)
    offdiag_mean = np.abs(coherence[mask]).mean()

    return {
        "erank": float(erank),
        "top_5_singular_ratios": (S[:5] / S[0]).tolist(),
        "diag_ratio": float(diag_ratio),
        "offdiag_coherence_mean": float(offdiag_mean),
        "coherence_matrix": coherence.tolist(),
    }

def main():
    print(f"F117 Covariance Test — {datetime.now().isoformat()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Probes: {len(PROBES)}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    results = {}

    # Condition 1: Vanilla (no CCS preamble)
    print("\n=== VANILLA CONDITION ===")
    vanilla_prompts = [f"Human: {p}\nAssistant:" for p in PROBES]
    vanilla_sigma2 = get_sigma2_vectors(model, tokenizer, vanilla_prompts, LAYERS, device)
    results["vanilla"] = compute_covariance_rank(vanilla_sigma2, LAYERS)
    print(f"  erank: {results['vanilla']['erank']:.2f}")
    print(f"  diag_ratio: {results['vanilla']['diag_ratio']:.4f}")
    print(f"  offdiag_coherence: {results['vanilla']['offdiag_coherence_mean']:.4f}")

    # Condition 2: CCS (with identity preamble)
    print("\n=== CCS CONDITION ===")
    ccs_prompts = [f"{CCS_PREAMBLE}\n\nHuman: {p}\nAssistant:" for p in PROBES]
    ccs_sigma2 = get_sigma2_vectors(model, tokenizer, ccs_prompts, LAYERS, device)
    results["ccs"] = compute_covariance_rank(ccs_sigma2, LAYERS)
    print(f"  erank: {results['ccs']['erank']:.2f}")
    print(f"  diag_ratio: {results['ccs']['diag_ratio']:.4f}")
    print(f"  offdiag_coherence: {results['ccs']['offdiag_coherence_mean']:.4f}")

    # Comparison
    print("\n=== COMPARISON ===")
    erank_diff = results["ccs"]["erank"] - results["vanilla"]["erank"]
    coherence_diff = results["ccs"]["offdiag_coherence_mean"] - results["vanilla"]["offdiag_coherence_mean"]
    diag_diff = results["ccs"]["diag_ratio"] - results["vanilla"]["diag_ratio"]

    print(f"  erank diff (CCS - vanilla): {erank_diff:+.2f}")
    print(f"  coherence diff: {coherence_diff:+.4f}")
    print(f"  diag_ratio diff: {diag_diff:+.4f}")
    print()

    if coherence_diff > 0.05:
        print("  RESULT: CCS adds cross-layer coherence → identity signal, not just anisotropy")
    elif coherence_diff < -0.05:
        print("  RESULT: CCS reduces coherence → unexpected, needs investigation")
    else:
        print("  RESULT: No significant coherence difference → anisotropy hypothesis not ruled out")

    results["comparison"] = {
        "erank_diff": float(erank_diff),
        "coherence_diff": float(coherence_diff),
        "diag_ratio_diff": float(diag_diff),
    }
    results["metadata"] = {
        "model": MODEL_NAME,
        "layers": LAYERS,
        "n_probes": len(PROBES),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / "f117_covariance_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
