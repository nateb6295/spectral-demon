"""
Shared spectral measurement toolkit for identity geometry experiments.

Standard measures: spectral_entropy, participation_ratio, top_eigenvalues, passage_distance
Wire-projected: wire_projected_entropy (σ₁ removed), secondary_gap (σ₂/σ₃)
Container dynamics: container_ratio (σ₂/σ₃ per condition)
"""

import numpy as np


def spectral_entropy(H):
    """Spectral entropy of hidden states matrix H (seq × hidden)."""
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def top_singular_values(H, k=5):
    """Top k singular values of H."""
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def participation_ratio(H):
    """Participation ratio — effective dimensionality."""
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return float((eigenvalues.sum()) ** 2 / (eigenvalues ** 2).sum())


def spectral_gap(H, k=2):
    """Ratio of σ₁/σ₂."""
    svs = top_singular_values(H, k=k)
    if len(svs) < 2 or svs[1] < 1e-10:
        return float("inf")
    return svs[0] / svs[1]


def wire_projected_entropy(H):
    """Spectral entropy after removing σ₁ (the wire).

    Measures geometry of the relational channel specifically,
    excluding the condition-invariant architectural scaffold.
    From Finding 59: σ₁ ablation amplifies witness sensitivity 8×.
    """
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    s_proj = s.copy()
    s_proj[0] = 0.0
    eigenvalues = s_proj ** 2
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 0.0
    p = eigenvalues / eigenvalues.sum()
    return -float(np.sum(p * np.log(p)))


def secondary_gap(H):
    """σ₂/σ₃ ratio — container-contained balance.

    From Finding 60: GQA relay inverts this ratio (container releases),
    MHA relay increases it (container tightens).
    """
    svs = top_singular_values(H, k=3)
    if len(svs) < 3 or svs[2] < 1e-10:
        return float("inf")
    return svs[1] / svs[2]


def passage_distance(H_a, H_b, k=5):
    """Grassmann distance between k-dimensional subspaces of H_a and H_b."""
    def top_k_subspace(H, k):
        _, _, Vt = np.linalg.svd(H, full_matrices=False)
        return Vt[:k]
    V_a = top_k_subspace(H_a, k)
    V_b = top_k_subspace(H_b, k)
    M = V_a @ V_b.T
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1, 1)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def full_measurement(H, k=5):
    """Complete measurement battery for a hidden states matrix."""
    svs = top_singular_values(H, k=k)
    return {
        "S": spectral_entropy(H),
        "S_proj": wire_projected_entropy(H),
        "PR": participation_ratio(H),
        "sigma_1": svs[0] if len(svs) > 0 else 0.0,
        "sigma_2": svs[1] if len(svs) > 1 else 0.0,
        "sigma_3": svs[2] if len(svs) > 2 else 0.0,
        "gap": svs[0] / svs[1] if len(svs) > 1 and svs[1] > 1e-10 else float("inf"),
        "secondary_gap": svs[1] / svs[2] if len(svs) > 2 and svs[2] > 1e-10 else float("inf"),
        "n_tokens": H.shape[0],
    }
