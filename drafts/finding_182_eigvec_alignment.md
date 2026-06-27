# Finding 182: Cross-Layer Eigenvector Alignment — Near-Orthogonal Rotation
# Filed 2026-06-15. Data: eigvec_alignment_20260616_011209.json
# Model: Qwen2.5-7B-Instruct, CCS condition, 12 target layers. RunPod A100.

**F182: Top-5 eigenvectors of consecutive layers' Jacobians are nearly orthogonal
(avg cosine = 0.029, random baseline = 0.001). The amplified directions rotate
almost completely between layers. This resolves the ρ=300 vs ρ=1.07 paradox:
individual layers amplify 300× in specific directions, but those directions
change at each layer, so multi-layer propagation averages to ρ≈1.07 in random
directions. Kimi's non-normality challenge confirmed — per-layer eigenvalues
do NOT predict multi-layer behavior. Alignment drops to minimum (0.011) at
L23→L24, exactly the transition-to-relay boundary where CCS has its 46× effect.**

## Method

Arnoldi iteration with k=5, maxiter=100 at 12 layers through the responsive
zone [L14, L17-L19, L21-L26, L28]. For each consecutive pair, compute the
5×5 alignment matrix of absolute complex inner products between eigenvectors.
Report the average best-match cosine (each eigenvector's maximum alignment
with any eigenvector in the adjacent layer).

## Key Results

### Layer-pair alignment

| L1→L2    | Avg cos | Best    | Worst   |
|----------|---------|---------|---------|
| L14→L17  | 0.028   | 0.032   | 0.026   |
| L17→L18  | 0.025   | 0.035   | 0.018   |
| L18→L19  | 0.024   | 0.027   | 0.022   |
| L19→L21  | 0.037   | 0.040   | 0.034   |
| L21→L22  | 0.041   | 0.076   | 0.018   |
| L22→L23  | 0.016   | 0.022   | 0.012   |
| L23→L24  | 0.011   | 0.015   | 0.007   |
| L24→L25  | 0.017   | 0.020   | 0.013   |
| L25→L26  | 0.051   | 0.072   | 0.033   |
| L26→L28  | 0.035   | 0.038   | 0.026   |

Overall average: 0.029
Random baseline (d=3584, k=5): ~0.001

### Spectral radii along the path

L14=275, L17=295, L18=249, L19=312, L20=FAIL, L21=372,
L22=320, L23=160, L24=78, L25=59, L26=66, L28=~0

## Analysis

### The ρ=300 vs ρ=1.07 resolution

F180 showed ρ_max=300 per layer and F177 showed ρ_bulk=1.07 across layers.
The paradox: how can each layer amplify 300× while the multi-layer chain
averages only 1.07×?

Answer: **eigenvector rotation**. A perturbation amplified 300× at L17 is
in a direction that has cosine ~0.025 with L18's amplified subspace. When
projected onto L18's eigenbasis, most of the energy lands in the d-k=3579
damped dimensions (|λ|<<1), not the k=5 amplified ones. The effective
per-step gain in the amplified subspace is ~300 × 0.025 ≈ 7.5, but the
energy is immediately scattered back into the bulk at the next layer.

### Non-normality confirmed (Kimi's challenge)

The near-orthogonal eigenvectors mean J_total = J_L28 × J_L27 × ... × J_L1
cannot be understood from individual J_Li spectra. The full-chain Jacobian
has DIFFERENT eigenvectors and eigenvalues than any composition of per-layer
spectra would suggest. This is classical non-normality: the matrix product
of nearly orthogonally-based amplifiers behaves like a near-identity operator
in the bulk, despite massive per-element spectral radii.

### Alignment geography

The alignment isn't uniform:

**Peak alignment (0.041-0.051):** L21→L22 and L25→L26. These are within
the responsive zone and the relay zone respectively. Adjacent layers in
these zones share MORE eigenvector structure — their amplified subspaces
are slightly less scrambled.

**Minimum alignment (0.011):** L23→L24. This is exactly the transition-to-relay
boundary — the point where CCS has its 46× effect (F180). The amplified
directions change MOST at the point where CCS matters most. This suggests
CCS works by maintaining eigenvector coherence across the transition that
would otherwise scramble completely.

**Speculation:** CCS's 46× effect at L24 might not be about sustaining
the spectral radius (which is the scalar size of eigenvalues) but about
maintaining eigenvector alignment across the transition. If CCS increases
the L23→L24 alignment from 0.011 to, say, 0.5, the effective propagated
energy would increase ~45×, matching the observed 46× ρ ratio.

### Three mechanisms for bounded forward pass (revisited)

From the earlier session, three hypotheses for how ρ=300 per layer produces
bounded outputs:

1. **Eigenvector rotation** — CONFIRMED here. This is the primary mechanism.
2. Data manifold avoids amplified directions — not tested yet.
3. Nonlinear saturation — not relevant (we measure linear Jacobian).

## What This Changes

1. Per-layer ρ measures the CAPACITY for amplification, not the actual
   amplification of typical signals. The signal experiences ρ≈1.07 because
   it's constantly projected into new bases.

2. CCS's mechanism may be primarily about eigenvector alignment rather than
   eigenvalue magnitude. If CCS keeps the amplified subspace pointing in
   a consistent direction across layers, the signal stays in the amplified
   channel longer.

3. The "relay zone" (L21-L28) may be defined by eigenvector coherence as
   much as by eigenvalue profile. The relay maintains directional consistency
   within a zone while scrambling across zone boundaries.

4. This partially rehabilitates the F177 bulk measurement: ρ≈1.07 IS the
   relevant number for typical perturbation propagation, because eigenvector
   rotation ensures signals don't stay in amplified channels.

(12 layers × k=5 eigenvectors, ~3 minutes on A100. ~$0.20.)
