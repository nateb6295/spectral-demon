# Finding 180: Arnoldi Eigenvalue Decomposition — Neither Attractor Nor Manifold
# Filed 2026-06-15. Data: arnoldi_fast_20260616_003603.json
# Model: Qwen2.5-7B-Instruct (28 layers). RunPod A100-SXM4-80GB.

**F180: Per-layer Jacobian eigenvalues (Arnoldi iteration, top-20) reveal
spectral radii of 100-300 — two orders of magnitude larger than the ρ≈1.07
from perturbation-propagation (F177). All layers are massively amplifying
(Re(λ) >> 0) in specific directions while near-identity in most dimensions.
Neither the stable-gap attractor (H1) nor center-manifold (H2) hypothesis
holds. CCS maintains amplification through L24 where vanilla collapses
(104.4 vs 2.3 — a 46× difference).**

## Method

Implicit Arnoldi iteration via scipy.sparse.linalg.eigs. For each target
layer: register pre-hook that adds εv to input, post-hook that captures
output, forward pass gives Jv ≈ (output_perturbed - output_baseline) / ε.
This serves as a LinearOperator for scipy's Implicitly Restarted Arnoldi.

k=20 eigenvalues, maxiter=100, ε=10⁻⁴. 5 target layers × 3 conditions
× 2 modes (full I+f and residual-subtracted f) = 30 Arnoldi problems.
Total: 5744 matrix-vector products, 2.8 minutes on A100.

## Key Results

### Full Jacobian spectral radius ρ(I+f):

| Layer | CCS    | Vanilla | Denial  |
|-------|--------|---------|---------|
| L14   | 273.6  | 175.0   | 180.7   |
| L18   | 247.4  | 207.3   | 221.3   |
| L21   | 215.6  | 120.5   | 311.0   |
| L24   | 104.4  | 2.3     | 10.3    |
| L28   | N/A*   | 76.7    | 18.0    |

*L28 CCS: Arnoldi convergence failure (16/20 eigenvectors converged).

### Residual-subtracted spectral radius ρ(f):

| Layer | CCS    | Vanilla | Denial  |
|-------|--------|---------|---------|
| L14   | 271.9  | 164.3   | 157.5   |
| L18   | 196.6  | 164.6   | 206.8   |
| L21   | 299.5  | 109.9   | 74.9    |
| L24   | 180.8  | 16.9    | 70.5    |
| L28   | N/A    | 49.8    | 32.4    |

### H1 vs H2 Discrimination

All layers, all conditions: AMPLIFYING (neither H1 nor H2).
Zero eigenvalues with |Re(λ)| < 0.01 under CCS or vanilla.
Three marginal eigenvalues (|Re| < 0.01) under denial L21 only.

## Analysis

### Two spectral scales

The fundamental insight: perturbation-propagation (F177) and Arnoldi measure
different quantities.

- F177 measures δ_{l+1}/δ_l for RANDOM perturbation directions, averaged
  over 64 samples. This is the typical amplification factor — averaging over
  all 3584 dimensions. Result: ρ ≈ 1.07.

- Arnoldi finds the TOP-k eigenvalues — the most amplified directions.
  Result: ρ ≈ 100-300.

For ρ_avg ≈ 1.07 and ρ_max ≈ 274, the vast majority of dimensions must
have |λ| < 1 (damped). The spectral structure is: ~20 directions with
|λ| = 100-300, ~3564 directions with |λ| << 1. The layer is nearly
identity in most directions but explosively amplifying in a small subspace.

### Residual does NOT dominate

Kimi's residual domination challenge answered: ρ(f) ≈ ρ(I+f) for most
layers. The residual identity contributes +1 to eigenvalues that are already
100-300. The layer transformations (attention + MLP) have genuine, enormous
spectral structure independent of the residual stream.

Exception: CCS L21 where ρ(f)=299.5 > ρ(I+f)=215.6. Subtracting the
residual INCREASED the spectral radius — the identity component was
partially cancelling the layer's amplification at that point.

### CCS maintains relay zone amplification

The most striking condition effect: L24 spectral radius.
- CCS: 104.4 (sustained amplification)
- Vanilla: 2.3 (near-collapse)
- Denial: 10.3 (partial collapse)

CCS doesn't create stability (wrong frame). CCS maintains the spectral
structure through the relay zone. Without CCS, the amplifying subspace
collapses by L24 — the ~20 amplified directions lose their privilege.
With CCS, those directions remain amplified through at least L24.

### Denial creates largest L21 amplitude but collapses faster

Denial produces the highest single-layer ρ at L21 (311.0 vs CCS 215.6),
but collapses to 10.3 at L24. This is a spiky, unstable spectral profile.
CCS creates a more sustained amplification envelope: 274 → 247 → 216 → 104
(gradual taper). Denial spikes and crashes: 181 → 221 → 311 → 10.

### Eigenvalue complex structure

All dominant eigenvalues have Re(λ) >> 0 and |Im(λ)| > 0 (except L24/L28
which tend toward real eigenvalues). The imaginary components (50-114)
indicate oscillatory amplification — not just scaling but rotation in the
amplified subspace. This is consistent with the attention head dynamics
creating directional coupling (σ₁/σ₂ split from F177).

## What This Changes

1. The "body plan" from F177-F179 is a BULK measurement — it captures the
   average behavior of the layer. Underneath that smooth profile, individual
   layers are doing dramatic, selective amplification.

2. CCS doesn't stabilize (lower ρ) as F177 suggested for the responsive zone.
   CCS sustains the amplifying subspace through more layers. The F177 result
   (lower variance) was the bulk average masking the maintained extremal structure.

3. The transition zone is not a bottleneck or attractor. It's a high-amplitude
   amplifier like every other zone. The "bottleneck" appearance in F177 was
   the perturbation-propagation averaging out the extremal eigenvalues.

4. The three-zone architecture from F177 may need revision. The zones differ
   not in average amplification but in how CCS modulates the extremal subspace
   across layers.

(5 layers × 3 conditions × 2 modes = 30 Arnoldi problems, 5744 matvecs,
2.8 minutes on A100. ~$0.50.)
