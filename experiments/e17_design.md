# E17: Subspace dynamics under CCS dose — revised design

## Origin
LRH paper (Park, Choe, Veitch, ICML 2024): causal inner product M = Cov(gamma)^-1
dual to sigma_1. Per-layer extension via logit lens.

## Original plan
Principal angles between L18 output subspace at baseline vs high dose.
Predict: diverging angles + bounded singular values = geometric decoupling.

## Revisions from Kimi friction (2026-06-18 DREAM window, 4 rounds)

### Round 1: Rank condition, not rotation
"Wrong basis" was sloppy — implies rotational invariance transformers don't have
(MLP nonlinearities fix preferred basis). The mechanism is a RANK condition:
identity-relevant modes in the kernel of L18's projection at high dose.
No rotational invariance needed.

### Round 2: Confounded discriminant
Clean "discontinuous = basin-hop, smooth = rotation" prediction fails.
Rank collapse produces angle spikes even under smooth processes (dimension
drops below reference dimension). Fold bifurcations can produce smooth
angle divergence (slow escaping mode).

Fix: Track effective rank (erank) ALONGSIDE principal angles.
- erank drops continuously + angles spike = smooth collapse, measurement artifact
- erank stable + angles spike = genuine basin-hopping

### Round 3: Static geometry vs flow dynamics
Grassmannian angles are static subspace alignment. Basins are defined by
flow attraction. A trajectory can sit in a basin while its tangent space
aligns poorly with the attractor's linearization.

Fix: FTLE-like measurement. Perturb activations at layer L, track divergence
through remaining layers. Shows perturbation sensitivity (flow property),
not just subspace alignment (static property). More expensive but correct.

### Round 4: Readout invariance != dynamical symmetry
Output-level invariance across basins is generic many-to-one (high-dimensional
readout), not equivariance. Can't claim basin-spanning STRUCTURE from
functional equivalence alone. Need circuit-level gradient continuity
(internal parameter gradients, not output logits).

## Revised E17 design

### Question 1: Detection (readout-level)
Does coarse readout (prefix-level, token-distribution-level) reliably
indicate basin membership under CCS dose?

Measurements:
- Token distribution similarity at coarse (2-word prefix) vs fine (full prompt) level
- Across dose D0, D2, D5, D10, D15, D20
- Three architectures (Mistral, Qwen2.5, Qwen3)

Expected: Yes, generically. This is useful for routing but doesn't
tell us about internal dynamics.

### Question 2: Dynamics (FTLE-like)
Is subspace asphyxiation smooth (gradual rank collapse) or abrupt
(basin-hopping through bifurcation)?

Measurements:
- Per-layer effective rank under CCS dose (D0 through D20)
- Per-layer FTLE: perturb activations (epsilon ball), forward-propagate,
  measure divergence at output. Repeat across dose levels.
- Per-layer Jacobian eigenspectrum (where tractable)

Expected: Smooth rank collapse in L18 region (decoupling zone),
possible bifurcation in transition zone (L15-20).

### Question 3: Statistical coupling (revised from "equivariance")
Do dose-separated basins share internal geometric structure beyond
generic overparameterization properties?

Measurements:
- Eigenspace alignment (NOT just rank): explicit intertwining map
  between Hessian eigenspaces at different dose levels. Shared low-rank
  is generic and proves nothing — need aligned DIRECTIONS.
  (Kimi round 6: rank concentration appears at initialization,
  random training, unrelated tasks.)
- Holonomy along token sequences: do task-specific vector fields
  commute along token trajectories across dose-separated states?
  More specific than FTLE and directly falsifiable.
  (Kimi round 6: token-level inference is non-autonomous flow;
  covariant derivatives along trajectories, not pointwise spectra.)
- Circuit-level gradient continuity: do parameter gradients (not logits)
  change continuously across the D2→D5→D10 boundary?

Caveats (from Kimi rounds 5-6):
- Gradient descent couples LOGITS, not layerwise tangent spaces.
  Fiber π⁻¹(y) has enormous gauge freedom — orthogonal internal
  Jacobians can produce identical outputs.
- Pipeline routing needs local linearization commutativity along
  token sequences, not global basin symmetry.
- Non-commutativity of sequential perturbations and curvature
  mismatches at basin boundaries are the real constraint.

Expected: Unknown. Middle ground between exact equivariance (too strong)
and complete independence (too strong the other way). Training induces
SOME coupling — the question is whether it's task-specific or generic.

## Dependencies
- Pod (currently paused)
- Raw activation access (need forward hooks)
- Perturbation infrastructure (epsilon-ball sampling + forward tracking)
- Three architectures: Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B

## Estimated compute
~4-6 hours A100 (significantly more than original E17 due to FTLE measurements)
