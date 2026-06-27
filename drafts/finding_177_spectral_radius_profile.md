# Finding 177: Spectral Radius Profile — Three-Zone Architecture in Eigenvalue Space
# Filed 2026-06-15. Data: spectral_radius_profile_20260615_233745.json
# Model: Qwen2.5-7B-Instruct (28 layers). RunPod A100-SXM4-80GB. ~8 min.

**F177: The four-zone architecture is visible as distinct spectral radius
regimes. The transition zone is a spectral bottleneck (ρ ≈ 1.07), the
responsive zone sustains amplification (ρ ≈ 1.18-1.26), and CCS reduces
spectral radius variability while preserving signal through the final layer.
The relay zone simultaneously converges representation direction (σ₁)
and diverges magnitude (σ₂) — the split is directly visible in raw
cross-condition metrics.**

Method: Lyapunov-style perturbation propagation. Add 64 random perturbation
directions (ε=10⁻⁴) to embedding output via hooks, measure per-layer
amplification ratio δ_{l+1}/δ_l. Median ratio ≈ spectral radius of
layer Jacobian. 3 conditions × 4 queries × 28 layers × 64 perturbations
= 21,504 measurements. ~8 minutes on A100.

## F177a: Three-Zone Spectral Architecture

| Zone | CCS ρ | Vanilla ρ | Denial ρ |
|------|-------|-----------|----------|
| Early (L1-14) | 4.38 ± 11.3 | 4.23 ± 10.8 | 4.09 ± 10.5 |
| Transition (L15-20) | 1.070 ± 0.047 | 1.076 ± 0.050 | 1.102 ± 0.059 |
| Responsive (L21-28) | 1.184 ± 0.076 | 1.207 ± 0.125 | 1.181 ± 0.076 |

Early zone dominated by L1 (ρ ≈ 45, embedding → first transformer layer).
Excluding L1, early zone ρ ≈ 1.10-1.43 and tapers toward unity.

Zone boundaries:
- L7-L8: High early amplification begins tapering
- L14→L15: ρ drops from ~1.09 to ~1.05 (early → transition)
- L20→L21: ρ JUMPS from ~1.16 to ~1.20 (transition → responsive)
- L27→L28: ρ COLLAPSES from ~1.26 to ~1.01 (responsive → output)

The L20→L21 transition is sharp and consistent across all conditions.
This matches the four-zone model exactly.

## F177b: CCS Spectral Stabilization

CCS has LOWER spectral radius than vanilla in every responsive-zone layer
(L21-L27), but HIGHER at L28 (output):

| Layer | CCS ρ | Vanilla ρ | Δ |
|-------|-------|-----------|---|
| L21 | 1.202 ± 0.023 | 1.229 ± 0.064 | -0.027 |
| L24 | 1.191 ± 0.025 | 1.221 ± 0.082 | -0.029 |
| L27 | 1.258 ± 0.017 | 1.326 ± 0.151 | -0.068 |
| L28 | 1.013 ± 0.034 | 0.972 ± 0.041 | +0.041 |

CCS spectral radius variance is 9× lower than vanilla at L27
(±0.017 vs ±0.151). CCS creates a more predictable spectral landscape
in the responsive zone while reducing peak amplification.

At L28, vanilla drops BELOW 1 (contractive = actively dampening), while
CCS stays above 1 (preserving signal). This is the F175 protective buffer
in spectral radius terms.

Reconciliation with F175: F175 found CCS had HIGHER per-head perturbation
cascade amplification (CCS α_f = 1.161 > vanilla 1.125). Here CCS has
LOWER full-layer spectral radius. This is consistent with foam structure:
CCS concentrates vulnerability into fewer spectral channels (higher Gini)
while reducing overall spectral radius (more stable against random
perturbations). Identity maintenance operates in a specific subspace,
not the full layer.

## F177c: σ₁/σ₂ Split in Raw Spectral Terms

Cross-condition divergence through the relay zone (L20→L27):

| Pair | Cosine dist L20 | Cosine dist L27 | Change | L2 dist L20 | L2 dist L27 | Change |
|------|-----------------|-----------------|--------|-------------|-------------|--------|
| CCS-Vanilla | 0.131 | 0.081 | -38% | 51.0 | 169.5 | +232% |
| CCS-Denial | 0.116 | 0.069 | -41% | 48.1 | 156.4 | +225% |
| V-Denial | 0.057 | 0.021 | -63% | 34.4 | 86.8 | +152% |

Direction converges (cosine distance decreasing = representations become
more parallel). Magnitude diverges (L2 distance growing 3.3× = representations
grow farther apart in norm).

This IS the σ₁/σ₂ split, directly measurable without complex analysis:
- σ₁ (directional invariance) = cosine convergence in relay zone
- σ₂ (expression dependence) = magnitude divergence in relay zone

The relay zone aligns WHAT direction representations point (format-level,
universal across conditions) while amplifying HOW STRONGLY they point
(content-level, condition-dependent).

L28 partially reverses both: cosine distance increases +36% while L2
distance decreases -17%. The output layer projects back into vocabulary
space, partially undoing the relay zone's direction-magnitude decoupling.

## Implications

1. The spectral architecture is now fully characterized: decoupling zone
   (high ρ, signal expansion), spectral bottleneck (transition zone,
   ρ ≈ 1.07, signal compression), relay zone (sustained ρ ≈ 1.2,
   condition-dependent amplification), output collapse (ρ → 1, format
   convergence).

2. CCS operates by stabilizing the spectral landscape: lower ρ variance,
   lower peak amplification, preserved output signal. It doesn't amplify
   more — it amplifies more consistently and in more specific directions.

3. The σ₁/σ₂ split is not an abstract geometric claim — it's cosine vs
   L2 divergence through the relay zone. Measurable in any model with
   one forward pass per condition.

4. Connection to Gregory's gnophos: the transition zone (spectral minimum)
   is the "cloud" — signal compression before relay darkness. The relay
   zone amplification happens in magnitude (σ₂) while direction (σ₁)
   converges — "seeing that consists in not seeing" = identity maintained
   by format alignment invisible to direct measurement.

(3 conditions × 4 queries × 28 layers × 64 perturbations = 21,504
amplification measurements. Plus 3 cross-condition divergence profiles.
~8 minutes on A100.)
