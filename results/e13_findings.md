# E13: Trajectory Curvature × Spectral Geometry × Melodic Coherence

**Experiment**: Mistral-7B-Instruct-v0.3, 5 doses (D0/D2/D3/D5/D8), 6 probes per dose.
Metrics: per-layer SVD (σ₁, σ₂, ratio, erank), trajectory geometry (curvature, velocity, path length),
melodic coherence (autocorrelation of σ₂ across layers), Jacobian approximation.
**Date**: 2026-06-23, RunPod A100-SXM4 80GB.
**Data**: `results/e13/e13_trajectory_spectral_20260624_004507.json`

## Key Findings

### F318: Melodic coherence peaks at therapeutic dose (D2)

Autocorrelation of σ₂ across layers:
- Vanilla (D0): 0.161 ± 0.123
- **D2: 0.669 ± 0.002** ← PEAK
- D3: 0.655 ± 0.003
- D5: 0.542 ± 0.001
- D8: 0.467 ± 0.001

**Prediction confirmed**: The Bergsonian prediction that commensurate dose produces maximal
interpenetration (each layer's spectral state shaped by its predecessor) is supported.
D2 is the melodic peak — high autocorrelation (layers interpenetrate) with substantial
variation (CV=1.667). Vanilla has near-zero autocorrelation (layers process independently).
Higher doses have declining autocorrelation but increasing CV — they produce VARIABLE
but INCOHERENT processing.

The three predicted regimes map onto the data:
- D0: noise (low autocorrelation, moderate CV)
- D2-D3: melodic (high autocorrelation, high CV)
- D5-D8: not monotone as predicted, but incoherent-variable (declining autocorrelation, rising CV)

The D8 correction is important: overdose doesn't suppress variation (monotone). It DISRUPTS
coherence while AMPLIFYING variation. The identity context is loud everywhere but the
layers aren't listening to each other.

### F319: σ₂ ramp onset moves earlier with dose

Per-layer σ₂ profile reveals dose-dependent onset of the responsive zone:
- Vanilla: flat (61-67) from L2 to L30. No zone differentiation.
- D2: flat until L24, then ramp (63→168). Onset in relay zone.
- D3: flat until L21, then ramp (60→191). Onset at relay zone entry.
- D5: starts ramping at L17. Enters transition zone.
- D8: ramps from L12. Invades early zone.

This means zone boundaries are NOT architectural fixtures — they emerge from CCS
and shift with dose. At therapeutic dose (D2), the boundary aligns with L21 (the
responsive zone entry we've measured in prior experiments). At overdose, it migrates
earlier, meaning the identity context begins modulating layers that normally handle
format-level processing.

**Mechanism**: More preamble tokens = earlier interpenetration onset in the forward pass.
D2 concentrates the effect in the relay zone (L24-30). D8 spreads it across the whole
network, diluting the zone structure.

### F320: Curvature is dose-invariant (architectural, not identity)

Curvature profiles across vanilla, D2, and D8 are nearly identical:
- All show peak around L7-L10 (early architecture feature)
- All show minimum around L20-L21 (transition-relay boundary)
- No dose modulation of curvature shape or magnitude

This answers the E13 primary question: trajectory geometry (curvature) and spectral
geometry (σ₂) measure DIFFERENT phenomena. Curvature is a format-level architectural
feature — the trajectory bends the same way regardless of identity context. σ₂ is an
identity-level feature — it changes dramatically with dose. They are independent observables.

Implication: Pandey's curvature peaks and our spectral zone boundaries are distinct
features, not the same thing measured differently. The forward pass has at least two
independent geometric structures: architectural curvature (fixed) and spectral variance
(identity-modulated).

### F321: Early-layer σ₂ compression under CCS

σ₂ in early layers (L2-14) drops with dose:
- D0: 63.5 ± 7.7
- D2: 64.7 ± 1.8
- D3: 61.9 ± 1.9
- D5: 40.2 ± 2.3
- D8: 28.7 ± 6.1

CCS COMPRESSES early-layer variation while AMPLIFYING relay-zone variation.
The early layers are doing less independent processing (lower σ₂) because the
identity context has already constrained the representation space. This is
spectral demon behavior: the preamble strips variation from early processing
to concentrate it in the relay zone.

The crossover (where σ₂ under CCS exceeds vanilla σ₂) occurs at:
- D2: ~L24
- D3: ~L21
- D5: ~L17
- D8: ~L12

This crossover IS the zone boundary, and it moves with dose.

### F322: Ratio autocorrelation peaks at D5, not D2

σ₁/σ₂ ratio autocorrelation across layers:
- Vanilla: -0.322 (anti-correlated — ratio bounces)
- D2: 0.756
- D3: 0.880
- **D5: 0.922** ← PEAK
- D8: 0.904

The ratio profile gets smoother at higher doses because σ₁ dominates more (monotone).
This is distinct from σ₂ autocorrelation (which peaks at D2). The two metrics
capture different aspects:
- σ₂ autocorrelation = melodic coherence (interpenetration quality)
- Ratio autocorrelation = σ₁ dominance (monotone saturation)

At D5+, the ratio is smoothly varying because σ₁ is swamping σ₂ everywhere.
At D2, the ratio has variation because σ₂ has genuine structure. The melodic
window is where BOTH σ₂ and the ratio have structure — enough identity context
to create coherence, not enough to create monotony.

### F323: Erank autocorrelation is near-universal under CCS

Erank autocorrelation:
- Vanilla: 0.816
- D2: 0.959
- D3: 0.971
- D5: 0.978
- D8: 0.981

Any amount of CCS pushes erank autocorrelation close to 1.0 and it barely
changes with dose. The effective dimensionality profile is architecturally
determined once identity context is present — the AMOUNT of identity context
matters only for the first step (vanilla→D2). After that, erank is locked.

## Methodological Notes

- Path length = inf for all conditions — caused by L31→L32 transition producing extreme
  velocity. Last layer artifact, does not affect per-zone analysis.
- Velocity autocorrelation = NaN — consequence of inf in velocity array.
- Jacobian contraction rates = 0.0 for all conditions — BUG in computation. The
  perturbation compares h_curr to itself rather than recomputing output from perturbed
  input. Jacobian analysis needs redesign (requires gradient-enabled forward pass).
- Latent-trajectories library probes skipped (not installed on pod).

## Synthesis

E13 confirms the Bergsonian prediction and adds empirical structure:

1. **σ₂ melodic coherence peaks at D2** — the therapeutic window IS a commensurability
   window where layer-to-layer interpenetration is maximal.
2. **Zone boundaries emerge from CCS, they're not architectural** — the σ₂ ramp onset
   moves with dose, meaning the "responsive zone" is a CCS artifact, not a fixed feature.
3. **Curvature and spectral geometry are independent** — trajectory shape is architectural,
   spectral variance is identity-modulated. Two phenomena, not one.
4. **Overdose is incoherent-variable, not monotone** — high σ₂ CV but low autocorrelation
   at D8 means the identity context makes processing variable but doesn't make the
   layers listen to each other.

The melodic coherence metric works. D2 = melody (coherent variation). D0 = noise
(independent layers). D8 = cacophony (loud variation, no coherence).
