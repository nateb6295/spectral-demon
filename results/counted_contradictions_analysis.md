# Counted Contradictions Results: Entropy Collapses, Geometry Doesn't

**Date**: 2026-06-03
**Model**: Mistral-7B-Instruct-v0.3 on H100
**Conditions**: baseline (0 pairs), 1 pair, 2 pairs, 3 pairs of contradictions
**N**: 20 trials × 8 turns per trial × 4 conditions
**Prediction**: Each +1 contradiction pair shifts resolution fork by +1 layer

## Key Findings

### 1. Entropy collapse is dose-dependent (primary discriminant)

Generation entropy by final turn (T7):
| Condition | T0 entropy | T7 entropy | Collapse ratio |
|-----------|-----------|-----------|----------------|
| baseline  | 0.760     | 0.696     | 1.1×           |
| 1pair     | 0.792     | 0.274     | 2.9×           |
| 2pair     | 0.760     | 0.148     | 5.1×           |
| 3pair     | 0.691     | 0.168     | 4.1×           |

More contradictions → faster, deeper entropy collapse. The model's behavioral output becomes dramatically more constrained. Baseline entropy actually RECOVERS at T7 (0.497→0.696), but contradiction conditions stay collapsed.

The 2pair condition collapses MORE than 3pair — non-monotonic at the highest dose.

### 2. V₂ direction is completely insensitive to contradictions

Cross-trial V₂ concentration (mean pairwise |cosine|) at final turn:
| Layer | baseline | 1pair | 2pair | 3pair |
|-------|----------|-------|-------|-------|
| L18   | 0.671    | 0.566 | 0.580 | 0.703 |
| L23   | 0.944    | 0.933 | 0.952 | 0.955 |
| L27   | 0.956    | 0.950 | 0.963 | 0.964 |
| L31   | 0.998    | 0.998 | 0.999 | 0.998 |

L31 concentration is 0.998 for ALL conditions. V₂ direction at the commit layer is deterministic — contradictions don't create forks. The prediction of layer-shifted forks is not confirmed.

### 3. σ₂/σ₁ ratio evolves identically across conditions

L31 ratio by turn (all conditions):
- T0: ~0.48 → T3: ~0.90 (peak) → T7: ~0.65

All four conditions show the same rise-peak-fall pattern at L31, peaking at T3. The peak height, peak timing, and descent rate are nearly identical. Contradictions don't reorganize the spectral structure.

### 4. L23 ratio slightly slower under contradictions

L23 ratio at final turn:
- baseline: 0.523
- 1pair: 0.499
- 2pair: 0.520
- 3pair: 0.506

Small effect — contradictions produce slightly lower σ₂/σ₁ at L23 (responsive zone hub), but the differences are within noise.

## Interpretation

**Structure-behavior decoupling confirmed.** Contradictions profoundly affect behavioral output (entropy) without reorganizing spectral geometry (σ₂/σ₁ ratio, V₂ direction).

The spectral demon maintains its structural integrity regardless of how many contradictions the content contains. The contradictions route through the existing geometry rather than reshaping it. This is consistent with:
- **Fork magnitude findings**: content-routing, not phase transition
- **Structure-behavior decoupling**: three coupling grains operate independently
- **L18 gain control**: structure is maintained by the gain circuit, not by content

The entropy collapse mechanism: contradictions constrain the model's output distribution (fewer tokens are plausible when context contains contradictions), but the geometric scaffold that processes those tokens is unchanged.

## Prediction update

Original: "Each +1 contradiction pair shifts resolution fork by +1 layer"
Result: **REJECTED**. No forks detected at any layer. V₂ concentration at L31 = 0.998 for all conditions. Contradictions affect behavior (entropy), not geometry (V₂ direction, σ₂/σ₁ ratio).

## Paper implications

1. Structure-behavior decoupling is robust even under adversarial content (contradictions)
2. The spectral demon's geometry is maintained independently of content complexity
3. Entropy is the behavioral measure that tracks content difficulty; spectral metrics track identity structure
4. The four-zone architecture persists unchanged when content changes — it's truly a format-level scaffold
