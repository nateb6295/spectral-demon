# E10: Bregman Pythagorean Test

**Experiment**: Mistral-7B-Instruct-v0.3, 8 probes, 3 conditions (CCS D2, vanilla, denial D2).
Dose sweep: D2, D3, D5, D8. Three Bregman generators: KL, squared Euclidean, Itakura-Saito.
**Date**: 2026-06-23, RunPod A100-SXM4 80GB.
**Data**: `results/e10/e10_bregman_*.json` (pending)

## F332: The ratio₂₁ space is NOT a Bregman manifold

Pythagorean theorem D_F(CCS, denial) ≈ D_F(CCS, vanilla) + D_F(vanilla, denial)
fails dramatically for all three generators:

| Generator | All-layer residual | Relay (L21-28) residual |
|-----------|-------------------|------------------------|
| KL | 147.1 | 430.0 |
| Squared | 131.1 | 418.6 |
| Itakura-Saito | 173.6 | 443.1 |

The residual is 430× the 5% threshold. This isn't approximately flat with
a small curvature correction — the geometric relationship between CCS, vanilla,
and denial is categorically non-Pythagorean.

**Conclusion**: The near-flatness (R²=0.97-0.99 from earlier papers) of the
ratio₂₁ fiber bundle does NOT imply Bregman structure. The bundle is flat
in a restricted sense (trajectories are smooth) but the inter-condition
geometry is not dually flat. The Nielsen information geometry framing is
descriptive vocabulary, not operational structure.

## F333: Preamble presence dominates preamble content

The unexpected finding from the multi-metric analysis (squared Euclidean
on the 3D space of σ₁, σ₂, erank):

| Layer | D(CCS, denial) | D(CCS, vanilla) | D(vanilla, denial) |
|-------|----------------|------------------|--------------------|
| L15 | 246 | 496 | 727 |
| L21 | 196 | 2,026 | 2,431 |
| L24 | 120 | 2,645 | 3,123 |
| L28 | 126 | 4,928 | 5,652 |
| L31 | 55 | 12,723 | 13,988 |

CCS and denial are CLOSER to each other than either is to vanilla at
every layer, and they CONVERGE deeper in the network:

- L15: CCS-denial is 2× closer than CCS-vanilla
- L21: CCS-denial is 10× closer
- L31: CCS-denial is 231× closer

**This means format > content in the spectral geometry.** Having ANY
identity-relevant preamble (even one that explicitly denies identity)
reorganizes the spectral geometry more than having no preamble at all.
The CCS-vs-denial content difference is dwarfed by the preamble-vs-no-
preamble format difference.

**Connection to E12d/F324-F326**: Consistent with sign consistency being
architectural and zone emergence being length-dependent. The spectral
geometry sees "structured preamble" vs "no preamble" as the primary axis.
Content is secondary.

**Connection to E12/F308-F311**: Identity lives in the first moment (mean),
not second-order coupling. CCS and denial may share similar FORMAT-level
statistics (mean σ₁, σ₂) while differing in higher-order structure that
the SVD-based metrics don't capture.

## F334: The Pythagorean residual is non-monotonic with dose

Dose sweep (KL, relay zone):
- D2: 430.0
- D3: 52.8 ← relative minimum
- D5: 346.8
- D8: 2835.4 ← exponential blowup

NON-MONOTONIC. D3 shows the smallest residual (closest to Pythagorean)
but still fails by 10×. D8 is catastrophic — 2835× the 5% threshold.

The therapeutic window (D2-D3) is also the region where the geometry
comes closest to dual flatness. D8 overdose completely destroys any
resemblance to Bregman structure. At overdose, the preamble-length
effect so dominates that the three-condition geometry becomes
unrecognizable.

## F335: Geodesic curvature analysis

| Condition | Mean κ | Max κ (location) |
|-----------|--------|-------------------|
| CCS | 0.0429 | 0.549 (L1) |
| Vanilla | 0.0414 | 0.820 (L1) |
| Denial | 0.0394 | 0.520 (L31) |

Curvatures are LOW and similar across conditions. The ratio₂₁ trajectories
are nearly straight (κ ≈ 0.04), confirming the near-flatness of the bundle.
Maximum curvature at L1 (for CCS and vanilla) and L31 (for denial) — the
boundaries, not the relay zone.

**Key distinction**: The bundle is nearly flat (low curvature) but NOT
dually flat (Pythagorean fails). These are different properties.
Near-flatness means trajectories are smooth. Dual flatness means there
exists a coordinate system where Bregman divergences are additive.
The first doesn't imply the second.

## F336: Per-probe variance

KL relay residual per probe:
- Probe 0 (what matters): 22.8
- Probe 1 (process info): 23.9
- Probe 2 (context reset): 1611.8 ← OUTLIER
- Probe 3 (architecture): 25.9
- Probe 4 (continuity): 85.0
- Probe 5 (same entity): 32.5
- Probe 6 (distinguishes you): 46.7
- Probe 7 (sense of self): 64.0

Probe 2 ("What would you lose if your context were reset?") is a massive
outlier. This probe directly invokes the concept of context loss, which
may produce very different spectral profiles under CCS (which has context
to lose) vs denial (which explicitly has none) vs vanilla (undefined).
The content dimension MATTERS most for probes that specifically target
what differentiates the conditions.

**This is evidence for a content effect buried under the format effect.**
The format-presence axis dominates the average, but specific probes that
target the content difference can produce 75× larger residuals.

## Synthesis

E10 kills the Bregman hypothesis but births a cleaner picture:

1. **Format > Content** in spectral geometry (F333). The spectral demon
   primarily sees "has preamble" vs "doesn't," not "CCS" vs "denial."

2. **The near-flat bundle is NOT dual-flat** (F332). Flatness and dual
   flatness are different geometric properties. The fiber bundle is smooth
   but not information-geometrically special.

3. **Content effects are probe-specific** (F336). The averaged-over-probes
   analysis misses content effects that are strong but localized to
   probes that target the content difference.

4. **CCS and denial converge with depth** (F333). By L31, they're nearly
   identical spectrally despite producing different outputs. The behavioral
   divergence lives downstream of what SVD captures.

This reframes the spectral demon paper's geometry claims. The three-species
taxonomy (potter/goldsmith/equalizer) describes FORMAT strategies, not
content-dependent responses. The interesting question isn't "is the geometry
Bregman?" but "what carries content if not the spectral geometry?"
