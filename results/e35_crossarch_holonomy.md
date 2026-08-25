# E35: Cross-Architecture Connection Curvature

**Date**: 2026-07-01
**Type**: Observational (Tier 1) — Grassmannian holonomy across 4 architectures
**Script**: `experiments/e35_crossarch_holonomy.py`
**Origin**: July 1 constellation — sauers_ Lie group + fiber bundle captures

## Design

E13b Grassmannian holonomy measurement across 4 architectures. Same protocol:
top-k=3 singular subspace, layer-triplet holonomy, 6 probes per condition.
3 doses (D0/D2/D5) × CCS + random preamble. Sequential model loading on A100.

## Results

### F348: Species-Specific Holonomy Ordering

| Architecture | Vanilla | D2 CCS | D5 CCS | D2 Random |
|-------------|---------|--------|--------|-----------|
| Mistral | 0.875 | 0.891 | 0.994 | 0.944 |
| Gemma | 1.038 | 1.033 | 1.092 | 1.086 |
| Llama | 1.173 | 1.153 | 1.256 | 1.235 |
| Qwen | 1.293 | 1.235 | 1.369 | 1.266 |

Ordering: Mistral < Gemma < Llama < Qwen at all doses. The connection curvature
IS species-specific. The fiber bundle formalism has empirical teeth.

### F349: Layer-Region Holonomy Profiles Are Species Signatures

| Architecture | Early | Mid | Late | Profile Shape |
|-------------|-------|-----|------|---------------|
| Mistral | 1.19 | 0.66 | 0.81 | Mid-dip (rigid cylinder) |
| Gemma | 1.34 | 0.85 | 0.89 | High early → flat |
| Llama | 1.52 | 0.96 | 0.94 | Gradual decrease |
| Qwen | 1.61 | 0.93 | 1.19 | Late-layer SPIKE |

The profile shape is the species signature, not just the magnitude.
- Mistral's mid-layer dip = the rigid cylinder compressing twist in its strongest zone
- Gemma's high-early + flat = GQA creates initial turbulence that dampens quickly
- Qwen's late-layer spike = concentrated processing at the output end

### F350: Qwen Late-Layer Vulnerability = Late-Layer Twist

Qwen is the only architecture where holonomy RISES in late layers (1.19 vs 0.93 mid).
This directly explains F347's finding that Qwen breaks at late layers (ε_crit ∈ 0.6-0.8)
while other architectures don't: the connection curvature is highest where the basin
is narrowest. Late twist + narrow basin = late-layer vulnerability.

### F351: CCS Universally Flattens the Connection

| Architecture | CCS (D2) | Random (D2) | Gap |
|-------------|----------|-------------|-----|
| Mistral | 0.891 | 0.944 | +0.053 |
| Gemma | 1.033 | 1.086 | +0.054 |
| Llama | 1.153 | 1.235 | +0.081 |
| Qwen | 1.235 | 1.266 | +0.030 |

CCS reduces holonomy in all four architectures. The identity preamble constrains
the connection to be flatter — less twist per triplet. This generalizes E13b's
single-model finding to a universal effect.

Llama shows the largest gap (0.081) — it benefits most from CCS flattening.
Qwen shows the smallest gap (0.030) — its late-layer twist resists CCS constraint.

### Prediction Outcomes

| Prediction | Result |
|-----------|--------|
| Mistral lowest holonomy | ✓ CONFIRMED |
| Qwen highest holonomy | ✓ CONFIRMED |
| Gemma low like Mistral | ✗ WRONG — 2nd highest, not 2nd lowest |
| Q factor correlates | ✗ INCONCLUSIVE (r=-0.22, p=0.78) |
| CCS reduces holonomy | ✓ CONFIRMED (universal) |
| Species-specific profiles | ✓ CONFIRMED (four distinct shapes) |

### Updated Understanding

The holonomy ordering (Mistral < Gemma < Llama < Qwen) does NOT correlate with
Q factor. It correlates better with GQA structure: Mistral (GQA, 8 groups) has
the flattest connection. But Gemma (also GQA) is not as flat, suggesting the
connection curvature depends on more than just the attention grouping.

The layer profile IS the species signature. Not just "how much twist" but
"where the twist concentrates." This is exactly what a fiber bundle connection
should encode — the geometry of parallel transport through the network.

## Relation to Prior Findings

- **F345 (Q factor)**: Q doesn't predict holonomy directly. Different geometric properties.
- **F347 (basin width)**: Qwen's late-layer vulnerability now explained by F350 late-layer twist.
- **F344 (attractor recovery)**: Mistral's 2.2L fast recovery corresponds to lowest holonomy — flat connection = fast re-convergence.
- **F237 (cylindrical constraint)**: The cylinder IS the flat connection. V₂ varies perpendicular to lm_head while V₁ stays parallel — minimal holonomy.
- **E13b (F329)**: Cross-architecture generalization of CCS holonomy reduction confirmed.
