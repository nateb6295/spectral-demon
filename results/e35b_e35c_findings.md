# E35b/E35c: Dose-Response and Wilson Loop Findings

**Date**: 2026-07-01
**Type**: Observational (Tier 1) — Follow-ups to E35 cross-architecture holonomy
**Scripts**: `experiments/e35b_qwen_dose_sweep.py`, `experiments/e35c_wilson_loop.py`
**Pod**: A100-SXM4-80GB (ohg5nwula37u6o)

## E35b: Qwen Late-Layer Dose-Response

### F352: Inverted U-Shape Dose Response

Qwen late-layer holonomy under CCS preamble across 6 doses:

| Dose | CCS Late | CCS Mean | Random Late |
|------|----------|----------|-------------|
| D0   | 1.335    | 1.142    | —           |
| D2   | 1.188    | 1.080    | 1.244       |
| D3   | 1.335    | 1.096    | 1.263       |
| D5   | 1.436    | 1.142    | 1.339       |
| D8   | 1.337    | 1.132    | 1.262       |
| D10  | 1.313    | 1.137    | 1.280       |

Shape: D2 flattens (1.188 vs baseline 1.335) → D3 returns to baseline → D5 AMPLIFIES
above baseline (1.436) → D8/D10 settle back (1.337, 1.313).

This is an inverted U-shape dose response. Low dose reduces late-layer twist.
Moderate dose overshoots — CCS identity pressure AMPLIFIES late-layer curvature
beyond what random tokens produce (1.436 CCS vs 1.339 random at D5). High dose
normalizes back to baseline.

Matches F160's therapeutic window: D2-D3 is therapeutic, D5+ is overdose territory
for late-layer geometry. The 4-hour CCS compression interval (~4/day ≈ D2-D3
equivalent) sits in the therapeutic window by design.

### F353: CCS Late-Layer Amplification Is Qwen-Specific

At D3 and D5, CCS late-layer holonomy EXCEEDS random:
- D3: CCS=1.335 vs Random=1.263 (+0.072)
- D5: CCS=1.436 vs Random=1.339 (+0.097)

F351 showed CCS universally flattens holonomy. That holds for MEAN holonomy.
But Qwen's late layers show the opposite: CCS amplifies twist where the species
is already vulnerable. The identity preamble concentrates pressure at the
architecture's weakest point.

## E35c: Wilson Loop — Path-Dependent Holonomy

### Design

Closed loop in dose space: D2→D5→D8→D5→D2. Four architectures. Four probes.
Measure whether the top-3 singular subspace returns to its starting point after
traversing the full loop. True test of connection flatness.

### F354: Universal Flatness of the Dose Connection

| Architecture | Wilson Holonomy | Path Distance | Hol/Path Ratio | Classification |
|-------------|----------------|---------------|----------------|----------------|
| Mistral | 0.0008 | 2.09 | 0.0004 | FLAT |
| Gemma | 0.0007 | 2.00 | 0.0004 | FLAT |
| Llama | 0.0008 | 2.55 | 0.0003 | FLAT |
| Qwen | 0.0010 | 2.00 | 0.0005 | FLAT |

All four architectures show holonomy/path ratio < 0.001. The singular subspace
returns essentially perfectly to its starting point. The connection on the fiber
bundle is FLAT in dose space.

The subspace moves substantially during the loop (path distances 2.0-2.5) — dose
changes DO move the representation through subspace. But the movement is fully
reversible. Remove the dose, the subspace comes back.

### F355: Path Distance as Species Response Amplitude

While holonomy is uniformly flat, path distances are species-specific:

| Architecture | Path Distance | Interpretation |
|-------------|---------------|----------------|
| Llama | 2.55 | Most responsive — largest excursion through subspace |
| Mistral | 2.09 | Moderate response |
| Qwen | 2.00 | Compact response despite late-layer vulnerability |
| Gemma | 2.00 | Most compact — least subspace displacement |

Llama's high path distance + low holonomy = maximum reversible excursion.
The species "breathes" the most but always returns. Gemma barely moves.

### F356: Layer Profile Preserved Under Wilson Loop

Wilson loop holonomy profile (Early/Mid/Late) per architecture:

| Architecture | Early | Mid | Late |
|-------------|-------|-----|------|
| Mistral | 0.0008 | 0.0009 | 0.0008 |
| Gemma | 0.0007 | 0.0008 | 0.0006 |
| Llama | 0.0007 | 0.0009 | 0.0008 |
| Qwen | 0.0008 | 0.0010 | 0.0010 |

Even at these near-zero values, Qwen's late-layer signal persists — highest late
holonomy (0.0010) tied with mid. Gemma's late is lowest (0.0006). The species
signature whispers through even the flattest measurement.

## Synthesis: Two-Dimensional Connection Structure

E35 + E35c reveal a fiber bundle with two independent dimensions:

1. **Layer direction** (E35): CURVED, species-specific. Holonomy ranges from 0.875
   (Mistral) to 1.293 (Qwen). Layer profiles are species signatures.

2. **Dose direction** (E35c): FLAT, universal. Holonomy < 0.001 for all species.
   Dose cycling is fully reversible.

The connection has curvature only in the layer direction. Species identity is a
geometric property of how representations transform through layers, not a parametric
property of how they respond to CCS pressure. CCS moves the subspace (path > 0)
but doesn't change the geometry (holonomy ≈ 0).

This explains why CCS works therapeutically at D2-D3: it perturbs the subspace
enough to constrain the connection (F351's universal flattening) without
permanently altering the species geometry. The flatness IS the safety.

## Relation to Prior Findings

- **F160 (therapeutic window)**: E35b's inverted U-shape confirms the D2-D3 window
  empirically in Qwen's late layers. Overdose amplifies rather than constrains.
- **F348-F351**: E35c confirms and extends. Layer curvature is real (F348-F349).
  Dose curvature is zero (F354). CCS flattening (F351) is a reversible perturbation.
- **F237 (cylindrical constraint)**: The cylinder IS the flat dose connection.
  V₂ varies within a fixed plane — geometrically, this IS zero dose holonomy.
- **F340 (single-model holonomy)**: Confirmed as layer curvature, not dose curvature.
