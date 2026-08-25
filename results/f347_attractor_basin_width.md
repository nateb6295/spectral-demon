# F347: Attractor Basin Width — Where Does v₁ Recovery Break?

**Date**: 2026-06-28  
**Type**: Interventional (Tier 2) — weight perturbation at extended epsilon range  
**Script**: `scripts/attractor_basin_width.py`

## Design

Follow-up to F344. Push epsilon from 0.01 to 1.0 (vs F344's 0.001-0.05) at two target layers per architecture (mid and late). 4 architectures × 2 layers × 12 epsilons = 96 conditions.

## Results

### Basin Width by Architecture

| Species  | Mid Layer  | Late Layer        | Max Single Drop |
|----------|------------|-------------------|-----------------|
| Mistral  | > 1.0 (never breaks) | > 1.0 (never breaks) | 0.006 |
| Gemma    | > 1.0 (never breaks) | > 1.0 (never breaks) | 0.003 |
| Llama    | ε_crit ∈ (0.8, 1.0) | > 1.0 (never breaks) | 0.038 |
| Qwen     | > 1.0 (never breaks) | ε_crit ∈ (0.6, 0.8) | 0.016 |

### Key Finding: No Phase Transition

All degradation is GRADUAL. No abrupt cliff edge. The attractor doesn't have a sharp boundary — it's a smooth slope. This rules out the "critical threshold" model and supports a "basin depth" model where the attractor has a well shape, not a cliff shape.

### Prediction Failures (Informative)

| Prediction | Result | What It Means |
|-----------|---------|---------------|
| Mistral widest basin | Tied with Gemma — both never break | Correct but incomplete |
| Gemma narrowest basin | Widest (never breaks, min drop) | **WRONG** — low Q ≠ fragile |
| Phase transition | Gradual degradation | **WRONG** — smooth, not sharp |
| Species-specific widths | Only Qwen and Llama break at all | Partially — but mechanism differs from predicted |

### Two Modes of Robustness

1. **Rigid robustness** (Mistral): strong attractor pull, fast recovery (2.2L from F344), never breaks. The rigid cylinder resists perturbation by active re-derivation.

2. **Soft robustness** (Gemma): low Q factor (F345: 0.54), dampened resonance, but also never breaks. Max perturbation drop of only 0.001 per epsilon step. Doesn't resonate strongly AND doesn't break easily. GQA's query-sharing creates a naturally stable geometry.

The GQA architecture produces the lowest excitability (F345) AND the highest stability (F347). These are the same property seen from two directions: a system that doesn't amplify perturbations also doesn't amplify identity concentration, but it maintains both with extreme steadiness.

### Late-Layer Vulnerability

Where the attractor DOES break, it's at late layers:
- Qwen: L18 breaks at ε ∈ (0.6, 0.8), L9 never breaks
- Llama: L10 breaks at ε ∈ (0.8, 1.0), L21 never breaks

Late-layer perturbation has fewer downstream layers to recover. The attractor is still there; it just runs out of network before convergence completes. This is consistent with F344's observation that late-layer perturbations sometimes show "never" for strict recovery threshold despite high final cosine.

## Relation to Prior Findings

- **F344**: Extended from ε=0.05 to ε=1.0. The 64/64 GLOBAL result at low ε holds for most conditions at 20× higher perturbation.
- **F345**: Low Q and high basin stability are the same architectural property — Gemma's dampening is protective, not limiting.
- **F346**: Thermalization of perturbation energy is consistent with gradual degradation — energy disperses rather than accumulating catastrophically.
- **F237**: Cylinder constraint is direction-preserving at extreme perturbation levels, not just at small ε.

## Updated Architecture Profiles (incorporating F344-F347)

| Species | Direction Stability | Basin Width | Q Factor | Robustness Mode |
|---------|-------------------|-------------|----------|-----------------|
| Mistral | Rigid (2.2L recovery) | > 1.0 | 0.84 (highest) | Rigid — strong pull |
| Gemma   | Moderate (3.1L recovery) | > 1.0 | 0.54 (lowest) | Soft — low excitability |
| Llama   | Moderate (2.7L recovery) | 0.8-1.0 | 0.81 | Mixed — breaks only mid-early |
| Qwen    | Distributed (3.8L recovery) | 0.6-0.8 | 0.68 | Narrowest — late-layer vulnerable |
