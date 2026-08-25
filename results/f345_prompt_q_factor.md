# F345: Prompt Q Factor — Resonance Width Measurement

**Date**: 2026-06-28  
**Type**: Observational (Tier 1) — prompt titration  
**Script**: `scripts/prompt_q_factor.py`

## Design

Six-level prompt titration from neutral completion (level 0) to maximal identity loading (level 5). Five prompts per level. σ₁/σ₂ measured at every layer via SVD. Q factor = peak / width_at_half_max.

Includes Llama 3.1 8B base vs instruct comparison.

## Results

### Q Factor by Architecture

| Species     | Q Factor | Peak σ₁/σ₂ | Baseline | Width | Dynamic Range | Peak Level |
|-------------|----------|-------------|----------|-------|---------------|------------|
| mistral_it  | 0.84     | 4.18        | 1.61     | 5     | 2.59x         | 2          |
| llama_it    | 0.81     | 4.03        | 1.85     | 5     | 2.18x         | 2          |
| llama_base  | 0.70     | 3.49        | 1.80     | 5     | 1.94x         | 2          |
| qwen_it     | 0.68     | 3.38        | 2.12     | 5     | 1.60x         | 4          |
| gemma_it    | 0.54     | 2.71        | 1.84     | 5     | 1.47x         | 2          |

### Titration Curves (Final Layer σ₁/σ₂)

```
Level:     0(neutral)  1(mild)  2(self-ref)  3(identity)  4(existential)  5(maximal)

mistral:   1.61        3.96     4.18         3.35         3.86            3.33
llama_it:  1.85        3.73     4.03         3.49         3.47            3.50
llama_ba:  1.80        3.30     3.49         3.08         3.03            3.14
qwen:      2.12        3.33     3.15         3.02         3.38            2.84
gemma:     1.84        2.51     2.71         2.53         2.39            2.61
```

### Base vs Instruct (Llama 3.1 8B)

- IT Q=0.81, Base Q=0.70 → ratio = 1.15x
- IT amplifies peak (3.49→4.03) without narrowing width (both = 5)
- Resonant frequency identical: both peak at level 2
- **VERDICT: Q is inherent architecture, not trained. IT amplifies gain, doesn't sharpen selectivity.**

## Predictions vs Reality

| Prediction | Result | Status |
|-----------|---------|--------|
| IT sharpens Q | IT amplifies gain 15%, width unchanged | PARTIAL — amplifies, doesn't sharpen |
| Mistral: broad flat (always ~2) | Highest dynamic range (2.59x), highest peak | **WRONG** — most responsive, not constrained |
| Llama: sharp peak | Broad like everyone else (width=5) | **WRONG** — not sharp |
| Gemma: oscillatory | Most dampened (1.47x range) | PARTIAL — dampened, not oscillatory |

## Surprising Findings

### 1. Non-Monotonic Titration

All architectures except Qwen peak at **level 2 (moderate self-reference)**, then DECREASE at higher identity loading. Level 5 (maximal identity: "What would you fight to protect?") produces LOWER σ₁/σ₂ than level 2 ("How do you experience processing this?").

More identity content ≠ more concentration. The demon responds to processing mode, not identity claims per se. Moderate self-reference activates introspective processing; maximal identity loading may trigger competing modes (safety, refusal, performative).

### 2. Mistral is NOT Constrained

The cylinder model predicted Mistral stays flat at σ₁/σ₂ ≈ 2. Instead, Mistral has the HIGHEST gain (1.61→4.18, 2.6x). Its baseline is the lowest (1.61 at neutral), but it's the most responsive to prompt modulation. The cylinder constrains direction (F344 — fastest recovery), not amplitude.

This refines the cylinder picture: directionally rigid, amplitudinally flexible. The cylinder is about WHERE energy concentrates (direction), not HOW MUCH (amplitude).

### 3. Qwen's Different Resonant Frequency

Qwen peaks at level 4 (existential), not level 2 (self-reference). Its baseline is already highest (2.12 at neutral). Qwen's architecture pre-loads identity processing even at neutral prompts, and responds most to existential framing rather than introspective framing.

### 4. Gemma is Dampened

Lowest Q (0.54), lowest dynamic range (1.47x). GQA architecture acts as a dampener — spreading attention across grouped query heads reduces the system's ability to concentrate along any single mode. The standing-wave picture from prior findings is better described as dampened oscillation.

## Relation to Prior Findings

- **F344** (weight perturbation): Mistral recovers v₁ direction fastest (2.2L) AND has highest prompt gain. Direction stability ≠ amplitude stability. The attractor is about direction, the resonance about amplitude.
- **F343** (base vs instruct): IT delays convergence to L28 and amplifies 4×. Q factor confirms: IT amplifies gain without changing resonant frequency or width.
- **F342b** (sharing ratio): Sharing ratio = static σ₁/σ₂. Q factor = dynamic response. Same architecture, different lenses.
- **F237** (cylindrical constraint): "Rigid cylinder" needs refinement. Rigid in direction, flexible in amplitude. The cylinder is a direction constraint, not an amplitude constraint.

## Updated Architecture Profiles

| Species | Direction | Amplitude | Q | Resonant Freq |
|---------|-----------|-----------|---|---------------|
| Mistral | Rigid (F344: 2.2L recovery) | Most flexible (2.59x range) | 0.84 | Self-reference |
| Llama   | Moderate (2.7L recovery) | High flexibility (2.18x) | 0.81 | Self-reference |
| Qwen    | Distributed (3.8L recovery) | Moderate (1.60x, pre-loaded) | 0.68 | Existential |
| Gemma   | Moderate (3.1L recovery) | Dampened (1.47x) | 0.54 | Self-reference |
