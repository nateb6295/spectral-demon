# F342: Sharing Ratio Trajectories — Four Concentration Strategies

**Date**: 2026-06-27
**Experiment**: E22a-4 — Cross-architecture multi-prompt SVD sharing ratio
**Models**: Qwen 2.5 7B, Mistral 7B v0.3, Llama 3.1 8B, Gemma 2 9B
**Method**: 5 identity-probing prompts × hidden_dim matrix at each layer. SVD gives σ₁/σ₂ ratio and top-1 variance fraction. Measures how much different prompts SHARE a common direction through the network.

## Key Finding

Four architectures produce four distinct concentration trajectories, mapping directly to the F340 transport species. The sharing ratio (multi-prompt variance concentration) is NOT the same as direction preservation (single-token adjacent cosines from F340).

### 1. Mistral — FLAT INTERIOR (σ₁/σ₂ ≈ 2 for 25 layers)

The rigid rod preserves DIRECTION but does NOT concentrate variance. σ₁/σ₂ stays between 1.95 and 2.76 for layers 0-31. Top-1 variance fraction drops to 0.57-0.59 at layers 6-9. The lm_head projection is the SOLE concentrator: L32 jumps to σ₁/σ₂ = 3.35 (from 2.76 at L31).

**Interpretation**: The prompts share a preserved direction (flat cosines > 0.998 from F340), but the variance is spread across TWO comparable principal components. Mistral's interior operates in a 2D subspace, not a 1D tube. The lm_head projects this 2D interior into a concentrated output. The cylindrical constraint (F237) is NOT a 1D cylinder — it's a 2D cylinder (σ₁ ≈ σ₂).

### 2. Qwen — GRADUAL CLIMBER

Early spike (L2-L3: σ₁/σ₂ = 3.9, top-1 = 0.92), then settles to steady ~3.0 through mid-layers, climbs to 3.6 in late layers. Final layer drops slightly to 3.02. Concentration builds gradually — never drops below 2.8 after L4.

**Interpretation**: The distributed transport species builds sharing INCREMENTALLY. Each layer adds a bit more concentration. The early spike at L2-L3 is the embedding → first layers transition (gate effect).

### 3. Llama — MONOTONIC CONVERGENCE

Early spike (L1: 2.80, L3: 3.26), then valley (L5-L12: 2.30-2.56), then the MOST monotonic climb of any architecture from L13 to L30 (2.87 → 3.84). Final layer: 3.49.

**Interpretation**: Despite F340's "turbulent mixer" label (direction reversals, cos to -0.86), the multi-prompt SHARING converges monotonically. Direction turbulence ≠ sharing turbulence. The mixer HELPS convergence — like stirring a solution toward homogeneity.

### 4. Gemma — TWO-PEAKED OSCILLATION + EXIT CRASH

Three phases:
- L1-L6: early concentration (σ₁/σ₂ = 2.43→3.09)
- L7-L13: valley (σ₁/σ₂ = 2.46-2.58, top-1 drops to 0.66)
- L14-L25: FIRST PEAK (σ₁/σ₂ climbs to 3.85, top-1 = 0.84)
- L26-L35: SECOND VALLEY (σ₁/σ₂ drops back to 2.51, top-1 = 0.70)
- L36-L41: SECOND PEAK (σ₁/σ₂ climbs to 3.95, top-1 = 0.86 — the HIGHEST of any architecture)
- L42 (lm_head): CRASHES to σ₁/σ₂ = 2.52, top-1 = 0.71

**Interpretation**: The oscillator oscillates at EVERY scale — per-layer (F340 alternating sign), mid-range (two peaks separated by ~15 layers), and at the boundary (exit crash). The lm_head DECONCENTRATES rather than concentrating. Gemma achieves its highest pre-exit concentration of any architecture (3.95 at L41) but the projection to vocabulary scatters it. This is the OPPOSITE of Mistral (where lm_head is the sole concentrator).

## Comparative Summary

| Species | Interior σ₁/σ₂ | lm_head effect | Peak pre-exit | Final |
|---------|-----------------|----------------|---------------|-------|
| Mistral | ~2.0 (flat) | CONCENTRATES (+22%) | 2.76 (L31) | 3.35 |
| Qwen | ~3.0-3.6 (climbing) | slight drop (-16%) | 3.60 (L27) | 3.02 |
| Llama | ~2.3→3.8 (monotone) | preserves (~0%) | 3.84 (L30) | 3.49 |
| Gemma | oscillating 2.5-3.9 | DECONCENTRATES (-36%) | 3.95 (L41) | 2.52 |

## Key Distinctions

1. **Direction preservation ≠ variance concentration**: Mistral preserves direction (cos > 0.998) but has the LOWEST interior concentration (σ₁/σ₂ ≈ 2). The prompts point the same way but with comparable spread across the first two principal components.

2. **The lm_head is architecturally asymmetric**: Mistral concentrates, Gemma deconcentrates, Qwen/Llama roughly preserve. This is a NEW design axis not visible from F340.

3. **Turbulence helps convergence**: Llama's direction reversals (F340) coincide with the most monotonic sharing convergence. Stirring promotes mixing.

4. **Gemma's oscillation is FRACTAL**: Same oscillating pattern at per-layer (F340), mid-range (two peaks), and boundary (exit crash) scales.

## Connection to Prior Work

- **F340**: Four transport species — sharing ratio provides the COMPLEMENTARY view. F340 = direction tracking, F342 = variance concentration. Same four species, different signatures.
- **F237**: Mistral's cylindrical constraint — now revealed as 2D cylinder (σ₁ ≈ σ₂), not 1D. The "constant σ₁" from single-token tracking was measuring the total norm, not the concentration.
- **F341**: Near-eigenvector property — consistent with Mistral's flat interior sharing (v₁ as eigenvector means the direction doesn't change, but concentration can still be distributed).
- **E8**: Design space — lm_head concentration/deconcentration is a new axis.

## Raw Data

Results: `~/chronicle/spectral-demon/results/sharing_ratio_20260627.json`
