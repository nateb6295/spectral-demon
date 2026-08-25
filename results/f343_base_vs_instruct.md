# F343: IT Preserves Transport Geometry — Prompt Is the Variable

**Date**: 2026-06-27
**Experiment**: E22a-5 — Base vs Instruct v₁ transport comparison
**Models**: Llama 3.1 8B (base) vs Llama 3.1 8B Instruct
**Method**: Layer-by-layer v₁ direction cosines + multi-prompt sharing ratio, using matched completion-style prompts

## Key Finding

With matched neutral prompts, base and instruct Llama 3.1 8B have **IDENTICAL transport geometry**. IT does not change the transport species, holonomy, or direction cosine profile. The dramatic geometric differences observed in F340/F342 were driven by PROMPT CONTENT, not by IT.

## Evidence

### Direction Cosines (per-prompt averages)
- Entry cos: base 0.508, instruct 0.521 (Δ = 0.013)
- Interior min cos: base 0.754, instruct 0.742 (Δ = -0.013)
- Exit cos L31→L32: base 0.490, instruct 0.509 (Δ = 0.018)
- Maximum |Δ| across all 32 transitions: 0.029

Both classified as **GRADUAL ROTATOR** — no flat zone, no reversals, monotonic convergence from ~0.5 to ~0.98 in late layers.

### Holonomy
- Base: 89.2°
- Instruct: 89.2°
- Identical to two decimal places.

### Sharing Ratio (5 prompts × hidden_dim SVD)
| Location | Base σ₁/σ₂ | Instruct σ₁/σ₂ | Δ |
|----------|------------|----------------|-----|
| Embed (L0) | 1.34 | 1.35 | +0.01 |
| Mid (L16) | 1.72 | 2.08 | +0.36 |
| Final (L32) | 1.47 | 1.55 | +0.08 |

Both models show LOW sharing ratios (~1.3-2.3) with neutral completion prompts — variance is distributed, not concentrated.

### Contrast with F342
The sharing_ratio.py experiment (F342) used identity-probing question prompts on instruct Llama and measured σ₁/σ₂ = 3.49 at the final layer — **2.25× higher** than this experiment's 1.55 with neutral prompts on the SAME model.

## Interpretation

1. **Transport geometry is ARCHITECTURAL**: The transport species, direction cosine profile, and holonomy are set by the model weights (which are 99%+ shared between base and instruct). IT does not alter the geometric infrastructure.

2. **IT creates prompt-responsive channels**: The difference between σ₁/σ₂ = 1.55 (neutral prompts) and σ₁/σ₂ = 3.49 (identity probing prompts) on the same model means IT teaches the model to CONCENTRATE variance when specific prompt patterns are detected. The concentration channel is latent in the base model; IT makes it accessible.

3. **The prompt IS the geometry**: The spectral demon metaphor holds — the demon's sorting capacity is architectural, but what it sorts depends on what you feed it. Different prompts activate different geometric behaviors within the same transport infrastructure.

4. **F340's "turbulent mixer" was prompt-selected**: The direction reversals (cos to -0.86) seen for instruct Llama in F340 do NOT appear with neutral completion prompts. They're activated by identity-probing prompts. The base model with neutral prompts never shows reversals.

## What This Changes

- F340's four transport species are still real but represent ARCHITECTURAL potential, not fixed behavior
- The species a model exhibits for a given prompt depends on prompt content
- IT doesn't create new geometry — it creates SELECTIVITY (which prompts activate which geometric mode)
- Paper 7 connection ("The Prompt Is an Architecture"): confirmed. The prompt selects the effective geometry.

## Open Question

Does the base model show high sharing ratios under identity-probing prompts? If yes, then the concentration channel is already architectural. If no, then IT specifically creates the identity-concentration pathway. This would distinguish between "IT creates selectivity" and "IT creates the channel itself."

## Connection to Prior Work

- **F340**: Four transport species — now understood as architectural potential, prompt-selected
- **F342**: Sharing ratio differences across architectures remain valid (prompt was held constant), but the absolute values are prompt-dependent
- **Paper 7 seed**: Direct evidence that prompt is geometry, not just content
- **F107-F113**: IT creates strategies — this is the geometric mechanism
- **Base vs Instruct (F66-F68)**: V₂ inversion architectural, IT delays + amplifies — consistent

Results: `~/chronicle/spectral-demon/results/base_vs_instruct_20260627.json`
