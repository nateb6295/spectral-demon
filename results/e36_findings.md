# E36: Constraint vs Construction — Effective Rank Under CCS

**Date**: 2026-07-02
**Runtime**: ~20 min on A100-80GB
**Models**: Mistral-7B, Gemma-2-9B, Llama-3.1-8B, Qwen-2.5-7B (all instruct)

## Prediction (from journal entry 109-110)

Original: CCS CONSTRAINS (reduces effective rank).
Refined: CCS REDISTRIBUTES (same rank, sharper spectral concentration).

## Results

### Effective Rank (Shannon entropy of normalized singular values)

| Model    | Vanilla | D2 CCS | D3 CCS | D5 CCS | D2 Rand | D3 Rand | D5 Rand |
|----------|---------|--------|--------|--------|---------|---------|---------|
| Mistral  | 5.4     | 36.0   | 47.9   | 64.3   | 37.8    | 50.3    | 65.0    |
| Gemma    | 5.8     | 45.4   | 59.5   | 77.2   | 50.1    | 67.1    | 74.2    |
| Llama    | 3.4     | 28.7   | 40.9   | 59.2   | 30.4    | 43.1    | 51.9    |
| Qwen     | 3.0     | 15.0   | 18.6   | 24.2   | 17.8    | 22.6    | 27.2    |

### Stable Rank (||A||_F² / ||A||_2² — concentration into σ₁)

| Model    | Vanilla | D2 CCS | D3 CCS | D5 CCS | D2 Rand | D3 Rand | D5 Rand |
|----------|---------|--------|--------|--------|---------|---------|---------|
| Mistral  | 1.4     | 1.8    | 1.8    | 1.9    | 2.0     | 2.1     | 2.2     |
| Gemma    | 1.2     | 1.5    | 1.6    | 1.8    | 1.6     | 1.7     | 1.9     |
| Llama    | 1.3     | 1.5    | 1.5    | 1.6    | 1.9     | 2.0     | 2.0     |
| Qwen     | 1.3     | 1.5    | 1.5    | 1.5    | 2.2     | 2.2     | 2.3     |

## Findings

### F357: CCS Redistributes, Not Constrains or Constructs
Both CCS and random preamble increase effective rank (token count effect — more
input tokens = more directions in the SVD). But CCS stable rank is CONSISTENTLY
LOWER than random across all four architectures. CCS concentrates more spectral
mass into σ₁ while maintaining similar or higher total dimensionality. This is
redistribution: not narrowing (constraint) or uniform expansion (construction)
but concentration into leading modes while spreading the tail.

### F358: Redistribution Strength Is Species-Specific
Stable rank gap (CCS vs Random at D3):
- Qwen: 1.5 vs 2.2 (gap = 0.7) — strongest concentration (sorter)
- Llama: 1.5 vs 2.0 (gap = 0.5) — strong concentration (relay)
- Gemma: 1.6 vs 1.7 (gap = 0.1) — weakest concentration (transition)
- Mistral: 1.8 vs 2.1 (gap = 0.3) — moderate (relay)

Species ordering by redistribution strength: Qwen > Llama > Mistral > Gemma.
Sorters concentrate hardest. The species signature appears in HOW MUCH the
spectral mass gets redistributed, not whether it does.

### F359: D5 Effective Rank Crossover
At D5, CCS overtakes random in effective rank for Gemma (77.2 vs 74.2) and
Llama (59.2 vs 51.9). Mistral and Qwen: CCS stays below random. The crossover
architectures are the ones with moderate-to-high holonomy (Gemma 1.033, Llama
1.153). CCS at high dose OPENS MORE DIRECTIONS for these species while
concentrating more into leading modes — spectral spreading with concentration.

### F360: Redistribution Prediction Confirmed
Journal entry 110's prediction: CCS redistributes rather than constrains.
CONFIRMED. Effective rank doesn't decrease (rejecting constraint). Stable rank
decreases relative to random (confirming concentration). The Wilson loop flatness
(F354) is geometrically consistent: if CCS operates as a unitary rotation
(redistribution), it has zero holonomy by construction.

## Connection to Therapeutic Window

The stable rank results suggest the therapeutic window might have a spectral
interpretation: D2-D3 produces moderate concentration (stable rank ~1.5-1.8)
while D5 produces strong concentration (same stable rank but higher effective
rank). The window is where concentration and dimensionality are balanced.
Overdose (D10+) would push stable rank even lower — maximum concentration,
identity collapsing to σ₁ alone. The screen getting too thin.
