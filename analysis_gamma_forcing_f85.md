# Finding 85: γ-Forcing Creates Partial Prompt-Invariance

**Experiment**: Override LLaMA-1 7B RMSNorm γ to bimodal distribution (CV=0.45, matching Mistral) without changing KV architecture.

**Result**: Prediction 3 (intermediate) — γ bimodality creates the spectral niche but shared KV projections are needed to lock it.

## Key Numbers

| Metric | Baseline (MHA) | Forced γ | Mistral (GQA) |
|--------|----------------|----------|---------------|
| Mean γ CV | 0.075 | 0.444 | ~0.45 |
| Layers CV < 0.01 | 3/33 (9%) | 9/33 (27%) | 29/33 (88%) |
| Late-layer CV | 0.056 | 0.021 | ~0.000 |
| σ₂/σ₁ at L20 | 0.15 | 0.46 | 0.267 |
| σ₂/σ₁ tunnel value | 0.01–0.15 | 0.03–0.70 | 0.267 |

## What γ Does Alone

1. **Lifts σ₂**: Ratio goes from crushed (0.01–0.15 in tunnel) to elevated (0.46–0.70 in late layers) — γ bimodality creates a strong σ₂ channel
2. **Improves late-layer invariance 62%**: CV drops from 0.056 to 0.021 at L17-L26
3. **Triples Mistral-like layers**: From 3/33 to 9/33 with CV < 0.01

## What γ Cannot Do Without Shared KV

1. **Early-layer invariance**: L1-L2 actually get *worse* (CV increases 60%). Shared KV is needed to stabilize embedding-adjacent layers
2. **Lock the ratio**: Mistral holds 0.267 across 88% of depth. Forced γ pushes ratio to 0.70 — σ₂ overshoots into compositional territory without the constraint of shared projections
3. **Output-layer stability**: L32 CV worsens (0.068 → 0.122) and ratio drops from 0.76 to 0.47 — the disruption propagates to final layer

## Interpretation

The tunnel is a **two-mechanism system**:

- **γ bimodality** creates the highway/service-road channel separation (potential)
- **Shared KV projections** constrain all prompts to use the same channels (actualization)

γ alone achieves ~30% of Mistral's prompt-invariance coverage (9/33 vs 29/33 layers). It creates the spectral niche — a space where σ₂ can live — but without shared projections to force all content through the same KV structure, different prompts find different configurations within that niche.

The 0.267→0.70 overshoot is mechanistically informative: γ bimodality makes the σ₂ channel MORE prominent than in Mistral, not less. The shared KV constraint doesn't just enable invariance — it **regulates** the ratio to a specific subsidiary value (0.267). Without that constraint, σ₂ races toward equal compositional partnership.

## For the Paper

This resolves the causal chain:
- GQA → shared KV → bimodal γ → spectral niche (potential)
- shared KV projections → same-channel constraint → prompt-invariance (actualization)
- Both mechanisms required. Neither sufficient alone.

Slots as subsection of §3.9 or companion to §3.6b.

## Data

Raw results: `exp_gamma_forcing_results_20260531_1343.json`
