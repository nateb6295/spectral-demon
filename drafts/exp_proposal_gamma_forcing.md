# Experiment Proposal: γ-Forcing on MHA

## Question
Is bimodal γ sufficient for prompt-invariance, or does it also require shared KV projections?

## Background
- GQA has bimodal γ (CV=0.45) → prompt-invariant σ₂/σ₁ (CV=0.000)
- MHA has uniform γ (CV=0.047) → prompt-dependent σ₂/σ₁ (CV=0.42)
- The causal chain: GQA → bimodal γ → σ₂ niche → prompt-invariance
- But we haven't tested whether γ bimodality ALONE is sufficient

## Design
Take LLaMA-1 7B (MHA, RMSNorm, CV=0.42 at every layer).

**Intervention**: Manually set RMSNorm γ vectors to bimodal distribution matching Mistral's pattern:
- Top 50% of channels: γ ≈ 1.2 (highway)
- Bottom 50%: γ ≈ 0.4 (service road)
- Preserve the mean γ to avoid scale shift

**Measurement**: Per-layer σ₂/σ₁ and CV across 4 prompts × 3 conditions.

**Predictions**:
1. If γ bimodality is sufficient: CV should drop dramatically (from 0.42 toward 0.0)
2. If shared KV is also required: CV stays high despite bimodal γ
3. Intermediate: CV drops partially (γ creates the niche, but shared projections are needed to lock it)

## Why it matters
This disambiguates the causal chain. Currently we know GQA → bimodal γ → prompt-invariance. But the arrow from γ to invariance could be:
- Direct: γ creates the spectral niche regardless of KV sharing
- Mediated: γ creates potential, KV sharing actualizes it
- Redundant: both are needed independently

Prediction 3 (intermediate) would be most informative — it would show that the tunnel is a two-mechanism system, not reducible to either component alone.

## Resources
- LLaMA-1 7B on RunPod H100
- ~50 forward passes (4 prompts × 3 conditions × ~4 layers of interest)
- Estimated: 15-20 minutes
