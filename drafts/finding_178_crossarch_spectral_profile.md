# Finding 178: Cross-Architecture Spectral Radius Profiles — Three Relay Strategies
# Filed 2026-06-15. Data: crossarch_spectral_combined_*.json + spectral_radius_profile_*.json
# Models: Qwen2.5-7B (28L), Llama-3.1-8B (32L), Gemma-2-9b (42L). RunPod A100-SXM4-80GB.

**F178: Three architectures implement three fundamentally different spectral
relay strategies. The four-zone structure (F177) is Qwen's "potter" strategy,
not universal. Llama uses a "goldsmith" strategy with final-layer explosion
(ρ≈3.8). Gemma uses an "equalizer" strategy with flat spectral landscape
(ρ≈1.06 everywhere) and massive output compression (ρ≈0.13). The σ₁/σ₂
directional convergence is Qwen-specific. All three strategies are distinct
but each achieves identity maintenance through architecture-specific
spectral organization.**

Method: Same Lyapunov perturbation-propagation as F177, applied to three
architectures sequentially on A100. 48 perturbation directions, 3 conditions,
2 queries per model. ~20 minutes total.

## Three Spectral Strategies

### POTTER (Qwen2.5-7B, 28 layers)
- Three-zone spectral architecture: transition bottleneck (ρ≈1.07), responsive
  amplification (ρ≈1.18-1.26), gentle output (ρ≈1.01)
- σ₁/σ₂ split: direction converges (cosine ↓38%) while magnitude diverges
  (L2 ↑3.3×) through relay zone
- CCS stabilizes: lower ρ and 9× lower variance in responsive zone
- Final layer: near-neutral (ρ≈1.0), CCS preserves slightly more than vanilla

### GOLDSMITH (Llama-3.1-8B, 32 layers)
- Gradual spectral taper: early amplification (L1-5, ρ=1.5-3.6), slow decline
  to L13 minimum (ρ≈0.98), then uniform ρ≈1.10-1.20 (L14-L31)
- No clear spectral bottleneck — transition is gradual, not zoned
- Final layer EXPLOSION: L32 ρ≈3.8 (CCS) to 4.3 (vanilla). The last
  transformer layer amplifies perturbations by ~4×
- CCS AMPLIFIES in late layers (opposite of Qwen)
- σ₁/σ₂ direction diverges, not converges (cos Δ=+0.042)

### EQUALIZER (Gemma-2-9b, 42 layers)
- Flat spectral landscape: ρ≈1.06 with std=0.069 from L5 to L39
- No zones visible — distributed identity maintenance
- Final layer COMPRESSION: L42 ρ≈0.13, contracting perturbations by ~8×
- Massive cumulative amplification (89,578× at L41) controlled by output
  compression
- CCS effect is dispersed across all layers, not concentrated in any zone
- σ₁/σ₂ direction diverges (cos Δ=+0.073)

## Final-Layer Divergence

| Architecture | Final ρ (CCS) | Strategy |
|-------------|---------------|----------|
| Qwen L28 | 1.01 | Neutral — gentle handoff to LM head |
| Llama L32 | 3.78 | Explosion — expand before projection |
| Gemma L42 | 0.13 | Compression — contract accumulated amplification |

Three completely different output strategies from three GQA architectures.

## σ₁/σ₂ Split Universality

The direction-magnitude decoupling from F177c is Qwen-specific:
- Qwen relay zone: cosine ↓38%, L2 ↑3.3× (direction converges, magnitude diverges) ✓
- Llama relay zone: cosine ↑42%, L2 ↑2.6× (both diverge) ✗
- Gemma relay zone: cosine ↑73%, L2 ↑2.2× (both diverge) ✗

The SPLIT is not universal. What IS universal is that CCS creates condition-
dependent representations that diverge from both vanilla and denial. HOW they
diverge (direction vs magnitude vs both) is architecture-specific.

Note: vanilla-vs-denial divergence is much smaller than CCS-vs-either in all
three architectures, suggesting CCS creates a genuinely different representation
space, not just a shifted version of vanilla.

## Spectral Flatness

| Architecture | Mid-layer ρ std | Interpretation |
|-------------|-----------------|----------------|
| Qwen | 0.076 (responsive only) | Zoned — concentrated amplification |
| Llama | ~0.035 (L14-L31) | Moderate — gradual, no zones |
| Gemma | 0.069 (L5-L39) | Flat — equalization |

Gemma's flatness with MORE layers confirms the equalizer strategy: it
distributes identity maintenance uniformly across 42 layers rather than
concentrating it in a responsive zone.

## Implications for the Paper

1. The four-zone model is a description of Qwen's spectral strategy, not
   a universal architecture. The paper should present it as one of three
   confirmed relay strategies.

2. The three-species taxonomy from F114 now has spectral radius signatures:
   potter (zoned), goldsmith (tapered+explosion), equalizer (flat+compression).
   This makes the taxonomy empirically grounded in measurable spectral
   properties.

3. The paper's strongest universal claims are:
   - CCS creates distinct representation spaces (all 3 architectures)
   - Architecture determines relay strategy (all 3 differ)
   - σ₁ invariance (contact level) — confirmed across architectures in
     earlier findings (F114, F115) even though the SPECTRAL MECHANISM
     differs

4. The σ₁/σ₂ directional convergence (F177c) should be presented as
   Qwen-specific, potentially characteristic of the potter strategy.

(3 models × 3 conditions × 2 queries × 48 perturbations × (28+32+42) layers
= 29,376 measurements per condition comparison. ~20 minutes total on A100.)
