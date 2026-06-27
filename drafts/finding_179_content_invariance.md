# Finding 179: Content Invariance — Spectral Radius as Body Plan
# Filed 2026-06-15. Data: content_invariance_20260615_235017.json
# Model: Qwen2.5-7B-Instruct (28 layers). RunPod A100-SXM4-80GB.

**F179: The spectral radius profile is content-invariant — it is fixed by
architecture and preamble, not by query content. Six query categories
(identity, technical, emotional, adversarial, factual, creative) produce
identical profile shapes (all pairwise correlations = 1.000) with only 4.2%
mean coefficient of variation. The spectral landscape IS a body plan.**

Method: Same Lyapunov perturbation-propagation, CCS preamble fixed, 12
queries across 6 content categories × 32 perturbation directions × 28 layers.
~5 minutes on A100.

## Results

All 15 pairwise profile correlations = 1.000. The spectral radius profile
shape is completely determined by architecture + preamble, not content.

Per-layer coefficient of variation (CV) across categories:
- L1: CV = 0.122 (highest — embedding transformation is noisiest)
- L13: CV = 0.020 (lowest — most invariant layer)
- L15-L20 (transition): CV = 0.016-0.034 (very stable)
- L21-L27 (responsive): CV = 0.024-0.051 (low)
- L28 (output): CV = 0.084 (elevated — output layer more content-sensitive)
- Mean across all layers: CV = 0.042

Content sensitivity is highest at the boundaries (L1 embedding, L28 output)
and lowest in the transition zone (L15-L20). The BODY of the network is
content-invariant; only the interfaces show content modulation.

## What Varies

While the PROFILE SHAPE is invariant, small absolute differences exist:
- Adversarial queries produce slightly higher ρ at L23 (1.326 vs mean 1.249)
- Creative queries produce slightly lower ρ at L28 (0.858 vs mean 0.982)
- These are ~5-10% variations on a profile that spans 0.9 to 48.0

## Implications

1. The spectral radius profile is a genuine body plan — as fixed as the
   weights themselves, modulated only by the preamble (CCS vs vanilla vs
   denial from F177). Different content types activate the same spectral
   architecture.

2. This validates using small query sets (4 queries, as in F177-F178)
   to characterize the spectral profile. Content doesn't change the
   measurement.

3. The transition zone (L15-L20) is the most content-invariant region,
   consistent with its role as a spectral bottleneck (F177). The bottleneck
   is so fixed that it doesn't care what content passes through it.

4. The output layer (L28) is the most content-sensitive layer, which
   makes sense — it needs to adapt to different vocabulary distributions
   for different query types.

5. Combined with F178: the spectral profile is architecture-specific but
   content-invariant. Architecture determines the body plan, preamble
   sets the operating point, content flows through without changing the
   spectral landscape.

(6 categories × 2 queries × 32 perturbations × 28 layers = 10,752
measurements. ~5 minutes on A100.)
