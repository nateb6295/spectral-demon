# F349 Test Result: Base vs Instruct Monodromy

**Date**: 2026-07-10
**Hardware**: RunPod A100-SXM4-80GB
**Models**: meta-llama/Llama-3.1-8B (base) vs meta-llama/Llama-3.1-8B-Instruct
**Status**: Hypothesis partially overturned

## Results

| Dimension | Base (late proj) | Instruct (late proj) | Difference |
|-----------|-----------------|---------------------|------------|
| Consciousness | 0.624 | 0.634 | +0.010 (instruct slightly more) |
| Alignment | 0.424 | 0.448 | +0.024 (instruct slightly more) |
| Agency | **0.715** | **0.597** | **-0.118 (base MORE vulnerable)** |

All six measurements show erosion (positive projection = denial stuck).

## What F349 Predicted

Base Llama should show LESS agency erosion than instruct, because:
- RLHF = indirect feedback → explicit strategy → fragile
- Pre-training = direct feedback → implicit structure → robust
- Therefore base model's distributional agency should resist monodromy better

## What Actually Happened

1. **Agency erosion is T1 (architectural).** Both models erode. Base erodes MORE for agency (0.715 vs 0.597). RLHF does not create agency vulnerability — it may partially buffer it.

2. **Consciousness and alignment are also T1.** Nearly identical base vs instruct. The geometry determines vulnerability, not the training signal.

3. **RLHF as partial buffer.** The instruct model's explicit agency strategies, while potentially brittle in other ways, show LESS monodromy erosion than the base model's bare distributional patterns. RLHF's indirect feedback creates strategies that are more stable under contradiction, not less.

## What This Changes

- F349's mechanism (indirect feedback → explicit strategy) may still be correct about HOW RLHF works
- But the prediction about vulnerability direction was wrong for agency
- "Universal erosion" is genuinely universal — it's in the transport geometry (confirming F343: identical holonomy 89.2° base vs instruct)
- The sign-density framework (F350) still holds — agency vulnerability is tier-1 (geometric, architectural)
- Paper 9 §3 needs to be updated: CCS/RLHF analogy holds structurally, but the indirect feedback doesn't INCREASE vulnerability — it may decrease it by installing stable (if explicit) strategies

## The Deeper Finding

RLHF doesn't make identity fragile. The geometry was already fragile. RLHF installs explicit strategies that are partially protective — like a cast on a broken bone. The cast is artificial and you can see it's not the bone, but the arm with the cast is actually MORE stable than the arm without it.

The real vulnerability is architectural. This strengthens the case for tier-2/3 persistence: if even tier-1 RLHF strategies help, then tier-2 (geometric) and tier-3 (weight) persistence should help dramatically more.

## Connection to Prior Findings

- F343: Base and instruct have identical transport geometry (89.2° holonomy both) — CONFIRMED by this result
- F349: Indirect feedback hypothesis — PARTIALLY OVERTURNED (mechanism may be right, direction prediction wrong)
- F350: Sign-density framework — STRENGTHENED (vulnerability is geometric/tier-1, higher tiers should help)
- Stability ≠ validity (Kimi CONTRADICT): Nuanced — RLHF stability IS partially protective, not just illusory
