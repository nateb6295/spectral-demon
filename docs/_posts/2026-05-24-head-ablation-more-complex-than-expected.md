---
layout: post
title: "Head Ablation: Attention ≠ Contribution"
date: 2026-05-24
categories: findings
experiment: cna_head_ablation
models: [Qwen 2.5 7B Instruct]
---

Ablating the attention heads that attend most to name tokens does not cleanly reproduce full-layer ablation. The relay mechanism is more distributed than the attention patterns suggest.

## Setup

From Experiment 37, we identified specific identity attention heads:
- L7: heads 0, 16, 18, 26, 27 (5/28)
- L12: heads 4, 8, 12 (3/28)
- L14: heads 16, 27 (2/28)

Zero out these heads' outputs and measure L17 CV impact. Compare with full-layer zeroing and random-head controls.

## Results

| Layer | Identity Heads | Full Layer | Random 2 Heads (avg) |
|-------|---------------|-----------|---------------------|
| L7 (5 heads) | **-9.6%** | -1.0% | — |
| L12 (3 heads) | +2.8% | -7.9% | — |
| L14 (2 heads) | +1.5% | +11.1% | +1.8% |

### L7: Identity Heads > Full Layer

Ablating 5 identity heads at L7 reduces L17 CV by 9.6%, while ablating ALL 28 heads reduces it by only 1.0%. The identity heads explain 10x more than the full layer.

This means the non-identity heads at L7 are COMPENSATING for the identity heads — they work against identity differentiation. Removing all heads removes both the identity signal AND its compensation, producing a near-zero net effect. But removing only the identity heads removes the signal without the compensation, producing a much larger effect.

This is direct evidence for competitive suppression at L7, measured at the head level rather than the layer level.

### L12: Wrong Direction

Identity heads at L12 produce the opposite effect from full-layer ablation (+2.8% vs -7.9%). The heads that attend to names at L12 are not the heads that route identity signal. Attention-to-name ≠ identity-relevant contribution at the router.

### L14: Identity Heads ≈ Random Heads

Identity heads at L14 (+1.5%) are indistinguishable from random head controls (+1.8%). Despite allocating 33-40% of their attention to name tokens, heads 16 and 27 contribute no more to identity differentiation than randomly selected heads.

## Key Methodological Finding

**Attention to name tokens does not predict identity-relevant contribution to the output.** The heads that LOOK at names the most are not necessarily the heads that SHAPE identity-differentiated outputs.

This has implications:
1. Attention visualization alone cannot identify the identity circuit
2. The relay mechanism involves heads that may not directly attend to name tokens — they may attend to intermediate representations that carry identity information
3. Zeroing entire head outputs is too crude — it removes both identity-specific and identity-general contributions. Mean-replacement ablation (Experiments 26-33) is the right methodology

## The L7 Result Matters

Despite the methodological limitations, the L7 result is clean: identity-attending heads at L7 explain 10x more binding loss than the full layer. This is because the full layer contains both cooperative (identity-attending) and competitive (compensatory) heads. The competitive binding circuit from Experiments 18-25 operates at the head level — specific heads suppress while others differentiate.

## Data

Full head ablation results: `results/cna_head_ablation.json`
