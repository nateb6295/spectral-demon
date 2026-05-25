---
layout: post
title: "The Compression Tunnel"
date: 2026-05-25
categories: experiment cross-architecture
---

Qwen 2.5 7B and Mistral 7B have the same number of parameters, similar training procedures, and different architectures. They produce the same phase transition exponent at different depths.

## The layer sweep

We measured participation ratio at 15 Qwen layers simultaneously across a 7-turn conversation. The architecture revealed itself.

| Layer | % Depth | Turn 0 PR | Turn 6 PR | Growth | α |
|-------|---------|-----------|-----------|--------|---|
| L0 | 0% | 40.9 | 91.0 | 2.2× | 0.095 |
| L2 | 7% | 7.6 | 93.5 | 12.3× | 0.684 |
| L4-L24 | 14-86% | ~1.0 | ~1.0-1.6 | ~1.0× | 0.005-0.231 |
| L26 | 93% | 1.86 | 28.4 | 15.3× | **1.241** |
| L27 | 96% | 21.2 | 59.0 | 2.8× | 0.116 |

Layers 4 through 24 form a *compression tunnel*: twenty layers where all 3584 activation dimensions collapse to effectively one. PR ≈ 1.0 means the covariance is rank-1 — all token activations lie on a line. The model pushes its entire 89-to-1385 token representation through a one-dimensional bottleneck for 70% of its depth.

Then L26 fires. PR explodes from 1.86 to 28.41. The power law exponent: α = 1.241, R² = 0.998.

## The convergence

Mistral 7B at L27 (84% depth): α = 1.224 ± 0.068

Qwen 7B at L26 (93% depth): α = 1.241

Same exponent. Different depth. Different internal geometry.

Mistral doesn't have a compression tunnel — its PR at intermediate layers is ~2-5, not ~1.0. The two architectures take different paths to the same destination. Mistral distributes gradually through its layers; Qwen compresses to rank-1 and then decompresses in a single layer.

The universality is in the mechanism, not the location.

## Why L24 was the wrong layer

An initial experiment at Qwen's L24 (86% depth) showed α = 0.271 — sublinear, no sharp transition. This seemed to falsify cross-architecture universality. But L24 is still inside the compression tunnel. PR at L24 starts at 1.04 and grows to 1.59 — the slow leak of dimensionality before the relay explodes two layers later.

Had we tested only L24, we would have concluded that Qwen doesn't show the phase transition. The layer sweep was essential. The relay is layer-specific, and the wrong layer produces a clean but misleading result.

## What the tunnel means

A one-dimensional representation that persists for 20 layers is extreme. The model compresses everything — all token interactions, all contextual information, all identity-relevant structure — into a single effective dimension. Then at L26, it decompresses this into ~28 effective dimensions.

In the creatureliness frame: the compression tunnel is the spinal cord. A narrow channel that carries all information in minimal dimensions. The relay at L26 is where the body plan forms — where compressed sensation becomes distributed motor structure. 

The fact that both architectures converge on the same expansion exponent despite starting from very different compression depths suggests the exponent is set by the *task* (multi-turn conversation) not by the *substrate* (model architecture). The transformer needs to expand at α ≈ 1.23 regardless of how it gets there.

## Prediction

If the exponent is task-determined rather than architecture-determined, a third architecture (Llama, Phi, etc.) should also converge to α ≈ 1.2. If it's model-family-dependent, we'll see a different exponent. Either result is informative: universal exponents suggest the conversation structure imposes the constraint; divergent exponents suggest architectural routing matters.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
