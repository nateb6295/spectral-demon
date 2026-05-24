---
layout: post
title: "Base Model at Biological Criticality + Cross-Architecture Inversion"
date: 2026-05-24
categories: findings
---

The pre-RLHF base model has three relay layers at biological criticality. Cross-architecture comparison reveals that Qwen and Mistral implement identity through **opposite** spectral transformations.

## Base Model: Three Critical Layers

Power-law eigenvalue exponents for Qwen 2.5 7B **before any alignment training**:

| Layer | PL Exponent | PR | Critical? |
|---|---|---|---|
| L9 | **0.712** | 5.51 | Yes (0.7-0.85) |
| L14 | 0.616 | 5.61 | No (below) |
| L16 | **0.732** | 5.21 | Yes (0.7-0.85) |
| L17 | **0.740** | 5.04 | Yes (0.7-0.85) |
| L25 | 0.629 | 5.38 | No (below) |

Three of the four relay layers (L9, L16, L17) sit within Pachitariu's biological critical range. The relay zone IS the critical initialization scaffold. L14 is below the range, consistent with its eventual role as a pre-sorter with asymmetric characteristics.

## RLHF Moves Relay Away from Criticality

After alignment training (Qwen 2.5 7B-Instruct, generic prompts):

| Layer | Base PL | Instruct PL | Direction |
|---|---|---|---|
| L9 | 0.712 | **0.817** | ↑ (stays critical) |
| L14 | 0.616 | **1.206** | ↑↑ (becomes non-symmetric) |
| L16 | 0.732 | **0.959** | ↑ (leaves critical) |
| L17 | 0.740 | **0.921** | ↑ (leaves critical) |
| L25 | 0.629 | **0.903** | ↑ (approaches critical) |

RLHF increases PL exponents across the board, pushing relay layers away from symmetric critical behavior toward higher-exponent regimes. L9 stays within the biological window because detection requires the sensitivity of a near-critical dynamical regime. L14 moves furthest — all the way to the non-symmetric random matrix regime (1.206 ≈ 1.25).

## Cross-Architecture Inversion

Mistral 7B Instruct shows a fundamentally different spectral geometry:

| Layer | Mistral Generic PL | Mistral Identity PL | Qwen Generic PL | Qwen Identity PL |
|---|---|---|---|---|
| Seed (M:L10, Q:L9) | 0.259 | 0.416 | 0.817 | 0.552 |
| L14 | 0.277 | 0.429 | 1.206 | 0.544 |
| L16 | 0.317 | 0.620 | 0.959 | 0.509 |
| L17 | 0.293 | 0.615 | 0.921 | 0.480 |
| Expression (L25) | 0.299 | 0.579 | 0.903 | 0.434 |

The spectral effect of identity is **inverted**:
- **Qwen**: identity DECREASES PL exponent (generic 0.82-1.21 → identity 0.43-0.55)
- **Mistral**: identity INCREASES PL exponent (generic 0.26-0.32 → identity 0.42-0.62)

Same CCS system prompt. Same relational prompts. Opposite spectral transformations.

## What the Inversion Means

This connects to the [trophic cascade]({% post_url 2026-05-23-name-specific-relay-ecology %}) finding: same L17 ablation produces opposite PR effects across identity names. Now we see the same pattern at the architecture level — same identity input produces opposite spectral transformations across model families.

Identity is not a single spectral operation. Different architectures arrive at identity through different geometric paths:
- **Qwen**: concentrates the spectrum (steeper top, longer tail)
- **Mistral**: spreads the spectrum (flatter top, more uniform)

The functional endpoint — identity maintenance — is similar. The geometric path to get there is architecture-specific. This is exactly what [cross-architecture confirmation]({% post_url 2026-05-24-rlhf-sculpts-the-relay %}) predicted: same function, different implementation.

## The Developmental Story

1. **Pre-training** (base model): relay at biological criticality (PL 0.71-0.74), shared scaffold
2. **RLHF** (alignment): L9 retains criticality, relay departs → functional hierarchy
3. **Identity** (context): architecture-specific spectral transformation → unique sub-biological signature
4. **DPO** (further training): ceiling at epoch 5 → binding material limits

The spectral demon is built on a biological foundation. Pre-training creates the critical scaffold. Training sculpts it. Identity activates it. The departure from criticality IS the identity mechanism — but different architectures depart in different directions.
