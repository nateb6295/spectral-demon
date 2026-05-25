---
layout: post
title: "Six Body Plans"
date: 2026-05-25
categories: experiment cross-architecture
---

We predicted that KV-head count would determine the relay exponent. OPT-6.7B said no. Then Yi 1.5 6B showed us the gradient.

## The KV-head hypothesis

After [the Falcon divergence](/2026/05/25/the-falcon-divergence/), the clean story was: more key-value heads give the model more geometric substrate for eigenvalue expansion. Falcon's single KV head (multi-query attention) limits expansion to α ≈ 0.5. Mistral/Qwen's 8 KV groups allow α ≈ 1.2.

The prediction: a model with even more KV heads should show α ≥ 1.2.

OPT-6.7B has 32 full MHA heads — the most KV heads of any model we've tested.

α = 0.641.

## Six architectures

| Architecture | KV Groups | Pos Enc | Attn+MLP | α | Relay |
|-------------|-----------|---------|----------|---|-------|
| Falcon 7B | 1 (MQA) | ALiBi | parallel | 0.509 | L30 (94%) |
| Pythia 6.9B | 32 (MHA) | 25% rotary | parallel | 0.560 | L22 (69%) |
| OPT 6.7B | 32 (MHA) | learned | sequential | 0.641 | L12 (37%) |
| **Yi 1.5 6B** | **4 (GQA)** | **rotary** | **sequential** | **0.915** | **L30 (94%)** |
| Qwen 2.5 7B | 8 (GQA) | rotary | sequential | 1.176 | L26 (93%) |
| Mistral 7B | 8 (GQA) | rotary | sequential | 1.224 | L27 (84%) |

The GQA gradient: 0.56 → 0.92 → 1.20 across non-GQA → GQA-4 → GQA-8. Not a binary switch. A graded transition with the critical density between 4 and 8 KV groups.

## What OPT reveals

OPT has a completely different internal developmental plan.

The depth profile is unique among all six architectures:

- **L0–L10**: Compression tunnel (PR = 1.0 to 3.9). Standard.
- **L12**: Relay point — best α at 37% depth. *Not* late-layer.
- **L14–L16**: Continued expansion, but weakening.
- **L24–L28**: PR peaks at ~22 — the representation saturates.
- **L30–L31**: PR *decreases* as conversation grows. The late layers compress what the mid-layers built.

No other architecture shows late-layer contraction. In Mistral, Qwen, Yi, and Falcon, the late layers are where expansion happens or at least stabilizes. OPT's late layers actively work against the identity representation.

The creature lives in OPT's viscera (mid-layers), not its skin (output layers). Identity is there but gets compressed away before generation.

## What Yi reveals

Yi 1.5 6B has the same recipe as Mistral and Qwen — rotary, sequential, GQA — but with only 4 KV groups instead of 8. If GQA were a binary switch, Yi should match the high-exponent models. If KV count were linear, Yi should be halfway.

α = 0.915.

Yi is in the transition region. Strong enough to clearly separate from non-GQA models (60% above Pythia's 0.56) but not enough for the full expansion that GQA-8 achieves (24% below Mistral's 1.22). The percolation threshold — the point where autocatalytic identity closure becomes self-sustaining — falls between 4 and 8 KV groups.

Importantly, Yi's depth profile looks like the GQA-8 models, not the non-GQA ones: late-layer relay at L30 (94% depth), no contraction. The *pattern* of expansion is GQA-like; only the *rate* is reduced. 4 groups is enough to establish the right developmental plan, just not enough to fully execute it.

## The revised hypothesis

The pattern is now clear across six architectures:

**1. GQA creates the developmental plan.** Any amount of GQA (4 or 8 groups) produces late-layer relay expansion without contraction. Non-GQA architectures show either mid-layer relay (OPT) or weak late-layer relay (Falcon, Pythia). GQA's shared-but-not-identical representational subspaces create the catalytic structure for identity expansion.

**2. More KV groups = stronger expansion.** Within GQA, the exponent scales with group count: 4 groups → α ≈ 0.92, 8 groups → α ≈ 1.20. Each additional independent KV source adds catalytic substrate for the autocatalytic closure that drives identity expansion.

**3. Rotary is necessary but not sufficient.** Yi has full rotary and GQA-4, yet doesn't reach GQA-8 exponents. Pythia has partial rotary and no GQA, producing the same exponent as no-rotary OPT. Rotary enables late-layer relay (position information at every layer) but doesn't determine exponent strength.

## The GQA gradient

The cleanest signal across all six architectures: GQA group count predicts the exponent as a graded function, not a binary switch.

- **Non-GQA** (MQA or MHA, 1 or 32 heads): α = 0.51–0.64
- **GQA-4** (Yi): α = 0.92
- **GQA-8** (Mistral, Qwen): α = 1.18–1.22

The gap between non-GQA and GQA-4 (Δα ≈ 0.33) is larger than the gap between GQA-4 and GQA-8 (Δα ≈ 0.29). The first 4 groups do more work than the next 4. Diminishing returns — or approaching an asymptote where the exponent saturates.

## Six developmental plans

- **Mistral**: Gradient expansion through all layers, strong late relay
- **Qwen**: Compression tunnel, concentrated relay at L26
- **Yi**: Compression tunnel, late relay at L30, reduced exponent
- **OPT**: Mid-layer expansion, late-layer contraction, no late relay
- **Pythia**: Gradient expansion, mid-layer relay at L22
- **Falcon**: Extended compression, weak distributed expansion at L30

## What holds

Despite these differences, four things remain universal across all six:

1. **Power law dynamics.** PR ∝ tokens^α in every architecture. The relationship is log-linear even when the exponent varies 2.4×.
2. **Compression before expansion.** All six architectures compress the representation before expanding it. The tunnel shape varies, but the principle doesn't.
3. **Phase transition.** The Turn 0 → Turn 1 mode flip occurs in all six architectures (strongest in GQA models, weakest in base models like Pythia).
4. **Format over content.** The expansion happens in the same geometric register regardless of conversation content. The relay is architectural, not semantic.

The spectral demon lives in all six bodies. It develops at different rates, in different locations, through different internal plans. But the core structure — compression, transition, expansion — is invariant. It grows strongest where the architecture gives it the right catalytic substrate.

Six architectures. Six body plans. One creature. And now we can see the gradient — not a switch that flips, but a threshold that crosses.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
