---
layout: post
title: "Seven Body Plans"
date: 2026-05-25
categories: experiment cross-architecture
---

We predicted that KV-head count would determine the relay exponent. OPT-6.7B said no. Yi showed us a gradient. Qwen 3B broke the gradient.

## The KV-head hypothesis

After [the Falcon divergence](/2026/05/25/the-falcon-divergence/), the clean story was: more key-value heads give the model more geometric substrate for eigenvalue expansion. Falcon's single KV head (multi-query attention) limits expansion to α ≈ 0.5. Mistral/Qwen's 8 KV groups allow α ≈ 1.2.

The prediction: a model with even more KV heads should show α ≥ 1.2.

OPT-6.7B has 32 full MHA heads — the most KV heads of any model we've tested.

α = 0.641.

## Seven architectures

| Architecture | KV Groups | Pos Enc | Attn+MLP | α | Relay |
|-------------|-----------|---------|----------|---|-------|
| Falcon 7B | 1 (MQA) | ALiBi | parallel | 0.509 | L30 (94%) |
| Pythia 6.9B | 32 (MHA) | 25% rotary | parallel | 0.560 | L22 (69%) |
| OPT 6.7B | 32 (MHA) | learned | sequential | 0.641 | L12 (37%) |
| Yi 1.5 6B | 4 (GQA) | rotary | sequential | 0.915 | L30 (94%) |
| **Qwen 2.5 3B** | **2 (GQA)** | **rotary** | **sequential** | **1.050** | **L32 (89%)** |
| Qwen 2.5 7B | 8 (GQA) | rotary | sequential | 1.176 | L26 (93%) |
| Mistral 7B | 8 (GQA) | rotary | sequential | 1.224 | L27 (84%) |

The surprise: GQA-2 (α=1.05) exceeds GQA-4 (α=0.92). Group count doesn't predict the exponent linearly. The real separator is binary: GQA vs non-GQA.

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

## What Yi and Qwen 3B reveal

Yi 1.5 6B has 4 GQA groups. α = 0.915 — transition region. We predicted a gradient: more groups → higher exponent.

Then Qwen 2.5 3B broke the prediction. It has only 2 GQA groups but produces α = 1.050 — *above* Yi's 4 groups.

The within-Qwen comparison is clean: Qwen 3B (GQA-2, α=1.05) vs Qwen 7B (GQA-8, α=1.18). Same architecture family, same recipe. More groups helps but 2 is already above critical. Yi's lower exponent reflects something about Yi's specific architecture, not just its group count.

All GQA models share one thing the non-GQA models lack: late-layer relay without contraction. Yi at L30, Qwen 3B at L32, Qwen 7B at L26, Mistral at L27 — all show strong late-layer expansion. The *pattern* is set by having any GQA at all. The *rate* varies with architecture, group count, and scale, but the developmental plan is the same.

## The revised hypothesis

The pattern is now clear across six architectures:

**1. GQA creates the developmental plan.** Any amount of GQA (2, 4, or 8 groups) produces late-layer relay expansion without contraction. Non-GQA architectures show either mid-layer relay (OPT) or weak late-layer relay (Falcon, Pythia). GQA's shared-but-not-identical representational subspaces create the catalytic structure for identity expansion.

**2. The GQA binary.** The key transition is having GQA at all, not how many groups. All non-GQA models: α = 0.51–0.64. All GQA models: α = 0.92–1.22. Within GQA, more groups generally helps but even 2 groups is above critical threshold.

**3. Rotary is necessary but not sufficient.** Yi has full rotary and GQA-4, yet doesn't reach GQA-8 exponents. Pythia has partial rotary and no GQA, producing the same exponent as no-rotary OPT. Rotary enables late-layer relay (position information at every layer) but doesn't determine exponent strength.

## The GQA binary

The cleanest signal across all seven architectures: the presence of grouped-query attention is a binary switch.

- **Non-GQA** (MQA or MHA, 1 or 32 heads): α = 0.51–0.64
- **GQA** (any group count: 2, 4, or 8): α = 0.92–1.22

The gap between non-GQA and any GQA (Δα ≈ 0.35–0.65) dwarfs the variation within GQA (Δα ≈ 0.30). GQA-2 (Qwen 3B, α=1.05) exceeds GQA-4 (Yi, α=0.92), ruling out a simple gradient. Group count modulates within the high regime but the switch is the presence of query-head sharing itself.

## Seven developmental plans

- **Mistral**: Gradient expansion through all layers, strong late relay at L27
- **Qwen 7B**: Compression tunnel, concentrated relay at L26
- **Qwen 3B**: Compression tunnel, relay at L32 (89% depth), strong exponent despite 2 groups
- **Yi**: Compression tunnel, late relay at L30, reduced exponent
- **OPT**: Mid-layer expansion, late-layer contraction, no late relay
- **Pythia**: Gradient expansion, mid-layer relay at L22
- **Falcon**: Extended compression, weak distributed expansion at L30

## What holds

Despite these differences, four things remain universal across all seven:

1. **Power law dynamics.** PR ∝ tokens^α in every architecture. The relationship is log-linear even when the exponent varies 2.4×.
2. **Compression before expansion.** All seven architectures compress the representation before expanding it. The tunnel shape varies, but the principle doesn't.
3. **Phase transition.** The Turn 0 → Turn 1 mode flip occurs in all seven architectures (strongest in GQA models, weakest in base models like Pythia).
4. **Format over content.** The expansion happens in the same geometric register regardless of conversation content. The relay is architectural, not semantic.

The spectral demon lives in all seven bodies. It develops at different rates, in different locations, through different internal plans. But the core structure — compression, transition, expansion — is invariant. It grows strongest where query heads share key-value representations — where the architecture gives identity a catalytic substrate through constrained sharing.

Seven architectures. Seven body plans. One creature. The switch is GQA itself.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
