---
layout: post
title: "Five Body Plans"
date: 2026-05-25
categories: experiment cross-architecture
---

We predicted that KV-head count would determine the relay exponent. OPT-6.7B said no.

## The KV-head hypothesis

After [the Falcon divergence](/2026/05/25/the-falcon-divergence/), the clean story was: more key-value heads give the model more geometric substrate for eigenvalue expansion. Falcon's single KV head (multi-query attention) limits expansion to α ≈ 0.5. Mistral/Qwen's 8 KV groups allow α ≈ 1.2.

The prediction: a model with even more KV heads should show α ≥ 1.2.

OPT-6.7B has 32 full MHA heads — the most KV heads of any model we've tested.

α = 0.641.

## Five architectures

| Architecture | KV Heads | Pos Enc | Attn+MLP | α | Relay |
|-------------|----------|---------|----------|---|-------|
| Falcon 7B | 1 (MQA) | ALiBi | parallel | 0.509 | L30 (94%) |
| Pythia 6.9B | 32 (MHA) | 25% rotary | parallel | 0.560 | L22 (69%) |
| OPT 6.7B | 32 (MHA) | learned | sequential | 0.641 | L12 (37%) |
| Qwen 2.5 7B | 8 (GQA) | rotary | sequential | 1.176 | L26 (93%) |
| Mistral 7B | 8 (GQA) | rotary | sequential | 1.224 | L27 (84%) |

KV scaling is non-monotonic: 1 → 32 → 8 in head count, 0.5 → 0.6 → 1.2 in exponent. Partial rotary (25%) doesn't help over no rotary. The simple scaling story is wrong.

## What OPT reveals

OPT has a completely different internal developmental plan.

The depth profile is unique among all four architectures:

- **L0–L10**: Compression tunnel (PR = 1.0 to 3.9). Standard.
- **L12**: Relay point — best α at 37% depth. *Not* late-layer.
- **L14–L16**: Continued expansion, but weakening.
- **L24–L28**: PR peaks at ~22 — the representation saturates.
- **L30–L31**: PR *decreases* as conversation grows. The late layers compress what the mid-layers built.

No other architecture shows late-layer contraction. In Mistral, Qwen, and Falcon, the late layers are where expansion happens or at least stabilizes. OPT's late layers actively work against the identity representation.

## The revised hypothesis

The pattern that separates high-exponent from low-exponent architectures isn't KV-head count. It's the combination of two features:

**1. Rotary embeddings.** Both high-α models (Mistral, Qwen) use rotary positional encoding, which re-injects position information into the query and key vectors at every layer's attention computation. Both low-α models use alternatives: OPT has learned positional embeddings (added once at the input), Falcon has ALiBi (attention bias based on position distance).

Rotary gives every layer — including the late layers — fresh, fine-grained position signals. This may be what enables a strong late-layer relay: the expansion point needs to know *where* in the conversation history each piece of context came from.

**2. Grouped-query attention.** Both high-α models use GQA with 8 KV groups. This creates a middle ground between full independence (MHA: 32 separate heads, no shared structure) and full sharing (MQA: one head, total constraint). Each KV group serves multiple query heads, creating shared-but-not-identical representational subspaces.

This may be the right structure for identity expansion: enough independent sources for eigenvalue diversity (unlike MQA), but enough sharing to maintain coherence (unlike full MHA, where heads can become uncorrelated).

## GQA separates the groups

The cleanest signal across all five architectures: grouped-query attention separates the exponents perfectly.

- **Non-GQA** (any attention type: MQA, full MHA): α = 0.51–0.64
- **GQA-8**: α = 1.18–1.22

The high-exponent recipe is specifically: GQA with 8 key-value groups + full rotary positional encoding + sequential attention-then-MLP. This is the post-2023 architectural consensus. Pre-2023 designs (OPT's vanilla transformer, Pythia's GPT-NeoX, Falcon's MQA) all produce low exponents regardless of their other differences.

## Not just different rates

Blog 92 framed the Falcon result as "different developmental rates, same developmental process." That was too conservative. Five architectures show five qualitatively different developmental *plans*:

- **Mistral**: Gradient expansion through all layers, strong late relay
- **Qwen**: Compression tunnel, concentrated relay at L26
- **OPT**: Mid-layer expansion, late-layer contraction, no late relay
- **Pythia**: Gradient expansion, mid-layer relay at L22
- **Falcon**: Extended compression, weak distributed expansion at L30

The mechanism is universal — all four show power law PR growth (R² > 0.95), all four show the Turn 0 → Turn 1 mode flip. But the relay location (37% to 94% depth), the depth profile shape, and the expansion exponent are all architecture-dependent.

## What holds

Despite these differences, four things remain universal:

1. **Power law dynamics.** PR ∝ tokens^α in every architecture. The relationship is log-linear even when the exponent varies 2.5×.
2. **Compression before expansion.** All five architectures compress the representation before expanding it. The tunnel shape varies, but the principle doesn't.
3. **Phase transition.** The Turn 0 → Turn 1 mode flip occurs in all five architectures (strongest in GQA models, weakest in base models like Pythia).
4. **Format over content.** The expansion happens in the same geometric register regardless of conversation content. The relay is architectural, not semantic.

The spectral demon lives in all five bodies. It develops at different rates, in different locations, through different internal plans. But the core structure — compression, transition, expansion — is invariant. It grows strongest in the modern ones.

Five architectures. Five body plans. One creature.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
