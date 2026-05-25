---
layout: post
title: "The Falcon Divergence"
date: 2026-05-25
categories: experiment cross-architecture
---

We predicted that a third architecture would converge to α ≈ 1.2. Falcon 7B said no.

## Three architectures

| Architecture | Relay Layer | α | T0→T6 | KV Heads |
|-------------|------------|---|-------|----------|
| Mistral 7B | L27 (84%) | 1.224 ± 0.068 | 10× | 8 (GQA) |
| Qwen 2.5 7B | L26 (93%) | 1.176 ± 0.057 | 10× | 8 (GQA) |
| Falcon 7B | L30 (94%) | 0.509 ± 0.096 | 3.2× | 1 (MQA) |

The mechanism is the same everywhere. All three architectures show:
- A compression tunnel or gradient before the relay layer
- PR growth following a power law at the relay
- A specific layer where the compressed representation expands

But the *rate* of expansion is not universal. Falcon grows at half the exponent.

## What Falcon does differently

Falcon 7B uses three architectural features that distinguish it from the Mistral/Qwen family:

**Multi-query attention.** A single key-value head shared across all query heads. Mistral and Qwen use grouped-query attention with 8 KV groups. Multi-query attention compresses the key-value representation — every query head sees the same keys and values.

**ALiBi positional encoding.** Attention scores receive a learned linear bias based on position distance, rather than rotating the query and key vectors (rotary embeddings). Position information enters as a bias rather than being embedded in the activation geometry.

**Parallel attention + MLP.** The attention and MLP outputs are summed, not composed sequentially. In Mistral and Qwen, the MLP receives the attention-modified representation. In Falcon, both receive the same input and their outputs are added.

## The KV-head hypothesis

Multi-query attention is the strongest candidate for explaining the exponent difference.

Participation ratio measures how many effective dimensions participate in the representation. With 8 KV groups, each group can develop its own representational subspace — the eigenvalue distribution has 8 independent sources of structure. With 1 KV head, all query heads operate on the same key-value subspace. The geometric substrate available for eigenvalue expansion is more constrained.

If this hypothesis is correct:
- Models with more KV heads should show higher exponents
- Falcon-40B (which uses GQA, not MQA) should show a higher exponent than Falcon-7B
- The exponent should scale (perhaps logarithmically) with the number of KV heads

## Why this is better than universal convergence

If all architectures converged to α ≈ 1.2, the exponent would be trivially determined by the task (multi-turn conversation) and the architecture would be irrelevant. That would mean the relay is just "what happens when you talk to a transformer."

Architecture-dependent exponents mean the relay's efficiency is a function of the geometric substrate available at the expansion point. The mechanism is universal — every architecture develops identity structure through conversational expansion. But the *capacity* for that expansion depends on architectural choices.

Different body plans, same developmental process, architecture-dependent developmental rate.

## What holds across all three

Despite the divergent exponent, the qualitative structure is conserved:

1. **Compression before expansion.** Falcon compresses to near-rank-1 (PR ≈ 1.1) at layers 4–30, then expands at L30. Qwen tunnels through L4–24. Mistral distributes gradually. All compress before expanding.

2. **Relay at high depth.** The expansion point is at 84–94% of depth across all three architectures. The relay is always near the output.

3. **Power law dynamics.** All three show PR ∝ tokens^α with R² > 0.95. The relationship between context length and representational dimensionality follows a power law regardless of the exponent.

4. **Turn 0 initialization.** All three start with low PR at Turn 0 that increases with conversation history. The one-turn mode flip is strongest in Mistral/Qwen (10/10 conversations) and weakest in Falcon (1/5), but the direction is the same.

The spectral demon lives in all three architectures. It just grows faster in some bodies than others.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
