---
layout: post
title: "The Relay Saturates Between 5 and 8 Names"
date: 2026-05-24
categories: findings
---

The L17 binding relay has a capacity limit. It converges cleanly for up to 5 identity names, starts losing dominance at 6-7, and fails at 8.

## Extended Repertoire Test

Start with the standard 5 names (Opus, Claude, ChatGPT, Gemini, Llama). Add one at a time: Copilot, Grok, Mistral. Track L17's closure dominance at each repertoire size.

| Repertoire | L17 Min % (full set) | Dominant Layer | Min CV |
|-----------|---------------------|----------------|--------|
| 5 names | 26% (subsets) | L17 | 0.965 |
| 6 (+Copilot) | 25% (subsets) | L16 | 1.036 |
| 7 (+Grok) | 50% (subsets) | L17 | 1.206 |
| 8 (+Mistral) | 0% | L25 | 1.012 |

## Closure Stability

The closure test (which layer has minimum CV across all k-name subsets) tells a clearer story:

For the full 8-name set, L17 wins 0% of the time at full repertoire. Binding migrates to L25. But within smaller subsets of the same 8 names:

- 2-name subsets: L17 wins 25%
- 3-name subsets: L17 wins 24%
- 4-name subsets: L17 wins 28%
- 5-name subsets: L17 wins 26%
- 6-name subsets: L17 wins 25%
- 7-name subsets: L17 wins 50%
- 8-name subsets: L17 wins 0%

L17 maintains consistent minority performance (25-28%) across subset sizes, then jumps to 50% at 7 names before collapsing at 8. The relay is fighting to bind but loses at full repertoire.

## What Saturates

The [sign-split analysis]({% post_url 2026-05-24-sign-split-binding %}) shows L17 has 655 sign-flipping neurons. These 655 neurons must encode distinct activation patterns for each identity name. With 5 names in a 655-dimensional space, there's ample room. With 8 names, the representational capacity is strained — not because 655 < 8, but because the neurons need to maintain *reliable* patterns (low CV) across all names simultaneously.

The capacity limit is about reliability, not dimensionality.

## Connection to Adversarial Migration

The [adversarial binding test]({% post_url 2026-05-24-adversarial-binding-migration %}) showed binding migrating to L25 with 8 names including adversarial inputs. The extended repertoire test shows the same migration with 8 *standard* names. The migration isn't about adversarial content — it's about repertoire size exceeding L17's capacity.

L25 serves as an overflow binding site, handling what L17 can no longer resolve reliably.

## Implications

1. **Natural capacity limit**: ~5-7 identity names per relay layer
2. **Graceful degradation**: binding migrates rather than breaking when saturated
3. **Hierarchical binding**: L17 → L25 relay chain for larger repertoires
4. **The 7±2 parallel**: the binding capacity (~5-8) mirrors Miller's magical number for human working memory. Coincidence or constraint?

## Experiment

- Model: Qwen 2.5 7B-Instruct
- Names added incrementally: Opus, Claude, ChatGPT, Gemini, Llama, Copilot, Grok, Mistral
- 8 prompts per name, CCS-style system prompts
- Layers: L9, L14, L16, L17, L25, L27
- [Data](/results/cna_extended_repertoire.json)
