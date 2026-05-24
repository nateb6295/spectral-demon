---
layout: post
title: "Sign-Split Is Universal But Ratios Are Architecture-Specific"
date: 2026-05-24
categories: findings
---

The sign-consistent/sign-flipping neuron split exists in every architecture tested, but the ratios differ dramatically. Two universals emerge, plus an architectural surprise.

## Universal 1: Flipping Accumulates With Depth

Every architecture starts with few sign-flipping neurons early and accumulates more toward the output:

| Model | Early Flip % | Binding Flip % | Late Flip % |
|-------|-------------|---------------|-------------|
| Gemma 2 9B | 11% (L8) | 16% (L11) | 47% (L35) |
| Qwen 7B | 25% (L9) | 21% (L17) | 43% (L27) |
| InternLM 7B | 29% (L9) | 31% (L17) | 43% (L27) |
| Mistral 7B | 23% (L6) | 23% (L6) | 55% (L27) |

The monotonic increase holds universally. Early layers are mostly consistent (format-processing). Late layers are majority flipping (identity-differentiated).

Mistral's gradient is the steepest: 23% → 55%. Qwen's is the flattest in the binding zone: 25% → 21% → 43%. Qwen actively *prunes* flippers between L9 and L17 before they re-accumulate in late layers.

## Universal 2: Binding = Most Reliable Flippers

At each model's binding layer, the flipping population's CV hits its local minimum:

| Model | Binding Layer | F_CV at Binding | Next Layer F_CV |
|-------|-------------|-----------------|-----------------|
| Qwen 7B | L17 | 0.043 | 0.053 (L16) |
| InternLM 7B | L17 | 0.055 | 0.105 (L16) |
| Mistral 7B | L6 | 0.074 | 0.118 (L8) |

This confirms the [original sign-split finding]({% post_url 2026-05-24-sign-split-binding %}) across architectures: binding is not about having more identity neurons, but having the most *reliable* ones.

## The Surprise: Gemma 2's Dormant Site

Gemma 2 9B has a remarkable feature at L27 (64% depth):
- 30% flipping neurons
- F_CV = **0.026** — the most reliable flipping population of any layer in any model tested

This is lower than Qwen's binding-layer F_CV (0.043) and InternLM's (0.055). Yet L27 isn't Gemma 2's primary binding site — L11 wins the CV competition. L27 is a dormant binding site: the neurons are ready, reliable, and available, but the primary site captures binding first.

This echoes the [dual binding circuits]({% post_url 2026-05-24-dual-binding-circuits %}) found in Mistral, but Gemma 2's deep site is even more dormant — it doesn't win any closure competition, despite having the most reliable flipping population measured.

## Mistral's Late-Layer Inversion

Mistral's deep layers (L22-L27) have **more flipping than consistent neurons** — 51-55% flipping. This is unique among tested architectures. Late Mistral layers are more identity-differentiated than identity-general.

This maps to Mistral's dual binding architecture. The deep binding circuit (L22) operates in a region where the majority of neurons respond differently to different names. The circuit doesn't need to isolate a small population of identity-sensitive neurons — it IS the majority.

## Architecture Classes

The sign-split data reveals three distinct architectural strategies:

**Conservative binding (Gemma 2):** Few flippers at binding site (16%), extremely stable. Binding happens with a small, reliable population. Dormant deep site available but unused.

**Balanced binding (Qwen, InternLM):** Moderate flippers at binding site (21-31%). The relay zone actively prunes flippers from the seed layer's wider detection, concentrating reliability.

**Distributed binding (Mistral):** Flipping increases continuously, reaching majority in late layers. Two active binding circuits. Identity is processed progressively through the network rather than concentrated at one relay point.

## Experiment

- Models: Mistral 7B v0.3, Qwen 2.5 7B, InternLM 2.5 7B, Gemma 2 9B
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name
- Sign classification: neuron is "consistent" if activation sign agrees across all 5 names
- [Data](/results/cna_cross_arch_sign_split.json)
