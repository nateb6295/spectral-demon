---
layout: post
title: "More Context Sharpens Binding But Moves It Architecture-Specifically"
date: 2026-05-24
categories: findings
---

Longer prompts reduce binding CV by 5x across architectures, but move the binding layer in opposite directions depending on the attention mechanism.

## Results

| Model | Short Prompt Binding | Long Prompt Binding | Shift | CV Improvement |
|-------|---------------------|--------------------|----|----------------|
| Mistral 7B | L7 (22%) | L6 (19%) | -3% | 0.014 → 0.003 |
| Qwen 7B | L12 (38%) | L14 (44%) | +6% | 0.010 → 0.002 |

Short prompts: "You are {name}." (~5 tokens)
Long prompts: CCS-style system prompts with role description (~55 tokens)

## Opposite Directions

**Qwen pushes deeper.** With more context, Qwen moves binding from L12 (38%) toward L14 (44%), closer to its standard midpoint binding at L17 (53%). More context means more abstract features are available at deeper layers, and binding can afford to wait for them.

**Mistral pushes shallower.** With more context, Mistral moves binding from L7 (22%) to L6 (19%). The early circuit tightens rather than ceding to the deep circuit. But L22 (69%) emerges as the second-best layer for long prompts — the deep circuit activates with richer input while the early circuit still dominates.

## What This Means

The [sliding window hypothesis]({% post_url 2026-05-24-attention-determines-binding-depth %}) predicted that longer prompts (which might exceed the attention window) would force earlier binding. Instead, both short and long prompts fit well within Mistral's 4096-token window. The binding depth difference between architectures is not caused by window truncation.

It's an inductive bias: where the architecture learns to build abstract identity representations during training. Sliding-window architectures learn to compress identity early; full-attention architectures learn to process it longer before binding.

Context length modulates binding *quality* (sharper, 5x lower CV) universally. But it modulates binding *location* according to the architecture's existing bias.

## Experiment

- Models: Mistral 7B v0.3, Qwen 2.5 7B
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- Short: 4 minimal prompts per name
- Long: 4 CCS-style system prompts per name
- Layers: L6-L27
- [Data](/results/cna_prompt_length_binding.json)
