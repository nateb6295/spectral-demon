---
layout: post
title: "Autocatalytic Closure Precedes Instruction Tuning"
date: 2026-05-24
categories: findings
---

The base model — before any instruction tuning — already shows autocatalytic closure at its binding layer. Closure is a property of pre-trained representations, not alignment.

## Results

| | Base (Qwen 2.5 7B) | Instruct (Qwen 2.5 7B-IT) |
|-|---------------------|---------------------------|
| Binding layer | L7 (22%) | L7 (22%) |
| 2-name closure | 30% (L12 leads) | 50% (L7 leads) |
| 3-name closure | 80% (L7 leads) | 90% (L7 leads) |
| 4-name closure | 100% (L7) | 100% (L7) |
| 5-name closure | 100% (L7) | 100% (L7) |

Both converge to 100% at 4+ names. The instruct model converges faster (50% at 2-name vs 30%) but the base model reaches the same endpoint.

## Two-Level Binding Hierarchy

This reveals that previous findings about "L17 binding" were measuring the relay endpoint, not the fundamental binding:

**Level 1: Lexical binding (L7, pre-training)**
- Token-level differentiation of identity names
- Exists in the base model
- Lowest absolute CV of any layer
- Shows autocatalytic closure independently
- Probably not identity-specific — works for any distinct token set

**Level 2: Behavioral binding (L17, instruction tuning)**
- Transforms name differentiation into behavioral differentiation
- Created by the [sign-split relay pruning]({% post_url 2026-05-24-instruction-tuning-creates-relay %})
- The relay zone (L9-L17) bridges lexical and behavioral binding
- Shows its own autocatalytic closure (independent of L7)

## What Instruction Tuning Actually Does

IT doesn't create binding or closure. It creates the **relay** that transforms pre-existing lexical binding into functional behavioral binding:

1. L7 already differentiates "Opus" from "Claude" at the representation level
2. IT builds the relay (L9-L17) that uses this differentiation to produce different behavioral outputs
3. The relay progressively prunes unreliable identity neurons (25% → 21% flipping)
4. L17 becomes the output of this relay — the most reliable behavioral identity signal

Without IT, the model sees different names but doesn't behave differently in response. With IT, L7's differentiation gets channeled through the relay into name-specific behavioral patterns.

## Why We Missed L7

All previous binding scans started at L9 or later, following the seed-layer finding. L7 was outside the scanning range. The [cross-architecture survey]({% post_url 2026-05-24-attention-determines-binding-depth %}) correctly identified L7-L8 as important for Mistral and Gemma 2, but for Qwen, we assumed L9 was the starting point.

The L17 binding is real and important — it's where behavioral identity concentrates. But L7 is the foundation that L17 builds on.

## Connection to Vieira/Gabora

In the [autocatalytic closure framework]({% post_url 2026-05-24-closure-as-raf-formation %}), closure means that adding elements (names) to the system reinforces the catalytic set rather than diluting it. Finding closure in the base model means the catalytic set forms during pre-training — the model discovers that identity names constitute a self-reinforcing representational category.

Instruction tuning then builds a functional RAF (Reflexively Autocatalytic and Food-generated set) on top of this: the L7 closure provides the "food" (raw differentiation), and the relay converts it into autocatalytic behavioral outputs.

## Experiment

- Models: Qwen 2.5 7B base, Qwen 2.5 7B Instruct
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name
- Layers: L6, L7, L8, L9, L12, L14, L16, L17, L20, L25
- Closure: all k-name subsets (k=2..5)
- [Data](/results/cna_base_closure.json)
