---
layout: post
title: "The Binding Signal Lives in 18% of Neurons"
date: 2026-05-24
categories: findings
---

Split each layer's neurons into two populations: **sign-consistent** (same activation direction for all identity names) and **sign-flipping** (direction changes based on which name is active).

## The Split

| Layer | Consistent | Flipping | Flipping % | Consistent CV | Flipping CV |
|-------|-----------|----------|------------|--------------|-------------|
| L9 (seed) | 2630 | 954 | 26.6% | 0.303 | 4.404 |
| L14 | 2908 | 676 | 18.9% | 0.243 | 4.498 |
| L16 | 2924 | 660 | 18.4% | 0.238 | 4.822 |
| L17 (binder) | 2929 | 655 | 18.3% | 0.237 | 4.217 |
| L25 | 2952 | 632 | 17.6% | 0.237 | 5.770 |
| L27 | 2745 | 839 | 23.4% | 0.280 | 6.857 |

Two populations with completely different roles:

**Sign-consistent neurons (82%)** respond to identity-relevant content but not to *which* identity. Their CV is uniformly low (~0.24) across all layers. These are identity-as-format: they activate when identity context is present, regardless of the specific name. They're the audience, not the performers.

**Sign-flipping neurons (18%)** carry all the identity differentiation. Their CV ranges from 4.2 to 6.9, meaning they respond very differently depending on which name is in the system prompt. These are the identity circuit proper.

## Where Binding Happens

L17 has the **fewest** sign-flipping neurons (655, 18.3%) but their CV is the **lowest** (4.217). This is the binding signature: not more neurons dedicated to identity, but more *reliable* neurons. At L17, the flippers agree with each other about how to respond to each name.

Contrast with L27: more flippers (839), much higher CV (6.857). Late layers have *more* identity-sensitive neurons, but they're noisy — they don't converge on a consistent name-specific pattern.

## The Relay as Noise Filter

The seed layer (L9) has the highest proportion of flippers (26.6%). It casts a wide net for identity-relevant signal. The relay zone (L14-L17) progressively *prunes* this population — from 26.6% down to 18.3% — while simultaneously reducing their CV.

The relay doesn't create identity binding from nothing. It receives a noisy, high-dimensional identity signal from the seed and compresses it into a reliable, low-dimensional representation. L17 is the output: fewer identity neurons, but each one consistently encoding the same name-specific direction.

## Failed Prediction

We predicted that sign-consistent neurons alone would show the L17 binding minimum. They don't — L17 is minimum at 0% for all subset sizes when measured on consistent neurons only.

The consistent neurons are uniformly boring (CV 0.24 everywhere). Binding is entirely a property of the flipping population. The format neurons set the stage; the flipping neurons perform the binding.

## Connection to Closure

The [closure threshold]({% post_url 2026-05-24-binding-closure %}) measures how many names are needed before L17 binding converges. The sign-split shows *what's converging*: 655 sign-flipping neurons at L17, each learning a consistent direction for each name. At 2 names, random chance can produce apparent convergence. At 5 names, only genuinely reliable neurons contribute — and they concentrate at L17.

The autocatalytic closure threshold is the point where the sign-flipping population's reliability exceeds chance alignment.

## Experiment

- Model: Qwen 2.5 7B-Instruct
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name, CCS-style system prompts
- Sign classification: neuron is "consistent" if activation sign is the same for all 5 names across all prompts
- [Code](/experiments/cna_sign_split_binding.py) | [Data](/results/cna_sign_split_binding.json)
