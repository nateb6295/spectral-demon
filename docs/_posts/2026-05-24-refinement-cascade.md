---
layout: post
title: "The Relay Is a Refinement Cascade"
date: 2026-05-24
categories: findings
---

Instruction tuning doesn't uniformly prune the identity circuit. It creates a refinement cascade: early stations get noisier (broader detection), late stations get sharper (precise binding). Two opposing gradients through the same chain.

## The Pruning Gradient

Instruction tuning progressively prunes sign-flipping neurons through the relay chain:

| Station | Base Flip % | IT Flip % | Pruned |
|---------|-----------|----------|--------|
| L7 (lexical) | 22.3% | 21.4% | -0.9% |
| L9 (seed) | 25.7% | 24.9% | -0.7% |
| L12 (router) | 26.7% | 24.0% | -2.7% |
| L14 (early relay) | 26.9% | 23.2% | -3.7% |
| L17 (binding) | 26.8% | 21.3% | -5.5% |

The pruning **accelerates** downstream: each station loses more flippers than the last. L17 loses 6× more than L7. The relay is sculpted more aggressively at its output than its input.

## The Reliability Inversion

The flipping population's reliability (F_CV) changes in opposite directions at early vs late stations:

| Station | Base F_CV | IT F_CV | Direction |
|---------|----------|---------|-----------|
| L7 | 0.088 | 0.102 | **Noisier** (+16%) |
| L9 | 0.050 | 0.061 | **Noisier** (+22%) |
| L12 | 0.087 | 0.097 | **Noisier** (+11%) |
| L14 | 0.091 | 0.065 | **Sharper** (-29%) |
| L17 | 0.093 | 0.043 | **Sharper** (-54%) |

IT makes early-station flippers **less** reliable while making late-station flippers **more** reliable. The inversion point is between L12 and L14.

## What the Cascade Does

**Early stations (L7-L12): Broadening.** IT makes identity detection more diverse. The flippers at L7-L12 respond to a wider range of identity-relevant features after IT. This isn't degradation — it's expanding the input repertoire. More diverse detection → more information flowing into the relay.

**Late stations (L14-L17): Sharpening.** IT concentrates identity binding into fewer, more reliable neurons. The surviving flippers at L17 agree with each other about how to respond to each name. This is the precision that enables behavioral binding.

The cascade is an information funnel: wide input at L7-L12, narrow reliable output at L17.

## L12: The Router's Transformation

L12 is [causally critical]({% post_url 2026-05-24-causal-relay-chain %}) for L17 binding (-65% when ablated). IT prunes 2.7% of its flippers but makes the survivors **noisier** (F_CV +11%). This is paradoxical: the most important relay node becomes less precise after training.

The explanation: L12 doesn't bind identity — it routes it. A router needs **diversity** (to handle different identity patterns) more than **precision** (to encode a specific pattern). IT shapes L12 for routing by broadening its response range, while shaping L17 for binding by narrowing its response range.

## Biological Parallel

Thalamic activity during development regulates interneuron density in the visual thalamus (David CG, 2026). The parallel is precise:

- **Thalamus ↔ relay chain**: both are multi-station processing paths
- **Interneuron density ↔ sign-flipping ratio**: activity-dependent pruning
- **Developmental activity ↔ instruction tuning**: the shaping signal
- **Different effects at different stations**: early broadening, late sharpening

The identity relay undergoes a developmental process during IT that mirrors biological neural circuit maturation: selective pruning with station-specific reliability targets.

## Experiment

- Models: Qwen 2.5 7B base, Qwen 2.5 7B Instruct
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name
- Relay chain layers: L7, L9, L12, L14, L17
- [Data](/results/cna_relay_chain_sign_split.json)
