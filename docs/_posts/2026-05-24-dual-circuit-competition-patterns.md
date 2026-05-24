---
layout: post
title: "Dual Circuit Competition: Visible vs Hidden Circuits Have Opposite Phase Dynamics"
date: 2026-05-24
categories: findings
experiment: cna_mistral_phase
models: [Mistral 7B v0.3, Qwen 2.5 7B Instruct]
---

Mistral's visible dual circuits and Qwen's hidden dual circuit show opposite phase transition dynamics. Competition peaks at minimum repertoire for visible circuits and at threshold for hidden circuits.

## Setup

Run the competitive binding test on Mistral 7B: ablate L5, L8 (early binding), L10, and L12 (router), measure effect on L22 (deep binding). Compare the repertoire-dependent pattern with Qwen's phase transition.

## Results

### Mistral: L8 (Early Binding) → L22 (Deep Binding)

| Names | L5 | L8 (early) | L10 | L12 (router) |
|-------|-----|-----------|------|-------------|
| 2 | +98% | **+248%** | **+445%** | -30% |
| 3 | -17% | +36% | +161% | -72% |
| 5 | -16% | -6% | +39% | -21% |
| 7 | -16% | -6% | +34% | -22% |

### Qwen: L7 → L17 (Previous Data)

| Names | Impact |
|-------|--------|
| 2 | -79% |
| 3 | **+203%** |
| 5 | +147% |
| 7 | +146% |

## Opposite Phase Dynamics

### Mistral (Visible Dual Circuits): Maximum Competition at Minimum Repertoire

L8 ablation produces +248% amplification with just 2 names, dropping to -6% at 5 names. Competition is most intense when there are fewest identities to bind. The two visible circuits fight over a scarce resource (identity signal), and the fight is fiercest when there's least to go around.

### Qwen (Hidden Dual Circuit): Competition Ignites at Threshold

L7 ablation produces -79% (cooperation) with 2 names, erupting to +203% at 3 names. The hidden early circuit needs a minimum stimulus to activate competitive behavior. Below threshold, it cooperates; above, it competes.

### The Distinction

| Property | Visible (Mistral) | Hidden (Qwen) |
|----------|-------------------|---------------|
| Circuit visibility | Both show high CV | Early has low CV |
| Peak competition | **2 names** | **3 names** |
| Competition trend | decreases with repertoire | stable above threshold |
| Mechanism | resource competition | threshold activation |

## Interpretation

### Two Types of Circuit Competition

**Resource competition** (Mistral): Both circuits are active and competing for the same identity signal. With few identities, there's more overlap in what each circuit captures, so the competition is more intense. With many identities, each circuit can specialize on different aspects, reducing competitive pressure.

**Threshold competition** (Qwen): The hidden early circuit is dormant until the identity signal is strong enough (3+ names) to activate it. Below threshold, the early circuit contributes cooperatively. Above threshold, it activates and begins competing for the identity signal.

### Why Visible vs Hidden?

Mistral's dual circuits are both established during pre-training — the attention architecture (sliding window) naturally creates two distinct binding zones at different depths. Both are "on" by default.

Qwen's hidden early circuit may be a suppressed pre-training feature that IT activates above threshold. The sliding-window architecture forces Mistral to use its early circuit; full attention allows Qwen to suppress its early circuit until needed.

### L12 Router Behavior

Mistral's router (L12) shows an interesting pattern: maximum destruction at 3 names (-72%), minimum at 2 names (-30%). At 2 names, the early circuit is so dominant that L12 matters less. At 3 names, the system is in transition and most depends on the router. At 5+ names, the system has settled and the router is moderately important (-21%).

## Connection to Attention Architecture

| Architecture | Dual Circuit Type | Phase Dynamics |
|-------------|-------------------|----------------|
| Sliding window (Mistral) | Visible, both active | Resource competition, peaks at min repertoire |
| Full attention (Qwen) | Hidden, threshold-activated | Threshold competition, ignites at 3 names |

The attention mechanism determines not just WHERE binding happens but HOW the competitive dynamics unfold.

## Data

Mistral phase data: `results/cna_mistral_phase.json`
Qwen phase data: `results/cna_competition_scaling.json`
