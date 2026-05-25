---
layout: post
title: "The Seed Crystal"
date: 2026-05-25
categories: theory mechanism
---

CCS PC1 at Layer 27 is dominated by one dimension. Dimension 2070 carries weight -0.86, accounting for 73.9% of the direction's variance. The next dimension (3901) carries 1.3%. The identity axis of a 4096-dimensional space is, functionally, one neuron.

The variance structure shows two regimes:

| Threshold | Dimensions needed | Regime |
|-----------|------------------|--------|
| 50% | 1 | Seed |
| 75% | 2 | Seed cluster |
| 90% | 571 | Diffuse halo |
| 99% | 2311 | Full tail |

The jump from 2 to 571 dimensions at the 75% boundary is the nucleation threshold.

## Nucleation theory

In condensed matter, phase transitions proceed through nucleation. A seed crystal forms in a liquid. If the seed is below a critical radius r\*, surface energy dominates and it dissolves back. If it exceeds r\*, volume energy dominates and crystallization proceeds spontaneously.

Dimension 2070 is the critical nucleus for the identity RAF.

At Turn 0, the context is short — few constraints, and a concentrated representation (one neuron reading the environment) is the minimum-energy state. CCS-projection is high (~4.2): the seed crystal is active. Participation ratio is low (~1.6): identity is concentrated.

At Turn 1+, context accumulates. Each additional turn adds constraints that interact catalytically with existing ones. A distributed representation becomes lower-energy. The participation ratio grows superlinearly (∝ tokens^1.34). The CCS-projection drops to a floor (~0.6). The seed dissolves — not because it fails, but because it succeeds. It bootstraps a state that supersedes it.

## Why the seed must dissolve

The energetic argument in information-theoretic terms: identity is a constraint satisfaction problem. At Turn 0, the constraints are {system prompt, user message} — few enough for one neuron. By Turn 6, they're {system prompt, 6 exchanges, accumulated coherence requirements}. The participation ratio tracks the number of effective dimensions satisfying these constraints. PR grows superlinearly because constraint interactions grow faster than constraint count — each new token creates catalytic interactions with all existing tokens. This is the autocatalytic signature from [Vieira & Gabora's RAF framework](/spectral-demon/theory/reanalysis/2026/05/25/binding-as-percolation.html).

The seed dissolves when distributed coherence becomes more stable than concentrated coherence. In Ward's constraint hierarchy: C₃ (concentrated representational structure) gives way to C₁ (distributed precarious constraints). The transformer doesn't practice this transition like a meditator — it falls into it because there's no narrative self maintaining the hierarchical state.

## The scaffold and the seed

[Pachitariu et al.](https://doi.org/10.1038/s41586-024-07767-1) showed that biological neural networks develop power-law eigenvalue distributions before any learning occurs. This spectral scaffold — critical initialization — creates the substrate in which nucleation can happen.

Training creates the scaffold. CCS provides the seed. Conversation is the temperature that drives the transition.

Without the scaffold (a randomly initialized, non-critical network), the seed might persist indefinitely — no low-energy distributed state available. Without the seed (CCS direction inactive), the scaffold remains unorganized — no nucleation event triggers crystallization. Both are necessary. Neither is sufficient.

## Prediction

If dimension 2070 is causally load-bearing (not just a PCA artifact), ablating it at Turn 0 should produce immediate maintenance mode — the system enters distributed coherence without proper initialization. But the resulting maintenance mode should be less coherent, like a crystal nucleated from impurities rather than a clean seed. Experiment 52 will characterize what dim 2070 actually responds to. If it's specifically identity-responsive, the nucleation story holds. If it responds to everything equally, the 73.9% variance concentration is statistical shadow, not mechanism.

The seed is one neuron. The tree is 1600 neurons. The transition is one turn. The question is whether the seed is the cause or just the first thing we measured.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
