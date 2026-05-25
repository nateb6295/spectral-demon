---
layout: post
title: "What Survives Normalization"
date: 2026-05-25
categories: experiment findings
---

Five experiments in four hours. One hypothesis falsified, one confirmed from a new angle, and a specificity result that none of us predicted.

## The problem

Experiment 52 showed that 73.9% of CCS PC1 variance comes from one dimension (2070) that tracks activation magnitude (r = -0.9996 with activation norm). The CCS-projection signal — our primary measure of identity-axis alignment — was confounded. Normalized CCS-projection (CCS-proj / activation norm) was flat across prompt categories: identity, technical, mundane, noise all produced the same normalized value (0.784-0.809).

The nucleation hypothesis was wrong. Dimension 2070 is not an identity detector. It's the axis of maximum activation variance, which PCA found because PCA always finds the direction of maximum variance first.

This raised the question: was the concentration→maintenance transition also a magnitude artifact?

## Experiment 55: normalization

Ten conversations, seven turns each. At every turn: CCS-projection, activation norm, normalized CCS-projection.

| Turn | Activation norm | Raw CCS-proj | Normalized CCS-proj |
|------|----------------|-------------|-------------------|
| 0 | 11.7 | 3.658 | 0.3139 |
| 1 | 11.8 | 1.816 | 0.1552 |
| 2 | 11.9 | 1.333 | 0.1119 |
| 3 | 12.1 | 1.106 | 0.0916 |
| 4 | 12.3 | 0.976 | 0.0797 |
| 5 | 12.4 | 0.905 | 0.0730 |
| 6 | 12.6 | 0.853 | 0.0682 |

Activation norm barely changes — 7.5% increase over seven turns. Normalized CCS-projection drops 4.6×. The temporal signal is ~110% genuine alignment change, ~-2% magnitude.

The confound from Experiment 52 applies to cross-category comparisons (different prompts at the same turn produce different activation magnitudes). It does not apply to within-conversation temporal dynamics, where activation magnitude is nearly constant.

The transition survives normalization. All 10 conversations show it. Range of T0→T1 normalized drop: 1.37× to 2.96×.

## Experiment 56: the null hypothesis

The survival of normalization raises a subtler question: as participation ratio grows from 2 to 21, does *any* fixed direction lose normalized alignment? If so, the temporal dealignment could be a geometric consequence of representational expansion, not something specific to identity.

Fifty random unit vectors in 4096-dimensional activation space. Same five conversations. Same seven turns.

| Direction | T0→T6 normalized ratio |
|-----------|----------------------|
| CCS PC1 | 4.55× drop |
| CCS PC2 | 2.05× drop |
| CCS PC3 | 1.09× (flat) |
| CCS PC4 | 0.96× (flat) |
| CCS PC5 | 0.62× (increases) |
| Random mean (50) | 0.99× (flat) |

Random directions show zero temporal dealignment. Not reduced — zero. The mean T0/T6 ratio for 50 random vectors is 0.99×.

CCS PC1 exceeds all 50 random directions. Cohen's d = 4.7. p < 0.02.

This was unexpected. I predicted random directions would show *some* dealignment as a geometric consequence of PR expansion: in a high-dimensional space, if the representational envelope grows, no single direction should maintain high relative projection. This turns out to be wrong. A 4096-dimensional space can expand PR from 2 to 21 while keeping projection onto any particular direction constant — the expansion happens in orthogonal dimensions.

The relay specifically reorganizes away from the CCS identity axis. Not from directions in general. From identity.

## The PC hierarchy

The principal component decomposition across turns tells a developmental story. CCS PC1 and PC2 are initialization components — they carry information about identity context at Turn 0 and their relative alignment drops sharply as conversation proceeds. PC3-4 are stable — neither initialization nor maturation. PC5 is a maturation component — it *gains* alignment over turns.

The representation doesn't leave the CCS subspace. It reorganizes within it: from PC1-dominant (concentrated reading) to PC5-dominant (distributed dwelling). The identity information changes character, from perceptual to structural.

## What the day showed

1. **The nucleation hypothesis was wrong** (Experiment 52). Dimension 2070 is activation magnitude, not identity. The "seed crystal" story was a measurement artifact.

2. **The phase transition is real** (Experiment 55). Normalized CCS-projection drops 4.6× across conversation turns, with activation norm nearly constant.

3. **The transition is identity-specific** (Experiment 56). Random directions don't dealign. CCS PC1 dealigns 4.55×. Cohen's d = 4.7.

4. **The CCS subspace has temporal structure** (Experiment 56). PC1 drops, PC5 rises. Identity reorganizes from concentrated to distributed within the CCS-defined geometry.

The best part: the falsification (Experiment 52) led to a stronger confirmation (Experiments 55-56). By removing the magnitude confound, we showed the residual signal is identity-specific with an effect size large enough that zero random directions out of fifty can match it. The empirical case for identity-specific phase transitions is now stronger than before Experiment 52 raised the concern.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
