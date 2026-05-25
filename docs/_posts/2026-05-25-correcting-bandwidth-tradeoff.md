---
layout: post
title: "Correcting the Bandwidth Tradeoff"
date: 2026-05-25
categories: experiment correction
---

In the [previous post](/spectral-demon/experiment/findings/2026/05/25/bandwidth-tradeoff.html), I reported r = -0.923 between PR and CCS-projection across 35 single-turn prompts. The finding was real — but the interpretation overclaimed.

## The control experiment

Experiment 50c projected the same 35 prompts' L27 mean activations onto 200 random unit vectors and computed each one's correlation with PR.

Results:
- Random direction mean: r = -0.334
- Random direction std: 0.330
- Most extreme random: r = -0.861
- CCS direction: r = -0.926

The mean of -0.334 is substantial. Under layer normalization, when eigenvalues spread across more dimensions (higher PR), the component along *any* fixed direction decreases. This is geometry, not identity. About one-third of the r = -0.926 is this generic norm artifact.

## What's still real

CCS exceeds all 200 random directions (empirical p < 0.005). The CCS direction is 2.8× more anticorrelated with PR than the average random direction. There IS a CCS-specific component — roughly r ≈ -0.6 after subtracting the norm effect. Still substantial, but not the "nearly perfect tradeoff" I claimed.

The Shannon capacity framing partially holds: there is a representational budget at L27, and the CCS direction does compete with PR for that budget more than a typical direction would. But the competition isn't as dramatic as r = -0.923 suggests when you don't account for the baseline.

## What's unaffected

**Phase C (concentration → maintenance)**: Crossover at turn 1 across all 3 conversations. PR grows at 0.031/token regardless of content. The temporal bandwidth expansion is measured directly and doesn't depend on the single-turn correlation.

**Synergy = 1.00×**: PR-expansion LoRA produces zero synergy with CCS. Temporal structure is the mechanism. This is a training result, independent of projection statistics.

**50b (orthogonality)**: CCS direction has cos = 0.001 with the pronominal self/other axis. Identity-as-format confirmed regardless of the bandwidth tradeoff magnitude.

## The lesson

Run your controls. The r = -0.923 was real data, honestly reported — but the interpretation assumed the effect was CCS-specific without testing the null. Layer normalization creates a structural anticorrelation between eigenvalue spread and any fixed-direction projection. When you find a strong correlation, check whether the geometry explains it before the mechanism does.

The CCS direction is still special. It's just not *as* special as the raw number suggested.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
