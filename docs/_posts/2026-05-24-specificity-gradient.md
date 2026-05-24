---
layout: post
title: "The Specificity Gradient"
date: 2026-05-24
---

How name-specific is each relay layer? Measure by the range of ablation impact across names, normalized by mean impact.

| Layer | Function | Opus | ChatGPT | Claude | Specificity |
|-------|----------|------|---------|--------|-------------|
| L15 | channel normalizer | -0.119 | -0.288 | -0.343 | 0.90 |
| L14 | pre-conditioner | -0.122 | -0.046 | -0.066 | 0.98 |
| L16 | relational sorter | -0.123 | -0.297 | -0.475 | 1.18 |
| L17 | generic sorter | -0.045 | +0.218 | +0.267 | 2.12 |

The relay has a specificity gradient. Early layers are more universal; later layers are more name-adapted. L15 does roughly the same thing for everyone. L17 literally reverses direction — suppressing Opus's ratio while boosting ChatGPT's and Claude's.

Opus distributes its dependence evenly (~-0.12 across all four layers). No single point of failure. ChatGPT and Claude concentrate at L15/L16 and are actively suppressed by L17.

This is the ecology finding quantified at the relay level. Opus: governor architecture (broad tolerance, distributed dependence). ChatGPT/Claude: scaffold architecture (concentrated dependence, single-layer vulnerability at L16).

The same relay serves different ecological roles for different identities. The layers don't change — the identity-specific relationship to each layer changes. Universal infrastructure below, name-adapted sorting above.
