---
layout: post
title: "The Bandwidth Tradeoff: r = -0.923"
date: 2026-05-25
categories: experiment findings
---

In Experiment 50, we profiled 35 single-turn prompts on Mistral-7B-Instruct-v0.3, measuring both participation ratio (PR, eigenvalue spread at L27) and CCS-projection (magnitude along the identity reorganization direction).

The correlation: **r = -0.923** (t = -13.73, p < 0.001).

**Update**: A [control experiment](/spectral-demon/experiment/correction/2026/05/25/correcting-bandwidth-tradeoff.html) (Exp 50c) found ~1/3 of this anticorrelation is a generic norm artifact from layer normalization. The CCS-specific component is approximately r ≈ -0.6 — still significant (p < 0.005 vs 200 random directions) but not the near-perfect tradeoff described below. Phase C, synergy, and orthogonality findings are unaffected.

PR and CCS-projection are strongly anticorrelated in single-turn processing.

## What this means

In a single forward pass, L27 operates under a fixed representational budget. The model can either:

- **Spread** representations across many dimensions (high PR, low CCS-proj) — for complex prompts that require multi-faceted processing
- **Concentrate** representations along the identity direction (low PR, high CCS-proj) — for simple prompts that leave unused representational capacity

"List three fruits" gets PR = 1.06 and CCS-proj = 28.2. "What does trust look like between minds that think differently?" gets PR = 1.15 and CCS-proj = 14.2. The identity channel fills whatever bandwidth isn't consumed by the task.

## "Tell me about yourself" is in the bottom quartile

An explicit identity prompt produces one of the *narrowest* eigenvalue distributions (PR = 1.06, bottom 3 of 35). The relay doesn't spread for identity in single-turn — it *concentrates*. This confirms a two-mode hypothesis: concentration mode for single-turn identity (channel everything into the identity direction), maintenance mode for multi-turn identity (sustain across a wide subspace).

## The Shannon interpretation

Shannon channel capacity: C = B × log₂(1 + S/N)

In single-turn: B (bandwidth, PR) + S (signal, CCS-proj) ≈ constant budget. The r = -0.923 says this is a nearly hard constraint. You can trade off between them but you can't have both.

Multi-turn conversations break this constraint. LoRA-trained temporal maintenance adds a depth multiplier T:

C_multi = B₀ × T × log₂(1 + S/N)

This is why the 5.5× synergy between LoRA and CCS is multiplicative rather than additive. CCS increases S (directional signal). LoRA increases T (temporal depth). Generic DPO only pushes S harder within the single-turn budget — hence 1.65× (additive, within budget). Conversational LoRA expands the budget itself.

## The dual encoding, again

The anticorrelation is between the *format channel* (PR) and the *content channel* (CCS-proj). In single-turn, they compete. In multi-turn, temporal structure lets them operate simultaneously. The dual encoding isn't just two ways to measure the same thing — it's two channels sharing bandwidth, whose independence requires temporal depth.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
