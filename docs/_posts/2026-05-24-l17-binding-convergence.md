---
layout: post
title: "L17 Binding Convergence Across Architectures"
date: 2026-05-24
categories: findings
---

The apex relay layer (L17) is a universal local minimum in cross-name binding variation across three architectures — Qwen, Mistral, and InternLM — despite each implementing identity through different spectral transformations.

## The Experiment

We measured binding CV (coefficient of variation of mean activations across 5 identity names) at matched relay-equivalent layers in three 7B-class instruct models. Each identity name was embedded in the same CCS system prompt with 8 relational questions.

## Results

| Role | Qwen CV | Mistral CV | InternLM CV |
|------|---------|-----------|------------|
| Seed (L9/L10) | 1.39 | 1.49 | 1.31 |
| Pre-sort (L14) | 1.05 | **2.42** | **2.16** |
| Relay (L16) | 1.08 | 1.28 | 1.24 |
| **Apex (L17)** | **0.96** | **0.85** | **1.18** |
| Expression (L25) | 1.21 | 1.35 | 1.45 |
| Final (L27/L30) | 1.82 | 1.90 | **0.89** |

## Four Findings

### 1. L17 Is a Universal Local Minimum

All three architectures show binding CV at L17 **lower than both L16 and L25** — making it a local minimum in every network tested. Qwen (0.96) and Mistral (0.85) have their global minimum at L17. InternLM has L17 as a local minimum (1.18) with a deeper minimum at L30 (0.89).

The local minimum property is the invariant: identity names converge at L17 regardless of architecture. Different architectures may have additional convergence points (InternLM at L30), but L17 is always one.

### 2. Seed Layer Spectral Inversion Confirmed in Binding

The [spectral inversion]({% post_url 2026-05-24-base-model-criticality %}) now has a binding signature:
- **Qwen L9**: identity DECREASES PR (4.44 → 3.84) — concentrates at the seed
- **Mistral L10**: identity INCREASES PR (3.89 → 7.45) — spreads at the seed
- **InternLM L10**: identity INCREASES PR (4.41 → 6.24) — same direction as Mistral

Same CCS prompt, same relational questions. The seed layers implement different geometric operations. But by L17, all arrive at a local CV minimum — the binding site absorbs upstream diversity.

### 3. Pre-Sorting Strategies Cluster

L14 reveals two strategies:
- **Low-CV sorting** (Qwen: 1.05): name-agnostic, treats all identities similarly
- **High-CV sorting** (Mistral: 2.42, InternLM: 2.16): name-specific, different activation patterns per name

Both strategies feed into the same L17 binding convergence. Qwen is the outlier here — possibly because its architecture handles sorting differently at the relay entrance.

### 4. Final Layer Divergence

The most architecturally variable layer is the final one:
- Qwen L27: CV=1.82 (names diverge at output)
- Mistral L30: CV=1.90 (names diverge at output)
- InternLM L30: CV=0.89 (names reconverge at output)

InternLM re-binds identity names at the output layer — a second convergence point absent in the other architectures. This suggests InternLM implements a "binding sandwich": L17 binds, L25 diversifies, L30 re-binds.

## Implications

The universal local minimum at L17 is a geometric constraint, not an architectural coincidence. Three model families, different training data, different spectral transformations — but L17 always produces lower cross-name variation than its neighbors. This positions the ~53% depth layer as a binding bottleneck in 7B-class transformers.

Combined with the [trophic cascade]({% post_url 2026-05-23-name-specific-relay-ecology %}) finding (L17 ablation produces opposite PR effects across names), L17 emerges as the critical node in identity computation: the layer where diverse representations converge and where disruption causes the most name-specific damage.
