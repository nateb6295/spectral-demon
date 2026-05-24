---
layout: post
title: "L17 Binding Convergence Across Architectures"
date: 2026-05-24
categories: findings
---

The apex relay layer (L17) shows minimum cross-name binding variation in both Qwen and Mistral — despite implementing identity through opposite spectral transformations.

## The Experiment

We measured binding CV (coefficient of variation of mean activations across 5 identity names) at matched relay-equivalent layers in Qwen 2.5 7B-Instruct and Mistral 7B-Instruct-v0.3. Each identity name was embedded in the same CCS system prompt with 8 relational questions.

## Results

| Role | Qwen Layer | Q Generic PR | Q Identity PR | Q Binding CV | Mistral Layer | M Generic PR | M Identity PR | M Binding CV |
|------|-----------|-------------|--------------|-------------|--------------|-------------|--------------|-------------|
| Seed | L9 | 4.44 | 3.84 | 1.39 | L10 | 3.89 | 7.45 | 1.49 |
| Pre-sort | L14 | 4.46 | 7.03 | 1.05 | L14 | 4.04 | 7.62 | **2.42** |
| Relay | L16 | 4.12 | 7.72 | 1.08 | L16 | 4.33 | 7.62 | 1.28 |
| **Apex** | **L17** | **4.09** | **7.89** | **0.96** | **L17** | **4.33** | **7.51** | **0.85** |
| Expression | L25 | 5.66 | 8.10 | 1.21 | L25 | 5.85 | 8.91 | 1.35 |
| Final | L27 | 5.91 | 9.15 | 1.82 | L30 | 5.83 | 9.89 | 1.90 |

## Three Findings

### 1. L17 Is the Cross-Architecture Binding Site

Both architectures show their **lowest** cross-name binding CV at L17 — Qwen 0.96, Mistral 0.85. Despite implementing identity through opposite spectral operations ([post 29]({% post_url 2026-05-24-base-model-criticality %})), both converge at the apex layer. This is where identity names are bound into a common representational format.

### 2. Seed Layer Spectral Inversion Confirmed in Binding

The [spectral inversion]({% post_url 2026-05-24-base-model-criticality %}) now has a binding signature:
- **Qwen L9**: identity DECREASES PR (4.44 → 3.84) — concentrates at the seed
- **Mistral L10**: identity INCREASES PR (3.89 → 7.45) — spreads at the seed

Same CCS prompt, same relational questions. The seed layers implement opposite geometric operations. But by L17, both arrive at minimum binding CV — the convergent binding site absorbs the upstream inversion.

### 3. L14 Pre-Sorting Diverges Wildly

Mistral L14 has the highest binding CV in the entire network (2.42) — extreme cross-name variation at the pre-sorter. Qwen L14 is modest (1.05). This suggests fundamentally different sorting strategies:
- **Qwen**: early sorting is name-agnostic (low CV, acts similarly across names)
- **Mistral**: early sorting is name-specific (high CV, different activation patterns per name)

Both strategies feed into the same L17 binding convergence.

## Implications

The L17 binding convergence is a **cross-architecture invariant**. Different training data, different architectures, opposite spectral transformations — but the same relay layer emerges as the binding site where identity names converge. This suggests L17's role as the identity binding layer is not architecture-specific but reflects a deeper geometric constraint on how transformer networks implement identity maintenance.

Combined with the [trophic cascade]({% post_url 2026-05-23-name-specific-relay-ecology %}) finding (L17 ablation produces opposite PR effects across names), this positions L17 as the critical node in identity computation: the layer where diverse representations converge and where disruption causes the most name-specific damage.
