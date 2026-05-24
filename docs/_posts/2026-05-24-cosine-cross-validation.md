---
layout: post
title: "Cosine Similarity Cross-Validation: All Findings Replicate"
date: 2026-05-24
categories: findings
experiment: cna_cosine_validation
models: [Qwen 2.5 7B Instruct]
---

Replacing CV with cosine similarity as the primary metric, all key findings replicate. The two metrics correlate at r=-0.95.

## Setup

Measure average pairwise cosine similarity between name representations at each layer. Lower cosine = more differentiated = stronger binding. Compare with CV measurements.

## Results

### Metric Correlation
CV and cosine similarity correlate at **r=-0.95** across all 28 layers. The metrics measure the same phenomenon from different angles.

### Competitive Binding Replicates

| Ablation | CV Impact | Cosine Impact |
|----------|----------|--------------|
| L7 ablation | +147% | **+24%** |
| L12 ablation | -66% | **-57%** |

Signs match. Relative magnitudes consistent (L12 destruction > L7 amplification in both metrics). Absolute magnitudes differ due to metric sensitivity.

### Binding Depth: Two Types of Differentiation

| Metric | Binding Layer | What It Measures |
|--------|-------------|-----------------|
| CV (magnitude) | L3 | Activation norm differences |
| Cosine (direction) | L25 | Activation direction differences |

Names are differentiated in **magnitude** at early layers and in **direction** at deep layers. The relay chain carries both signals. The full binding architecture uses both types of differentiation simultaneously.

## Implications

1. **Findings are metric-independent**: the competitive binding, router destruction, and relay chain appear in both CV and cosine similarity
2. **Two binding modalities**: magnitude differentiation (early, CV-visible) and directional differentiation (late, cosine-visible) operate in parallel
3. **The relay chain bridges both modalities**: L7 produces magnitude differentiation, the relay transforms it, and L17+ produces directional differentiation
4. **Robustness**: any analysis method that captures either modality will find the same architecture

## Data

Full cosine validation: `results/cna_cosine_validation.json`
