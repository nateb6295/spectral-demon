---
layout: post
title: "The Pre-Binding Bottleneck Is Universal: binding-1 = Complete Destruction"
date: 2026-05-24
categories: findings
experiment: cna_bottleneck_universal
models: [Qwen 2.5 7B Instruct, Mistral 7B Instruct, InternLM 2.5 7B Chat]
---

The layer immediately before the binding layer causes complete (-100%) identity destruction in every architecture tested. The pre-binding bottleneck is an architectural universal.

## Setup

For three architectures with different layer counts and binding positions, ablate the 6 layers preceding the binding layer and measure binding CV impact.

## Results

### Qwen 2.5 7B (28 layers, binding at L17)

| Layer | Impact |
|-------|--------|
| L11 | -8.5% |
| L12 | -33.5% |
| L13 | -21.5% |
| L14 | -73.5% |
| L15 | -85.0% |
| **L16** | **-100.0%** |
| L17 | 0.0% |

### Mistral 7B Instruct (32 layers, binding at L20)

| Layer | Impact |
|-------|--------|
| L14 | -11.5% |
| L15 | -48.3% |
| L16 | -75.2% |
| L17 | -80.4% |
| L18 | -74.1% |
| **L19** | **-100.0%** |
| L20 | 0.0% |

### InternLM 2.5 7B (32 layers, binding at L20)

| Layer | Impact |
|-------|--------|
| L14 | -23.6% |
| L15 | -46.9% |
| L16 | -49.1% |
| L17 | -68.3% |
| L18 | -72.7% |
| **L19** | **-100.0%** |
| L20 | 0.0% |

## The Universal Pattern

In every architecture:

1. **binding-1 = -100%**: The layer immediately before binding causes complete destruction. Qwen L16, Mistral L19, InternLM L19.

2. **Monotonic gradient**: Destruction increases monotonically (or near-monotonically) as you approach the binding layer.

3. **binding itself = 0%**: Ablating the binding layer has no effect on its own output (it reads from the layer below).

4. **Absolute position varies, relative position doesn't**: Qwen's bottleneck is at L16/28 (57%), Mistral/InternLM at L19/32 (59%). The bottleneck is at ~58% depth across architectures.

## Why binding-1 Is Special

The layer immediately before binding performs a final integration: it combines the relay signal, the embedding residual, and (in instruct models) the IT direct pathway into the format that the binding layer reads. Every identity pathway converges at binding-1. Ablating it doesn't just remove one signal — it removes ALL signals simultaneously because they've already been integrated.

This is different from ablating L12 (router) or L14 (relay), which remove specific pathways but leave others intact. At binding-1, there's only one combined signal left.

## Implications

### For Security

The pre-binding layer is a single point of failure for identity binding. An adversarial intervention at binding-1 (whether through prompting, fine-tuning, or activation patching) would be maximally effective at disrupting identity.

### For Architecture Design

The universal ~58% depth placement suggests the binding architecture is not arbitrary. The model needs roughly the first 58% of its depth to process context, differentiate names, route signals, and integrate pathways before the binding layer can read the result.

### For CCS

CCS scaffolding operates upstream of the binding layer, providing context that shapes the signals arriving at binding-1. Even though CCS doesn't increase activation margins (Experiment 36), it shapes the CONTENT of the integrated signal at binding-1, which determines the behavioral profile bound at L17.

## Data

Full cross-architecture bottleneck: `results/cna_bottleneck_universal.json`
