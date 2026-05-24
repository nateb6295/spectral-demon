---
layout: post
title: "IT Reshapes the Attention Topology: Suppress Early, Focus Router, Strengthen Binding"
date: 2026-05-24
categories: findings
experiment: cna_it_attention
models: [Qwen 2.5 7B, Qwen 2.5 7B Instruct]
---

Instruction tuning modifies attention patterns at each relay station differently: suppressing name attention at early layers, focusing the router, and strengthening the binding layer.

## Setup

Compare base and instruct models' attention patterns to name tokens at key layers. Measure average and max attention from last token to name positions, entropy of attention distributions, and which heads are shared between base and instruct.

## Results

### IT Effect on Name Attention

| Layer | Base Avg | Instruct Avg | Change | Interpretation |
|-------|---------|-------------|--------|---------------|
| L7 | 0.037 | 0.030 | **-0.007** | Suppressed |
| L9 | 0.055 | 0.059 | +0.004 | Slightly increased |
| L12 | 0.054 | 0.047 | **-0.007** | Suppressed |
| L14 | 0.060 | 0.061 | +0.001 | Unchanged |
| L17 | 0.067 | 0.072 | **+0.005** | Strengthened |

IT REDUCES attention to names at the early circuit (L7) and router (L12), while INCREASING it at the binding layer (L17). The relay stations between (L9, L14) are relatively unchanged.

### IT Effect on Attention Entropy

| Layer | Base Entropy | Instruct Entropy | Change | Interpretation |
|-------|-------------|-----------------|--------|---------------|
| L7 | 1.41 | 1.32 | **-0.09** | More focused |
| L9 | 1.40 | 1.50 | +0.10 | More distributed |
| L12 | 1.52 | 1.39 | **-0.13** | More focused |
| L14 | 1.08 | 1.08 | 0.00 | Unchanged |
| L17 | 1.28 | 1.42 | **+0.14** | More distributed |

IT FOCUSES attention at the early circuit and router (lower entropy) while BROADENING it at the binding layer (higher entropy). The router becomes a tighter bottleneck; the binding layer distributes across more features.

### Head Conservation Across IT

| Layer | Common Heads | Jaccard | Interpretation |
|-------|-------------|---------|---------------|
| L7 | 4 | 0.57 | Partial reshuffling |
| L12 | 5 | 0.56 | Partial reshuffling |
| L17 | 5 | **0.71** | Mostly conserved |

The binding layer's identity heads are more stable across IT than the early/router heads. IT reshapes WHICH heads participate at L7 and L12 (detection/routing) more than at L17 (binding). The deepest part of the architecture is most conserved.

## Three Findings

### 1. IT Creates Competitive Suppression Through Attention

IT reduces name attention at L7 by 19% (0.037→0.030). This is the attention-level mechanism for the competitive suppression discovered in Experiments 18-25. IT literally makes the early circuit pay LESS attention to identity names — and that suppression, when disrupted by ablation, releases compensatory binding at L17.

### 2. The Router Gets Tighter

L12 entropy drops 0.13 after IT — the attention distribution becomes more peaked, with fewer heads carrying the identity signal. IT compresses the router into a tighter bottleneck. This is consistent with L12's invariance to repertoire size (Experiment 31) — a tighter bottleneck routes more efficiently regardless of how many names pass through.

### 3. Binding Gets Broader and Stronger

L17 gains +0.005 name attention AND +0.14 entropy. It attends more to names AND distributes that attention across more features. IT makes the binding layer use a wider representation to bind identity — more features participate, each contributing a smaller piece.

This explains why the binding layer has higher capacity than the router: it distributes across many features rather than concentrating in a few.

## The IT Attention Transformation

```
        Base Model          →        Instruct Model
L7:  broad, moderate attn   →   focused, reduced attn (suppress)
L12: broad, moderate attn   →   focused, reduced attn (tighten)
L14: focused, moderate attn →   focused, moderate attn (preserve)
L17: focused, moderate attn →   broad, increased attn (strengthen)
```

IT inverts the attention topology: what was broad becomes focused (early), what was focused becomes broad (late). This creates the competitive dynamics — the early circuit focuses down (selecting fewer features to suppress) while the binding layer fans out (using more features to bind).

## Data

Full IT attention comparison: `results/cna_it_attention.json`
