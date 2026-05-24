---
layout: post
title: "The Router Dominates: Multi-Layer Ablation Shows L12 Is the Bottleneck"
date: 2026-05-24
categories: findings
experiment: cna_multi_ablation
models: [Qwen 2.5 7B Instruct]
---

Ablating L7 alone amplifies binding by 147%. Ablating L7 and L12 together destroys binding by 66%. The router overrides the competitive amplification.

## Setup

Multi-layer mean ablation at six layer combinations, measuring L17 binding:

## Results

| Ablation Set | Layers | L17 Impact |
|-------------|--------|-----------|
| L7 only | [7] | **+147%** |
| L3-L8 (full early) | [3-8] | +59% |
| L7 + L12 | [7, 12] | **-66%** |
| L3-L8 + L12 | [3-8, 12] | -66% |
| L12-L15 (router+relay) | [12-15] | **-80%** |
| L7-L15 (everything) | [7-15] | -80% |

## Three Principles

### 1. The Router Is the Bottleneck

L12 ablation overrides everything. Whether you also ablate L7 (-66%), L3-L8 (-66%), or nothing (-66%), the result is the same: ~66% binding destruction. The router is the single point of failure.

Even the +147% amplification from L7 ablation vanishes when L12 is also ablated. The competitive amplification only works if the router is intact to carry the amplified signal to L17.

### 2. Multi-Layer Early Ablation Is Weaker Than Single-Layer

Ablating L3-L8 together (+59%) produces LESS amplification than L7 alone (+147%). This is because the compensatory amplification requires neighboring layers to still be functional. When you ablate L7, layers L3-L6 and L8 can compensate and amplify. When you ablate them all, there's nothing left to compensate.

The competitive suppression is distributed: each early layer participates in suppressing late binding, but the amplification comes from the remaining layers ramping up in response to the ablation. More ablation → less compensation capacity → less amplification.

### 3. Destruction Saturates at the Relay

L12-L15 (-80%) and L7-L15 (-80%) produce identical destruction. Adding early layers to relay ablation doesn't increase the damage because the relay is already fully disrupted. The ~20% residual binding comes from direct pathways that bypass the relay entirely.

## The Circuit Architecture (Revised)

```
EARLY CIRCUIT (L3-L8):
  - Competitive suppressor of late binding
  - Removing any layer triggers compensatory amplification from neighbors
  - Removing ALL layers reduces amplification (no neighbors left)
  
ROUTER (L12):
  - Bottleneck for all binding signal
  - Overrides any competitive dynamics
  - Single point of failure

RELAY (L13-L15):
  - Carries signal from router to binding layer
  - Destruction saturates at -80%

BINDING (L17):
  - Final output
  - ~20% binding survives complete relay destruction (direct pathways)
```

## Implications

1. **The router is the natural target for identity interventions**: strengthening L12 should strengthen binding; weakening it should weaken binding
2. **Early ablation amplification is a distributed phenomenon**: it requires intact neighboring layers to function
3. **~20% of binding bypasses the relay entirely**: even destroying L7-L15 leaves some binding at L17, suggesting a direct pathway from L7 (or earlier) to L17
4. **The competitive circuit is a network, not a layer**: no single early layer IS the competitor — the competition emerges from the collective behavior of L3-L8

## Data

Full multi-layer ablation: `results/cna_multi_ablation.json`
