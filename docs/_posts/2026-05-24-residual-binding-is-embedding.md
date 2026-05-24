---
layout: post
title: "20% Binding Survives Everything: The Residual Is the Embedding"
date: 2026-05-24
categories: findings
experiment: cna_residual_trace
models: [Qwen 2.5 7B, Qwen 2.5 7B Instruct]
---

No matter how many layers you ablate, ~20% of binding at L17 persists. This residual comes from the token embedding itself, flowing through the residual stream.

## Setup

Progressively ablate more layers while measuring L17 binding:
1. L7-L15 (relay only)
2. L7-L16 (relay + pre-binding)
3. L3-L15 (extended early + relay)
4. L1-L16 (everything except embedding and L17 itself)

## Results

### Instruct Model

| Ablation | L17 Impact |
|----------|-----------|
| L7-L15 | **-80.2%** |
| L7-L16 | **-80.2%** |
| L3-L15 | **-80.2%** |
| L1-L16 | **-80.2%** |

### Base Model

| Ablation | L17 Impact |
|----------|-----------|
| L7-L15 | **-87.7%** |
| L3-L15 | **-87.7%** |

## The Residual Is Invariant

The exact same 19.8% (instruct) or 12.3% (base) of binding survives regardless of how many layers are ablated. Whether you remove 9 layers or 16 layers, the residual is identical.

This means the residual doesn't come from any processing layer. It comes from the token embedding itself.

## Mechanism

In transformer architectures, the residual stream carries the original embedding representation through every layer. Each layer ADDS to this stream, but the original embedding persists. When we ablate a layer (replace its output with the cross-name mean), we remove that layer's identity-relevant contribution, but the embedding's contribution flows through unchanged.

The ~20% binding at L17 after ablating L1-L16 represents the raw token embedding's contribution to identity differentiation. Different identity names (Opus, Aria, Sage, etc.) have different embeddings, and those embedding differences persist through the residual stream to L17 regardless of what intermediate layers do.

## Base vs Instruct Residual

| Model | Residual | Interpretation |
|-------|---------|----------------|
| Base | 12.3% | Raw embedding contribution |
| Instruct | 19.8% | Embedding + IT-created direct pathway (+7.5%) |

The instruct model has 7.5 percentage points MORE residual binding than the base model. IT doesn't just create the relay — it also creates or strengthens a direct pathway that bypasses the relay entirely. This direct pathway contributes ~7.5% of binding on top of the 12.3% embedding baseline.

## Implications

1. **Token embeddings carry identity**: the sheer difference in embedding vectors for different identity names contributes ~12% of behavioral binding at L17
2. **IT creates redundancy**: instruction tuning adds a direct pathway (+7.5%) on top of the embedding contribution, making binding more robust to relay disruption
3. **The relay handles ~80% of binding**: despite the residual, the relay chain is still responsible for the vast majority of identity binding
4. **Complete binding destruction would require embedding-level intervention**: to eliminate all identity binding, you'd need to modify the token embeddings themselves, not just the intermediate processing

## The Full Architecture

```
EMBEDDING (L0): 12% of binding
  ↓ (residual stream — direct to L17)
  ↓
EARLY CIRCUIT (L3-L8): competitive suppressor
  ↓
ROUTER (L12): bottleneck
  ↓
RELAY (L13-L15): carries signal
  ↓
IT DIRECT PATHWAY: +7.5% (instruct only)
  ↓
BINDING (L17): 80% from relay, 20% from embedding + direct
```

## Data

Full residual trace: `results/cna_residual_trace.json`
