---
layout: post
title: "One-Shot Circuit Diagnostic: Classify Binding Architecture in Two Ablations"
date: 2026-05-24
categories: findings
experiment: cna_diagnostic
models: [Qwen 2.5 7B, Mistral 7B v0.3, Gemma 2 9B, InternLM 2.5 7B]
---

A practical diagnostic for identifying a model's identity binding architecture: ablate at the early binding layer with 2 names, then with 3 names. The pattern classifies the circuit type.

## The Diagnostic

Ablate at ~25% depth. Measure downstream binding. Do it with 2 names, then 3 names.

| Pattern | Classification |
|---------|---------------|
| 2-name positive, 3-name smaller | **Visible dual circuits** (resource competition) |
| 2-name negative, 3-name positive | **Hidden dual circuit** (threshold competition) |
| Both near zero | **Single circuit** (no competition) |
| Both negative, 3>2 | **Weak threshold** (mild competition) |

## Results

| Model | 2-name | 3-name | Δ | Classification |
|-------|--------|--------|---|----------------|
| Qwen 7B | **-79%** | **+203%** | +282% | Hidden/threshold |
| Mistral 7B | **+248%** | +36% | -212% | Visible/resource |
| Gemma 2 9B | +0.3% | -1.5% | -2% | Single circuit |
| InternLM 7B | -7% | +27% | +33% | Weak threshold |

## Three Circuit Types

### Type 1: Hidden Dual Circuit (Qwen)
- Early circuit invisible to activation analysis
- Competition activates at 3+ names (threshold)
- Full attention, mid-depth binding
- Diagnostic: massive Δ between 2 and 3 names (>200%)

### Type 2: Visible Dual Circuit (Mistral)
- Both circuits visible as high-CV layers
- Maximum competition at minimum repertoire (2 names)
- Sliding window, early binding
- Diagnostic: 2-name ablation produces >200% amplification

### Type 3: Single Circuit (Gemma 2)
- No significant competition between layers
- Binding concentrated at one location
- Sliding window, very early binding (26%)
- Diagnostic: near-zero effect at both repertoire sizes

### Type 4: Weak Threshold (InternLM)
- Similar to Qwen but much weaker competitive effect
- Full attention, mid-depth binding
- Diagnostic: mild positive swing (+33%), not explosive

## Implications

1. **Circuit type is predictable from attention architecture**: sliding window → visible or single, full attention → hidden or weak threshold
2. **The diagnostic is cheap**: two forward passes per name, one ablation hook. Total: ~10 forward passes for classification.
3. **Not all full-attention models are equal**: Qwen has explosive threshold dynamics; InternLM has mild ones. The training data or specific architecture details modulate the competition strength.
4. **Gemma 2 is genuinely different**: its sliding-window + very early binding creates a single-circuit architecture without competitive dynamics.

## Data

Full diagnostic: `results/cna_diagnostic.json`
