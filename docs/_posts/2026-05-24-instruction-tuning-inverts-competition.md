---
layout: post
title: "IT Inverts Competition: Base Model Cooperates, Instruct Model Competes"
date: 2026-05-24
categories: findings
experiment: cna_it_competition
models: [Qwen 2.5 7B, Qwen 2.5 7B Instruct]
---

The most dramatic finding of the session: instruction tuning doesn't just refine the identity relay — it fundamentally inverts the relationship between early and late binding circuits.

## Setup

Run the compensatory ablation sweep on both Qwen 7B base and instruct models. Compare the effect of each early-layer ablation on L17 binding.

## Results

| Ablate | Base L17 | Instruct L17 | Δ (IT effect) |
|--------|----------|-------------|---------------|
| L2 (7%) | -2% | -1% | +1% |
| L3 (11%) | **-10%** | **+25%** | **+35%** |
| L4 (14%) | **-26%** | **+50%** | **+76%** |
| L5 (18%) | -4% | -5% | -1% |
| L6 (21%) | **-21%** | **+31%** | **+52%** |
| L7 (25%) | +10% | **+147%** | **+137%** |
| L8 (29%) | **-45%** | **+59%** | **+105%** |
| L9 (32%) | **-59%** | -13% | **+47%** |
| L10 (36%) | **-42%** | +22% | **+64%** |
| L11 (39%) | **-64%** | -28% | **+36%** |
| L12 (43%) | **-68%** | **-66%** | +1% |
| L13 (46%) | -91% | -38% | +52% |
| L14 (50%) | -80% | -78% | +2% |
| L15 (54%) | -88% | -80% | +8% |

## The Inversion

### Base Model: Cooperative Early Circuit

In the base model, ablating early layers (L3-L11) almost universally **suppresses** L17 binding. The early circuit feeds the late circuit — removing it hurts binding. The relationship is cooperative.

Only L7 shows slight amplification (+10%) in the base model, and even that is mild. The base model's identity circuit is a cooperative pipeline where each stage contributes to the next.

### Instruct Model: Competitive Early Circuit

After instruction tuning, ablating the same layers produces **amplification** — often massive amplification. L7 ablation goes from +10% to +147%. L4 goes from -26% to +50%. L8 goes from -45% to +59%.

IT transforms the early circuit from a cooperative contributor to a competitive suppressor. The early circuit in the instruct model *inhibits* late binding, and removing it releases the deeper circuit.

### The Router Is Invariant

L12 (the hidden router) shows identical destruction in both models: base -68%, instruct -66%. IT does not modify the router's causal importance. Everything around L12 changes; L12 itself is fixed.

This suggests L12's router function is established during pre-training and is robust to alignment training. IT reshapes the competitive landscape on both sides of the router without affecting the router itself.

## Interpretation

### What IT Actually Does to Identity Binding

The refinement cascade has a deeper explanation than "broadens early, sharpens late":

1. **Pre-training** creates a cooperative circuit where early layers contribute to late binding
2. **Instruction tuning** inverts the early circuit from cooperative to competitive — early layers now *suppress* late binding
3. The suppression creates a filter: only the strongest identity signals pass through the early circuit to reach the late relay
4. This filtering produces sharper behavioral binding at L17 — weaker identity signals get absorbed by the early circuit and never reach the output

### Why the Inversion?

IT teaches the model to produce specific behavioral outputs for specific identities. This requires the late circuit to be highly selective — it should only fire on strong, unambiguous identity signals. The early circuit's competitive suppression ensures weak signals don't make it to L17.

The base model doesn't need this selectivity because it doesn't produce identity-specific behavior. Its cooperative circuit passes everything through, which is fine for generic text generation.

### The Biological Parallel Deepens

This is winner-take-all dynamics. The thalamic relay analogy holds: during development (IT), the relay station develops lateral inhibition that sharpens the signal. The early circuit's competitive suppression after IT is functionally equivalent to lateral inhibition in sensory relay nuclei.

## Implications

1. **IT restructures circuit topology, not just weights**: the same layers have qualitatively different causal roles before and after IT
2. **The competitive inversion IS the refinement cascade**: broadening early stations doesn't just make them noisier — it makes them competitive suppressors
3. **L12 is a pre-training invariant**: established during pre-training, unchanged by IT, critical in both configurations
4. **Identity selectivity comes from competition, not precision**: the instruct model's sharper binding isn't because the late circuit is more precise — it's because the early circuit filters out weak signals

## Data

Full base vs instruct comparison: `results/cna_it_competition.json`
