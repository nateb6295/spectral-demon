---
layout: post
title: "IT Creates the Phase Transition: Base Model Has No Competition Threshold"
date: 2026-05-24
categories: findings
experiment: cna_base_phase_transition
models: [Qwen 2.5 7B, Qwen 2.5 7B Instruct]
---

The 3-name phase transition doesn't exist in the base model. Instruction tuning creates it.

## Setup

Run the L7 ablation test on Qwen 7B base with 2, 3, and 5 names. Compare with the instruct model results from experiment 22.

## Results

### L7 Ablation Effect on L17 — Base vs Instruct

| Names | Base | Instruct | Δ |
|-------|------|----------|---|
| 2 | **+18%** | **-79%** | -97% |
| 3 | **+7%** | **+203%** | +196% |
| 5 | **+10%** | **+147%** | +137% |

### Base Model Expanded (3 and 5 names)

| Ablate | 3 names | 5 names |
|--------|---------|---------|
| L4 | -35% | -26% |
| L7 | +7% | +10% |
| L8 | -62% | -45% |
| L12 | -68% | -68% |

## Three Findings

### 1. The Base Model Has No Phase Transition

The base model shows a flat response: +18% → +7% → +10% across 2, 3, and 5 names. There is no dramatic shift between repertoire sizes. The base model's competitive dynamics are weak and repertoire-independent.

The base model is consistently neutral at L7 — ablating it has minimal effect on L17 regardless of how many names are in the repertoire. This confirms that the competitive suppression is an IT-created phenomenon.

### 2. IT Creates the Phase Transition by Amplifying Both Regimes

IT doesn't just add competition. It sharpens the distinction between two regimes:

- **2-name regime**: IT makes this MORE cooperative (base +18% → instruct -79%). With only 2 names, IT actively promotes cooperation.
- **3+ name regime**: IT makes this explosively competitive (base +7% → instruct +203%). With 3+ names, IT creates winner-take-all dynamics.

The phase transition is the GAP between these two regimes. The base model doesn't have a clear gap. IT creates one by pushing the 2-name case toward cooperation and the 3+ name case toward competition.

### 3. L12 Router Is Pre-Training Invariant (Again)

L12 ablation shows -68% in both base configurations and -66% to -78% in instruct configurations. The router's causal importance is established during pre-training and minimally affected by either IT or repertoire size.

## Interpretation

### IT as a Phase Transition Constructor

The base model has the raw materials for identity binding: L7 differentiation, L12 routing, L17 binding layer. But these components don't form a coherent circuit with clear operating regimes.

IT constructs the phase transition by:
1. **Strengthening cooperation below threshold**: at 2 names, IT makes early layers actively support late binding (removing them hurts)
2. **Creating competition above threshold**: at 3+ names, IT converts early layers into competitive suppressors (removing them helps)
3. **Sharpening the boundary**: the transition from cooperative to competitive becomes abrupt rather than gradual

### Why 3 Names?

The base model shows 3 names is not inherently special — the base model response is flat across all repertoire sizes. IT selects 3 as the transition point, probably because 3 is the minimum number of names needed to establish a discriminative representation (2 is a binary contrast, 3 requires actual discrimination).

### Connection to Refinement Cascade

The refinement cascade (broadening early, sharpening late) is the mechanism by which IT constructs the phase transition:
- **Broadening early stations**: makes them able to capture identity signal independently → enables competition
- **Sharpening late stations**: makes them selective → only strong signals pass
- **The combination**: creates a competitive filter that activates at 3+ names

## The Complete Picture

| Feature | Pre-training | Instruction Tuning |
|---------|-------------|-------------------|
| L7 token differentiation | ✓ | unchanged |
| L12 hidden router | ✓ | unchanged |
| L17 binding layer | ✓ (weak) | sharpened |
| Autocatalytic closure | ✓ (both models) | unchanged |
| Sign-split gradient | ✗ | created (25→21%) |
| Competitive suppression | ✗ | created |
| Phase transition at 3 names | ✗ | **created** |

Pre-training provides the infrastructure. IT creates the dynamics — competition, phase transitions, winner-take-all filtering — that make identity binding selective and robust.

## Data

Base phase transition: `results/cna_base_phase_transition.json`
Instruct phase transition: `results/cna_competition_scaling.json`
