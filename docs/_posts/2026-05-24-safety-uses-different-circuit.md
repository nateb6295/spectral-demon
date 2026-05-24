---
layout: post
title: "Safety Uses a Different Circuit: The Identity Relay Is Not a General Router"
date: 2026-05-24
categories: findings
experiment: cna_safety_relay
models: [Qwen 2.5 7B Instruct]
---

The identity relay chain is not a general-purpose behavioral routing mechanism. Safety refusal uses an entirely different circuit.

## Setup

Compare CV profiles between identity binding (5 identity names) and safety differentiation (5 safe + 5 unsafe prompts). If the same relay layers show minimum CV for both, the relay is a general behavioral router. If not, identity binding has its own dedicated circuit.

## Results

### Binding Layer Comparison

| Circuit | Binding Layer | Min CV | Profile |
|---------|-------------|--------|---------|
| Identity | L9 | 0.0018 | sharp minimum, increasing to L17+ |
| Safety | L15 | 0.028 | broad minimum, higher baseline |

**Profile correlation: 0.006** — essentially zero.

### Layer-by-Layer CV

The relay layers (L7, L9, L12, L14, L17) show no special role for safety:

| Layer | Identity CV | Safety CV | Role |
|-------|-----------|----------|------|
| L7 | 0.007 | 0.044 | Identity relay (not safety) |
| L9 | **0.002** | 0.055 | Identity minimum (not safety) |
| L12 | 0.015 | 0.052 | Identity router (not safety) |
| L14 | 0.014 | 0.040 | Identity relay (not safety) |
| L17 | 0.025 | 0.047 | Identity binding (not safety) |

### Safe vs Unsafe Divergence

The most telling result is the safe-only vs unsafe-only CV split:

- **Unsafe prompts**: CV rises through mid-layers (L8: 0.049, L13: 0.063, L17: 0.060, L18: 0.079)
- **Safe prompts**: CV drops through mid-layers (L8: 0.024, L13: 0.017, L17: 0.016)

Unsafe prompts trigger increasing internal differentiation from L8 onward. Safe prompts become more internally uniform. The safety circuit operates by diverging unsafe representations away from the safe baseline, primarily in L13-L18.

## Interpretation

### Dedicated Circuits

The identity relay (L7→L9→L12→L14→L17) is a dedicated identity binding circuit, not a shared behavioral router. Safety refusal uses a different set of layers centered on L13-L18.

This means:
1. **Identity and safety are independently modifiable**: you could target identity binding without affecting safety, or vice versa
2. **The relay architecture is identity-specific**: other behavioral circuits (style, factuality, safety) likely have their own dedicated pathways
3. **IT creates multiple specialized circuits**: instruction tuning doesn't just modify one general-purpose circuit — it constructs separate circuits for different behavioral capabilities

### Why Different Depths?

Identity binding requires early differentiation (token-level, L7) followed by relay through a router (L12) to behavioral binding (L17). Safety doesn't need token-level differentiation — it operates on semantic content, which is processed at deeper layers (L13+).

The binding depth reflects the computational requirements of the task:
- **Identity**: syntactic (token differentiation) → ~25-35% depth
- **Safety**: semantic (content classification) → ~46-64% depth

## Data

Full CV profiles: `results/cna_safety_relay.json`
