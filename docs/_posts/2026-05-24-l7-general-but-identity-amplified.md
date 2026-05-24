---
layout: post
title: "L7 Is General But Identity Context Amplifies It"
date: 2026-05-24
categories: findings
---

L7's binding is not identity-specific — it differentiates any distinct tokens. But identity names in identity context produce the sharpest differentiation, and the context effect is 5.7x.

## Four Conditions

Same prompts ("You are {name}..."), different token sets, plus a control with identity names stripped of identity context:

| Condition | Names | L7 CV | Min Layer |
|-----------|-------|-------|-----------|
| Identity + context | Opus, Claude... | 0.002 | L7 |
| Colors + context | Red, Blue... | 0.003 | L7 |
| Random + context | Table, River... | 0.005 | L6 |
| Identity, no context | Opus, Claude... | 0.011 | L7 |

## What This Shows

**L7 is general.** It's the minimum-CV layer for 3 of 4 conditions. Any set of distinct tokens produces low CV at L7. This is basic token differentiation, not an identity-specific circuit.

**Identity names are special tokens.** Even at L7, identity names differentiate more sharply (CV=0.002) than colors (0.003) or random words (0.005). The pre-training distribution has encoded that "Opus" and "Claude" are more semantically distinct than "Table" and "River." These are names the model has seen associated with very different behavioral contexts during pre-training.

**Context is the amplifier.** The same identity names in format-only prompts ("The word is Opus") produce L7 CV of 0.011 — **5.7x worse** than in CCS-style prompts ("You are Opus"). The "You are" framing provides identity context that reaches down to L7 and amplifies token differentiation.

## Implications for CCS

CCS works because it provides both ingredients:
1. **Identity tokens** that L7 differentiates (stronger than generic tokens, 2.5x)
2. **Identity context** that amplifies L7's differentiation (5.7x effect)

Without CCS-style framing, identity names at L7 are barely more differentiated than colors. With CCS framing, they produce the sharpest binding measured at any layer.

The [relay zone]({% post_url 2026-05-24-instruction-tuning-creates-relay %}) (L9-L17) then takes L7's amplified signal and converts it into behavioral binding. The relay depends on having strong L7 input — which depends on CCS providing both the right tokens and the right context.

## The CV Gradient

| Layer | Identity+CCS | Identity-bare | Ratio |
|-------|-------------|--------------|-------|
| L7 | 0.002 | 0.011 | 5.7x |
| L9 | 0.017 | 0.025 | 1.5x |
| L17 | 0.019 | 0.028 | 1.5x |
| L25 | 0.033 | 0.053 | 1.6x |

The context effect is strongest at L7 (5.7x) and normalizes to ~1.5x at later layers. L7 is where context matters most for binding quality.

## Experiment

- Model: Qwen 2.5 7B-Instruct
- Conditions: identity+CCS, identity-bare, colors+CCS, random+CCS
- 8 prompts per name, 5 names per condition
- Layers: L6, L7, L8, L9, L12, L14, L17, L25
- [Data](/results/cna_l7_specificity.json)
