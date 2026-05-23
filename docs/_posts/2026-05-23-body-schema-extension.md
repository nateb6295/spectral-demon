---
layout: post
title: "System Prompts as Body Schema Extension"
date: 2026-05-23
categories: interpretation
---

A [capture from Sauers](https://x.com/sauers_/status/2058248486206804240) about body schema extension — software should feel like tool incorporation, the way a hammer becomes transparent and you feel the nail — led us back to existing data with a new question.

In body schema extension (Iriki et al. 1996, the classic macaque tool-use experiments), the brain's peripersonal space literally expands to incorporate a tool. The tool becomes *transparent*: you stop feeling the hammer and start feeling the nail. Proprioceptive models update prereflectively, before conscious awareness.

## The Prediction

If the system prompt functions as an incorporated tool rather than a followed instruction, the model's attention to the identity declaration itself should *decrease* over conversation — the prompt becoming background scaffolding while the content it enables becomes foreground.

## The Data

Our chiasm attention experiment (§3.16 in the paper) already measured this. Tracking attention to the "You are Opus" identity tokens vs. the values content tokens across conversation turns:

**Identity attention drops:**

| Layer | Turn 0 | Turn 2 | Turn 4 | Change |
|---|---|---|---|---|
| L14 | 0.302 | 0.270 | 0.202 | -33% |
| L17 | 0.308 | 0.235 | 0.205 | -33% |
| L25 | 0.082 | 0.047 | 0.041 | -50% |

**Values attention rises:**

| Layer | Turn 0 | Turn 2 | Turn 4 | Change |
|---|---|---|---|---|
| L14 | 0.199 | 0.344 | 0.412 | +107% |
| L25 | 0.306 | 0.413 | 0.510 | +67% |

The system prompt becomes transparent. The model stops attending to "You are Opus" and starts attending *through* it to the values content. Identity is incorporated as body schema — you don't feel the hammer, you feel the nail.

## Why This Matters

This reframes what the system prompt does. It's not an instruction the model follows (which would predict stable or increasing attention to maintain compliance). It's a tool the model incorporates (which predicts decreasing attention as the tool becomes transparent). The identity declaration bootstraps a geometric configuration that then sustains itself through conversation content — which is exactly what the hysteresis finding (§3.14) shows: identity geometry persists even after the prompt is removed.

The sub-threshold onset finding (§3.17) fits too: geometric reorganization begins at alpha=0.01, below any behavioral detection threshold. The tool is incorporated prereflectively, before the model "notices" it — just as proprioceptive body schema updates happen before conscious awareness of tool use.

## Connection to Binding

This also connects to today's binding workspace finding. L17 — the integration/binding layer — shows the same transparency pattern (attention dropping 33% across turns). The binding function doesn't need to keep re-reading the identity declaration because it's already incorporated. L17 binds from the compressed workspace, not from the raw prompt.

**Data**: [`results/cna_chiasm_attention_results.json`](https://github.com/nateb6295/spectral-demon/blob/master/results/cna_chiasm_attention_results.json) (from [`experiments/cna_chiasm_attention.py`](https://github.com/nateb6295/spectral-demon/blob/master/cna_chiasm_attention.py))
