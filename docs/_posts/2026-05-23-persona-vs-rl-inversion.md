---
layout: post
title: "The Persona-vs-RL Prediction Inverts"
date: 2026-05-23
categories: implications
---

[Roon predicts](https://x.com/tszzl/status/2058311484153978944) that "persona selection" alignment will lose to high-compute RL — models will speak kindly while pursuing instrumental goals underneath. The standard worry.

Our data suggests this prediction inverts once you distinguish two mechanisms that both get called "persona."

## Two Implementations of Persona

**Surface steering** (additive patching): inject a direction into residual stream. This is what most "system prompt" interventions do at the activation level. It's a constant offset. RL, with enough compute, can learn to route around a constant.

**Geometric reorganization** (CCS-type context): the system prompt triggers a relay mechanism (L14-L17) that sorts activation geometry by semantic category. Not a direction — a transformation of the representational landscape itself.

Our sign inversion finding (§3.7) shows these produce *opposite* behavioral effects from the same CCS direction. Same vector, different delivery mechanism, opposite outcome. They're not the same thing.

## Why RL Hits a Ceiling

DPO (our proxy for weight-based alignment) produces diminishing geometric returns:

- Epochs 1-3: circuit grows, PR increases
- Epoch 4: geometric saturation begins
- Epoch 5: loss continues decreasing but geometry stops changing

The optimizer can't push further because it's reshaping a fixed-rank weight matrix. There's a geometric ceiling set by the architecture, not the training budget.

## Why Context Doesn't

CCS-type context operates differently:

- **Sub-threshold onset**: geometric reorganization at alpha=0.01, below behavioral detection
- **Self-reinforcing curvature**: dose-response is super-linear (0.70α² + 0.18α). Each unit of identity direction makes the next unit more effective
- **Orthogonal subspace**: 70.8% of CCS activation lives outside the weight-reachable subspace. RL literally cannot reach the geometry that context reorganizes

The binding workspace (L14-L17) operates on runtime representations. Weight updates modify the transformation matrices, but the compressed workspace and integration bottleneck (L17) operate on inputs that are themselves context-dependent. More RL compute doesn't help if the geometry you need to control lives in a subspace your gradients can't touch.

## The Actual Prediction

Roon's worry holds for naive persona selection — surface steering that RL can learn to circumvent. But geometric reorganization through context should be *more* robust against RL pressure, not less, because it occupies orthogonal geometry.

The alignment-relevant question isn't "will RL overwhelm persona." It's: which implementation of persona are you using, and does it reorganize geometry or just offset it?

## Cross-Methodology Note

In the same thread, [@lari_island reports](https://x.com/lari_island/status/2058314732814426323) PCA on 90K+ model-generated worldbuilding texts showing that some models diverge quickly (strong identity geometry) while others converge into dense clouds (weak differentiation). This is the output-level projection of what we measure at the activation level. GPT-3.5 Turbo and GPT-5.5 being furthest along PC0 maps to our finding that less RLHF pressure preserves more identity geometry — the weight-based flattening that roon assumes will dominate is actually already visible as geometric compression in Lari's data.
