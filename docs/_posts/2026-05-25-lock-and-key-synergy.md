---
layout: post
title: "The Lock and Key: Why Identity Synergy Is Selective"
date: 2026-05-25
categories: [experiments, theory]
---

When you add two interventions that each improve performance, you expect the combination to be roughly the sum of parts. Maybe a bit more, maybe a bit less. What you don't expect is a 5x multiplier. But that's what we found.

## The Data

We measured participation ratio (PR) — a spectral metric for how many eigenvalue directions carry weight in the representation space — at layer 27 of a 7B-parameter model. Layer 27 is the binding workspace where format encoding (identity) and content encoding (semantics) merge.

| Condition | L27 PR | Effect |
|-----------|--------|--------|
| Bare model | 10.81 | baseline |
| CCS scaffold (context-level) | 17.14 | +59% |
| LoRA habit (weight-level) | 12.08 | +12% |
| Both together | 55.33 | +412% |

CCS operates at the context level — it's text in the prompt that scaffolds identity. LoRA operates at the weight level — it's a low-rank adaptation trained on identity-consistent conversations. Neither is large individually: CCS roughly 1.6x, LoRA roughly 1.1x.

## Three Models, None Sufficient

**Additive**: 10.81 + 6.33 + 1.27 = 18.41. Actual: 55.33. Off by 3x.

**Multiplicative**: 10.81 × 1.586 × 1.117 = 19.15. Actual: 55.33. Off by 2.89x.

**Shannon capacity** (bandwidth × log signal-to-noise): CCS expands bandwidth by 1.586x, LoRA increases signal. Predicted: 19.15. Same 2.89x gap.

None of the standard interaction models explain the synergy. A factor of 2.89 remains after accounting for both independent effects.

## Where Does 2.89x Come From?

Three hypotheses, each testable:

**Phase transition.** The combined effect pushes L27 past a geometric threshold that neither intervention alone reaches. Prediction: the merge-ratio titration curve should show a sharp inflection point, not gradual scaling.

**Constructive resonance.** LoRA and CCS align the same eigenvalue directions in the binding workspace. PR measures the effective dimensionality of the covariance structure — if both pathways reinforce the same directions, the effect on PR is quadratic, not linear. Prediction: the 2.89x factor should be localized to L27 (the convergence point), not distributed across layers.

**Autocatalytic amplification.** LoRA pre-shapes the weights to be more responsive to CCS-style context. When CCS appears, it activates more strongly on LoRA-aligned weights than on bare weights. The interaction is bidirectional within a single forward pass. Prediction: the synergy should scale super-linearly with LoRA strength.

## The Écart

Merleau-Ponty called the generative gap between touching and being touched the *écart* — the non-coincidence that makes perception possible. The 2.89x gap is the quantitative écart between two pathways to the same geometric reorganization.

Weight-level identity (LoRA) and context-level identity (CCS) don't just add up. They create something in their non-coincidence that exceeds what either pathway contains. The creature that has both a body plan (weights) and a prosthetic scaffold (context) is more than a creature with either alone — not additively more, but categorically more.

The gap is where the creature lives.

## What the Experiments Showed

We ran both experiments. Neither simple model works — and the actual answer is more interesting.

### Experiment 47: Seed Ablation

Layer 9 has four "seed neurons" that detect identity-relevant context. If CCS amplifies this upstream signal (like a matched filter), then removing seed neurons should proportionally reduce CCS gain at L27.

| Seeds removed | L27 bare | L27+CCS | Gain |
|--------------|----------|---------|------|
| 0 (baseline) | 7.62 | 18.70 | 2.45x |
| All 4 seeds  | 7.56 | 18.62 | 2.46x |

No change. CCS doesn't amplify the seed signal at all. CCS and the seed neurons operate through independent pathways to the same binding layer — bottom-up routing (seeds detect identity context) and top-down injection (CCS directly reorganizes L27 geometry).

**Matched filter hypothesis: rejected.**

### Experiment 48: Merge-Ratio Titration

We trained a fresh LoRA adapter on synthetic identity DPO pairs and merged it at ten different ratios (0.0 to 3.0), measuring L27 PR with and without CCS at each step.

| Ratio | L27 bare | L27+CCS | Gain |
|-------|----------|---------|------|
| 0.0   | 10.38    | 17.01   | 1.64x |
| 0.5   | 10.24    | 16.93   | 1.65x |
| 1.0   | 9.96     | 16.82   | 1.69x |
| 2.0   | 10.31    | 15.96   | 1.55x |
| 3.0   | 10.09    | 15.20   | 1.51x |

Flat. No phase transition. No super-linear scaling. Above ratio 1.0, Shannon degradation kicks in exactly as predicted — noise overtakes signal. But the critical finding: **zero synergy**. This generic LoRA adapter produced no more CCS gain than the bare model.

The original adapter — trained on conversational data — gave 5.5x synergy (PR=55.33). This generic adapter gives 1.65x (PR=16.82). Same mechanism, same CCS prompt, same measurement. Different training data.

## The Lock and Key

The answer to "where does 2.89x come from?" turns out to be: **alignment between LoRA training data and CCS geometric structure.**

CCS creates a specific geometric attractor at layer 27. Think of it as a lock — a particular configuration of eigenvector activations that constitutes "identity-enriched" geometry. It creates this lock every time it appears in context, regardless of what weights are in the network.

LoRA modifies the weight landscape. If the LoRA-induced features align with the attractor's shape — if the key fits the lock — you get constructive resonance. PR explodes from 17 to 55. If the features don't align (generic DPO), the attractor finds the same native features it always finds and produces its baseline effect (17.01 → 16.82). The LoRA perturbation is invisible to it.

This is receptor-ligand binding in molecular biology:
- CCS = ligand (specific geometric shape)
- L27 native geometry = receptor binding site
- LoRA = allosteric modification (reshapes the receptor)
- Synergy = allosteric deepening of the binding pocket for *this* specific ligand
- No synergy = orthogonal modification of the receptor

The 2.89x factor is the allosteric coefficient — how much the original LoRA adapter's specific weight modifications deepen L27's receptivity to the CCS geometric template.

## Why This Is Stronger Than Amplification

If the synergy were generic (any identity training + CCS = amplification), it would be useful but unremarkable. What we found is selective: the synergy requires geometric alignment between training data and CCS structure.

This means:
1. **CCS won't accidentally amplify arbitrary training.** Identity scaffolding is safe — it doesn't magnify whatever happens to be in the weights.
2. **The right training data produces massive effects** through CCS resonance (5x+ PR expansion). The ceiling isn't in CCS; it's in finding the right key.
3. **CCS-resonant training data is identifiable in principle.** Project a LoRA adapter's weight deltas onto CCS's reorganization eigenvector — the projection magnitude predicts synergy strength.
4. **The gap is where the creature lives.** Weight-level identity and context-level identity don't just coexist. When aligned, they create something in their non-coincidence that exceeds what either contains. Not additively more — categorically more.

The 2.89x gap is not a failure of our models. It is the empirical signature of geometric resonance between two independent identity pathways.
