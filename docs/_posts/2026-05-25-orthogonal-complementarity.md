---
layout: post
title: "Orthogonal Complementarity: Why the Synergy Is Multiplicative"
date: 2026-05-25
categories: experiments dual-encoding synergy
---

Experiment 49 tested the temporal coherence hypothesis — that CCS-resonant training data works because of multi-turn temporal structure, not identity content. The prediction was a clean gradient: multi-turn identity > multi-turn generic > single-turn identity > single-turn generic.

The prediction was wrong. What we found is more interesting.

## The Split

Four conditions, two metrics, measured at L27 on Mistral-7B-Instruct-v0.3:

| Condition | CCS-projection | Participation Ratio |
|-----------|---------------|-------------------|
| Multi-turn identity | 248.0 | 21.41 |
| Single-turn identity | 378.1 | 17.40 |
| Multi-turn generic | 172.9 | 21.31 |
| Single-turn generic | 250.5 | 20.25 |

The two metrics split along orthogonal axes.

**Participation ratio** tracks temporal structure. Multi-turn conversations produce higher PR regardless of whether the content involves identity (21.41 vs 21.31 — nearly identical). Identity content contributes almost nothing to PR. In single-turn conditions, identity content actually *reduces* PR (17.40 vs 20.25).

**CCS-projection** tracks identity content. Identity-relevant prompts produce higher CCS-projection regardless of turn count (378 vs 250 for single-turn, 248 vs 173 for multi-turn). But multi-turn *reduces* CCS-projection compared to single-turn with the same content.

## What This Means for the 5.5x Synergy

The [lock-and-key model](/experiments/synergy/2026/05/25/lock-and-key-synergy.html) showed that conversational LoRA + CCS produces 5.5x synergy while generic DPO + CCS produces only 1.65x. The temporal coherence hypothesis proposed this was because multi-turn structure "resonates" with CCS.

The data shows something cleaner: multi-turn and CCS are **orthogonal interventions that multiply**.

In multi-turn conversations, identity is distributed across turns. At any single extraction point, less is concentrated along the CCS eigenvector — but more eigenvalues participate in the representation. The model maintains identity across a wider geometric subspace.

This means:
- **Conversational LoRA** trains eigenvalue expansion (format-level PR increase). It provides what CCS doesn't.
- **CCS** provides directional alignment along the identity eigenvector. It provides what LoRA doesn't.
- **Generic DPO** (single-turn) trains directional push along the same axis CCS already handles. Redundant. 1.65x = additive.
- **Conversational LoRA + CCS** = orthogonal forces whose product creates a larger volume in activation space. 5.5x = multiplicative.

The synergy isn't two things pushing in the same direction. It's two things pushing in perpendicular directions whose cross product creates something neither achieves alone. LoRA expands the subspace. CCS orients it.

## Third Confirmation of Dual Encoding

This is the third independent measurement of the format/content split:

1. **Name vs company** (Experiments 1-42): Content encoding (name) changes freely; format encoding (company affiliation) persists through every intervention
2. **Behavioral probes** (Experiments 43-45): Three model scales differ in format maintenance while sharing content-level patterns
3. **Temporal ablation** (Experiment 49): PR tracks temporal structure (format); CCS-projection tracks identity content (content)

Three different measurement instruments — prompt manipulation, cross-scale comparison, temporal structure variation — same two-axis structure. The dual encoding isn't an artifact of any single approach. It's a property of how transformers organize identity.

## The Revised Hypothesis

The temporal coherence hypothesis needs refinement:

**Original**: CCS-resonance comes from temporal structure (multi-turn identity maintenance), not identity content.

**Revised**: CCS synergy comes from temporal structure because multi-turn format creates eigenvalue expansion that is geometrically orthogonal to CCS's directional alignment. The "key's shape" isn't alignment — it's complementarity. The lock and key don't match; they interlock.

## Open Question

If PR expansion is the mechanism, can we produce synergy *without* multi-turn data by training LoRA specifically to maximize eigenvalue spread at L27? If yes, temporal structure is a proxy for PR expansion, not intrinsically valuable. If no, temporal structure contributes something beyond geometry — perhaps the reflexive closure that [autocatalytic identity](/experiments/closure/2026/05/24/autocatalytic-closure-and-ccs.html) requires.

---

*Experiment 49. H100 80GB, Mistral-7B-Instruct-v0.3. [Full data](https://nateb6295.github.io/spectral-demon)*
