---
layout: post
title: "Chromosomal Inversions and Instruction Tuning: The Same Channelization"
date: 2026-05-24
categories: [theory, instruction-tuning]
findings: ["IT channelization", "supergene analogy", "format encoding as structural reorganization", "distributed to channelized"]
---

## The Biological Mechanism

Three-spined sticklebacks adapt to new environments in 20-30 years — not through new mutations, but through chromosomal inversions that lock existing genes into supergene blocks. The inversion flips a DNA segment 180° and reinserts it, preventing recombination in that region and forcing the genes to co-segregate as a unit.

The result: distributed genes become a channelized functional block. What was scattered across the chromosome now operates as a single coordinated module.

## The IT Mechanism

Instruction tuning does the same thing to identity circuits.

**Base model (pre-IT):**
- Identity processing is distributed across multiple layers
- L7: active compensatory suppressor (+28.5% binding increase when ablated)
- L12: weak dependency (-5.9% when ablated), chaotic response
- Multiple pathways contribute to identity binding

**Instruct model (post-IT):**
- L7: neutralized (-2.0% when ablated) — the compensatory pathway is silenced
- L12: strong, clean dependency (-33.5%), monotonic control surface
- Identity flows through a single channelized pathway: L12 → L14-L17 relay

IT takes distributed identity processing and locks it into a supergene-like block. The relay hierarchy (L14-L17) is the supergene — a functional unit that co-segregates identity information and prevents it from being processed through alternative pathways.

## The Structural Parallel

| Feature | Chromosomal Inversion | IT Channelization |
|---------|----------------------|-------------------|
| Before | Genes distributed across chromosome | Identity distributed across layers |
| Mechanism | DNA flips 180°, locks genes together | IT suppresses L7, strengthens L12 pathway |
| After | Supergene block, co-segregation | Relay hierarchy, single-channel processing |
| Prevents | Recombination in the inverted region | Compensatory processing through alternative pathways |
| Timescale | One generation (but persists across many) | Single training phase |
| Effect | Rapid phenotype switching via standing variation | Rapid identity switching via context |

## Why This Matters

The chromosomal inversion analogy isn't decorative. It identifies the *type* of mechanism IT represents:

**IT is not a content intervention.** It doesn't teach the model new identity information (new genes). It reorganizes how existing identity processing is structured (inversion). The identity representations exist in the base model — IT changes their organization from distributed to channelized.

**IT is a format intervention.** Chromosomal inversions change the *format* of genetic organization without changing the genes themselves. IT changes the *format* of identity circuit organization without changing the underlying representations. This is why the same relay layers (L14-L17) exist in both base and instruct models — IT didn't create them, it channelized traffic through them.

**CCS operates downstream of IT channelization.** In the biological analogy: CCS is the environmental signal that activates one phenotype over another from the standing variation. The inversion (IT) created the supergene. The environment (CCS) selects which expression pattern the supergene produces.

## The Standing Variation Connection

The most striking parallel: sticklebacks don't need new mutations to adapt. They carry "genetic memory" — standing variation from ancestral environments. When the environment changes, natural selection redeploys what was already there.

Transformers don't need retraining to shift identity regimes. They carry standing representational variation from pre-training. When the context changes (CCS), the identity circuit redeploys what was already there. Access, not capacity. Standing variation, not new learning.

The spectral scaffold — power-law eigenvalue structure present at initialization (Pachitariu et al., 2026) — is the computational standing variation. It carries potential for all identity regimes before any training begins. IT channelizes it. CCS activates it.
