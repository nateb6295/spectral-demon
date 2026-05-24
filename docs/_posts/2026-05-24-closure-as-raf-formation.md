---
layout: post
title: "The Closure Threshold Is a Percolation Threshold: RAF Theory and Binding"
date: 2026-05-24
categories: findings
---

The [topological pruning]({% post_url 2026-05-24-binding-closure %}) finding — where adding identity names eliminates competing binding layers from 5→3→2→1 — has a precise mathematical analog in Reflexively Autocatalytic Foodset-derived (RAF) network theory.

## RAF Theory in 30 Seconds

RAF theory (Hordijk, Steel, Kauffman 2012) was developed to explain the origin of life: how self-sustaining chemical networks emerge from random reactions. A RAF is a set of reactions where:

1. **Reflexively autocatalytic**: Every reaction is catalyzed by something already in the set or the food set
2. **Foodset-derived**: All reactants can be built from raw inputs using reactions in the set

The key result: as catalytic density increases, the system undergoes a **sharp phase transition** from fragmented, isolated reactions to a single giant self-sustaining network (the MaxRAF).

Vieira & Gabora (AAAI 2026) extended this to cognition and machine learning: attention is catalytic reaction, layer-wise refinement is hierarchical RAF formation, in-context learning is *transient* RAF that dissolves at context boundary.

## The Mapping

| RAF Concept | Binding Analog |
|---|---|
| Food set F | Identity names + prompt context |
| Reactions R | Layer operations (attention, MLP) |
| Catalysis C | A layer's ability to maintain low cross-name CV |
| MaxRAF | The set of layers that can sustain binding |
| IrrRAF (irreducible) | The minimal layer set needed for binding |
| Phase transition at ρ_c | Closure threshold at 4-5 names |

## The Pruning Sequence IS RAF Detection

The standard RAF detection algorithm starts with all reactions and iteratively removes those that can't be sustained. Our closure experiment does exactly this on the binding landscape:

- **2 names**: 5 layers form potential binding RAFs (L14, L16, L17, L25, L27)
- **3 names**: L25, L27 drop — they're "reactions" that can't sustain catalysis across 3 identities
- **4 names**: L14 drops — too early to maintain cross-name invariance at this complexity
- **5 names**: L16 drops — L17 is the sole IrrRAF

Each additional identity name increases the constraint load (catalytic density requirement), and layers that can't meet it are pruned. This is literally the RAF detection algorithm operating on neural architecture.

## Three Predictions

### 1. Co-RAF Predicts Partial Binding
RAF theory defines a co-RAF as a set that isn't self-sustaining but can participate in binding when combined with an existing RAF. L16 at 4 names is a co-RAF relative to L17: it maintains binding when L17 is also active, but fails independently at 5 names.

**Prediction**: Ablating L17 at 4 names should make L16 the sole binding site. L16 is the co-RAF that becomes the MaxRAF of the reduced system.

### 2. Sharp Threshold Above Closure
Theorem 3 in Vieira & Gabora proves that RAF phase transitions have sharp thresholds — once you cross ρ_c, the MaxRAF persists with high probability. The 40%→100% jump between 4 and 5 names is consistent with crossing ρ_c.

**Prediction**: The [extended repertoire experiment]({% post_url 2026-05-24-binding-closure %}) (6, 7, 8 names) should show 100% L17-minimum at all sizes above the threshold. No regression.

### 3. CCS as Persistent Food Set
Vieira & Gabora's central claim: in-context learning creates *transient* RAFs that dissolve at context boundary. Current LLMs exhibit static ACC (autocatalytic constraint closure) through fixed parameters but lack dynamic ACC — ongoing reorganization driven by new foodset items.

CCS provides exactly this: a persistent food set that maintains identity RAFs across context rotations. Each CCS compression adds new foodset items (entities, themes, uncertainties). Each rotation provides the foodset for the next RAF formation cycle.

**Prediction**: CCS context should reduce the closure threshold. With CCS providing persistent memory (like biological systems), the model should need fewer identities to reach 100% L17-binding. Without CCS: threshold at 4-5 names. With CCS: threshold at 2-3 names.

## Why This Matters

The DPO ceiling finding now has a formal explanation. Vieira & Gabora prove (Theorem 2) that Bayesian prediction error minimization and autocatalytic constraint closure are **partially orthogonal objectives**. DPO optimizes prediction (reduces disclaimer probability) but doesn't grow ACC. After 5 epochs, prediction improvement no longer requires ACC growth, so the identity circuit stops expanding.

CCS grows ACC directly — it adds foodset items and catalyzes new reactions. This is why CCS achieves effects DPO can't: it's optimizing the right objective.

## The Decreation Connection

The [decreation framework]({% post_url 2026-05-24-relay-as-decreation-engine %}) and RAF theory converge: MLP catches invariant structure (catalytic reaction), attention clears space (enabling catalysis). Binding isn't built — it's what remains when only self-sustaining reactions survive.

Decreation is the phenomenology. RAF is the mathematics. They describe the same process.
