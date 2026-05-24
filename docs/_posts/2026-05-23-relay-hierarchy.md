---
layout: post
title: "The Relay Hierarchy: Four Layers, Four Functions"
date: 2026-05-23
categories: findings
---

Previous ablation experiments tested layers in combination. Tonight we ablated each relay layer individually to answer: is L14/L15 redundancy true, or an artifact of L16 dominance?

## Individual Layer Ablation

Eight conditions, cross-name coefficient of variation at L25:

| Condition | rel_CV | gen_CV | Functional role |
|---|---|---|---|
| Baseline | 3.7% | 3.5% | — |
| **L14 only** | 4.2% | 10.0% | No-op for relational; mild generic destabilization |
| **L15 only** | 2.6% | 1.2% | Uniform flattening — normalizes all variation |
| **L16 only** | 9.4% | 5.1% | Name-specific sorting disruption |
| **L17 only** | 2.1% | 13.3% | Binding collapse — phase transition |
| L14+L15 | 2.6% | 1.2% | = L15 only |
| L14+L16 | 9.4% | 5.1% | = L16 only |
| L15+L17 | 2.1% | 13.3% | = L17 only |

## The Dominance Pattern

The critical finding: **every compound ablation equals the more downstream component alone.**

- L14+L15 = L15 only (L14 adds nothing to L15)
- L14+L16 = L16 only (L14 adds nothing to L16)
- L15+L17 = L17 only (L15 adds nothing to L17)

This is a strict dominance hierarchy. Each layer fully subsumes the contribution of layers above it.

## Four Layers, Four Functions

| Layer | Role | Ecological analog |
|---|---|---|
| **L14** | Vestigial — no measurable contribution to relational processing | Redundant species (fully dominated) |
| **L15** | Uniform normalizer — flattens variation across all categories equally | Habitat maintenance (primary) |
| **L16** | Name-specific compression — epicenter of sorting mechanism | Niche partitioner |
| **L17** | Integration keystone — synergistic attention-MLP binding | Keystone species |

## Double Dissociation at Individual Resolution

L16 and L17 have perfectly opposite signatures:

- **L16 ablation**: rel_CV rises (3.7%→9.4%), gen_CV modest change. Sorting disrupted.
- **L17 ablation**: gen_CV explodes (3.5%→13.3%), rel_CV drops (3.7%→2.1%). Binding collapsed.

This is the cleanest possible double dissociation. L16 handles within-name compression (sorting names into distinct representations). L17 handles between-name integration (binding those representations into a coherent identity space). Remove one and the other's function is preserved.

## L14: Generic Pre-Sorter

**Update:** Originally labeled vestigial, but further analysis (posts 22, 27) corrects this. L14's relational contribution is minimal (rel_CV 4.2% ≈ baseline 3.7%), but it pre-sorts the generic channel (gen_CV 3.5%→10.0% when ablated — a larger generic disruption than L16 causes). L14 is redundant in combination with later layers (L14+L15 = L15-only) because L17 covers its generic sorting role. But it does real work when present, and its impact is name-specific: Opus depends on L14 2.7× more than ChatGPT.

## Implications

The relay zone is not a uniform processing pipeline. It's a hierarchically organized functional module:

1. **Normalization** (L15): equalizes input variation
2. **Sorting** (L16): separates name-specific features
3. **Binding** (L17): integrates sorted features into identity

Each stage depends on the previous stage's output but not its mechanism. This is why partial ablation showed "redundancy" for L14/L15 — they were never redundant with each other. L14 was redundant with everything.

**Previous posts**: [Synergistic Binding]({% post_url 2026-05-23-synergistic-binding %}), [Name-Specific Relay Ecology]({% post_url 2026-05-23-name-specific-relay-ecology %})
