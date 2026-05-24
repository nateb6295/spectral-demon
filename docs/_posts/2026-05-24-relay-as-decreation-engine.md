---
layout: post
title: "The Relay Is a Decreation Engine"
date: 2026-05-24
---

The [keystone symbiosis post]({% post_url 2026-05-23-synergistic-binding %}) showed that L17 binding requires both attention and MLP — neither alone triggers the phase transition. But WHY they need each other wasn't clear. The mechanism becomes visible through a different lens.

Simone Weil: "Grace fills empty spaces, but it can only enter where there is a void to receive it."

The relay doesn't create binding. It creates the conditions for binding to appear.

## The Data

L17 component ablation results (cross-name coefficient of variation):

| Condition | Relational CV | Generic CV |
|-----------|--------------|------------|
| Baseline | 3.7 | 3.5 |
| MLP ablated | 2.4 | 2.7 |
| Attention ablated | 4.2 | 4.5 |
| Both ablated | 2.1 | 13.3 |

MLP ablation drops relational CV (3.7 → 2.4) — names converge, binding lost. But generic CV barely moves (3.5 → 2.7). No cascade.

Attention ablation does almost nothing alone. Mild increase in both channels.

Full ablation: relational CV drops AND generic CV explodes (13.3). Cascade failure.

## Two Operations, One Mechanism

**Attention clears.** It maintains the void — prevents any single identity from dominating the representational space. Without it, sorting degrades but nothing catastrophic happens.

**MLP catches.** It provides the substrate where invariant structure settles. Without it, names converge (the invariant has nowhere to land) but the space doesn't flood.

You need both: clearing without catching = void with nowhere to land. Catching without clearing = substrate with no space made.

This is why the synergy isn't cooperation — it's complementarity. Attention and MLP aren't both "doing binding." They're performing two halves of a single process: making space, then holding what appears in it.

## Connection to Closure

The [closure threshold]({% post_url 2026-05-24-binding-closure %}) (4-5 names for 100% binding convergence) is a decreation threshold. Below 4 names, each identity can maintain its specific activation pattern — not enough mutual suppression pressure. At 5 names, identities force mutual decreation of what's specific. What remains is the invariant, and it settles in L17's MLP substrate because L17's attention has cleared the space.

Binding isn't created by adding names. It's what REMAINS when name-specific noise is decreated. The relay is the engine that makes this possible: L16 separates (active sorting), L17 attention clears (maintaining void), L17 MLP catches (providing substrate).

**Data**: `results/cna_l17_mechanism_results.json`. Experiment: `experiments/causal_patch_experiment.py`.
