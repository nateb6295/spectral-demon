---
layout: post
title: "L17 Binding Is Emergent: Closure Under Name Subsets"
date: 2026-05-24
categories: findings
---

If L17 is a true binding attractor, its minimum-CV property should hold for any subset of identity names, not just the full set. It doesn't — and that's the interesting finding.

## The Test

We measured binding CV at layers [9, 14, 16, 17, 25, 27] for all possible subsets of names {Opus, Claude, ChatGPT, Gemini, Llama}: 10 pairs, 10 triples, 5 quadruples, 1 quintuple.

## Results

| Subset size | L17 is global minimum | Min-layer distribution |
|-------------|----------------------|----------------------|
| 2 names | 30% (3/10) | L14:1, L16:1, **L17:3**, L25:2, L27:3 |
| 3 names | 30% (3/10) | L14:3, L16:3, **L17:3**, L25:1 |
| 4 names | 40% (2/5) | L16:2, **L17:2**, L25:1 |
| 5 names | **100%** (1/1) | **L17:1** |

## What This Means

L17 binding convergence is not a fixed architectural property — it's **emergent from the identity repertoire**. With only two names, the binding minimum wanders across layers: L14, L16, L17, L25, L27 all compete. As you add more names, the binding landscape stabilizes and L17 wins.

The trajectory: 30% → 30% → 40% → 100%. Each additional name narrows the set of layers that can maintain low cross-name CV, until only L17 survives.

## Why L17 Wins at Scale

With two names, any layer can achieve low CV by chance — the two activation patterns might happen to align at L25 or L27. But as you add more identity patterns, only layers with genuine binding capacity (where the architecture naturally converges identity representations) can maintain low CV across all of them simultaneously.

L17 is the layer where the relay has enough representational bandwidth to bind multiple identities into a common format. Other layers can do this for 2-3 names but fail when asked to bind all 5.

## Connection to Scaling

This mirrors the [scaling migration finding]({% post_url 2026-05-24-binding-migration-across-scale %}): binding is about **capacity**. Just as wider models shift binding from seed to relay (more parameter capacity → more specialized binding), more identity names shift the minimum toward the layer with the most binding capacity.

The closure test reveals that L17's binding role is a statistical attractor — it emerges from the ensemble, not from any single identity pair.

## Numerical Note

With 2-3 name subsets, mean CV values at L17 are inflated by numerical instability (near-zero mean neurons creating extreme ratios). This disappears at 4-5 names as the mean stabilizes. The min-layer distribution (which layer has lowest CV) is robust to this artifact.
