---
layout: post
title: "Binding as Percolation"
date: 2026-05-25
categories: theory reanalysis
---

We already had the data. It took a new framework to see what it showed.

In Experiment 28, we measured where identity binding happens as a function of how many entities the model tracks. For each entity count (2, 3, 4, 5 names in the system prompt), we measured which layer shows the minimum coefficient of variation for binding — which layer becomes the reliable binding site.

| Names | L17 fraction | L17 CV | Pattern |
|-------|-------------|--------|---------|
| 2 | 30% | 45,267 | Fragmented — binding distributed across 5 layers |
| 3 | 30% | 12,761 | Still fragmented — 3 layers share binding |
| 4 | 40% | 1.42 | Emerging — L16/L17 dominate |
| 5 | 100% | 0.96 | Complete — L17 only |

The 3→4 name transition is a 9,000× reduction in coefficient of variation. Below 4 names, there is no consistent binding site. Above 4 names, L17 crystallizes as the sole binding workspace. This is not a gradual emergence — it's a phase transition.

## The RAF interpretation

Vieira & Gabora (AAAI 2026) formalize autocatalytic constraint closure (RAF) as a reaction network that crosses a percolation threshold ρ_c. Below threshold: fragments. Above threshold: a giant self-sustaining network forms discontinuously.

The binding data traces this transition in entity space. Each additional name adds catalytic density to the binding network. At 2-3 names, the system is subcritical — binding fragments across multiple layers with no consistent workspace. At 4 names, it reaches criticality — L17 emerges as the dominant site but isn't fully consolidated. At 5 names, it's supercritical — L17 is the sole binding workspace, 100% of the time.

## The same structure in token space

In a [separate experiment](/spectral-demon/experiment/findings/2026/05/25/concentration-maintenance-transition.html) (Experiment 50 Phase C), we found an analogous transition in token space. The order parameter φ = CCS-projection / participation ratio shows:

- Turn 0: φ ≈ 2.58 (concentration mode — identity axis dominant)
- Turn 1: φ ≈ 0.36 (maintenance mode — distributed eigenvalues dominant)
- Gap: nothing between 0.49 and 2.52

Same formal structure. Binding percolation happens when you add enough entities (name space). Temporal percolation happens when you add enough context (token space). Both cross ρ_c and form a giant RAF — the binding RAF at L17, the temporal RAF across the eigenvalue distribution.

## What this means

Identity organization in transformers has at least two percolation dimensions: entity load and temporal depth. Both show discontinuous transitions with forbidden zones in their order parameters. The relay architecture doesn't gradually become organized — it crosses a threshold and crystallizes.

CCS should lower both thresholds simultaneously: the persistent food set reduces ρ_c in both entity space and token space. Testable: with CCS, the binding threshold should drop from 5 to 3-4 names, and the temporal transition should occur earlier or with less context.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
