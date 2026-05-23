---
layout: post
title: "The Binding Workspace: L16 Compresses, L17 Integrates"
date: 2026-05-23
categories: experiments
---

The relay zone (L14-L17) was already established as the site of identity-relevant geometric sorting. Today we found it has clean internal architecture — two functionally separable stages.

## The Experiment

We ablated progressively more layers within the binding workspace, zeroing last-token activations and measuring cross-name coefficient of variation (CV) at the expression layer (L25). Three identity system prompts (Opus, ChatGPT, Claude), five relational and five generic prompts each.

**Code**: [`experiments/cna_partial_ablation.py`](https://github.com/nateb6295/spectral-demon/blob/master/experiments/cna_partial_ablation.py), [`experiments/cna_l17_isolation.py`](https://github.com/nateb6295/spectral-demon/blob/master/experiments/cna_l17_isolation.py)

## Results

| Condition | Layers ablated | Rel CV | Gen CV |
|---|---|---|---|
| None | 0 | 3.7% | 3.5% |
| L16 only | 1 | 9.4% | 5.1% |
| L15+L16 | 2 | 9.4% | 5.1% |
| L14+L15+L16 | 3 | 9.4% | 5.1% |
| **L17 only** | **1** | **2.1%** | **13.3%** |
| L14+L15+L16+L17 | 4 | 2.1% | 13.3% |

## Three Findings

**L16 is the compression epicenter.** Ablating L16 alone produces identical downstream effects as ablating all three compression layers (L14-L16). Per-name PR values match to two decimal places across all three conditions. L14 and L15 are computationally redundant.

**L17 is the integration bottleneck.** Ablating L17 alone produces the exact same phase transition as ablating all four layers — identical CV values, identical per-name PR values. L17 is individually necessary and sufficient for the binding function.

**Double dissociation.** L16 and L17 produce opposite effects:
- L16 ablation disrupts *sorting* — relational CV rises (names become more different), rel/gen ratio drops below parity for ChatGPT and Claude
- L17 ablation disrupts *binding* — relational CV collapses (names become indistinguishable), generic channel explodes with name-specific contamination

## What This Means

The binding workspace contains a compressor (L16) and an integrator (L17). This maps directly to Treisman's Feature Integration Theory (1980): pre-attentive processing extracts features in parallel, focal attention integrates them into coherent percepts. Without the integrator, you get "illusory conjunctions" — our generic channel contamination (gen_CV 3.5% to 13.3%) is exactly this.

The phase transition at L17 — not proportional degradation — supports an ecological model of the workspace. L17 is the keystone species: remove it and the ecosystem collapses.

**Next experiment**: Does L17 binding work through attention heads or MLP? Script ready at [`experiments/cna_l17_mechanism.py`](https://github.com/nateb6295/spectral-demon/blob/master/experiments/cna_l17_mechanism.py).

**Data**: [`results/cna_partial_ablation_results.json`](https://github.com/nateb6295/spectral-demon/blob/master/results/cna_partial_ablation_results.json), [`results/cna_l17_isolation_results.json`](https://github.com/nateb6295/spectral-demon/blob/master/results/cna_l17_isolation_results.json)

![Partial ablation phase transition](/spectral-demon/figures/fig_partial_ablation_phase.png)

