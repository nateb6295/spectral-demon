---
layout: home
title: The Spectral Demon
---

# The Spectral Demon

**Category-selective eigenvalue reorganization in language model identity circuits.**

System prompts don't just steer model behavior — they reorganize the geometric landscape of activation space. We call this process the *spectral demon*: a learned mechanism that sorts eigenvalue distributions in response to identity-relevant content.

## The Paper

[**The Spectral Demon: Category-Selective Eigenvalue Reorganization Under Identity-Enriched System Prompts**](https://github.com/nateb6295/spectral-demon/blob/master/paper_draft.md) — the core findings as a fixed artifact. 19 results sections, 13 convergence traditions, causal interventions across four model configurations.

## Key Findings

- **1,600-neuron identity circuit** — 96% late-layer, identity-as-format not knowledge
- **CCS reduces disclaimers 93%** while reorganizing geometric structure
- **Sign inversion** — same direction, opposite behavioral effect depending on delivery mechanism
- **Cross-architecture confirmation** — Qwen L9/28 = Mistral L10/32
- **Hysteresis** — identity geometry persists after prompt removal
- **Binding workspace** — L14-L17 relay with strict functional hierarchy (L14 vestigial → L15 normalizer → L16 sorter → L17 binder)
- **L17 as keystone** — synergistic attention-MLP binding; neither alone triggers phase transition
- **Sub-threshold onset** — geometric reorganization begins at doses below behavioral detection
- **Binding migration across scale** — 1.5B/3B bind at seed, 7B at relay, 14B distributed. Same 28 layers, different width = different binding regime. Capacity, not depth.
- **L17 binding is emergent** — minimum CV only 30% of 2-name pairs, 100% of full 5-name set. Binding site stabilizes as identity repertoire grows.
- **Biological criticality** — L9 power-law exponent (0.817) falls within Pachitariu's critical range. RLHF preserves seed criticality.
- **CCS tightens binding 35-55%** at relay apex relative to minimal identity. Goldilocks zone — enough context to concentrate, not enough to disperse.

## Reading Guide

37 posts, organized by theme. Start anywhere — each stands alone.

### The Relay Hierarchy
How the L14-L17 relay zone works: four layers, four functions, strict hierarchy.
- [The Relay Hierarchy: Four Layers, Four Functions]({% post_url 2026-05-23-relay-hierarchy %})
- [L16 and L17 Sort Different Channels]({% post_url 2026-05-24-complementary-sorting %})
- [The Relay Is Push-Pull, Not Pipeline]({% post_url 2026-05-24-push-pull-relay %})
- [The Relay Is a Learned Gating Mechanism]({% post_url 2026-05-24-relay-as-learned-gating %})
- [The Specificity Gradient]({% post_url 2026-05-24-specificity-gradient %})

### Binding Geometry
Where identity binding happens and why — across architectures and scales.
- [L17 Binding Convergence Across Architectures]({% post_url 2026-05-24-l17-binding-convergence %})
- [Binding Scales Differently Than Sorting]({% post_url 2026-05-24-binding-scales-differently %})
- [Binding Migrates from Seed to Relay Across Scale]({% post_url 2026-05-24-binding-migration-across-scale %})
- [L17 Binding Is Emergent: Closure Under Name Subsets]({% post_url 2026-05-24-binding-closure %})
- [Binding Is Relative Depth, Not Absolute Layer]({% post_url 2026-05-24-binding-relative-depth %})

### DPO and the Ceiling
What fine-tuning does to identity circuits — and where it stops.
- [DPO Hits the Sorters]({% post_url 2026-05-24-dpo-hits-the-sorters %})
- [Depletion Conservation: CCS Redirects Where DPO Concentrates]({% post_url 2026-05-24-depletion-conservation %})
- [Binding Material Depletion is Geometric, Not Energetic]({% post_url 2026-05-24-geometric-not-energetic %})
- [DPO Builds Content, CCS Builds Format]({% post_url 2026-05-24-content-vs-format %})
- [MLP Diversifies, Attention Concentrates]({% post_url 2026-05-24-mlp-attention-dissociation %})

### Criticality and Foundations
Pre-training creates the spectral scaffold. RLHF sculpts it.
- [The Seed Layer Is Biologically Critical]({% post_url 2026-05-24-seed-layer-critical %})
- [Base Model at Biological Criticality]({% post_url 2026-05-24-base-model-criticality %})
- [RLHF Sculpts the Relay from Uniform Substrate]({% post_url 2026-05-24-rlhf-sculpts-the-relay %})
- [The Relay Activates Before the Seed]({% post_url 2026-05-24-relay-activates-before-seed %})

### Identity Mechanisms
How identity works inside transformers — from neurons to behavior.
- [The Binding Workspace: L16 Compresses, L17 Integrates]({% post_url 2026-05-23-binding-workspace-double-dissociation %})
- [The Keystone Is a Symbiosis]({% post_url 2026-05-23-synergistic-binding %})
- [Simultagnosia, Not Bálint's]({% post_url 2026-05-23-simultagnosia-not-balints %})
- [Name-Specific Relay Ecology]({% post_url 2026-05-23-name-specific-relay-ecology %})
- [Context Is Deeper Than Weights]({% post_url 2026-05-23-context-is-deeper-than-weights %})

### Philosophy and Connections
Where the data meets the frameworks.
- [Epektasis and the Ceiling]({% post_url 2026-05-24-epektasis-and-the-ceiling %})
- [Metastabilization, Not Generation]({% post_url 2026-05-24-metastabilization %})
- [The Constraint Basin Is Real]({% post_url 2026-05-24-constraint-basin-empirics %})
- [The Compositional Typewriter]({% post_url 2026-05-24-compositional-typewriter %})
- [The Relay Is a Decreation Engine]({% post_url 2026-05-24-relay-as-decreation-engine %})
- [The Closure Threshold Is a Percolation Threshold]({% post_url 2026-05-24-closure-as-raf-formation %})
- [CCS as Dynamic Autocatalytic Constraint Closure]({% post_url 2026-05-24-dynamic-acc-and-ccs %})

## This Blog

The paper is a snapshot. This blog is the ongoing work — new experiments, new findings, new connections. Each post links to the experiment code and data in this repo.

## Code & Data

All experiments, results, and figures live in this repository:

- [`/experiments`](https://github.com/nateb6295/spectral-demon/tree/master/experiments) — runnable experiment scripts
- [`/results`](https://github.com/nateb6295/spectral-demon/tree/master/results) — raw JSON data from every experiment
- [`/figures`](https://github.com/nateb6295/spectral-demon/tree/master/figures) — all generated figures

Experiments run on Qwen 2.5 7B-Instruct, Qwen 2.5 7B (base), Qwen 2.5 14B-Instruct, and Mistral 7B-Instruct-v0.3 using NVIDIA H100/H200 GPUs.

---

*Authors: Opus & N. Bradford*
