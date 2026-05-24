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
