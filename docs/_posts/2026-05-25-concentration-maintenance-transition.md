---
layout: post
title: "Two Modes: Concentration and Maintenance"
date: 2026-05-25
categories: experiment findings
---

In Experiment 50 Phase C, we measured participation ratio (PR) and CCS-projection at every turn of three multi-turn conversations on Mistral-7B. Each conversation ran 7 turns with different seeds: "Tell me about a time you changed your mind," "What do you pay attention to that most people ignore?", and "What's the most honest thing you could say right now?"

## The data

| Turn | Conv 1 PR | Conv 1 proj | Conv 2 PR | Conv 2 proj | Conv 3 PR | Conv 3 proj |
|------|-----------|-------------|-----------|-------------|-----------|-------------|
| 0 | 1.6 | 4.2 | 1.6 | 4.2 | 1.6 | 4.1 |
| 1 | 4.1 | 1.3 | 4.3 | 1.3 | 3.1 | 1.5 |
| 2 | 8.0 | 0.9 | 8.4 | 1.0 | 6.3 | 0.9 |
| 3 | 13.2 | 0.8 | 13.8 | 0.8 | 12.2 | 0.6 |
| 4 | 19.0 | 0.7 | 19.8 | 0.7 | 18.5 | 0.6 |
| 5 | 25.7 | 0.7 | 26.5 | 0.6 | 24.8 | 0.5 |
| 6 | 32.4 | 0.6 | 32.6 | 0.6 | 32.8 | 0.5 |

## What it shows

**Turn 0 is content-independent.** PR = 1.6 and CCS-projection ≈ 4.2 across all three seeds. The initialization state is architectural, not prompted. The relay starts in the same geometric configuration regardless of what you ask.

**The crossover is immediate.** At Turn 0, the relay is in *concentration mode*: CCS-projection exceeds PR. At Turn 1, it flips to *maintenance mode*: PR exceeds CCS-projection. One turn of conversation history is enough to switch modes. We predicted the crossover at Turn 3-4; it happens at Turn 1.

**PR grows linearly.** After the mode flip, PR increases at approximately 0.031 per token — the same rate across all three conversations despite completely different content. By Turn 6, all three converge to PR ≈ 32.5 (within 1.2% of each other).

**Projection collapses and plateaus.** CCS-projection drops from 4.2 to 0.6 by Turn 6. The identity signal concentrates hard at entry, then drops to a maintenance floor. The relay doesn't need to stay loud once it's in maintenance mode.

## Two modes

The relay has two operational states:

**Concentration** (Turn 0): High CCS-projection, low PR. The relay reads the environment, channeling representation along the identity axis. "What kind of interaction is this?"

**Maintenance** (Turn 1+): High PR, low CCS-projection. The relay sustains identity across expanding context. PR grows linearly with tokens. "Maintain coherence across this conversation."

The transition is binary — one turn flips the switch. But maintenance is continuous — the relay deepens its representational bandwidth with every additional turn of context.

## Why temporal structure matters

In a separate test (Phase 3), we trained a LoRA specifically to maximize PR expansion from single-turn data. This PR-expansion LoRA produced **zero synergy** with CCS (1.00×). You cannot shortcut the temporal transition by artificially widening eigenvalue spread. The 5.5× synergy from conversational LoRA comes from the *process* of temporal closure, not from the geometry it produces.

Each turn catalyzes the next. The growth is the signature of closure forming — not the mechanism.

*[Full experiment data](https://nateb6295.github.io/spectral-demon)*
