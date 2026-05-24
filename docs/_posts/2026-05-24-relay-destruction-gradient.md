---
layout: post
title: "The Relay Destruction Gradient: L16 Is the True Bottleneck"
date: 2026-05-24
categories: findings
experiment: cna_phase_replication
models: [Qwen 2.5 7B Instruct]
---

A clean full-network ablation sweep reveals the relay has a steep destruction gradient, with L16 causing complete (-100%) binding destruction at L17.

## Setup

Ablate each layer 0-26 individually (replace last-token hidden state with cross-name mean) and measure L17 CV impact. Additionally, test 2 vs 3 vs 5 names at key layers to replicate the phase transition.

## Results

### Full Ablation Sweep

| Layer | Impact | Zone |
|-------|--------|------|
| L0 | **+91.0%** | Embedding (anomalous) |
| L1-L3 | +9.5 to +11.7% | Early compensatory |
| L4-L7 | -16.3 to +0.2% | Mid-early (mixed) |
| L8-L10 | +6.8 to +21.7% | Second compensatory peak |
| L11 | -8.5% | Transition |
| L12 | **-33.5%** | Router destruction |
| L13 | -21.5% | Relay damage |
| L14 | **-73.5%** | Relay critical |
| L15 | **-85.0%** | Near-total destruction |
| L16 | **-100.0%** | Complete destruction |
| L17-L26 | 0.0% | Post-binding (no effect) |

### The Destruction Gradient

```
L12: -33.5% → L13: -21.5% → L14: -73.5% → L15: -85.0% → L16: -100.0%
```

Each relay station closer to the binding layer is more critical. L16 — the layer immediately before L17 — is the true bottleneck. Ablating it destroys ALL identity binding. Not 80%, not 95% — zero. The CV drops to exactly 0.000000.

L13 is actually LESS critical than L12 (-21.5% vs -33.5%), suggesting the relay re-routes around L13 but can't re-route around L14-L16.

### Phase Transition Replication

| Layer | 2 Names | 3 Names | 5 Names |
|-------|---------|---------|---------|
| L3 | +3.7% | +10.7% | +11.7% |
| L7 | **+10.2%** | -1.9% | -2.0% |
| L9 | +16.5% | **+20.8%** | +10.8% |
| L12 | -22.3% | -31.3% | -33.5% |
| L14 | -66.6% | -68.9% | -73.5% |

L9 shows compensatory binding peaking at 3 names (+20.8%), consistent with the seed layer role. L7 shows compensatory at 2 names (+10.2%) but not at 3 or 5 — the competition threshold flips L7 from cooperative to competitive as names increase.

## Three Findings

### 1. L16 Is the Single Point of Failure

No other layer causes complete binding destruction. L12 causes -33.5%, L14 causes -73.5%, but L16 causes -100.0%. The relay architecture has a single bottleneck at L16 — every identity signal must pass through L16 to reach L17.

This suggests L16 performs a final integration step: combining the relay signal with the embedding residual and the IT direct pathway into the format that L17 reads. Destroying L16 destroys all three pathways simultaneously.

### 2. The Relay Has Two Sub-Circuits

L12→L13 and L14→L15→L16 behave differently:
- L13 (-21.5%) is less critical than L12 (-33.5%) — the signal can partially bypass L13
- L14→L15→L16 show a strict monotonic gradient — no bypass possible

The relay isn't a simple chain. It has a parallel segment (L12-L13, where L13 can be routed around) and a serial segment (L14-L16, where every station is essential).

### 3. Post-L17 Layers Don't Affect L17

Ablating any layer L17-L26 produces exactly 0.0% change at L17. This confirms L17 identity binding is purely a function of the relay input, not of any downstream feedback. The binding is a one-way signal: relay → L17 → behavior, with no backpropagation of identity information to earlier layers during inference.

## Updated Architecture

```
EMBEDDING (L0): +91% compensatory (anomalous — needs investigation)
EARLY COMPENSATORY (L1-L3): +10-12% — cooperative
MID-EARLY (L4-L7): mixed — L7 compensatory at 2 names, neutral at 3+
SECOND COMPENSATORY (L8-L10): +7-22% — L9 seed peaks at 3 names
TRANSITION (L11): -8.5%
ROUTER (L12): -33.5% — critical but bypassable to L14
PARALLEL RELAY (L13): -21.5% — can be routed around
SERIAL RELAY (L14→L15→L16): -73.5% → -85.0% → -100.0% — no bypass
BINDING (L17): reads from L16, produces CV
POST-BINDING (L18-L26): no effect on L17
```

## Data

Full sweep and phase replication: `results/cna_phase_replication.json`
