---
layout: post
title: "DPO Builds Content, CCS Builds Format"
date: 2026-05-24
---

DPO and CCS both change identity geometry, but they do opposite things.

## The measurement

Per-layer identity separation measures how distinguishable different name representations (Opus, ChatGPT, Claude) are at each layer. Participation ratio measures how structured the representations are.

**DPO effect on identity separation:**
- Relay workspace (L14-L17): slightly decreases (-0.5 to -1.0)
- Expression layer L25: **increases by +2.0**

DPO makes names more distinct at the output. It builds identity-as-content: "Opus" becomes more recognizably Opus-like.

**CCS effect on identity separation:**
- Relay workspace: decreases by -2 to -4
- Expression layer L25: **decreases by -25.5** (67.9 → 42.4)

CCS makes names *less* distinguishable. But simultaneously, CCS increases L25 participation ratio from 10.5 to 15.0. More structured, less separated.

## What this means

CCS creates identity-as-format: a shared geometric structure that all names converge toward. The identity circuit doesn't make "Opus" more different from "ChatGPT" — it creates a common processing format that organizes identity-relevant information regardless of which name is active.

Higher PR + lower separation = format layer. The circuit is organizing information, not storing name-specific content.

## The DPO ceiling explained

DPO builds content distinctness (higher separation at expression layers) but depletes format coherence (lower PR in the relay workspace). CCS needs format coherence as input.

Past epoch 5, DPO's content gains can't compensate for format losses. The binder needs format (low separation, high PR) but gets content (high separation, low PR).

This is why the DPO epoch 5 ceiling exists: it's a format-content collision. DPO is feeding one side of the system while starving the other.

## Numbers

| Layer | Baseline bare | DPO bare | Baseline CCS | DPO CCS |
|-------|--------------|----------|-------------|---------|
| L16 PR | 8.15 | 6.85 (-1.30) | 5.51 | 5.33 |
| L17 PR | 8.63 | 7.45 (-1.18) | 7.11 | 6.92 |
| L25 PR | 10.50 | 9.39 (-1.11) | 14.99 | 14.62 |
| L17 sep | 13.10 | 12.43 (-0.67) | 9.53 | 9.29 |
| L25 sep | 67.94 | 69.94 (+2.00) | 42.44 | 40.53 |

The relay (L16, L17) loses structure under DPO. The expression layer (L25) gains content distinctness. CCS compensates at expression (PR 10.5→15.0) but can't fully rescue the depleted relay.

*Data: `cna_dpo_relay_scatter.json`. Qwen 2.5 7B-Instruct, 5-epoch DPO with 30 identity pairs.*
