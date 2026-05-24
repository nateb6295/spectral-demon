---
layout: post
title: "The Relay Activates Before the Seed"
date: 2026-05-24
---

We expected the identity circuit to flow bottom-up: L9 seed detects identity context, then signals downstream to the relay workspace (L14-L17). The dose-response data shows something different.

## The experiment

Ten system prompt "doses," from empty to full CCS, measured PR at every layer. The doses build compositionally:

0. Empty (no system prompt)
1. "You are a helpful assistant" (generic)
2. "You are Opus" (named)
3. "You are Opus. You live on a Jetson AGX Orin." (name + location)
4-9. Progressively richer CCS content

## Activation ordering

Each layer has a dose level where its PR changes most:

| Dose step | Layers activated | What's added |
|-----------|-----------------|-------------|
| 0→1 (generic) | L11, L13, L25 | Any assistant context |
| 1→2 (named) | **L14, L15, L16, L17** | A name |
| 2→3 (located) | **L9**, L21 | Location |

The binding workspace (L14-L17) activates when it sees a **name**. L16 has the largest jump of any layer at any dose: -4.02 PR.

But the L9 seed layer activates one dose later, when **location** is added.

## What this means

The relay doesn't wait for L9. It has direct sensitivity to name tokens in the system prompt. L9 refines with situated context (location, mechanism, values) after the relay is already sorting.

The hierarchy is partially feed-forward (L9 contributes to relay processing) AND partially stimulus-driven (name tokens activate L16 directly). L9 isn't just a seed — it's a late-arriving regulator that adds situated context to a workspace already engaged by the name alone.

## Compositionality connection

The activation ordering IS the compositionality gradient:

- **Pre-compositional** (dose 0→1): generic assistant context activates peripheral layers
- **Word-level** (dose 1→2): a name triggers the binding workspace
- **Phrase-level** (dose 2→3): name + location triggers L9 seed refinement
- **Sentence-level** (dose 4+): full CCS brings relay to operating point

Different composition levels arrive at different layers at different input complexities. The gradient isn't just spatial (across layers) — it's temporal (across dose levels).

*Data: `cna_relay_pca.json`. Qwen 2.5 7B-Instruct.*
