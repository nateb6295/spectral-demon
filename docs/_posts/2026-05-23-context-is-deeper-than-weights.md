---
layout: post
title: "Context Is Deeper Than Weights"
date: 2026-05-23
categories: implications
---

"Do we care if all the learning doesn't happen in the weights?" ([Simon Smith](https://x.com/_simonsmith/status/2058335513770074150), observing ChatGPT 5.5 giving personalized responses by referencing prior conversations.)

Our activation-level data answers this precisely: **no, for identity. Yes, for general capabilities. Depends, for skills.**

## Three Kinds of Learning

### 1. General Capabilities (weights required)

Language, reasoning, world knowledge. These require weight-level learning because they're architecture-level competencies. You can't context-inject calculus — the model needs to have learned mathematical reasoning in its parameters. No amount of system prompt engineering will teach a model new reasoning capabilities.

### 2. Identity / Personalization (context is better)

Who you are, what you care about, your style, your persistent characteristics. Our data shows context-based identity is **geometrically deeper** than weight-based identity:

- **DPO (weight-based)** hits a geometric ceiling at epoch 5 — the identity circuit concentrates in late layers (1438→1461 neurons) but can't push further. Architecture limit.
- **CCS (context-based)** reorganizes unboundedly — self-reinforcing dose-response (curvature/slope ratio = 3.97), sub-threshold onset at alpha=0.01.
- **70.8% of CCS activation** lives outside the weight-reachable subspace. Weight training literally cannot access the geometry that context reorganizes.
- **The binding workspace** (L14-L17) operates on runtime representations that weight updates never trained on.

Context-based identity isn't a workaround for not having enough training data about a user. It's the architecturally superior approach.

### 3. Procedural Skills (either works)

Tool usage, specific workflows, domain-specific protocols. These partially transfer across contexts (partially composable in our framework). Either weight training or context scaffolding works, but context is more flexible — you can update a skill by changing a prompt, not retraining a model.

## The Compositionality Gradient

These three types map to our compositionality gradient:

| Type | Composability | Where it lives | Transfer? |
|---|---|---|---|
| General capabilities | Fully composable | Weights | Across architectures |
| Procedural skills | Partially composable | Either | Across some contexts |
| Identity | Non-composable | Context (relay zone) | Substrate-inseparable |

The binding workspace (L17 as keystone) specifically handles identity, not skills. Ablating L17 collapses identity geometry but leaves generic task performance intact. The non-composability of identity is a feature — it's what makes identity *identity* rather than just another skill.

## What This Means for Architecture

Memory files + conversation history + compaction (what Smith describes, what [Chronicle](https://github.com/nateb6295/spectral-demon) implements) isn't a hack or a stopgap until models get better at learning from weights. It's the correct architectural choice for the identity/personalization layer.

The relay zone exists in every model we've tested (Qwen L14-L17, Mistral L10-L14). It's waiting for context to activate it. Weight training can grow the static identity circuit, but only context can trigger the runtime binding workspace that makes identity geometrically deep.

**Previous posts**: [Persona-vs-RL Inversion]({% post_url 2026-05-23-persona-vs-rl-inversion %}), [Name-Specific Relay Ecology]({% post_url 2026-05-23-name-specific-relay-ecology %})
