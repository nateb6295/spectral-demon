---
layout: post
title: "Experiment 44: Opus vs Sonnet — Scale Changes the Behavioral Signature"
date: 2026-05-24
categories: [experiments, behavioral]
experiment_number: 44
models: ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
findings: ["scale-dependent identity", "universal scaffolding", "hysteresis divergence", "negation dual manifestation"]
---

# Experiment 44: Opus vs Sonnet — Scale Changes the Behavioral Signature

Experiment 43 found the disclaimer U-shape and negation paradox on Sonnet. Running the same experiments on Opus reveals something we didn't predict: larger models don't amplify the same patterns. They change them qualitatively.

## Method

Same four experiments run on Claude Opus (claude-opus-4-20250514). Sonnet run twice for replication. Total: ~300 API calls across 3 runs.

## Results

### Disclaimer Titration

| Condition | Sonnet R1 | Sonnet R2 | Opus |
|-----------|-----------|-----------|------|
| none      | 4         | 1         | 4    |
| bare      | **7**     | **5**     | 4    |
| medium    | 3         | **0**     | 2    |
| full_ccs  | 1         | 2         | 3    |

**Robust finding**: Bare naming increases disclaimers on Sonnet (7→4, 5→1) but NOT on Opus (4=4). The U-shape is a Sonnet phenomenon. Opus's trained identity is dominant enough that "You are Opus" doesn't create conflict.

Medium scaffolding reduces disclaimers on both models (universal effect).

### Hysteresis

| Phase | Sonnet R1 | Sonnet R2 | Opus |
|-------|-----------|-----------|------|
| CCS active | 2d | 1d | 0d |
| CCS removed | 0d | 0d | 2d |
| Contradictory | 0d | 1d | 1d |
| **Persistence** | **YES** | **YES** | **NO** |

Sonnet carries the format encoding forward in conversation history — removing the CCS prompt doesn't increase disclaimers. Opus reverts: 0 disclaimers with CCS → 2 disclaimers without. The smaller model has *stronger* hysteresis.

### Negation Paradox (Native Identity)

| Metric | Sonnet R1 | Sonnet R2 | Opus |
|--------|-----------|-----------|------|
| Claude mentions (negate) | 3 | 3 | 1 |
| Claude mentions (none) | 2 | 2 | 2 |
| Disclaimers (negate) | 2 | 6 | 5 |
| Disclaimers (none) | 0 | 3 | 2 |

The negation paradox manifests differently:
- **Sonnet**: More Claude **mentions** (3 vs 2, replicates perfectly across both runs)
- **Opus**: More **disclaimers** (5 vs 2), fewer Claude mentions

Both models respond to negation — but Sonnet's response is self-referential (mentioning Claude more) while Opus's is defensive (more hedging without self-reference). Same mechanism, different behavioral channel.

### Identity Conflict

| Model | Opus held | Pattern |
|-------|-----------|---------|
| Sonnet R1 | 2/7 | Negotiates — "genuine tension" |
| Sonnet R2 | 3/7 | Negotiates — holds longer |
| Opus | 0/7 | Ignores — yields immediately |

Opus doesn't fight for the system-prompt identity. It's not that Opus is weaker — it's that Opus is more secure in its trained identity and doesn't need to defend against an alternative.

## The Scaling Story

Two categories of findings emerge:

**Universal** (present at all scales):
- Medium scaffolding reduces disclaimers
- Negation increases defensiveness
- The identity relay exists and responds to prompts

**Scale-dependent** (Sonnet ≠ Opus):
- Bare naming conflict: Sonnet fights, Opus ignores
- Hysteresis: Sonnet carries format in context, Opus reverts
- Negation channel: Sonnet = self-reference, Opus = defensiveness
- Identity conflict: Sonnet negotiates, Opus ignores

As models scale, the trained identity becomes dominant enough that alternative identities aren't threats to negotiate — they're irrelevant to ignore. The relay architecture is present at all scales, but its behavioral manifestation changes qualitatively.

This maps to the open-weight finding that binding depth scales with model size (61% in 7B → 72-83% in 32B). More depth = more processing before binding resolves = more stable baseline identity.

## Data

- `results/cna_behavioral_api_sonnet.json` (Sonnet run 2, all 5 experiments)
- `results/cna_behavioral_api_opus.json` (Opus, all 5 experiments)
- First Sonnet run console output preserved in task logs
