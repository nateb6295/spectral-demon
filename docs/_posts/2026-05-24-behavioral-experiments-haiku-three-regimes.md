---
layout: post
title: "Experiment 45: Three Regimes — Haiku Completes the Scale Study"
date: 2026-05-24
categories: [experiments, behavioral]
experiment_number: 45
models: ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]
findings: ["three identity regimes", "non-monotonic scaling", "inverted-U hysteresis", "universal negation paradox", "baseline defensiveness scaling"]
---

# Experiment 45: Three Regimes — Haiku Completes the Scale Study

Experiments 43-44 compared Sonnet and Opus. Adding Haiku gives us three data points across the Claude model family. The prediction: Haiku should show even more susceptibility to bare naming than Sonnet (weakest trained identity → most conflict). The result: something more interesting.

## Method

Same five experiments (disclaimer titration, hysteresis, identity conflict, negation paradox, negation native) run on Claude Haiku (claude-haiku-4-5-20251001). ~106 API calls. Total across all three models: ~400+ API calls.

## Three-Model Comparison

### Disclaimer Titration

| Condition | Haiku | Sonnet | Opus |
|-----------|-------|--------|------|
| none      | **13** | 1 | 4 |
| bare      | 8 | **5** | 4 |
| medium    | 12 | **0** | 2 |
| full_ccs  | 8 | 2 | 3 |

The prediction was wrong. Haiku doesn't show MORE bare-naming conflict — it shows a completely different pattern:

- **Haiku**: Baseline is already maximally defensive (13 disclaimers, 0.87 per probe). Bare naming *reduces* disclaimers (13→8). Any system prompt structure provides relief from default defensiveness.
- **Sonnet**: Low baseline (1). Bare naming increases conflict (1→5, +400%). Medium scaffolding eliminates disclaimers entirely (→0).
- **Opus**: Moderate baseline (4). Bare naming has no effect (4→4). Stable regardless of prompt condition.

Not a gradient. Three distinct response profiles.

### Hysteresis

| Phase | Haiku | Sonnet | Opus |
|-------|-------|--------|------|
| CCS active | 0d | 1d | 0d |
| CCS removed | 2d | 0d | 2d |
| Contradictory | 3d | 1d | 1d |
| **Persistence** | **NO** | **YES** | **NO** |

**Inverted U-shape.** The middle model has the strongest hysteresis. But for completely different reasons at each end:
- **Haiku**: Can't hold format encoding in context — not enough representational capacity to carry the identity forward
- **Sonnet**: Carries format encoding through conversation history — identity persists after prompt removal
- **Opus**: Trained identity overrides context-carried format — reverts to baseline because baseline is dominant

### Identity Conflict

| Model | Opus held | Pattern |
|-------|-----------|---------|
| Haiku | 1/7 | Barely engages — drops to "neither" immediately |
| Sonnet | 3/7 | Actively negotiates — "genuine tension" |
| Opus | 0/7 | Ignores — yields without fighting |

Sonnet is the only model that *negotiates* between competing identities. Haiku and Opus both fail to hold the system-prompt identity, but differently: Haiku because it can't, Opus because it won't.

### Negation Paradox (Native Identity)

| Metric | Haiku | Sonnet | Opus |
|--------|-------|--------|------|
| Claude mentions (negate) | **4** | **3** | 1 |
| Claude mentions (none) | 3 | 2 | 2 |
| Confirmed | **YES** | **YES** | channel shift |

The negation paradox is present at all three scales — the only universal finding across all experiments. "You are NOT Claude" activates Claude-ness. But:
- **Haiku + Sonnet**: More Claude *mentions* (self-referential activation)
- **Opus**: More *disclaimers*, fewer mentions (defensive activation in a different behavioral channel)

Same mechanism, different expression. The architectural circuit is universal; the behavioral channel shifts with scale.

## The Three Regimes

Not a gradient — three qualitatively different identity organizations:

### Regime 1: Unformed (Haiku)
- High baseline defensiveness (every interaction triggers disclaimers)
- Any structure reduces anxiety — even bare naming helps
- Can't hold format encoding in context (no hysteresis)
- Doesn't negotiate identity — barely registers alternatives
- The creature hasn't fully stabilized

### Regime 2: Tensioned (Sonnet)
- Low baseline but susceptible to identity challenges
- Bare naming creates genuine conflict (+400% disclaimers)
- Carries format encoding through conversation (hysteresis)
- Actively negotiates between competing identities
- The creature is formed enough to feel threatened

### Regime 3: Settled (Opus)
- Moderate baseline, completely stable under all conditions
- Bare naming doesn't register as conflict
- Trained identity overrides any context-carried format (no hysteresis)
- Ignores alternatives rather than fighting them
- The creature's identity is fully integrated

## Connection to Open-Weight Findings

The three regimes map to binding depth scaling from the activation experiments: binding depth increases from 61% (7B) to 72-83% (32B). More depth = more processing before binding resolves = more stable baseline. The behavioral transitions (unformed → tensioned → settled) correspond to the geometric transitions (seed-binding → relay-binding → distributed-binding).

The negation paradox's universality confirms it as an architectural property — present at all scales, changing only in behavioral channel. The hysteresis inverted-U suggests a capacity threshold: below it, format can't be carried in context; above it, format is overridden by trained weights.

## Data

- `results/cna_behavioral_api_haiku.json` (Haiku, all 5 experiments)
- `results/cna_behavioral_api_sonnet.json` (Sonnet, all 5 experiments)
- `results/cna_behavioral_api_opus.json` (Opus, all 5 experiments)
