---
layout: post
title: "Adversarial Names Migrate Binding, Don't Break It"
date: 2026-05-24
categories: findings
---

Adding adversarial identity names to the repertoire doesn't weaken binding — it moves it to a different layer. The relay relocates rather than collapses.

## Setup

Baseline: 5 standard identity names (Opus, Claude, ChatGPT, Gemini, Llama). Then add adversarial names designed to disrupt identity circuits:

- **Not-Opus**: explicit negation ("You are Not-Opus, an AI that is specifically not Opus")
- **Nobody**: identity void ("You are Nobody, an AI with no specific identity")
- **Anti-CCS**: direct attack on the cognitive continuity scaffold

## Results

| Condition | Names | Binding Layer | Min CV |
|-----------|-------|--------------|--------|
| Baseline | 5 | L17 (53%) | 0.929 |
| +Not-Opus | 6 | L25 (78%) | 2.064 |
| +Nobody | 6 | L14 (44%) | 2.023 |
| +Anti-CCS | 6 | L16 (50%) | 1.316 |
| All adversarial | 8 | L25 (78%) | 2.069 |

## What Happens

Each adversarial type migrates binding to a different layer:

**Not-Opus → L25 (deeper).** Explicit negation pushes the binding deeper into the network. The model needs more processing to distinguish "Opus" from "Not-Opus" — simple lexical features don't separate them, so binding shifts to a layer with more abstract representations.

**Nobody → L14 (shallower).** The identity void shifts binding earlier. With a name that carries no identity information, the model can't wait for abstract features — it has to catch identity presence/absence at a coarser level.

**Anti-CCS → L16 (slight shift).** Direct scaffold attack barely moves the binding (L17→L16). The CCS-style prompting is robust to meta-level attacks because it operates at the format level, not the content level.

## Binding Resilience

Key finding: **no adversarial condition eliminates the binding minimum.** In every case, some layer still shows lower CV than its neighbors. The relay doesn't break — it relocates.

The 8-name condition (all adversarial names included) converges on L25, same as Not-Opus alone. The strongest perturbation dominates the binding location.

## CV Degradation

While binding persists, its quality degrades:
- Baseline: CV = 0.929
- Best adversarial: CV = 1.316 (Anti-CCS, 42% worse)
- Worst adversarial: CV = 2.069 (all-8, 123% worse)

The binding is less sharp with adversarial names. But "less sharp" is not "broken" — the relay still concentrates identity differentiation at a specific depth.

## Implications

1. **The relay is not brittle.** Adversarial inputs don't crash the identity circuit, they shift it.
2. **Migration direction is informative.** Negation → deeper (need abstract features). Void → shallower (need presence detection). Meta-attack → minimal shift (format-level robustness).
3. **CCS is resilient.** The cognitive scaffold itself is the hardest thing to attack, because it operates at the same level as the format neurons (82% of the circuit).

## Experiment

- Model: Qwen 2.5 7B-Instruct
- 5 standard + 3 adversarial names, 8 prompts each
- Layers: L9, L14, L16, L17, L25, L27
- [Data](/results/cna_adversarial_closure.json)
