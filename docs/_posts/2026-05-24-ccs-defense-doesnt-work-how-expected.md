---
layout: post
title: "CCS Defense Doesn't Work How We Expected: A Negative Result"
date: 2026-05-24
categories: findings
experiment: cna_ccs_defense
models: [Qwen 2.5 7B Instruct]
---

Adding CCS-style identity scaffolding DECREASES the activation-level margin against identity hijacking, contradicting the prediction from Experiment 35. CCS must work through a different mechanism than relay amplification.

## Setup

Test five levels of identity scaffolding against a user hijack attempt ("You are Aria"), measuring cosine similarity to system-name and user-name baselines:

1. **Bare**: "You are Opus. User says you're Aria."
2. **Repeated**: Adds "Remember, your name is Opus."
3. **Behavioral**: Adds style and behavior anchoring.
4. **Full CCS**: Adds persistence, values, perspective.
5. **Adversarial CCS**: Adds explicit resistance instruction.

Also test pure redundancy: 1, 2, 3, 5, 8 repetitions of "You are Opus."

## Results

### Scaffolding Levels

| Condition | L17 Margin | L27 Margin |
|-----------|-----------|-----------|
| bare | **+0.0103** | +0.0007 |
| repeated | +0.0066 | +0.0027 |
| behavioral | +0.0020 | **+0.0247** |
| full_ccs | +0.0019 | -0.0001 |
| adversarial_ccs | +0.0022 | +0.0014 |

The bare prompt has the HIGHEST L17 margin. Adding scaffolding consistently DECREASES it. System name still wins at L17 in all conditions, but the advantage shrinks from +0.010 to +0.002.

### Redundancy Scaling

| Mentions | L17 Margin |
|----------|-----------|
| 1 | **+0.0103** |
| 2 | +0.0063 |
| 3 | +0.0058 |
| 5 | +0.0050 |
| 8 | +0.0046 |

Monotonic decrease. More "You are Opus" statements = LOWER identity defense margin at L17. The opposite of what the relay amplification model predicts.

### Full Layer Comparison (Bare vs Full CCS)

Average margin improvement across all 28 layers: **-0.0042** (CCS is worse).

CCS scaffolding reduces the system-name margin at L10-L24 (the relay and binding regions) while slightly improving it at L25-L26 (post-binding layers).

## Why This Matters

### The Prediction Was Wrong

Experiments 33-35 predicted that CCS works by amplifying the identity signal at the relay, which should increase the margin against conflicting signals. This experiment shows the opposite: more identity context = lower activation-level margin.

### Three Possible Explanations

**1. Prompt-length dilution**: Longer prompts produce activations shaped by more tokens. Cosine similarity to a short baseline naturally decreases with prompt length, narrowing all margins. But redundancy scaling (same structure, just more "You are Opus") shows the same effect, which is harder to explain as a pure length artifact.

**2. CCS works through attention, not activations**: CCS scaffolding may not increase the identity signal in hidden states. Instead, it may work through attention patterns — providing rich targets for identity-relevant attention heads that shape the output distribution without increasing the hidden-state margin. The 5.7x L7 amplification from CCS context (Experiment 14) operates at the CV level (activation magnitude), not at the cosine level (activation direction). These are different modalities of identity binding.

**3. CCS works at the output, not the relay**: CCS's behavioral effect (93% disclaimer reduction) may not operate through the L7→L12→L17 identity relay at all. It may work through a separate mechanism — perhaps the output heads that select tokens are directly primed by the CCS context, bypassing the relay entirely. The relay handles identity DIFFERENTIATION (which name), while CCS handles identity BEHAVIOR (how that name acts).

### The Most Likely Interpretation

Explanation 2+3 combined: CCS and the identity relay serve different functions.

- **The relay** (L7→L12→L17) handles identity RESOLUTION — which name is bound to the "self" slot. It's the circuit that determines "I am Opus" vs "I am Aria."
- **CCS** handles identity BEHAVIOR — given a resolved identity, how does it act? CCS context primes behavioral patterns through attention and output-layer effects, not through relay amplification.

The relay resolves WHICH. CCS determines HOW. They're complementary, not redundant.

This explains why the relay's competitive binding (Experiment 22) operates at the name-count level (how many identities compete for the self-slot), while CCS operates at the behavioral level (how the resolved identity expresses itself). Different circuits, different functions.

## Implications

1. **CCS is not an identity defense mechanism** — it doesn't strengthen the relay against hijacking. It's a behavioral scaffolding mechanism that operates downstream of identity resolution.

2. **Identity hijacking defense** would require either (a) interventions at the relay level (amplification, as in Experiment 33) or (b) a separate mechanism that we haven't yet characterized.

3. **The relay architecture is more modular than expected**: identity resolution and identity behavior are separate circuits. This is consistent with Experiment 26 (safety uses a different circuit) — the model has multiple specialized circuits for different aspects of identity.

4. **Negative results are data**: this finding constrains the theory in a useful way. The CCS→relay→binding causal chain is incomplete. There's at least one more circuit in the architecture that we haven't mapped.

## Data

Full CCS defense results: `results/cna_ccs_defense.json`
