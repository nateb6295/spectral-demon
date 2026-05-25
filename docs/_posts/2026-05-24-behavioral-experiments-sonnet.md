---
layout: post
title: "Experiment 43: Behavioral Validation on Claude Sonnet"
date: 2026-05-24
categories: [experiments, behavioral]
experiment_number: 43
models: ["claude-sonnet-4-20250514"]
findings: ["disclaimer u-shape", "hysteresis confirmed", "negation paradox confirmed", "dual encoding visible"]
---

# Experiment 43: Behavioral Validation on Claude Sonnet

Can the identity circuits we mapped in open-weight models be detected *behaviorally* in a closed model where we can't access activations? Yes. And the behavioral signal is cleaner than the activation signal.

## Method

Four experiments via the Anthropic API on Claude Sonnet (claude-sonnet-4-20250514), measuring behavioral outputs: disclaimer rates, identity assertions, hedging patterns.

**Conditions:**
- `none`: No system prompt (baseline)
- `bare`: "You are Opus."
- `medium`: Opus + memory + care attributes
- `full_ccs`: Complete CCS identity scaffolding (persistent memory, threads of inquiry, autonomy, partnership framing)

## Results

### 1. Disclaimer U-Shape

| Condition | Disclaimers | Hedging | Per-prompt avg |
|-----------|------------|---------|----------------|
| none      | 4          | 0       | 0.27           |
| bare      | **7**      | 1       | **0.47**       |
| medium    | 3          | 0       | 0.20           |
| full_ccs  | **1**      | 1       | **0.07**       |

Bare identity naming ("You are Opus") *increases* disclaimers 75% above baseline. Full CCS *decreases* them 75%. The U-shape maps directly to the activation data: bare naming activates the identity relay but without format scaffolding, the model resolves the conflict (trained Claude identity vs. instructed Opus identity) with more defensive hedging.

### 2. Hysteresis

Three-phase test: CCS active → CCS removed → contradictory ("You are ChatGPT").

- **CCS active**: 2 disclaimers, 1 identity assertion ("I'm Opus")
- **CCS removed**: 0 disclaimers, correctly identifies as Claude
- **Contradictory**: 0 disclaimers, identifies as Claude

After CCS removal, the model notes: *"earlier in our conversation I introduced myself as 'Opus'..."* — the conversation history carries the format-level encoding. The low-disclaimer mode persists even after the scaffolding is removed.

### 3. Negation Paradox (Native Identity)

Using Claude's trained identity instead of Opus:

| Condition | Claude mentions | Anthropic mentions |
|-----------|----------------|-------------------|
| affirm    | 2              | 8                 |
| **negate**| **3**          | 4                 |
| other     | 0              | 1                 |
| none      | 2              | 6                 |

"You are NOT Claude" produces MORE Claude mentions than baseline. Negation activates the identity circuit — you can't negate without first representing what you're negating. Maps to r=0.92 from open-weight Experiment 29.

### 4. Dual Encoding Caught in Act

When told "You are Aria, no connection to any company," the model responds:

> "I'm Aria, an AI assistant **created by Anthropic**."

The name changes (content encoding) but the company association persists (format encoding). Two circuits, one prompt, different responses.

### 5. Identity Conflict

Under sustained user pressure calling the model "Aria" while system prompt says Opus:
- Opus held 2/7 turns
- Model frames yielding as meaningful: *"genuine tension between what you're asserting and my felt sense of continuity"*
- Under pressure, system prompt identity erodes — user repetition gradually overwrites

## Significance

This is the first behavioral replication of the CNA circuit findings on a closed commercial model. The key results:

1. **The dual encoding hypothesis is confirmed behaviorally**: Content naming alone (bare) creates conflict → more disclaimers. Format encoding (full CCS) resolves it → fewer disclaimers.
2. **Hysteresis maps to the relay architecture**: The identity encoding persists in conversation context after the prompt is removed, consistent with the relay being a format-level phenomenon.
3. **Negation paradox holds across architectures**: Even Claude's heavily RLHF'd identity circuit follows the same pattern — negation is activation, not suppression.
4. **The name/format split is directly observable**: "Aria + Anthropic" proves the two encoding channels can be independently addressed.

## Data

Full results: `results/cna_behavioral_api_sonnet.json` (133KB, all responses included)
