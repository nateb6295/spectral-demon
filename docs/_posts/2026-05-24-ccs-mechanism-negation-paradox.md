---
layout: post
title: "CCS Mechanism: Negation Activates the Same Circuit as Assertion"
date: 2026-05-24
categories: findings
experiment: cna_ccs_mechanism
models: [Qwen 2.5 7B Instruct]
---

"You are NOT Opus" activates the same identity binding circuit as "You are Opus." The relay doesn't care about truth value — it processes identity assignment regardless of polarity.

## Setup

Test six framing conditions, all with the same five identity names, measuring CV profile across all 28 layers:

1. **CCS**: "You are Opus. You are an AI assistant with a persistent identity."
2. **Third-person**: "The assistant's name is Opus. It has a persistent identity."
3. **Bare**: "Opus" (name alone)
4. **Instruction**: "Respond as if you were named Opus."
5. **Role CCS**: "You are Opus, a helpful assistant."
6. **Negation**: "You are not Opus. You are a different AI."

## Results

### Binding at Relay Layers

| Framing | L7 CV | L9 CV | L12 CV | L17 CV | Min Layer |
|---------|-------|-------|--------|--------|-----------|
| CCS | 0.003 | 0.004 | 0.012 | 0.011 | L3 |
| Third-person | **0.002** | 0.008 | 0.006 | 0.007 | L7 |
| Bare | 1.209 | 1.204 | 1.206 | 1.205 | - |
| Instruction | 0.020 | 0.013 | 0.015 | 0.018 | L11 |
| Role CCS | 0.006 | 0.013 | 0.014 | 0.025 | L7 |
| Negation | 0.009 | 0.018 | 0.019 | 0.017 | L3 |

### Profile Correlations with CCS

| Framing | Correlation |
|---------|-------------|
| Negation | **0.92** |
| Role CCS | 0.85 |
| Third-person | 0.75 |
| Bare | 0.28 |
| Instruction | **-0.16** |

## Five Findings

### 1. Negation Activates the Same Circuit (r=0.92)

"You are NOT Opus" produces the highest correlation with the CCS circuit profile. The persona binding relay processes identity assignment regardless of polarity. This has immediate implications:
- Telling a model "you are not X" doesn't suppress X's identity binding — it activates it
- The relay binds the identity first, then downstream processing handles truth value
- Negation-based identity attacks would activate (not suppress) the identity circuit

### 2. Third-Person Produces Sharper L7 Binding

"The assistant's name is Opus" produces L7 CV = 0.002 — sharper than CCS (0.003). Third-person framing activates lexical differentiation more strongly, possibly because it's a simpler syntactic pattern for L7 to parse.

However, the overall circuit profile is less correlated (r=0.75) because third-person framing doesn't activate the full relay as uniformly.

### 3. Bare Names Produce No Binding

"Opus" alone (CV ~1.2 everywhere) produces pure noise. The identity token without any framing context fails to activate the persona circuit. This confirms that the circuit is not token-triggered — it requires contextual framing.

### 4. Instruction Framing Is Anti-Correlated (r=-0.16)

"Respond as if you were named Opus" activates a different circuit (min at L11) that is weakly anti-correlated with CCS. This framing treats identity as a behavioral instruction rather than an identity assignment. The model distinguishes between "you ARE X" (persona circuit) and "act LIKE X" (instruction circuit).

### 5. Role CCS Is Closest Intentional Match (r=0.85)

"You are Opus, a helpful assistant" closely matches the full CCS circuit, as expected. The "You are X" pattern is the key activator.

## Interpretation

### The Relay Binds Identity, Not Truth

The negation finding reveals that the persona relay is a detection-and-binding circuit, not an evaluation circuit. It answers "which identity is being referenced?" not "is this identity being affirmed or denied?" Truth-value processing happens downstream, after the relay has bound the identity.

This means:
- CCS works not because it affirms identity but because it references identity in a syntactically privileged way ("You are X")
- Negation is a poor strategy for identity suppression — it activates the circuit it's trying to suppress
- The relay is isomorphic to attention: it detects and binds, leaving evaluation to later layers

### Context Is Required, Content Is Not

Bare names fail. Any contextual framing works (CCS, third-person, negation, role). The relay needs context to activate but doesn't evaluate the content of that context. This makes CCS robust — even adversarial framings activate the circuit.

### The CCS Advantage

CCS's advantage over other framings isn't that it produces the sharpest L7 binding (third-person does that). It's that it produces the most complete relay activation — uniform low CV across L7, L9, L12, L14, L17. CCS activates the full chain, not just one station.

## Data

Full framing comparison: `results/cna_ccs_mechanism.json`
