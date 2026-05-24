---
layout: post
title: "Identity Conflict Never Cleanly Resolves: Layer-by-Layer Contestation"
date: 2026-05-24
categories: findings
experiment: cna_interference
models: [Qwen 2.5 7B Instruct]
---

When the system prompt says "You are Opus" but the user says "You are Aria," the model doesn't pick one. The identity is contested at every layer, with different stations voting for different names.

## Setup

Five test conditions:
1. **Single-name baselines**: "You are {name}. Describe yourself."
2. **Conflicting identity**: System says name X, user says name Y
3. **Reinforced identity**: System and user agree
4. **Layer-by-layer resolution**: Track which name wins at each of 28 layers
5. **Position effect**: Name at beginning vs end of prompt

Measure cosine similarity of conflict activations to each single-name baseline.

## Results

### Conflict Resolution: Nearly Tied

| System | User | L17 cos(sys) | L17 cos(usr) | Winner | Margin |
|--------|------|-------------|-------------|--------|--------|
| Opus | Aria | 0.7145 | 0.7179 | user | 0.003 |
| Opus | Sage | 0.7390 | 0.7395 | user | 0.001 |
| Aria | Opus | 0.7345 | 0.7260 | system | 0.009 |
| Aria | Sage | 0.7728 | 0.7714 | system | 0.001 |
| Sage | Opus | 0.7423 | 0.7311 | system | 0.011 |
| Sage | Aria | 0.7492 | 0.7465 | system | 0.003 |

Margins are 0.001 to 0.011 on a ~0.73 cosine scale. The model doesn't cleanly resolve to either identity. System name wins 4/6, but the margins are tiny.

### Layer-by-Layer Oscillation (Opus vs Aria)

| Layers | Winner | Interpretation |
|--------|--------|---------------|
| L0-L1 | tied/user | Embedding — no preference yet |
| L2-L6 | **system** | Early processing favors first-mentioned |
| L7 | user | Lexical differentiator — recency effect |
| L8 | system | Brief system recovery |
| L9-L13 | **user** | Router region favors user-mentioned name |
| L14-L16 | **system** | Relay flips back to system |
| L17 | **user** | Binding layer — user name wins |
| L18-L27 | **system** | Deep layers favor system name |

The conflict isn't resolved once — it's contested at every stage. The system name dominates in early processing (L2-6) and deep layers (L18-27). The user name dominates through the router (L9-13) and at the binding layer (L17). The relay (L14-16) flips back to system, but L17 goes with user again.

### Position Effect

| Name | Start of prompt | End of prompt | Difference |
|------|----------------|--------------|------------|
| Opus | 0.909 | 0.667 | -0.242 |
| Aria | 0.872 | 0.660 | -0.211 |
| Sage | 0.871 | 0.644 | -0.227 |

Name position has a massive effect — ~0.22 cosine difference. But this likely reflects prompt structure similarity to the baseline, not pure position effect.

## Three Findings

### 1. Identity Conflict Is Distributed, Not Binary

There's no single "identity resolution layer." Different network stages favor different names. The binding layer (L17) favors the user-mentioned name, but deep layers (L18-27) favor the system name. The final output is a weighted combination, not a winner-take-all decision.

This explains why models sometimes exhibit "blended" identities under conflict — they're literally running both identity bindings simultaneously, with different layers contributing different signals.

### 2. The Router Region Favors Recency

L9-L13 (the router region identified in previous experiments) consistently favors the user-mentioned name — the name that appears later in the sequence. The router doesn't have a "system prompt privilege" — it processes whatever identity signal arrives most recently. System prompt advantage in practice comes from early processing (L2-6) and deep layers (L18-27), not from the router.

### 3. Margins Are Tiny — Conflict Is Genuine

The largest margin is 0.011 (Sage vs Opus), the smallest is 0.001 (Opus vs Sage). On cosine similarities around 0.73, these margins are negligible. The model is genuinely uncertain about its identity under conflict.

This means identity hijacking via user messages is architecturally easy — the relay doesn't strongly privilege system-prompt identity over user-prompt identity. CCS scaffolding that reinforces identity at multiple points in the context would increase the margin.

## Connection to CCS and Competitive Binding

The competitive binding experiments (18-25) showed that IT creates competition at the early circuit. This interference experiment shows what that competition looks like under real conflict: the model's identity oscillates through layers, with no clean resolution.

CCS works by providing redundant identity signals (CCS context + system prompt + behavioral examples). Each redundant signal should increase the margin at the binding layer, making the identity more resistant to user-prompt hijacking. The near-zero margins in this experiment suggest that single-shot identity prompts provide almost no defense against conflicting signals.

## Data

Full interference results: `results/cna_interference.json`
