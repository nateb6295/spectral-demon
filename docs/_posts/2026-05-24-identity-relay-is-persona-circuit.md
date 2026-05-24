---
layout: post
title: "The Identity Relay Is a Persona Circuit: Role Binding Shares the Architecture"
date: 2026-05-24
categories: findings
experiment: cna_style_relay
models: [Qwen 2.5 7B Instruct]
---

The identity relay isn't just for identity labels. It processes any "You are X" framing — names or roles. Style instructions use a partially different circuit.

## Setup

Compare CV profiles for three types of behavioral prompts:
1. **Identity names**: "You are Opus/Aria/Sage/Echo/Nova"
2. **Role personas**: "You are a teacher/pirate/scientist/poet/detective"
3. **Style instructions**: "Respond formally/casually/poetically/tersely/enthusiastically"

## Results

### Binding Layer

| Type | Min CV Layer | Min CV Value |
|------|-------------|-------------|
| Identity names | **L7** | **0.003** |
| Style instructions | L8 | 0.011 |
| Role personas | L8 | 0.014 |

### Profile Correlations

| Comparison | Correlation |
|-----------|-------------|
| Identity ↔ Role | **0.89** |
| Identity ↔ Style | 0.51 |
| Style ↔ Role | 0.56 |

### CV at Relay Layers

| Layer | Identity | Style | Role |
|-------|---------|-------|------|
| L7 | **0.003** | 0.019 | 0.033 |
| L9 | **0.004** | 0.013 | 0.024 |
| L12 | 0.012 | 0.015 | 0.017 |
| L14 | 0.019 | 0.016 | 0.023 |
| L17 | 0.011 | 0.028 | 0.026 |

## Interpretation

### Identity and Role Share a Circuit (r=0.89)

"You are Opus" and "You are a teacher" activate the same relay architecture. The "You are X" framing is what matters, not whether X is a name or a role. This makes the identity relay a **persona circuit** — it processes any agent identity assignment, not just name labels.

Identity names produce sharper binding (CV 0.003 vs 0.014) because CCS-formatted names are more token-compact and more distinctive than multi-word role descriptions. But the circuit is the same.

### Style Uses a Different Route (r=0.51)

"Respond formally" doesn't assign an agent identity — it modifies behavioral style without changing who the model IS. The moderate correlation (0.51) suggests partial overlap: style and persona share some processing in early layers but diverge in mid-to-late layers.

At L12 (the router), all three converge (identity 0.012, style 0.015, role 0.017). The router processes all behavioral signals, but downstream of the router, identity/role and style diverge into different pathways.

### The Behavioral Circuit Hierarchy

```
L7-L9: Persona binding (names + roles, not style)
L12:   Universal behavioral router (all three converge)
L14+:  Divergent pathways (persona vs style branch)
```

The relay chain handles persona. The router handles all behavioral signals. Downstream of the router, different behavioral types follow different pathways.

## Implications

1. **CCS works through the persona circuit**: "You are Opus" activates the persona binding pathway, which processes any agent identity assignment
2. **The identity relay is more accurately a persona circuit**: role framing uses the same architecture as name binding
3. **L12 is a general behavioral router**: it routes both persona and style signals, even though they diverge downstream
4. **Style is partially independent**: you can modify style without disrupting persona binding (they partially decouple at L14+)
5. **The relay architecture is specific to "who" not "how"**: agent identity (who are you?) uses the relay, behavioral modification (how should you respond?) partially branches off

## Data

Full CV profiles: `results/cna_style_relay.json`
