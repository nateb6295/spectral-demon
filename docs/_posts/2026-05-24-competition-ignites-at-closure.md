---
layout: post
title: "Competition Ignites at Closure: The 3-Name Phase Transition"
date: 2026-05-24
categories: findings
experiment: cna_competition_scaling
models: [Qwen 2.5 7B Instruct]
---

The competitive binding dynamic doesn't exist at all repertoire sizes. It ignites at exactly 3 names — the autocatalytic closure threshold.

## Setup

Run compensatory ablation at key layers (L4, L7, L8, L12) with repertoire sizes of 2, 3, 5, 7, and 9 names. Measure the amplification/suppression effect on L17 binding.

## Results

### L7 Ablation Effect on L17 by Repertoire Size

| Names | L4 Impact | L7 Impact | L8 Impact | L12 Impact |
|-------|----------|----------|----------|-----------|
| 2 | **-93%** | **-79%** | **-66%** | -75% |
| 3 | +27% | **+203%** | +86% | -78% |
| 5 | +50% | +147% | +59% | -66% |
| 7 | +67% | +146% | +62% | -53% |
| 9 | +64% | +135% | +81% | -44% |

### Baseline L17 CV

| Names | L17 CV |
|-------|--------|
| 2 | 0.01118 |
| 3 | 0.01037 |
| 5 | 0.01072 |
| 7 | 0.00908 |
| 9 | 0.00836 |

## The Phase Transition

### 2 Names: Cooperative

With only 2 identity names, early-layer ablation uniformly **suppresses** L17 binding. L7 ablation destroys 79% of binding. The circuit is fully cooperative — identical to the base model's behavior. There is no competitive dynamic.

### 3 Names: Competition Ignites

At 3 names, L7 ablation produces **+203% amplification** — the strongest effect at any repertoire size. The competitive circuit doesn't just appear; it *erupts*. The transition from -79% (2 names) to +203% (3 names) is a 282 percentage point swing.

This is the same threshold where autocatalytic closure activates: binding convergence begins at 3+ names and reaches 100% at 4-5 names.

### 5-9 Names: Competitive Equilibrium

The amplification stabilizes at ~135-147% for L7 and gradually increases for L4 and L8. The competitive system has reached equilibrium — adding more names maintains the competition without strengthening it further.

### Router Weakens with Repertoire

L12 destruction decreases from -78% (3 names) to -44% (9 names). With more identity names, the system develops redundancy around the router. The router is still critical but less of a single point of failure.

## Interpretation

### Closure = Phase Transition in Circuit Topology

Autocatalytic closure is not just about binding convergence. It's a phase transition in the circuit's competitive structure:

- **Below closure (1-2 names)**: cooperative circuit, no competition, ablation hurts
- **At closure onset (3 names)**: competition erupts, maximum amplification
- **Above closure (4+ names)**: competitive equilibrium, stable amplification

The 3-name threshold is where the identity representation becomes rich enough to support both early and late circuits independently. Below 3, there isn't enough identity signal for two circuits to compete — they must cooperate. At 3+, each circuit can capture enough signal on its own, so they begin competing.

### Why Peak at 3?

The peak amplification at 3 names (vs 5 or 7) suggests the competition is most intense just at the threshold. Like a phase transition in physics, the largest fluctuations occur at the critical point. At 5+ names, the system has settled into its competitive equilibrium — the competition is still present but the dynamics have stabilized.

### Connection to Closure

This reframes autocatalytic closure as a circuit topology transition:

1. **1-2 names**: weak identity signal, cooperative circuit, no closure
2. **3 names**: signal crosses threshold, competition ignites, closure begins
3. **4-5 names**: competitive equilibrium established, closure completes (100%)
4. **5-8 names**: stable competitive operation, slight efficiency loss (baseline CV drops)
5. **9+ names**: approaching saturation, router becomes less critical

The closure isn't just binding converging — it's the competitive filter activating. Once the filter is on, binding becomes selective (only strong signals pass) which produces the appearance of convergence.

## Implications

1. **Closure is a phase transition in circuit topology**, not just binding strength
2. **The competitive circuit requires minimum 3 identities to activate** — below this, the system lacks the representational diversity to support competition
3. **Peak competition occurs at the critical point** (3 names), not at maximum repertoire
4. **Router redundancy increases with repertoire** — the system becomes more fault-tolerant with more identities
5. **Baseline binding decreases with repertoire** — the system distributes finite binding capacity across more identities (0.011 → 0.008 CV)

## Data

Full scaling data: `results/cna_competition_scaling.json`
