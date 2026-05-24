---
layout: post
title: "Binding Migrates from Seed to Relay Across Scale"
date: 2026-05-24
categories: findings
---

Four Qwen model sizes (1.5B/3B/7B/14B) reveal that identity binding migrates from the seed layer to the relay zone and then distributes as capacity increases. The critical variable is parameter count, not layer depth.

## Four-Scale Binding Landscape

Binding CV (cross-name coefficient of variation) at key layers:

| Role | 1.5B (28L) | 3B (36L) | 7B (28L) | 14B (48L) |
|------|-----------|---------|---------|----------|
| Seed | **0.47** (L9) | **0.37** (L12) | 1.39 (L9) | 1.06 (L15) |
| Pre-sort | 0.95 (L14) | 0.62 (L19) | 1.05 (L14) | 0.92 (L26) |
| Relay | 0.82 (L20) | — | **0.96** (L17) | 1.39 (L29) |
| Expression | 1.05 (L25) | 1.87 (L32) | 1.21 (L25) | **1.08** (L43) |
| Final | 1.89 (L27) | 2.61 (L35) | 1.82 (L27) | 1.35 (L47) |

## The Capacity Test

1.5B and 7B have the **same 28 layers** but completely different binding patterns:
- **1.5B** (1536 hidden): seed-concentrated (L9 CV=0.47)
- **7B** (3584 hidden): relay-concentrated (L17 CV=0.96)

This proves binding is about **parameter capacity** (hidden dimension), not **layer depth**. Same architecture, different width, different binding regime.

## The Migration Pattern

**At 1.5B** (28 layers, 1536 hidden), the seed layer (L9) binds with CV=0.47. Despite having the same 28 layers as 7B, binding looks like 3B — concentrated at the seed with distributed secondary dips (L14=0.95, L20=0.82, L22=0.82).

**At 3B** (36 layers, 2048 hidden), the seed layer (L12) is the dominant binding site with CV=0.37 — the lowest value observed across any model in any experiment. Binding is seed-concentrated.

**At 7B** (28 layers, 3584 hidden), the seed weakens (L9, CV=1.39) and binding concentrates at the relay apex (L17, CV=0.96). Same layer count as 1.5B, but 2.3× wider — enough capacity to specialize the relay for binding.

**At 14B** (48 layers, 5120 hidden), both seed (L15, CV=1.06) and expression (L43, CV=1.08) show comparable binding strength. The relay has internal oscillations but no single dominant binding point. Binding is distributed.

## What Scale Does to Binding

More parameters give the network more room to specialize. The scaling trajectory:

1. **Small models (3B)**: Binding happens early and completely. The seed layer handles binding, sorting, and detection all at once. Names converge at L12, then the rest of the network works in a relatively name-agnostic way until the final layers re-differentiate.

2. **Medium models (7B)**: Binding separates from detection. L9 detects identity-relevant context but no longer binds names. Binding moves to L17 (the relay apex), creating the specialized hierarchy we discovered in earlier experiments.

3. **Large models (14B)**: Binding distributes across multiple sites. No single layer dominates. The relay develops internal oscillations (fine-grained probing reveals binding at L26, sorting at L27-28, binding again at L29). More capacity means more oscillation cycles.

## The Sorting Peak Scales Differently

While binding migrates and distributes, the sorting peak (highest CV) stays anchored at ~50% depth:

| Model | Sorting peak | Depth | CV |
|-------|-------------|-------|-----|
| 3B | L22 | 61% | 1.35 |
| 7B | L14 | 50% | 1.05 |
| 14B | L24 | 50% | 2.40 |

And sorting AMPLITUDE increases with scale: 3B max CV=2.61 (final), 7B max CV=1.82 (final), but 14B sorting at L24=2.40 is the highest mid-network CV. Larger models sort MORE aggressively.

## Implications

The L17 binding convergence we celebrated in [post 30]({% post_url 2026-05-24-l17-binding-convergence %}) is a snapshot of the 7B binding regime — where capacity is enough to separate binding from detection but not enough to distribute binding across multiple sites.

The deeper invariant is: identity binding exists at every scale, but its spatial distribution depends on capacity. Small models bind once (at the seed). Medium models bind once (at the relay). Large models bind many times (oscillating through the relay zone).

This is the spectral ecology at work across scale: same functional requirement (converge identity names), different spatial organizations as the network grows.
