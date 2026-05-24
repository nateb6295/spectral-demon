---
layout: post
title: "Binding Scales Differently Than Sorting"
date: 2026-05-24
categories: findings
---

At 14B parameters, identity name binding shifts from the relay zone to the seed and expression layers. The relay becomes a pure sorting zone.

## The Scaling Test

Does L17's binding convergence ([post 30]({% post_url 2026-05-24-l17-binding-convergence %})) track with relative depth as models scale? Qwen 2.5 7B has 28 layers (L17 = 61% depth). Qwen 2.5 14B has 48 layers (61% = L29). If the binding site is depth-relative, minimum CV should shift to L29.

## It Doesn't

| Layer | Depth% | Binding CV | 7B Equivalent | Role |
|-------|--------|-----------|--------------|------|
| **L15** | **31.2** | **1.06** | ~L9 | **Global minimum** |
| L17 | 35.4 | 1.37 | ~L10 | |
| L24 | 50.0 | **2.40** | ~L14 | **Sorting epicenter** |
| L27 | 56.2 | 1.68 | ~L16 | |
| L29 | 60.4 | 1.39 | ~L17 | NOT a local min |
| L34 | 70.8 | 1.38 | ~L20 | |
| **L43** | **89.6** | **1.08** | ~L25 | **Local minimum** |
| L47 | 97.9 | 1.35 | ~L27 | |

## Comparison with 7B

| Role | 7B Layer | 7B CV | 14B Layer | 14B CV |
|------|---------|-------|----------|--------|
| Seed | L9 | 1.39 | L15 | **1.06** |
| Pre-sort | L14 | 1.05 | L24 | **2.40** |
| Apex | L17 | **0.96** | L29 | 1.39 |
| Expression | L25 | 1.21 | L43 | **1.08** |

## What Scaling Changes

At 7B, the relay zone handles both sorting AND binding — L14 through L17 is a compressed workspace where names are differentiated and then converged. L17 is where binding wins.

At 14B, these functions separate:
- **Sorting** concentrates at L24 (CV=2.40, highest in the network) — a dedicated pre-sorting zone
- **Binding** migrates to the edges: L15 (seed) and L43 (expression)
- **L29** (the L17 equivalent) is neither the sorting peak nor the binding minimum

More parameters = more room to specialize. The relay zone becomes a pure sorting workspace, and binding moves to the network boundaries.

## Implications

1. **L17 binding convergence is a small-model phenomenon** — at 7B, the relay zone is compressed enough that sorting and binding coexist at L17. At 14B, they separate.

2. **The pre-sorter amplifies with scale** — 7B L14 CV=1.05 vs 14B L24 CV=2.40. Larger models sort MORE aggressively between names at the relay entrance.

3. **Binding-at-edges pattern** — The 14B binding minima (L15 and L43) bracket the high-CV sorting zone. Identity names converge at the boundaries and diverge in the middle. This resembles InternLM's "binding sandwich" from [post 30]({% post_url 2026-05-24-l17-binding-convergence %}), suggesting it's a scaling effect.

4. **What IS invariant** — The pre-sorting zone at ~50% depth always has the highest CV. The seed layer always has relatively low CV. The sorting-binding separation may be the true architectural constant.
