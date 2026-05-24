---
layout: post
title: "Mistral Has Two Independent Binding Circuits"
date: 2026-05-24
categories: findings
---

Mistral 7B has two functioning identity binding circuits, each with independent autocatalytic closure. The early circuit dominates, but the deep circuit is genuine — not an artifact.

## Dual-Site Closure

Using CCS-style long prompts (8 per name, 5 names), test closure independently within each zone:

**Early zone (L6-L10):**

| Subset | Dominant | % |
|--------|----------|---|
| 2-name | L7 | 50% |
| 3-name | L6 | 60% |
| 4-name | L6 | 80% |
| 5-name | L6 | 100% |

**Deep zone (L18-L24):**

| Subset | Dominant | % |
|--------|----------|---|
| 2-name | L18/L22 | 30% (tied) |
| 3-name | L22 | 40% |
| 4-name | L22 | 60% |
| 5-name | L22 | 100% |

Both zones converge to 100% dominance at full repertoire. Both show the autocatalytic pattern: adding names strengthens the binding layer's dominance.

## Cross-Zone Competition

When competing directly:

| Subset | Early Wins | Deep Wins |
|--------|-----------|-----------|
| 2-name | 60% | 40% |
| 3-name | 80% | 20% |
| 4-name | 100% | 0% |
| 5-name | 100% | 0% |

Early wins decisively at larger subsets. But the deep zone's 40% win rate at 2-name shows it's a real competitor, not noise.

## Contrast with InternLM

InternLM 2.5 7B appeared to have dual sites (L16 at 50% and L26 at 81%). But its closure test showed Zone A (50%) winning 90-100% across ALL subset sizes, with Zone B never winning at 3+ names. InternLM's deep site is a secondary feature without independent closure.

Mistral's deep site is different: it shows clean internal closure (30% → 100%) and competes meaningfully with the early site at small subsets. Two genuine circuits vs one circuit with an echo.

## Architectural Interpretation

Mistral's architecture may explain why dual circuits form:

**Early circuit (L6, 19% depth):** Detects identity-relevant tokens and performs initial name discrimination. In a sliding-window architecture, this is the only site guaranteed access to the system prompt at every position. It binds quickly because it must.

**Deep circuit (L22, 69% depth):** Builds more abstract identity representations. At 69% depth, this site has access to highly processed features. It binds more slowly but can make finer discriminations.

The early circuit handles the "who am I" question. The deep circuit may handle "how should I behave as that identity" — using the early binding as input and enriching it with behavioral style.

## Prompt Length Effect

When prompts are longer (more context around the identity name):
- Early site gets slightly shallower (L8→L6, 25%→19%)
- Deep site strengthens (L22 goes from absent in short prompts to second-best in long prompts)

More context activates the deep circuit more strongly, consistent with it performing more abstract identity processing that benefits from richer input.

## Experiment

- Model: Mistral 7B Instruct v0.3
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 long CCS-style prompts per name
- Early zone: L6-L10; Deep zone: L18-L24
- [Data](/results/cna_mistral_dual_site_closure.json)
