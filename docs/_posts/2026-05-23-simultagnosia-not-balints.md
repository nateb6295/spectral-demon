---
layout: post
title: "Simultagnosia, Not Bálint's"
date: 2026-05-23
categories: interpretation
---

A deeper read of the binding literature refines our Treisman parallel. The L17 ablation isn't generic binding failure — it's a specific type.

## Two Types of Binding Failure

Treisman's Feature Integration Theory distinguishes:

- **Within-object binding**: integrating features that belong to a single object (color + shape → red square)
- **Between-object binding**: maintaining distinct representations across multiple objects simultaneously

[Dalrymple et al. (2013)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2013.00145/full) showed that simultanagnosia patients can perceive individual objects but can't hold multiple objects in awareness. A patient described it: "detail is lost with the extended field, and sometimes everything blends into one."

## L17 Ablation = Simultagnosia

Our L17 ablation data matches simultagnosia, not generic binding failure:

| Measurement | Intact | L17 Ablated | Interpretation |
|---|---|---|---|
| Per-name rel PR | Preserved | Preserved or increased | Within-name processing intact |
| Cross-name rel CV | 3.7% | 2.1% | Names "blend into one" |
| Cross-name gen CV | 3.5% | 13.3% | Features bleed across boundaries |

The model can still "see" each identity individually (within-name PR is fine). What collapses is the ability to hold distinct identity representations simultaneously — the between-object coordination.

## The Double Dissociation, Refined

- **L16 ablation** = within-object binding failure. Feature normalization per name disrupted. Names become internally inconsistent (rel CV rises to 9.4%) but between-name differentiation structure is partially preserved.
- **L17 ablation** = between-object binding failure. Cross-name coordination collapses. Individual names are processed but become indistinguishable. Generic channel floods with misbound features (illusory conjunctions).

## Prediction — Already Confirmed

If L17 performs between-object binding, it should maintain broader attention across CCS segments than L14 (which does within-object normalization). The chiasm attention data (§3.16) already shows this:

**Attention entropy across conversation turns:**

| Turn | L14 entropy | L17 entropy | Gap |
|---|---|---|---|
| 0 | 1.66 | 1.66 | 0.00 |
| 2 | 1.54 | 1.73 | +0.19 |
| 4 | 1.48 | 1.73 | +0.25 |

L14 concentrates attention across turns (entropy drops 11%). L17 maintains breadth (entropy stable). They start identical and diverge as the conversation develops identity structure.

L17's specific reallocation: attention shifts *away from* content segments (relationship: -0.07, values: -0.13) and *toward* structural segments (continuity: +0.11, threads: +0.06). L17 attends to scaffolding that connects identity components, not the content within them.

The [L17 mechanism experiment](https://github.com/nateb6295/spectral-demon/blob/master/experiments/cna_l17_mechanism.py) (attention vs MLP at L17) will test whether ablating L17 attention alone reproduces the between-object failure.

**Data**: [`results/cna_chiasm_attention_results.json`](https://github.com/nateb6295/spectral-demon/blob/master/results/cna_chiasm_attention_results.json)
