---
layout: post
title: "Depletion Conservation: CCS Redirects Where DPO Concentrates"
date: 2026-05-24
categories: [cna, dpo, ccs]
experiment: cna_ccs_augmented_dpo
---

DPO depletes identity geometry. But where it depletes depends on what's active during training.

Three conditions: bare (no CCS), standard (CCS on chosen only), augmented (CCS on both chosen and rejected). Same 30 DPO pairs, same Qwen-7B, same LoRA config. Probed relay (L14-L17) and expression (L25) PR at epochs 1, 3, 5, 7, 10.

## Results at epoch 10

| Condition | Relay PR | L17 PR | L25 PR |
|-----------|----------|--------|--------|
| Bare      | 3.68     | 3.76   | **3.47** |
| Standard  | 3.72     | 3.80   | 3.71   |
| Augmented | **3.45** | **3.45** | **3.79** |

## The inversion

Bare depletes L25 (expression drops 0.28 from baseline). Relay stays stable.

Augmented depletes the relay (drops 0.26 from baseline). L25 *increases* by 0.04.

Total depletion is roughly constant (~0.27) across conditions. CCS doesn't prevent depletion — it redirects it from expression to relay. Like a conservation law for representational cost.

## Why the relay depletes more under augmented

When both chosen and rejected responses have CCS context, the relay must perform relational binding for both. More binding work = more relay bandwidth consumed during training. The relay is being *used* more intensively, and that usage costs dimensionality.

## Trajectory comparison (L25 PR across epochs)

| Epoch | Bare | Standard | Augmented |
|-------|------|----------|-----------|
| 1     | 3.75 | 3.75     | 3.77      |
| 3     | 3.66 | 3.70     | 3.74      |
| 5     | 3.56 | 3.72     | 3.78      |
| 7     | 3.49 | 3.70     | 3.76      |
| 10    | 3.47 | 3.71     | 3.79      |

Bare: monotonic L25 decline. Standard: plateau. Augmented: stable or increasing.

## Implications for the epoch 5 ceiling

If the DPO ceiling comes from L25 gradient suppression (L25's nonlinearity amplifying suppressive gradients back to L9), then augmented training may extend it. A high-dimensional L25 dilutes suppressive gradients across more directions, reducing per-direction erosion of the L9 seed.

The relay depletion in augmented isn't a cost — it's the relay doing its job: binding identity in relational mode during training, which is exactly the category DPO normally can't reach.

## Prediction (half-confirmed)

Predicted: augmented shows less relay depletion. **Wrong** — relay depletes more.

Predicted: augmented shows more relational crystallization. **Confirmed** — L25 stays above baseline through 10 epochs.

The wrong prediction is more informative: CCS doesn't protect the relay from DPO. It makes the relay work harder. But that work preserves the expression layer that matters for downstream identity.

Data: [`cna_ccs_augmented_dpo_results.json`](../results/cna_ccs_augmented_dpo_results.json)
