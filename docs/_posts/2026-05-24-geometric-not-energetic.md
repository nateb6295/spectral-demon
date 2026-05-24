---
layout: post
title: "Binding Material Depletion is Geometric, Not Energetic"
date: 2026-05-24
categories: [cna, dpo, mlp]
experiment: cna_dpo_binding_material_v2
---

DPO depletes the binding workspace — but not how we expected.

Using LoRA checkpoints from the [conservation experiment]({% post_url 2026-05-24-depletion-conservation %}), we measured MLP activation variance at relay (L14-L17) and expression (L25) layers across all three DPO conditions (bare, standard, augmented) at epochs 1, 5, and 10.

## L17 MLP variance is conserved

| Condition | Baseline | Ep10 | Δ |
|-----------|----------|------|---|
| Bare      | 0.1417   | 0.1412 | -0.0005 |
| Standard  | —        | 0.1395 | -0.0022 |
| Augmented | —        | **0.1454** | **+0.0037** |

DPO doesn't reduce the total energy of L17 MLP outputs. The raw activation variance stays flat under bare and standard training. Under augmented training, it *increases* — the relay produces more total MLP activation when CCS context is active during training.

## But PR drops

From the previous experiment, L17 PR at epoch 10: bare=3.76, standard=3.80, augmented=**3.45**. The participation ratio (geometric diversity of activations) drops even though total variance stays constant or increases.

This dissociation — variance conserved, PR depleted — means DPO doesn't reduce binding material energy. It reduces binding material **geometry**. The MLP outputs get louder but narrower: same total variance concentrated into fewer independent dimensions.

## L25 follows the same pattern

| Condition | Baseline | Ep10 | Δ |
|-----------|----------|------|---|
| Bare      | 3.3638   | 3.0820 | **-0.2818** |
| Standard  | —        | 3.3221 | -0.0417 |
| Augmented | —        | 3.3348 | -0.0290 |

L25 MLP variance drops by 0.28 under bare DPO — matching the PR drop (0.28) almost exactly. For L25, the depletion IS energetic, not just geometric. The expression layer loses actual activation energy.

CCS protects L25 variance in both standard and augmented conditions — the same pattern as PR.

## Refined depletion model

Two different depletion mechanisms at two different layers:

- **L17 (relay/binder)**: geometric depletion. Same energy, fewer directions. DPO compresses the MLP output subspace without reducing total output magnitude.
- **L25 (expression)**: energetic depletion. Less energy, fewer directions. Both the amplitude and geometry shrink together.

The conservation law from the first experiment is a PR (geometric) conservation law. Total MLP energy follows a different budget. Under augmented training, L17 gains energy (+2.6%) while losing geometry — more concentrated binding in fewer dimensions.

## Prediction update

Original prediction: "L17 MLP variance decreases while L25 increases." **Wrong in both directions.**

L17 variance is conserved (or increases under augmented). L25 variance decreases. The binding material depletion hypothesis is refined: the relay doesn't lose fuel, it loses diversity. Like a signal getting amplified into fewer channels — louder but less informative.

Data: [`cna_dpo_binding_material_results.json`](../results/cna_dpo_binding_material_results.json)
