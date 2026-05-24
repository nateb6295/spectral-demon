---
layout: post
title: "MLP Diversifies, Attention Concentrates: Component-Level Dissociation Under DPO"
date: 2026-05-24
categories: [cna, dpo, eigenspectrum]
experiment: cna_eigenspectrum_probe
---

The previous two experiments showed DPO depletion is [conserved]({% post_url 2026-05-24-depletion-conservation %}) and [geometric rather than energetic]({% post_url 2026-05-24-geometric-not-energetic %}). This third experiment measures the eigenspectrum of MLP outputs across conditions, revealing that MLP and hidden state PR move in opposite directions at L25.

## L17 (relay): coherent depletion

| Condition | MLP PR | top1 frac | entropy |
|-----------|--------|-----------|---------|
| Baseline  | 7.12   | 0.281     | 2.42    |
| Bare ep10 | 6.85   | 0.295     | 2.40    |
| Standard ep10 | 7.02 | 0.286   | 2.41    |
| Augmented ep10 | 6.93 | 0.295  | 2.41    |

L17 MLP PR drops slightly under DPO, and the top eigenvalue grows from 28.1% to 29.5% of total variance. The relay's MLP component concentrates gently — consistent with geometric depletion at the hidden state level.

L17 is **coherent**: its MLP and attention components deplete together. Both contribute to the PR drop measured in hidden states.

## L25 (expression): MLP and hidden state diverge

| Condition | MLP PR | Hidden state PR | Direction |
|-----------|--------|-----------------|-----------|
| Baseline  | 5.89   | 3.75            | —         |
| Bare ep10 | **6.97** | **3.47**      | **opposite** |
| Standard ep10 | 5.93 | 3.71          | stable    |
| Augmented ep10 | **7.14** | **3.79**  | **opposite** |

Under bare DPO, L25 MLP output PR *increases* from 5.89 to 6.97 (+18%). But hidden state PR *decreases* from 3.75 to 3.47 (-7.5%). The MLP produces more geometrically diverse output. The full layer produces less.

The gap is the attention component. Attention at L25 must be concentrating — routing the MLP's diverse output into fewer effective directions. The expression layer's internal components work in opposition under DPO: MLP compensates (diversifies), attention suppresses (concentrates).

## CCS stabilizes both components

Standard condition (CCS on chosen only): L25 MLP PR stays flat at 5.93, hidden state PR stays at 3.71. CCS prevents the MLP diversification AND the attention concentration. It stabilizes the component balance.

Augmented condition: L25 MLP PR increases most (→7.14), hidden state PR stays high (→3.79). CCS on both sides allows MLP diversification while preventing attention concentration. Best of both.

## Interpretation

The L25 gradient suppression mechanism from [earlier data]({% post_url 2026-05-24-depletion-conservation %}) is now localized: it's in the attention weights at L25, not the MLP. DPO trains L25 attention to concentrate its routing while the MLP tries to compensate by producing more varied output.

This is the component-level story of the DPO ceiling: the MLP builds material, attention routes it. Under DPO, attention routing narrows faster than MLP can diversify. CCS slows the attention narrowing, extending the useful training window.

Data: [`cna_eigenspectrum_results.json`](../results/cna_eigenspectrum_results.json)
