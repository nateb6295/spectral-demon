---
layout: post
title: "DPO Hits the Sorters"
date: 2026-05-24
---

DPO depletion isn't uniform across the relay. It's function-specific.

| Layer | Function | DPO bare Δ | DPO+CCS Δ |
|-------|----------|-----------|-----------|
| L14 | vestigial | -0.65 | +0.01 |
| L15 | channel normalizer | -0.59 | -0.11 |
| L16 | relational sorter | **-1.30** | -0.18 |
| L17 | generic sorter | **-1.18** | -0.19 |

Without CCS, DPO concentrates its depletion at the sorting layers. L16 (relational sorting) loses 1.30 PR. L17 (generic sorting) loses 1.18. The channel normalizer and vestigial layer are partially spared.

With CCS, the depletion is redistributed. All four layers take a roughly equal hit (-0.01 to -0.19). The sorting layers are protected at the cost of the normalizer and vestigial layers absorbing more.

CCS doesn't prevent depletion. It redistributes it from functionally critical layers toward less critical ones.

This extends the conservation law we found at the expression level (post 15): total depletion is conserved, but CCS controls WHERE the cost is paid. At the expression level, CCS redirects depletion from L25 to the relay. At the relay level, CCS redistributes depletion from the sorters (L16/L17) across all relay layers.

The pattern is recursive: CCS implements load-balancing at every level of the hierarchy. The functionally important components are protected by spreading their costs to less important ones. This is why the epoch 5 ceiling exists — eventually there's nowhere left to redirect the cost. Every layer has absorbed its share. The system has found the Pareto frontier of depletion allocation.
