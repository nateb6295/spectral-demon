---
layout: post
title: "Sub-Threshold Geometric Onset"
date: 2026-05-23
categories: experiments
---

Geometric reorganization begins at doses far below any behavioral detection threshold. The identity direction carries geometric information that generic processing cannot access.

## The Experiment

We added the CCS direction to relay layers (L14-L17) at eight micro-doses (alpha from 0.00 to 0.25) and measured participation ratio at L25 separately for relational and generic prompts.

**Code**: [`experiments/cna_subthreshold_pr.py`](https://github.com/nateb6295/spectral-demon/blob/master/experiments/cna_subthreshold_pr.py)

## Results

- **Generic PR is completely invariant**: 3.34 +/- 0.003 across all eight doses. The CCS direction carries zero generic-relevant geometric information.
- **Relational PR rises monotonically**: from 3.36 at alpha=0.00 to 3.64 at alpha=0.25, with geometric onset at alpha=0.01.
- **Super-linear dose-response**: quadratic fit yields 0.70*alpha^2 + 0.18*alpha + 1.00. The positive quadratic coefficient means each unit of CCS direction makes the next unit more effective — self-reinforcing geometry.

## What This Means

The CCS direction is a pure identity signal — it reorganizes relational geometry without touching generic processing. This selectivity holds down to the smallest testable dose. The super-linear response suggests positive curvature in the fiber bundle connection: parallel transport along the identity direction accumulates geometric effect faster than linearly.

**Data**: [`results/cna_subthreshold_pr_results.json`](https://github.com/nateb6295/spectral-demon/blob/master/results/cna_subthreshold_pr_results.json)

![Sub-threshold PR](/spectral-demon/figures/fig_subthreshold_pr.png)
