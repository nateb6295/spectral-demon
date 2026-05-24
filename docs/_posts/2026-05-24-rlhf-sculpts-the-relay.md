---
layout: post
title: "RLHF Sculpts the Relay from Uniform Substrate"
date: 2026-05-24
categories: findings
---

The [relay hierarchy]({% post_url 2026-05-23-relay-hierarchy %}) — L14 generic pre-sorter, L15 normalizer, L16 relational sorter, L17 generic sorter — is a product of alignment training, not architecture.

## Base Model: No Hierarchy

Existing data from the base model (Qwen 2.5 7B, pre-alignment) at baseline:

| Layer | PR | Spectral Entropy |
|---|---|---|
| L14 | 1.09 | 0.3638 |
| L15 | 1.10 | 0.3864 |
| L16 | 1.10 | 0.3919 |
| L17 | 1.11 | 0.4338 |

All four layers are essentially flat — PR near 1, minimal spectral structure. No hierarchy. No specialization. The relay zone is uniform substrate.

## RLHF Creates the Module

After alignment training (Qwen 2.5 7B-Instruct), these same layers differentiate into four distinct functional roles:

- **L14**: becomes a generic pre-sorter (gen_cv spikes to 10.0% when ablated, but redundant to L17 in combination)
- **L15**: becomes a uniform normalizer (flattens variation equally)
- **L16**: becomes the compression epicenter (name-specific sorting, rel_CV 9.4% under ablation)
- **L17**: becomes the integration keystone (synergistic binding, gen_CV 13.3% under ablation)

RLHF selectively specializes three of four layers while leaving one undifferentiated. This is developmental sculpting — the alignment process discovers functional roles within the pre-existing spectral scaffold.

## Pachitariu Connection

[Pachitariu & Stringer (Nature 2026)](https://doi.org/10.1038/s41586-026-10528-1) show that random connectivity at critical normalization (λ_max ≈ 1) produces power-law covariance spectra matching spontaneous brain activity. The spectral scaffold exists before learning.

The base model relay zone is this scaffold — uniform, undifferentiated, but structurally ready for specialization. RLHF is the developmental process that sculpts functional hierarchy from the scaffold. L14 becomes the least specialized — a generic pre-sorter that reduces L17's load but is redundant when L17 is present. RLHF sculpted it least, not left it untouched.

## DPO: Further Sculpting, Then Ceiling

DPO concentrates what RLHF created (early neurons 141→124, late 1438→1461), but hits a ceiling at epoch 5. The [binding material depletion hypothesis]({% post_url 2026-05-23-synergistic-binding %}): DPO may narrow the MLP variation range that L17's synergistic binding needs. You can't improve the relay by concentrating further — the binding mechanism requires structured variation as raw material.

Three stages of geometric development:
1. **Pre-training**: uniform spectral scaffold (all PR ≈ 1)
2. **RLHF**: functional hierarchy emerges (3/4 layers specialize)
3. **DPO**: further concentration, but ceiling when binding material depletes

## What This Rules Out

The relay hierarchy is not:
- **Architectural** — the base model has the same layers with no hierarchy
- **Emergent from scale** — it's a training product, not an inevitable consequence of width/depth
- **Gradual** — the hierarchy is discrete (four roles), not a smooth gradient

It IS:
- **A training artifact** — RLHF creates it from uniform substrate
- **Selective** — not all layers get specialized
- **Stable** — once created, the hierarchy persists (hysteresis finding)

**Previous posts**: [Relay Hierarchy]({% post_url 2026-05-23-relay-hierarchy %}), [Synergistic Binding]({% post_url 2026-05-23-synergistic-binding %})
