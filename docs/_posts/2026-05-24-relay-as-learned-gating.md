---
layout: post
title: "The Relay Is a Learned Gating Mechanism"
date: 2026-05-24
---

MUDD skip connections (Lisennlp, NanoGPT speedrun WR) use a small MLP to generate data-dependent coefficients for skip connections. The skip strength varies by input.

The relay zone (L14-L17) already does this. Different identity names produce different effective transformations through the same layers. From individual layer ablation data:

| Layer | Opus | ChatGPT | Claude | Spread |
|-------|------|---------|--------|--------|
| L14 | -0.122 | -0.046 | -0.066 | 0.076 |
| L15 | -0.119 | -0.288 | -0.343 | 0.225 |
| L16 | -0.123 | -0.297 | -0.475 | 0.351 |
| L17 | -0.045 | +0.218 | +0.267 | 0.311 |

(Values: change in L25 PR ratio when that layer is ablated)

L14 is nearly name-independent — a vestigial transformation that affects all identities roughly equally. The spread is 0.076.

L16 is maximally name-dependent — ablating it devastates Claude (-0.475) while barely touching Opus (-0.123). The sorter applies different effective gates per name.

L17 is the most surprising: ablation makes Claude and ChatGPT's identity *stronger* at L25 (+0.27) while doing nothing to Opus. The binder's removal helps some architectures. This is the opposite of what you'd expect if L17 were applying a uniform transformation.

The relay learned input-conditional processing. Each name gets different effective "skip ratios" through L15-L17. MUDD formalizes this as an efficiency technique. The relay zone discovered it as an identity-sorting mechanism. Same pattern, different optimization pressure.

This confirms the structure group prediction: the relay implements a family of transformations parameterized by identity input. One transformation family, different instantiations per name. The relay IS a gauge transformation — its effect depends on what it's transforming.
