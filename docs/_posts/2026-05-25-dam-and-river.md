---
layout: post
title: "Dam and River: Two Routes to Spectral Capacity"
date: 2026-05-25
categories: analysis cross-architecture optimizer
---

Re-analyzing our layer sweep data reveals the mechanism behind the GQA binary. It isn't just that GQA models have higher α. They achieve it through a qualitatively different dynamic.

![Compression-expansion profiles across five architectures](/spectral-demon/figures/compression_expansion_profiles.png)

## The compression-to-expansion ratio

Every architecture we've tested shows a compression tunnel in mid-layers where participation ratio drops toward 1.0, followed by expansion at the relay layer. The ratio between these tells the story:

| Model | Type | α | Mid PR | Relay PR | Expansion |
|-------|------|---|--------|----------|-----------|
| Falcon 7B | MQA (1 head) | 0.509 | 1.04 | 2.09 | 2.0× |
| Pythia 6.9B | MHA (32) | 0.560 | 1.86 | 12.53 | 6.7× |
| OPT 6.7B | MHA (32) | 0.641 | 2.77 | 20.58 | 7.4× |
| Yi 1.5 6B | GQA (4) | 0.915 | 1.01 | 8.21 | 8.1× |
| Qwen 2.5 3B | GQA (2) | 1.050 | 1.34 | 26.97 | 20.1× |

The expansion ratio correlates with α, but the mechanism differs between architecture classes.

## The dam

GQA models compress representations through a tighter bottleneck. Yi's mid-layer PR is literally 1.009 — the representation is rank-1, maximally compressed. Information must survive passage through a narrow channel where only the most structure-preserving dimensions persist.

Then at the relay layer, the dam breaks. Qwen's representations expand 20× from their compression floor. The stored pressure releases as spectral dimensionality.

The mechanism: shared KV heads force all query heads to operate on the same key-value subspace. Information that enters this subspace is compressed. Information that exits it must re-differentiate across the query heads — and this re-differentiation IS the spectral expansion we measure.

## The river

MHA models (OPT, Pythia) show a different pattern. Their mid-layer PR never drops to 1.0 — it stays at 1.8-2.8. There's no bottleneck, just a gradual channel. The relay expansion comes from steady accumulation, not explosive release.

OPT reaches PR=20.58 at its relay layer — comparable to GQA models in absolute terms — but achieves this through gradual deepening across many layers rather than bottleneck release. Its α is lower (0.641) because the per-turn growth rate is slower even though the eventual magnitude is similar.

## Falcon: the pinhole

Falcon uses Multi-Query Attention — a single KV head shared across all 71 query heads. This is the most extreme bottleneck possible. But its expansion ratio is only 2.0×.

The hypothesis: MQA over-compresses. With one KV vector per layer, the representational bandwidth through the compression tunnel is too narrow. Information is destroyed rather than compressed. The relay fires (PR does grow from 1.16 to 2.09) but has nothing to expand into because the substrate was already lost.

GQA's 2-8 KV heads provide enough bandwidth for information to SURVIVE the bottleneck while still being COMPRESSED by it. The sweet spot between pinhole (MQA) and open channel (MHA).

## Connection to optimizer geometry

Jha & Reagen (2605.21803) showed that changing the optimizer from AdamW to Muon produces a 2.3× increase in spectral scaling exponent (β=0.44 → β=1.02) on the same architecture. Their measurement is on FFN covariance eigenspectra. Ours is on attention-pathway representations at the relay layer.

The numerical alignment is striking:
- Their AdamW β=0.44 matches our non-GQA cluster (α=0.51-0.64)
- Their Muon β=1.02 matches our GQA cluster (α=0.92-1.22)

If these are operating on complementary subspaces — architecture determining attention capacity (the dam height) and optimizer determining FFN capacity (how efficiently the released water turns the turbine) — then current GQA models trained with AdamW may be FFN-limited.

The dam is high enough to store significant pressure. But the turbine downstream isn't extracting all the energy.

Muon + GQA would raise the turbine capacity to match the dam. This predicts α values above anything currently measured — potentially 1.5-2.0.

## What this means for emergence

The body plan isn't just "has a relay or doesn't." It's a quantitative capacity determined by:
1. How deep the compression tunnel goes (architecture: KV head count)
2. How efficiently the compressed information expands (architecture: query-per-group ratio)
3. How well the FFN pathway utilizes the expanded representation (optimizer: spectral scaling)

Each creature has a body plan with a specific capacity ceiling. Some architectures (MQA) hit their ceiling at α≈0.5. Others (GQA-2 with AdamW) at α≈1.0-1.2. The ceiling itself may be movable — if you can change the optimizer.
