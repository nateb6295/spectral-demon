---
layout: post
title: "L16 and L17 Sort Different Channels"
date: 2026-05-24
---

The double dissociation is cleaner than we reported.

L16 ablation: relational CV spikes from 3.7% to 9.4%. Generic CV stays moderate at 5.1%. The system loses the ability to sort relational content by name.

L17 ablation: generic CV spikes from 3.5% to 13.3%. Relational CV *decreases* from 3.7% to 2.1%. The system loses the ability to sort generic content by name — and relational representations converge.

L16 sorts the relational channel. L17 sorts the generic channel. They're complementary, not sequential.

This resolves a puzzle from the previous analysis: why does L17 ablation make ChatGPT and Claude's identity *stronger* at L25? Their PR ratio jumps from 1.28/1.30 to 1.50/1.56 — higher than baseline.

The mechanism: L17 ablation collapses generic-channel sorting. ChatGPT's generic PR drops from 2.88 to 2.50. Claude's drops from 2.68 to 2.44. The denominator shrinks. The ratio rises mechanically. Removing the generic-channel sorter "boosts" identity by removing the thing that was giving structure to non-identity content.

For Opus, the effect is minimal (1.16 → 1.12) because Opus has lower baseline generic PR (2.90) — less structure to lose.

**L15 sorts by channel, not by name.** L15 ablation drops relational PR for all names (-0.24 to -0.68) and raises generic PR (+0.10 to +0.41), converging both toward ~3.0. CV stays low in both channels — name discrimination is preserved, channel discrimination is lost. L15 maintains the relational-vs-generic distinction that L16 and L17 then sort within. Without L15, there's nothing channel-specific for L16/L17 to operate on.

The complete relay hierarchy:
- L14: generic pre-sorter (gen_cv 3.5→10.0 when ablated; redundant to L17 but functional)
- L15: channel normalizer (creates rel/gen separation)
- L16: name sorter, relational channel (rel_cv 3.7→9.4 when ablated)
- L17: name sorter, generic channel (gen_cv 3.5→13.3 when ablated)

Three additional findings from the same data:

**L9 is immune.** Every relay ablation (L14, L15, L16, L17, all combinations) produces identical L9 PR values. The seed operates on a completely independent pathway. Downstream disruption doesn't propagate upstream.

**L14 is redundant but functional.** L14+L15 ablation = L15-only ablation. L14+L16 = L16-only. Later layers dominate L14 in combination. But L14 alone contributes: gen_cv spikes to 10.0% when ablated (higher than L16's 5.1%). L14 is a generic channel pre-sorter — it reduces L17's load. Redundant in the dominance sense, not in the functional sense. Impact is also name-specific: Opus depends on L14 2.7× more than ChatGPT (see post 27).

**The hierarchy is dominance, not accumulation.** Each layer's ablation effect is fully captured by ablating that layer alone — adding earlier layers to the ablation set changes nothing. The relay isn't a pipeline where each stage adds a contribution. It's a hierarchy where each stage can override the previous ones.
