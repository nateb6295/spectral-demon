---
layout: post
title: "The Constraint Basin Is Real"
date: 2026-05-24
---

494 compression events. Entity counts from 13 to 43. Gist drift measured as 1 minus Jaccard similarity between consecutive semantic gists.

| Entities | N | Mean drift | Frozen (%) | High drift (%) |
|----------|---|-----------|------------|----------------|
| ≤15 | 40 | 0.670 | 10.0 | 47.5 |
| 16-20 | 119 | 0.635 | 5.9 | 55.5 |
| 21-25 | 109 | 0.414 | 36.7 | 35.8 |
| 26-30 | 94 | 0.373 | 44.7 | 30.9 |
| 31-35 | 98 | 0.257 | 58.2 | 19.4 |
| 36+ | 34 | 0.190 | 70.6 | 17.6 |

The pattern is monotonic: more entities = less gist dynamism = more frozen compressions. Entity accumulation mechanically constrains the compression operator's ability to update the semantic gist.

The transition happens at 20-25 entities: frozen rate triples (6% → 37%), high-drift rate drops (56% → 36%). Above 30, the system is more often frozen than changing.

The longest frozen runs — 11 consecutive compressions with no gist change — occurred at entity counts of 33-34 and 39. These are exactly the counts at which the split brain was most severe (StrongSync at 0.176-0.272).

The metastable window is 15-20 entities. The system explores freely (5.9% frozen, 55.5% high-drift) while maintaining enough structure for continuity. Our entity cap emerged empirically at 19. It sits at the top of this window — maximum structural closure before dynamism collapses.

The constraint continuum from the metastabilization post maps directly:

```
< 15 entities    →  under-constrained (high exploration, low persistence)
  15-20 entities →  metastable (high exploration AND persistence)
  20-30 entities →  transition (increasing rigidity)
> 30 entities    →  over-constrained (frozen, split brain)
```

This isn't a parameter we tuned. The entity cap was set by orphan repair's convergence point. The fact that it lands in the metastable window is the system finding its own operating point — exactly what a self-organizing system should do.
