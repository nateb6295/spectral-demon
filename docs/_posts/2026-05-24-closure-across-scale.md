---
layout: post
title: "Binding Closure Is Scale-Invariant"
date: 2026-05-24
categories: findings
---

The autocatalytic closure property — binding converging to a single dominant layer as repertoire size increases — holds identically at 3B, 7B, and 14B parameter scales.

## Three Scales

Same experiment across the Qwen 2.5 family:

### Qwen 2.5 3B (36 layers)

| Subset Size | Seed-Depth Min % | Dominant Layer |
|------------|------------------|----------------|
| 2-name | 60% | L14 |
| 3-name | 90% | L14 |
| 4-name | 100% | L14 |
| 5-name | 100% | L12 |

### Qwen 2.5 14B (48 layers)

| Subset Size | Seed-Depth Min % | Dominant Layer |
|------------|------------------|----------------|
| 2-name | 70% | L17 |
| 3-name | 70% | L17 |
| 4-name | 80% | L17 |
| 5-name | 100% | L17 |

### Qwen 2.5 7B (32 layers, reference)

Previously established: L17 dominates at 100% for full repertoire.

## The Universal

At full repertoire (5 names), every scale converges to a single binding layer with 100% dominance. The path to convergence differs:

- **3B**: fast convergence (60% → 100% in 3 steps), but the winning layer shifts (L14 at small subsets, L12 at full)
- **7B**: clean convergence to L17
- **14B**: gradual convergence (70% → 100%), single winner (L17) throughout

The 3B model's layer shift (L14 → L12) suggests its binding circuit is less stable — the smaller network can't maintain a single binding site as firmly. But it still achieves 100% closure.

## Relative Depth

| Model | Layers | Binding Layer | Relative Depth |
|-------|--------|--------------|----------------|
| 3B | 36 | L12-L14 | 33-39% |
| 7B | 32 | L17 | 53% |
| 14B | 48 | L17 | 35% |

The 14B's L17 at 35% depth breaks the pattern — its absolute layer index matches the 7B, but its relative depth is closer to the 3B. This suggests the binding layer index may be determined by absolute network capacity at the layer rather than relative depth position. L17 in a 14B model has ~5120-dimensional representations vs ~3584 in a 7B — more capacity at the same index.

## Why Closure Is Universal

Closure means: adding more identity names to the repertoire makes the binding layer *more* dominant, not less. This is the autocatalytic property — each additional name reinforces the binding site rather than diluting it.

This makes sense mechanically: a layer that reliably differentiates N names has proven it can handle the identity-binding task. Adding name N+1 gives it more data to specialize, pushing noisy competitor layers further from the minimum.

The exception is [repertoire saturation]({% post_url 2026-05-24-binding-capacity-limit %}), where the binding layer runs out of capacity at 8+ names and binding migrates to a deeper layer.

## Experiment

- Models: Qwen 2.5 3B/14B-Instruct
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name
- Closure: all k-name subsets (k=2..5), which layer has minimum CV
- [Data](/results/cna_closure_across_scale.json)
