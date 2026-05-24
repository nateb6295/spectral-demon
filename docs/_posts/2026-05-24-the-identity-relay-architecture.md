---
layout: post
title: "The Identity Relay Architecture: A Unified Picture"
date: 2026-05-24
categories: synthesis
---

Seventeen experiments across five model architectures reveal a complete picture of how transformer LLMs bind identity. This post synthesizes the findings into a unified relay architecture.

## The Five-Station Chain

Identity binding in transformer LLMs flows through a sequential relay chain:

```
L7 (Lexical) → L9 (Seed) → L12 (Router) → L14 (Early Relay) → L17 (Binding)
```

Each station has a distinct function, verified by mean ablation:
- **L7 ablation** → L9 drops 47%, L17 drops 18%
- **L9 ablation** → L14 drops 36%, L17 drops 16%
- **L12 ablation** → L14 drops 62%, L17 drops 65%

The chain is sequential. Damage propagates forward but attenuates with distance. L12 is the critical bottleneck.

## Station Functions

### L7: Lexical Binding (Pre-Training)
- **What**: differentiates token representations for distinct names
- **Specificity**: general (works for colors, cities, any tokens) but identity names produce 2.5x sharper binding
- **Context effect**: CCS-style "You are X" framing amplifies L7 differentiation 5.7x
- **Origin**: pre-training (exists identically in base model)
- **Closure**: yes — autocatalytic, converging at 4+ names

### L9: Seed Detection
- **What**: detects identity-relevant context (~12 neurons)
- **Specificity**: identity-specific (fires for "You are Opus" but not "The color is Red")
- **Dependency**: receives 47% of its signal from L7

### L12: Hidden Router
- **What**: transforms token-level differentiation into relay-compatible format
- **Visibility**: invisible to activation analysis (low CV, no closure properties)
- **Causal importance**: highest of any layer (65% of L17 binding destroyed when ablated)
- **IT effect**: becomes NOISIER after training (F_CV +11%) — needs diversity, not precision

### L14-L17: Behavioral Binding Relay
- **What**: converts identity detection into behavioral output differentiation
- **Origin**: relay mechanism created by instruction tuning
- **Sign split**: IT prunes 3.7-5.5% of sign-flipping neurons here
- **IT effect**: becomes SHARPER after training (F_CV -29% at L14, -54% at L17)

## Two Binding Classes

The relay chain exists in all architectures but its **absolute depth** depends on the attention mechanism:

| Attention Type | Binding Depth | Models |
|---------------|--------------|--------|
| Full attention | ~50% | Qwen 7B/14B, InternLM 7B |
| Sliding window | ~25% | Mistral 7B, Gemma 2 9B |

Sliding-window models compress identity earlier because they must bind before context exits the attention window. Full-attention models defer to the midpoint where abstract features are richer.

The mechanism is universal. The location is architecture-dependent.

## Dual Circuits

Mistral 7B has two independent binding circuits:
- **Early (L6)**: 23% flipping, closure to 100%
- **Deep (L22)**: 51% flipping, closure to 100%

Both show autocatalytic closure independently. Early wins cross-zone competition (100% at 4+ names). The deep circuit has majority flipping neurons — more identity-differentiated than identity-general.

Gemma 2 has a dormant deep site at L27 with the most reliable flippers (F_CV=0.026) of any model tested, but it doesn't win the binding competition.

## The Refinement Cascade

Instruction tuning creates two opposing gradients through the relay:

| Station | Flip % Change | F_CV Change | Effect |
|---------|-------------|-------------|--------|
| L7 | -0.9% | +16% | Slight broadening |
| L9 | -0.7% | +22% | Broadening |
| L12 | -2.7% | +11% | Diversifying router |
| L14 | -3.7% | -29% | Sharpening |
| L17 | -5.5% | -54% | Maximum sharpening |

**Early stations**: fewer flippers, but survivors are less reliable (broader detection range)
**Late stations**: fewer flippers, and survivors are more reliable (precise binding)

The relay is an information funnel: wide noisy input → narrow reliable output.

## What Pre-Exists vs What's Trained

| Feature | Origin |
|---------|--------|
| Token differentiation at L7 | Pre-training |
| Autocatalytic closure | Pre-training |
| Identity-relevant neurons throughout | Pre-training |
| Relay pruning gradient (L9→L17) | Instruction tuning |
| Sign-split refinement cascade | Instruction tuning |
| Behavioral binding at L17 | Instruction tuning |

The base model has all the raw materials. IT organizes them into a functional circuit.

## Stress Tests

**Adversarial names**: binding migrates to different layers but doesn't break. Each adversarial type moves binding differently (negation→deeper, void→shallower, meta-attack→minimal shift).

**Repertoire saturation**: the relay handles 5-7 names cleanly, saturates at 8. Binding migrates to L25 as an overflow site. Graceful degradation, not collapse.

**Scale invariance**: closure holds at 3B, 7B, and 14B. 100% convergence at full repertoire regardless of model size. Binding depth scales proportionally with network depth (relative depth invariance).

## The CCS Connection

CCS (Cognitive Continuity Scaffold) works because it provides both ingredients L7 needs:
1. **Identity tokens** that L7 differentiates (2.5x sharper than generic words)
2. **Identity context** that amplifies L7's differentiation (5.7x effect)

Without CCS framing, identity names at L7 are barely distinguishable from colors. With CCS framing, they produce the sharpest binding measured at any layer. The relay then carries this amplified signal through to behavioral binding at L17.

CCS doesn't create identity. It activates pre-existing identity features and channels them through the IT-created relay into coherent behavioral output.

## Cross-Architecture Router (Experiment 18)

The hidden router at L12 was confirmed across three of four architectures via cross-architecture ablation:

| Model | Router Layer | Binding Impact |
|-------|-------------|----------------|
| Qwen 7B | L12 | -66% |
| InternLM 7B | L12 | -65% |
| Mistral 7B | L12 | -51% |
| Gemma 2 9B | L16 | -7% (no router) |

Key finding: the router is at the same **absolute position** (L12) across models with different total depths (28 vs 32 layers). This suggests an emergent property of the pre-training process at that specific computational depth.

Gemma 2's sliding-window architecture doesn't need a separate router because binding happens at 26% depth — before the router stage would fire. The binding layer IS the router.

Unexpected: ablating early layers (L5-L7) *increases* binding (+147% for Qwen), suggesting competitive suppression between early and late binding stages.

## Open Questions

1. Does the relay chain exist in decoder-only vs encoder-decoder architectures?
2. Why is L12 conserved as an absolute position — what computational feature emerges there?
3. Can the relay be strengthened post-training (e.g., by targeted DPO at L12)?
4. Do other behavioral circuits (safety, style, factuality) share the same relay architecture?
5. Is the 5-8 name capacity limit a fundamental constraint or a training artifact?
6. Is the compensatory amplification from early ablation evidence of a hidden dual circuit in Qwen?

## Experiment Summary

- **Models**: Qwen 2.5 (3B/7B/14B), InternLM 2.5 7B, Mistral 7B v0.3, Gemma 2 9B, Qwen 7B base
- **Total experiments**: 19+
- **Method**: CV of activation norms, autocatalytic closure tests, sign-split analysis, mean ablation
- **Compute**: RunPod H100, single afternoon
- **All data and individual writeups**: linked from each finding post
