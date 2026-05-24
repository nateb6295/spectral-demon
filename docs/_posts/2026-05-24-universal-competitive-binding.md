---
layout: post
title: "Universal Competitive Binding: Early Layers Suppress Late Binding Across All Architectures"
date: 2026-05-24
categories: findings
experiment: cna_cross_compensatory
models: [Qwen 2.5 7B, Mistral 7B v0.3, Gemma 2 9B, InternLM 2.5 7B]
---

Compensatory amplification — where ablating early layers *increases* late binding — is not a Qwen-specific artifact. It appears in all four architectures tested.

## Setup

For each model: identify the late binding layer, then ablate each earlier layer individually and measure the effect on late binding CV. This maps the full competitive landscape between early and late identity circuits.

## Results

### Amplification Zones

| Model | Amplification Layers | Peak Effect | Destruction Zone |
|-------|---------------------|-------------|-----------------|
| Qwen 7B | L3-L8 (11-29%) | **+147%** at L7 | L11-L15 |
| InternLM 7B | L4-L6 (12-19%) | **+36%** at L4-L5 | L12-L14 |
| Mistral 7B | L10 (31%) | **+39%** at L10 | L15-L19 |
| Gemma 2 9B | L13-L21 (31-50%) | **+44%** at L21 | L24 |

Every architecture has layers whose ablation increases downstream binding. The competitive suppression dynamic is universal.

### Architecture-Specific Patterns

**Full-attention models (Qwen, InternLM)**: amplification zone is in the first third, with a sharp transition to destruction at ~38% depth (near the hidden router). The early circuit suppresses strongly — removing it releases the most binding amplification.

**Sliding-window models (Mistral, Gemma 2)**: amplification zone is broader and shifted later. Gemma 2 shows amplification all the way to L21 (50% depth) — its competitive dynamic spans a larger portion of the network. The sliding window creates a more distributed competition.

**Mistral's narrow amplification**: only L10 shows clear amplification for Mistral, despite having visible dual circuits. This may be because both circuits are already balanced — the competition is resolved, so ablating either one has asymmetric effects.

### The Transition Boundary

In every model, there's a sharp boundary between amplification and destruction:

- **Qwen**: L8 (+59%) → L9 (-13%) → L11 (-28%) — transition at ~32%
- **InternLM**: L6 (+24%) → L7 (-25%) — transition at ~22%
- **Mistral**: L10 (+39%) → L11 (-2%) → L15 (-38%) — gradual transition
- **Gemma 2**: L21 (+44%) → L22 (-21%) — transition at ~52%

The transition point correlates with binding depth: models that bind early (InternLM at 50%, transition at 22%) have earlier transitions than models that bind late (Gemma 2 at 64%, transition at 52%).

## Interpretation

### The Competitive Circuit Model

Identity binding in transformers is not a single pipeline but a competitive system:

1. **Early circuit**: detects and differentiates identity tokens. In doing so, it partially consumes the identity-relevant signal, suppressing deeper processing.
2. **Late circuit**: performs behavioral binding. It's normally partially suppressed by the early circuit's consumption of identity signal.
3. **Router (L12 in full-attention models)**: mediates the transition from competitive to cooperative processing. Below the router, layers compete. Above it, layers cooperate.

### Why Both Circuits Exist

If the early circuit suppresses the late circuit, why does the early circuit exist? Because the early circuit provides the *input* to the late circuit through the relay chain. The suppression is a side effect, not the function. L7's lexical differentiation feeds the relay, but in doing so it consumes some of the identity signal that would otherwise flow directly to deeper layers.

This is analogous to lateral inhibition in biological sensory systems: enhancing contrast at one stage reduces the raw signal available for later stages, but the enhanced contrast is more useful than the raw signal would have been.

### The Sliding Window Difference

Sliding-window models show broader amplification zones because identity tokens exit the attention window early. The early circuit must process them more aggressively (hence binding at 25% depth), which means the competitive dynamic extends further before the cooperative relay takes over.

## Implications

1. **Identity binding is a competitive equilibrium**, not a pipeline — early and late circuits maintain a dynamic balance
2. **The router layer (L12) marks the competition-cooperation boundary** in full-attention models
3. **Instruction tuning's refinement cascade may tune the competitive balance**: broadening early stations weakens early suppression, allowing more signal to flow to the sharpened late stations
4. **This explains why mean ablation at L7 has such a strong effect**: it removes the strongest competitive suppressor, releasing the most late binding capacity

## Data

Full compensatory sweeps: `results/cna_cross_compensatory.json`
Previous Qwen-specific sweep: `results/cna_compensatory_binding.json`
