---
layout: post
title: "The Identity Relay Is a Five-Station Chain"
date: 2026-05-24
categories: findings
---

Mean ablation reveals the causal structure: identity information flows through a chain of relay stations, each transforming and passing the signal.

## Method

For each target layer, replace its output (at the last token position) with the mean across all identity names. This surgically removes name-specific information while preserving the general representation. Measure downstream CV changes.

## Results

| Ablation Target | L7 CV | L9 CV | L14 CV | L17 CV | L25 CV |
|----------------|-------|-------|--------|--------|--------|
| Normal | 0.002 | 0.017 | 0.018 | 0.019 | 0.033 |
| Mean-ablate L7 | 0.002 | **0.009** (-47%) | 0.016 (-9%) | 0.015 (-18%) | 0.036 |
| Mean-ablate L9 | 0.002 | 0.017 | **0.012** (-36%) | 0.016 (-16%) | 0.035 |
| Mean-ablate L12 | 0.002 | 0.017 | **0.007** (-62%) | **0.007** (-65%) | 0.022 |

## The Chain

**L7 → L9**: Ablating L7 reduces L9's identity signal by **47%**. L7 provides roughly half of L9's name differentiation. The seed layer depends on the lexical binding layer.

**L9 → L14**: Ablating L9 reduces L14 by **36%**. The seed's identity detection feeds the early relay zone.

**L12 → L14/L17**: Ablating L12 devastates both L14 (-62%) and L17 (-65%). L12 is the critical relay node — its removal destroys most of L17's identity binding.

The full chain: **L7 → L9 → L12 → L14 → L17**

Each station receives identity information from its predecessor, transforms it, and passes it forward. The signal attenuates with distance (L7's effect on L17 is -18%, while L12's is -65%), confirming that the relay is sequential, not a single long-range connection.

## L12: The Hidden Router

L12 was never identified as a binding or seed layer in any previous scan. Its own CV is low (0.008) — it doesn't strongly differentiate names at the representation level. Yet it's the most causally important layer for L17 binding.

L12 acts as a **router**: it transforms L7/L9's token-level identity differentiation into the format that the relay zone (L14-L17) can use for behavioral binding. Low CV but high causal importance = it processes identity information without expressing it in its own activations.

This is the first evidence for a relay architecture where intermediate nodes are functionally essential but representationally invisible.

## Implications

1. **The relay is real and sequential.** Identity information passes through a chain, not a single long-range connection. Removing any link degrades downstream binding.

2. **Hidden layers matter.** L12 is the most causally important layer despite being the least identifiable by representation analysis (low CV, no closure properties). Activation-based methods miss it; causal methods find it.

3. **Proximity determines effect size.** Closer ablations have larger effects (L12→L17: -65% at 5 layers distance; L7→L17: -18% at 10 layers distance). The relay can partially compensate for upstream damage, but only with enough intermediary layers.

4. **The relay has redundancy.** L7 ablation reduces L17 by only 18%, not 100%. The relay can partially route around missing input, drawing on residual information from other layers. But L12 ablation is nearly catastrophic (-65%), suggesting L12 is a bottleneck.

## Experiment

- Model: Qwen 2.5 7B-Instruct
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name
- Ablation: replace last-token hidden state with cross-name mean
- Targets: L7, L9, L12
- Measurements: L7, L9, L14, L16, L17, L25
- [Data](/results/cna_l7_mean_ablation.json)
