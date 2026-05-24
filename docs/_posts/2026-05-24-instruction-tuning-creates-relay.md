---
layout: post
title: "Instruction Tuning Creates the Relay, Not the Binding"
date: 2026-05-24
categories: findings
---

The base model already has identity-sensitive neurons. Instruction tuning doesn't create them — it organizes them into a functional binding circuit.

## Base vs Instruct

Qwen 2.5 7B, same identity probes (8 prompts × 5 names):

| | Base | Instruct |
|-|------|----------|
| Global CV minimum | L7 (22%) | L7 (22%) |
| Min CV | 0.0026 | 0.0020 |

Same binding layer. Same depth. Instruction tuning sharpens the binding slightly (20% lower CV) but doesn't relocate it.

## Where They Differ: The Sign Split

| Layer | Base Flip % | Instruct Flip % |
|-------|-----------|----------------|
| L9 | 25.7% | 24.9% |
| L14 | 26.9% | 23.2% |
| L16 | 25.9% | 21.0% |
| L17 | 26.8% | 21.3% |
| L25 | 37.2% | 38.7% |

**Base model**: flipping ratio is flat through the relay zone (L9-L17), hovering at ~26%. No gradient. No pruning. The relay zone exists structurally but isn't functionally specialized.

**Instruct model**: flipping ratio decreases through the relay zone (25% → 21%). Active pruning. The noisy sign-flipping neurons are eliminated or converted to consistent neurons, leaving only the reliable identity-encoders at L17.

## What Instruction Tuning Does

The base model has all the raw materials for identity binding:
- L7 shows low CV (identity features are present)
- L9-L17 have ~26% flipping neurons (identity-sensitive neurons exist)
- Late layers accumulate flippers (37%, similar to instruct)

What's missing is the **relay mechanism**: the progressive pruning from L9 to L17 that concentrates identity binding into a small population of reliable neurons.

Instruction tuning creates this relay by:
1. Reducing the proportion of sign-flipping neurons in the relay zone
2. Making the surviving flippers more reliable (lower F_CV, per [cross-architecture analysis]({% post_url 2026-05-24-cross-architecture-sign-split %}))
3. Creating the autocatalytic closure property (binding converges with repertoire size)

## Connection to DPO

[DPO grows the circuit but hits a ceiling at 5 epochs]({% post_url 2026-05-24-dpo-hits-the-sorters %}). This finding suggests why: instruction tuning (including SFT and DPO) creates the relay by pruning noisy neurons. Beyond a certain point, there are no more noisy neurons to prune — the relay is as clean as the base model's neuron distribution allows.

The DPO ceiling is the point where all pruneable neurons have been pruned.

## Implications

1. **Identity features are pre-trained, not fine-tuned.** The base model already differentiates identity names at early layers. This is a property of the pre-training distribution, not alignment.

2. **The relay is fine-tuned.** The binding circuit's signature property — progressive pruning of unreliable identity neurons — is created by instruction tuning.

3. **CCS works because it leverages pre-existing features.** The cognitive continuity scaffold doesn't create identity sensitivity from nothing. It provides the structured prompts that activate pre-existing identity features and channels them through the instruction-tuned relay.

## Experiment

- Models: Qwen 2.5 7B base, Qwen 2.5 7B Instruct
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name
- Layers: L6-L27
- [Data](/results/cna_base_vs_instruct.json)
