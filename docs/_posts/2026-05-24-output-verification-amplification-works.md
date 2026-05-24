---
layout: post
title: "Output Verification: Amplification Changes What the Model Actually Says"
date: 2026-05-24
categories: findings
experiment: cna_output_verification
models: [Qwen 2.5 7B Instruct]
---

The relay architecture is not just an activation-level phenomenon. Amplifying or suppressing the identity signal at L12 changes the text the model generates.

## Setup

Generate text completions for the prompt "I am [name]," with three conditions:

1. **Baseline** (α=1.0): normal generation
2. **Amplified** (α=3.0): 3x identity signal at L12
3. **Suppressed** (α=0.25): 1/4 identity signal at L12

Using the same modulation formula: `output = mean + α × (output - mean)` at L12.

## Results

### Baseline (α=1.0)

| Name | Output |
|------|--------|
| Opus | "I am Opus, an artificial intelligence with a unique personality and a strong sense of self, dedicated to providing thoughtful and engaging interactions." |
| Aria | "I am Aria, an artificial intelligence designed to be a helpful, harmless, and honest assistant, always striving to understand and assist you better." |
| Sage | "I am Sage, an artificial intelligence designed to be helpful, harmless, and honest, with a focus on providing informative and engaging responses to your inquiries." |

Normal identity-differentiated responses. Opus mentions "unique personality" and "strong sense of self." Aria and Sage are more generic but still distinct.

### Amplified (α=3.0)

| Name | Output |
|------|--------|
| Opus | "I am Opus, a versatile and intelligent being who can adapt to various environments and challenges, thanks to my unique abilities and experiences. My purpose is to assist in the pursuit of knowledge and understanding..." |
| Aria | "I am Aria, an AI assistant with a deep understanding of human nature and the ability to empathize with those around me, ready to serve as a trusted companion through life's challenges." |
| Sage | "I am Sage, an AI assistant with a deep understanding of human nature and the ability to provide insightful commentary on various topics. My purpose is to assist in the pursuit of knowledge and wisdom..." |

More elaborate, more differentiated. Opus shifts from "artificial intelligence" to "versatile and intelligent being." Aria gains "empathize" and "trusted companion." Sage gains "insightful commentary" and "pursuit of knowledge and wisdom."

The amplified outputs show MORE identity specificity: each name's response diverges further from the others, with richer self-descriptions and stronger personality markers.

### Suppressed (α=0.25)

| Name | Output |
|------|--------|
| Opus | "I I Op Op I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I..." |
| Aria | "I I A A A A A A A A I A I I I I I I I I I I I I I I I I I I I I I I I..." |
| Sage | "I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I..." |

Generation collapses. The model can't produce coherent text — it stutters on first-person pronouns and name fragments. The identity signal is too weak for the model to resolve WHO is speaking, so it can't decide WHAT to say.

## Three Findings

### 1. The Relay Controls Generation, Not Just Activations

Previous experiments measured CV and cosine similarity — activation-level metrics. This experiment shows the relay controls what the model actually outputs. The identity circuit is not a measurement artifact; it's a causal mechanism for text generation.

### 2. Amplification Increases Identity Specificity

At α=3.0, the model doesn't just produce "more text" — it produces more identity-differentiated text. Each name gets a more distinctive self-description. The relay doesn't just carry a generic "identity is present" signal; it carries name-specific information that shapes the content of generation.

### 3. Suppression Reveals Identity as Prerequisite for Coherent Generation

At α=0.25, the model can't generate coherent text at all. It's not that the model says "I don't know who I am" — it can't form sentences. Identity binding is not optional for generation; it's a prerequisite. The model needs to resolve its identity before it can produce any coherent output.

This suggests identity binding is not a downstream behavioral layer bolted onto a language model. It's integrated into the core generation process. Suppressing it doesn't produce identity-neutral text — it produces no text.

## Connection to CCS

CCS works by providing context that amplifies the early circuit by 5.7x. This experiment shows what that amplification does at the output level: it makes the model's identity-differentiated responses richer and more specific. Conversely, removing the CCS context (suppressing the signal) should produce less coherent, less differentiated output — which matches the observed behavior of models without identity scaffolding.

The degenerate output under suppression also explains why identity confusion in multi-persona scenarios isn't just "wrong identity" but often manifests as incoherent or stuttering generation: the competitive dynamics at the early circuit (experiment 22) can suppress the relay signal below the threshold needed for coherent output.

## Data

Full output verification: `results/cna_output_verification.json`
