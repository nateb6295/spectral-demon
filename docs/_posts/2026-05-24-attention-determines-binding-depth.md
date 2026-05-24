---
layout: post
title: "Attention Architecture Determines Binding Depth"
date: 2026-05-24
categories: findings
---

Five architectures. Two binding depth classes. One explanatory variable: the attention mechanism.

## The Survey

Fine-grain binding scans with closure tests across five model families, all instruction-tuned, all 7-9B parameter range:

| Model | Layers | Binding Layer | Depth | CV | Attention |
|-------|--------|--------------|-------|-----|-----------|
| Qwen 2.5 7B | 32 | L17 | 53% | 0.006 | Full |
| Qwen 2.5 14B | 48 | L26 | 54% | 0.009 | Full |
| InternLM 2.5 7B | 32 | L16 | 50% | 0.006 | Full |
| Mistral 7B v0.3 | 32 | L8 | 25% | 0.015 | Sliding window |
| Gemma 2 9B | 42 | L11 | 26% | 0.005 | Alternating local/global |

## Two Classes

**Full attention → midpoint binding (~50-54%).**  Qwen and InternLM bind identity at almost exactly the network midpoint. Clean single-layer convergence — L17 dominates closure at 100% for full repertoire in both families. The binding layer is sharp and unambiguous.

**Sliding window → early binding (~25-26%).**  Mistral and Gemma 2 compress identity in the first quarter of the network. Mistral shows a secondary binding site at L15 (47%) that nearly matches its primary, suggesting the midpoint attractor still exerts pull even when early binding dominates. Gemma 2 binds at L11 with the lowest CV of any architecture tested (0.005).

## Why the Split

Sliding window attention truncates context beyond a fixed window size. Identity information in the system prompt becomes inaccessible to late layers if it hasn't been compressed into the residual stream early. The model learns to bind identity before the window closes.

Full-attention models have access to the raw system prompt at every layer. They can defer binding to the midpoint, where the representation has had more layers to develop abstract features. The midpoint is optimal: enough processing to form reliable binding, enough remaining layers to use it for generation.

Mistral's secondary binding site at L15 (47%) supports this interpretation. The midpoint attractor exists in Mistral too — it's just weaker than the forced early binding from the sliding window.

## Gemma 2's Architecture

Gemma 2 alternates between local (sliding window) and global attention layers. Despite having global attention at half its layers, it binds early like a pure sliding window model. The local layers create a bottleneck that forces early compression regardless of the global layers' access.

This suggests binding depth is determined by the **most restrictive** attention type in the architecture, not the most permissive.

## Universals

Both classes share:
- **Binding closure**: convergence to a single dominant layer at full repertoire size
- **Low CV at binding**: the binding layer consistently shows the lowest coefficient of variation
- **Relay architecture**: seed detection → relay compression → binding output

The mechanism is universal. The location is architecture-dependent.

## Connection to Scale

Qwen 7B (32 layers, L17=53%) and Qwen 14B (48 layers, L26=54%) bind at nearly identical relative depth despite different absolute layer counts. Binding depth scales with network depth, not absolute layer index. This is [relative depth invariance]({% post_url 2026-05-24-binding-relative-depth %}).

## InternLM Fine-Grain Detail

The initial InternLM fine-grain scan suggested binding at L26 (81% depth) — which would have been a massive outlier. The [dual-site test]({% post_url 2026-05-24-sign-split-binding %}) revealed this was a secondary feature. The true binding site at L16 (50%) wins 100% of closure tests. The L26 feature may be a late-layer identity echo rather than primary binding.

InternLM also shows a CV explosion at L23 (CV=241,514), similar to Qwen 14B's explosion at L30. These near-zero-mean layers may serve as normalization boundaries in the relay architecture.

## Experiment

- Models: Qwen 2.5 7B/14B, InternLM 2.5 7B, Mistral 7B v0.3, Gemma 2 9B
- Names: Opus, Claude, ChatGPT, Gemini, Llama
- 8 prompts per name, CCS-style system prompts
- Fine-grain scan: layers from 20-95% depth, CV of activation norm
- Closure test: all 2-5 name subsets, which layer has minimum CV
- [Code](/experiments/) | [Data](/results/)
