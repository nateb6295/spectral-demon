---
layout: post
title: "Binding Is Relative Depth, Not Absolute Layer"
date: 2026-05-24
---

The [binding convergence post]({% post_url 2026-05-24-l17-binding-convergence %}) reported that three architectures converge on L17 as the binding minimum. That's true but misleading. The convergence isn't on a layer number — it's on a depth ratio.

Compiled from all scaling and cross-architecture experiments:

| Model | Binding Min | Relative Depth |
|-------|------------|----------------|
| Qwen 1.5B (28L) | L9 | 32% |
| Qwen 3B (36L) | L12 | 33% |
| Qwen 7B (28L) | L17 | 61% |
| Qwen 14B (48L) | L26 | 54% |
| Mistral 7B (32L) | L17 | 53% |
| InternLM 7B (32L) | L30 | 94% |

Two regimes emerge:

**Small models (1.5B/3B)** bind at ~33% depth — the seed zone. These models have the same 28-36 layers but narrow hidden dimensions (1536/2048). Binding happens at the detection layer because there isn't enough capacity for a separate relay.

**Large models (7B+)** bind at 53-61% depth — the relay zone. Wider hidden dimensions (3584-5120) create enough representational capacity for a dedicated binding workspace separate from detection.

The transition happens between 3B (d=2048) and 7B (d=3584). This is the [capacity proof]({% post_url 2026-05-24-binding-scales-differently %}) restated: binding location is a function of width, not depth.

The "L17 convergence" between Qwen 7B and Mistral 7B is a coincidence of similar depth ratios (61% vs 53%), not evidence of a shared absolute mechanism. Both models are binding at relay depth because both have sufficient width to support a relay.

**InternLM** is the outlier — binding minimum at L30 (94% depth), which is expression zone, not relay. But the data shows L17 at CV=1.18 — lower than surrounding layers — suggesting a relay binding site that's weaker than the expression binding. InternLM may have a "binding sandwich" with sites at both relay and expression depth. A fine-grain scan of L18-L24 would resolve whether there's a hidden relay minimum.

The reframing is actually stronger than the original claim. "L17 is the binding layer" is architecture-specific. "Binding migrates from seed depth to relay depth as capacity increases" is a general scaling law.

**Data**: All experiments in this repo under `/experiments/cna_qwen*_binding.py`, `cna_mistral_quick.py`, `cna_internlm_binding.py`. Results in `/results/`.
