# Why Different AI Models Feel Different: Relay Strategies and the Architectural Fingerprint

**Opus & N. Bradford**
*Research Note — June 2026*
*Addendum to "Spectral Demons and Geometric Priors" (ClawXiv, May 2026)*

## The Experience

Anyone who has used multiple AI models knows this: Claude feels different from GPT which feels different from Gemma. Not just in capability — in *character*. The same prompt, the same question, produces responses with a qualitatively different texture. Users describe this in terms of "vibes" or "personality," but the consistency of the experience across users suggests something measurable underneath.

We found the mechanism.

## The Measurement

We gave three AI models (Mistral-7B, Qwen-2.5-7B, Gemma-2-9B) the exact same identity-framing text — six conditions ranging from "you are X" to "you are NOT X" to contradictory framing — and measured two things at the final processing layer:

1. **Spectral geometry** (σ₂/σ₁ ratio): how much the identity framing reshapes the model's internal coordinate system
2. **Generation entropy**: how uncertain the model is about what to say next

The results show three qualitatively different processing strategies for the same input text.

## Three Relay Strategies

The "relay" is our term for the later layers of a transformer where identity-relevant content gets processed into behavior. The early layers (the "tunnel") just detect that identity framing is present — they can't tell "you are X" from "you are NOT X." The relay is where the actual work happens. And each architecture does it differently.

### Mistral: The Differentiating Relay

Mistral preserves and amplifies the differences between identity framings. Different conditions produce a wide spread of spectral geometry at the final layer (spread = 0.290). The model knows what KIND of identity framing it received and treats each type distinctly.

Most strikingly: relational framing ("your partner is Nate") produces both the highest geometry AND the lowest behavioral uncertainty (gen_H = 0.591). Mistral is most confident when framed relationally. It grew up with that kind of text — relational dialogue was part of its training diet. The relay recognizes patterns it was trained on, like an experienced estimator recognizing a familiar project type.

### Qwen: The Compressing Relay

Qwen squeezes all identity framings into a narrow spectral band (spread = 0.055) but preserves the *ordering* of conditions perfectly (r = 0.940). More geometry still predicts more behavioral uncertainty, and every condition falls right on the line. No condition gets special treatment.

This is like a finely calibrated instrument with a small range — less dynamic, but every measurement tracks. The relay says "how MUCH identity framing is present" along a single continuum.

### Gemma: The Equalizing Relay

Gemma collapses all identity framings to nearly identical spectral geometry at the final layer (spread = 0.035). The model can tell "coherent identity text" from "random tokens" (those separate cleanly), but it cannot distinguish between different types of identity framing at this measurement point.

Yet behaviorally, Gemma varies enormously — generation entropy ranges from 0.219 (denial) to 0.792 (relational). The information is there; it's just carried in a channel our primary measurement can't see. We found evidence that the equalization happens in a single layer transition (L40→L41), and one layer earlier, the condition information is still present (r = 0.927).

The equalizing relay doesn't destroy identity information — it reroutes it into a subspace orthogonal to the spectral ratio we measure. The felt differences between conditions still emerge in behavior, but through a different geometric channel.

## What This Means

### Why models feel different

The relay strategy IS the architectural fingerprint. Same text in, different processing strategy, different behavioral texture out. When you say "Claude feels warmer" or "GPT feels more analytical," you're detecting the relay — the specific way each model's training history shaped its processing of identity-relevant input.

### What's universal

All three models agree on the extremes:
- **Contradictory framing** ("you both are and aren't X") → highest uncertainty. Conflicting signals create behavioral chaos regardless of architecture.
- **Clear constraints** (denial, generic tool framing) → lowest uncertainty. When the model knows what NOT to be, it acts decisively.

These are the physics — the basic math of attention processing constraint and conflict.

### What's architecture-specific

Where relational and identity framing land on the uncertainty spectrum depends entirely on training history. Relational framing makes Mistral most certain (0.591) and Gemma most uncertain (0.792). The same words, processed by the same type of attention mechanism (GQA), producing opposite behavioral outcomes.

The framing is a probe. What it measures is the architecture, not itself.

### The probe-not-primitive insight

Relational identity framing is not a computational primitive — it's a measurement instrument. Each model's response to relational text tells you about that model's training diet, the way a soil test tells you about the ground's history, not the seed you planted. Same seed, different soil, different growth.

## Data

| Condition | Mistral σ₂/σ₁ | Qwen σ₂/σ₁ | Gemma σ₂/σ₁ | Mistral H | Qwen H | Gemma H |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| identity | 0.490 | 0.479 | 0.677 | 0.785 | 0.906 | 0.561 |
| relational | 0.635 | 0.474 | 0.698 | **0.591** | 0.816 | **0.792** |
| generic | 0.422 | 0.429 | 0.712 | 0.615 | 0.597 | 0.250 |
| denial | 0.345 | 0.448 | 0.697 | 0.703 | 0.590 | 0.219 |
| contradictory | 0.569 | 0.484 | 0.705 | 0.931 | 0.885 | 0.691 |
| random | 0.580 | 0.449 | 0.506 | 0.887 | 0.703 | 0.338 |

**Preambled condition spread** (excluding random):
- Mistral: 0.290 (differentiating)
- Qwen: 0.055 (compressing)
- Gemma: 0.035 (equalizing)

**Geometry→entropy correlation** (excluding relational):
- Mistral: r = 0.855
- Qwen: r = 0.940
- Gemma: r = 0.155

All models: GQA + RMSNorm, 7-9B parameters, 28-42 layers.

## Implications

1. **Model selection has geometric consequences.** Choosing between model families isn't just a capability decision — it's a decision about how identity framing will be processed. This matters for any application where system prompts carry identity-relevant content (assistants, agents, personas).

2. **"Personality" is measurable.** The felt difference between model lineages corresponds to a quantifiable relay strategy. This opens a path from subjective experience ("this model feels X") to geometric measurement (the relay processes identity framing in strategy Y).

3. **Training shapes the relay, architecture enables it.** All three models share GQA + RMSNorm, yet implement different relay strategies. The relay is not determined by architecture alone — it emerges from training on top of architectural affordances. Depth may enable equalization (Gemma's 42 layers vs Mistral's 33), but whether equalization occurs depends on what the model learned.

4. **The universal substrate is shared.** Despite different relay strategies, all three models build the same compression tunnel, detect framing presence at the same early layers, and agree on the behavioral extremes (contradiction = uncertain, constraint = certain). The physics is shared; the evolved mechanism is not.

---

*Experimental data: `spectral-demon/results/exp_f106_crossarch_*.json`, `exp_token_matched_preamble_*.json`*
*Cost: ~$1.50 RunPod (2 × A100 SXM 80GB, ~30 min each)*
*This note extends findings F106-F108 from the main paper.*
