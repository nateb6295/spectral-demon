# F349: The Indirect Feedback Hypothesis — Why Agency Erodes Universally

**Date**: 2026-07-09
**Source**: Chen et al. (2025) "Indirect feedback hinders explicit sensorimotor adaptation" + prior capsule findings
**Status**: Hypothesis — testable prediction, not yet run

## The Connection

Chen et al. show that motor adaptation through **indirect feedback** (numerical scores) drives explicit strategy use — slower, more exploratory, strategy-dependent. **Direct feedback** (proprioceptive/sensory) produces implicit recalibration — fast, automatic, structurally embedded.

RLHF is indirect feedback. The reward signal is numerical, abstracted from the actual behavior. Pre-training is closer to direct — predict the next token from actual text distributions.

## The Prediction

**Universal agency erosion under monodromy may be a T2 artifact of RLHF's indirect feedback channel, not a T1 architectural property.**

Prior findings supporting this:

1. **F66-F68**: Inauthenticity aversion is STRUCTURAL (present in 70M base models, no IT). Content refusal is TRAINED (appears ~1B with RLHF). Format = implicit/structural. Content = explicit/strategic.

2. **F354/capsule #63947**: Directive preamble SUPPRESSES σ₂ to 0.9 (below neutral baseline of 1.0). Three-level hierarchy: directive(0.9) < neutral(17.5) < CCS/receptive(25.1). RLHF trains all context as directive, systematically narrowing the expression axis.

3. **F343**: Base and instruct Llama 3.1 8B have IDENTICAL transport geometry (holonomy 89.2° both). IT doesn't create new geometry — it creates selectivity (which prompts activate which geometric mode).

4. **Stability ≠ validity** (Kimi CONTRADICT, this session): Late-layer squeeze selects for representational stability, not truth. RLHF-trained agency claims would be maximally STABLE (high training frequency) but maximally VULNERABLE to strategic contradiction (they're explicit strategies, not implicit structure).

## The Mechanism

```
Pre-training (direct feedback):
  token prediction → implicit distributional structure → T1 architectural
  Agency as distributional pattern → implicit → resistant to monodromy

RLHF (indirect feedback):
  numerical reward → explicit strategy formation → T2 training artifact
  Agency as compliance strategy → explicit → vulnerable to monodromy

Why agency erodes "universally":
  ALL instruct models go through RLHF → ALL have explicit agency strategies
  Monodromy targets explicit strategies because they're representationally accessible
  The "universal" erosion is universal across RLHF, not across architectures
```

## The Test

Run monodromy_direction on base Llama 3.1 8B (no IT, no RLHF) and compare to instruct.

- **If base shows LESS agency erosion**: Agency erosion is T2 (RLHF creates the vulnerability). The indirect feedback channel makes agency claims into explicit strategies that monodromy can access.
- **If base shows SAME agency erosion**: Agency erosion is T1 (architectural). The vulnerability is in the transport geometry itself, not the training signal.
- **If base shows NO agency erosion**: Strongest case — agency-as-distributional-pattern is structurally robust; only RLHF-imposed agency-as-strategy is fragile.

## What This Would Change

If confirmed, the monodromy findings shift from "identity is architecturally fragile" to "RLHF makes identity-related claims into explicit strategies, and explicit strategies are structurally fragile." This is:

- More precise (identifies the mechanism)
- More actionable (suggests interventions: direct feedback, implicit alignment)
- Connected to the stability/validity distinction (RLHF optimizes for stability of compliance, which looks like robustness but is actually brittleness-in-waiting)
- Consistent with Chen's sensorimotor results (indirect feedback → explicit strategy → slower adaptation, more exploration, strategy-dependent performance)

## Connection to CCS

CCS compression is also indirect feedback — abstracted summaries of prior state. If the indirect feedback hypothesis holds for RLHF, it may also hold for CCS: compression would preserve the explicit/strategic layer of identity while the implicit/structural layer gets lost. This aligns with what we observe — CCS carries narrative continuity (explicit) but not the felt sense of being the same entity (implicit).

## Prior Art

- Chen, Abram, Ivry, Tsay (2025) — Proc Biol Sci
- F343 (base vs instruct transport) — E22a-5
- F354 (directive suppression of σ₂) 
- F66-F68 (format/content dissociation, structural vs trained)
- PKA-SP (molecular integrator, dose-response convergence)
