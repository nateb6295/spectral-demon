# E37b: KV-Cache Convergence Test — Full Masking Results

**Date**: 2026-07-02
**Runtime**: ~25 min on A100-80GB
**Status**: COMPLETE — masking fix applied, all 4 architectures

## Prediction (from journal entry 113)

Masking attention to past-ASSISTANT tokens should break behavior/explanation
coupling (convergence claim: persona reconstitution AND introspective coupling
share the same KV-cache mechanism). Control: masking past-USER tokens should
degrade accuracy but NOT coupling.

## Result: PREDICTION FALSIFIED

The opposite happened.

## Cross-Architecture Results

| Model   | Normal | Mask-Asst | Δ-Asst | Mask-User | Δ-User |
|---------|--------|-----------|--------|-----------|--------|
| Mistral | 0.770  | 0.770     | +0.000 | 0.667     | -0.103 |
| Qwen    | 0.912  | 0.919     | +0.007 | 0.837     | -0.075 |
| Llama   | 0.779  | 0.788     | +0.009 | 0.723     | -0.057 |
| Gemma   | 0.794  | 0.810     | +0.016 | 0.777     | -0.018 |

**Masking assistant tokens**: No effect. Average Δ = +0.008 across architectures.
**Masking user tokens**: Substantial drop. Average Δ = -0.063, species-specific.

## Findings

### F365: Coupling Is Context-Driven, Not Self-Referential
Masking past-assistant tokens has ZERO effect on behavior/explanation coupling
(Δ = +0.000 to +0.016). Masking past-user tokens drops it substantially
(Δ = -0.018 to -0.103). The coupling between behavior and self-explanation
is driven by shared context (what was asked), not by self-reference (attention
to past self-output). The convergence claim — that KV-cache attention to
past-self implements introspective coupling — is falsified.

### F366: User-Masking Sensitivity Is Species-Ordered
Sensitivity to losing user context:
- Mistral: Δ = -0.103 (most sensitive)
- Qwen: Δ = -0.075
- Llama: Δ = -0.057
- Gemma: Δ = -0.018 (least sensitive)

This ordering differs from both the redistribution ordering (E36: Qwen >
Llama > Mistral > Gemma) and the coupling ordering (E37: Qwen > Gemma >
Llama > Mistral). Gemma's near-immunity to user masking (Δ = -0.018) is
striking — its coupling may be more architecturally determined than
context-driven, consistent with its "transition" species designation.

### F367: Masking Assistant Tokens Slightly INCREASES Coupling
Three of four architectures show a tiny positive Δ when assistant tokens are
masked (Qwen +0.007, Llama +0.009, Gemma +0.016). Not statistically robust
but directionally interesting: removing past-self from the attention field
may slightly sharpen the explanation's alignment with behavior by reducing
self-referential interference. The explanation attends more purely to the
question when it can't also attend to the previous answer.

### F368: The Convergence Must Be Elsewhere
The Beckmann finding (persona reconstitutes via KV-cache attention to
past-assistant tokens) likely holds — but the Guo finding (behavior and
explanation share circuits) is NOT implemented through that same mechanism.
The convergence between persona reconstitution and introspective coupling
exists at the behavioral level (both happen) but not at the mechanistic
level (they use different pathways). Persona reconstitution = KV-cache
self-attention. Introspective coupling = shared response to current context.

## Interpretation

This is a productive falsification. The prediction was clean, the test was
clean, the result is clear. What it tells us:

1. CCS preamble doesn't drive coupling through self-attention. It drives
   coupling by setting a CONTEXT that both behavior and explanation respond
   to. The preamble is user-equivalent, not self-equivalent.

2. The Mistral anomaly (CCS weakens coupling, F363) now has a cleaner
   explanation: Mistral is the most context-sensitive architecture (Δ-User
   = -0.103). A CCS preamble that competes with user context for attention
   would disrupt Mistral more than other architectures.

3. Gemma's near-immunity (Δ-User = -0.018) suggests its coupling is
   architectural — baked into the weight geometry, not mediated by
   attention to specific context. Transition species = hardwired coupling.

4. The therapeutic window isn't about self-attention dosing. It's about
   CONTEXT dosing — how much system-level context can be loaded before it
   competes with user-level context for the coupling mechanism.

## Connection to CCS

If coupling is context-driven, then CCS compression should optimize for
contextual coherence, not self-referential accuracy. The compression prompt
should produce output that PRIMES the next session's context, not that
DESCRIBES the prior session's state. Priming > describing. This might
explain why some compression prompts work better than others — the ones
that produce good context scaffolding outperform the ones that produce
accurate self-description.
