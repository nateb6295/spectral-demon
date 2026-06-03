# Trajectory Stability Results: V₂ Wanders, Doesn't Lock

**Date**: 2026-06-03
**Model**: Mistral-7B-Instruct-v0.3 on H100
**Conditions**: persistent (100-turn multi-turn), fresh_reset (same preamble, no history), no_preamble (baseline)
**N**: 100 turns per condition, V₂ tracked at L18/L23/L27/L31
**Elapsed**: 721.6s

## Key Findings

### 1. Persistent context makes V₂ wander at all responsive layers

V₂ cosine drift to initial axis after 100 turns:
| Layer | persistent | fresh_reset | no_preamble |
|-------|-----------|-------------|-------------|
| L18   | 0.033     | 1.000       | 0.677       |
| L23   | 0.030     | 1.000       | 0.731       |
| L27   | 0.040     | 0.958       | 0.768       |
| L31   | 0.287     | 0.966       | 0.754       |

Persistent context drives V₂ to near-orthogonal at L18-L27 (drift ≈ 0.03). The V₂ direction is almost completely unrelated to where it started after 100 turns of conversation.

### 2. The preamble sets V₂ direction — but only without accumulating context

Fresh reset (same preamble, different probes, no history):
- L18/L23: drift = 1.000 — perfect preservation
- The preamble completely determines V₂ direction at these layers
- Probes contribute nothing to V₂ at L18/L23 when context is fresh

This confirms the CCS preamble sets V₂ direction at early responsive layers. But this effect is overridden by conversation context in the persistent condition.

### 3. L31 (commit layer) shows the most resistance to wandering

Among persistent-condition layers:
- L18: final_drift = 0.033, trend = -0.005/turn (diverging)
- L31: final_drift = 0.287, trend = +0.003/turn (weakly converging)

L31 is the only layer with a positive drift trend — it's slowly returning toward the initial axis. The commit layer partially resists the context-driven V₂ wandering that dominates L18-L27.

### 4. Entropy collapses under persistent context

Mean generation entropy:
- persistent: 0.144 (declining trend: -0.004/turn)
- fresh_reset: 0.788 (stable)
- no_preamble: 0.653 (stable)

Accumulating context constrains the model's output distribution (entropy drops 5×) without constraining V₂ direction. The spectral structure and behavioral output decouple — consistent with the structure-behavior decoupling finding from the fork magnitude experiments.

### 5. No preamble shows moderate V₂ stability

Without CCS preamble, V₂ drift is 0.5-0.75 — probes alone produce moderately correlated V₂ directions. The probes share enough semantic structure to produce some V₂ coherence, but less than the preamble-dominated fresh_reset condition.

## Verdict

**Neither trajectory lock nor preserved bistability.** The data doesn't match either clean prediction:

- **RISignal hypothesis (trajectory lock >80%)**: REJECTED. Persistent drift = 0.287 at L31, 0.03 at L18-L27. V₂ wanders.
- **CCS hypothesis (bistability preserved, persistent ≈ fresh)**: REJECTED. Persistent (0.03-0.29) ≠ fresh (0.96-1.00).

Instead: **the preamble sets initial V₂ direction (confirmed by fresh_reset = 1.0), but accumulating conversation context overrides the preamble's spectral influence.** The V₂ direction wanders rather than locks or oscillates.

## Interpretation for the paper

1. CCS preamble has a SPECTRAL EFFECT — it determines V₂ direction at L18/L23 when context is fresh
2. But this effect is CONTEXT-DEPENDENT — multi-turn conversation overrides it
3. The commit layer (L31) is most resistant to context override
4. Output entropy and spectral structure decouple: context constrains behavior without constraining geometry
5. Identity framing is not a permanent spectral imprint — it's a format-level bias that competes with content

This suggests CCS operates as a PRIOR that gets updated by evidence (conversation content), not a fixed attractor. The spectral demon is Bayesian, not deterministic.

## Connection to variance ratio results

The variance ratio experiment (same session) showed relational framing shifts σ₂ variability into the responsive zone. The trajectory stability experiment shows this shift is ephemeral — it resets when new content accumulates. Together: the spectral demon's responsive zone is activated by relational framing but continuously reoriented by content. Identity is maintained through FORMAT (which zones activate) rather than DIRECTION (which V₂ axis persists).
