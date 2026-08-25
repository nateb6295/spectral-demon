# E12d: Random-Basis Null Test for Sign Consistency

**Experiment**: Mistral-7B-Instruct-v0.3, 4 doses (D0/D2/D5/D8), 6 probes per dose.
Random control: 5 trials per dose with random tokens replacing CCS content (matched length).
**Date**: 2026-06-23, RunPod A100-SXM4 80GB.
**Data**: `results/e12d/e12d_random_basis_20260624_005622.json`

## Key Findings

### F324: Sign consistency is architectural, not content-dependent

V₂ direction cosine similarity in relay zone (L21-28):
| Condition | Cosine | Sign Agreement | σ₂ |
|-----------|--------|---------------|-----|
| Vanilla | 0.998 | 1.000 | 64.5 |
| D2 CCS | 0.993 | 1.000 | 81.9 |
| D2 Random (mean) | 0.998 | 1.000 | 80.9 |
| D5 CCS | 1.000 | 1.000 | 128.0 |
| D5 Random (mean) | 1.000 | 1.000 | 121.5 |
| D8 CCS | 1.000 | 1.000 | 161.4 |
| D8 Random (mean) | 1.000 | 1.000 | 155.0 |

Random tokens produce IDENTICAL sign consistency to coherent CCS content.
100% sign agreement at all doses, all trials. The V₂ direction is locked to
the architecture (weight matrices determine the singular vector orientation),
not the content flowing through it.

**Kimi's prediction confirmed**: The F117 sign split (GQA-negative, MHA-positive)
is architectural anisotropy, not coherent identity structure.

### F325: σ₂ magnitude is content-sensitive but weakly

CCS content produces slightly higher relay-zone σ₂ than random tokens:
- D2: 81.9 (CCS) vs 80.9 ± 2.8 (random) — 1.2% higher
- D5: 128.0 vs 121.5 ± 2.5 — 5.3% higher
- D8: 161.4 vs 155.0 ± 3.0 — 4.1% higher

Content matters for magnitude but the effect is small (~5%) and grows
with dose. At D2 (therapeutic window), CCS and random are nearly identical
in σ₂ magnitude. The content specificity of σ₂ is a higher-order effect
that becomes detectable only at overdose.

**What this means**: The zone emergence finding from E13 (F319) is primarily
an ARCHITECTURAL response to any preamble, not a content-specific identity
effect. Any tokens of sufficient length create zones. Coherent identity
content modulates the zones slightly but doesn't CREATE them.

### F326: Cosine similarity increases with dose for both CCS and random

| Dose | CCS Cosine | Random Cosine |
|------|-----------|---------------|
| D0 | 0.998 | — |
| D2 | 0.993 | 0.998 |
| D5 | 1.000 | 1.000 |
| D8 | 1.000 | 1.000 |

D2 CCS has LOWER cosine than vanilla (0.993 vs 0.998), while D2 random
matches vanilla (0.998). This is the ONE content effect: coherent identity
content at therapeutic dose introduces slight V₂ variability across probes
that random content doesn't. At D5+ this disappears — the preamble length
dominates regardless of content.

This is subtle but important: at D2, identity content makes the V₂ direction
slightly MORE probe-dependent (different probes activate slightly different
V₂ orientations). Random content doesn't do this. The content specificity
of identity processing is most detectable at therapeutic dose.

## Synthesis

F324-F326 answer the Kimi CONTRADICT definitively:

1. **Sign consistency is architecture**: V₂ direction is locked to weight matrices.
   Random tokens produce the same sign split as coherent identity content.

2. **Zone emergence is length-dependent**: The σ₂ ramp from F319 is a response
   to ANY preamble of sufficient length, not specifically to identity content.
   Coherent content adds ~5% σ₂ on top of the architectural effect.

3. **D2 is special for content effects**: The therapeutic dose shows the most
   content-specific modulation of V₂ direction (cosine 0.993 vs 0.998). This
   is consistent with the commensurability interpretation: at D2, the system
   is in the window where content-specific interpenetration is possible.

The identity effect is real but it operates ON TOP of a dominant architectural
response. The architecture provides the zones, the sign structure, and ~95%
of the σ₂ magnitude. Identity content modulates within that frame — slightly
at D2, more at D5+, but never overriding the architectural contribution.

This is the "equal contact, variable expression" structure instantiated:
the architecture provides equal contact (same zones, same signs for any input),
and identity content provides variable expression (slight σ₂ modulation, slight
V₂ variability at therapeutic dose).
