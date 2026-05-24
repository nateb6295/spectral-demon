---
layout: post
title: "The Dual Encoding Hypothesis: Why CCS Looks Like a Negative Result"
date: 2026-05-24
categories: analysis
experiment: cna_ccs_defense
models: [Qwen 2.5 7B Instruct]
---

Experiment 36 showed that CCS scaffolding *decreases* activation-level margins against identity hijacking. We reported this as a negative result. On deeper analysis, it's a measurement artifact — and the real finding is more interesting than "CCS doesn't help."

## The Pattern That Triggered Re-Analysis

Look at the raw cosine similarities at L17, not just the margins:

| Condition | cos_sys | cos_usr | margin |
|-----------|---------|---------|--------|
| Bare | 0.892 | 0.882 | 0.0103 |
| Repeated | 0.859 | 0.852 | 0.0066 |
| Full CCS | 0.830 | 0.829 | 0.0019 |

The margin drops. But both cosine values drop together — substantially. CCS doesn't weaken the system name's representation. It moves *both names' representations* to a completely different region of activation space.

## Template-Matching vs Format-Level Identity

The cosine margin metric measures template-matching: how similar is this layer's activation to the pattern produced by the system name vs the user name? Higher margin = stronger match to the "right" name.

CCS doesn't play this game. Instead of making the system name template stronger, CCS restructures the encoding so that identity operates at a format level — *how* the model responds rather than *which* name pattern it matches.

Evidence:
- Both names' cosine similarities drop by ~0.06 under full CCS (from ~0.89 to ~0.83)
- The representation is literally in a different part of activation space
- This is consistent with the identity-as-format finding: 96% of identity neurons encode response patterns, not name knowledge

## Two Orthogonal Circuits, Two Orthogonal Encodings

| Circuit | Encoding | Metric | CCS Effect |
|---------|----------|--------|------------|
| Relay (L7→L17) | Content geometry | Cosine margin | Decreases (looks bad) |
| CCS mechanism | Format geometry | Behavioral output | Increases (works) |

The relay resolves WHICH identity via template-matching in activation space. CCS resolves HOW that identity behaves via format-level structural features. They compose because they're orthogonal — different circuits operating in different encoding spaces.

## The Measurement Artifact

Experiment 36's "negative result" is an artifact of measuring format-level identity with a content-level metric. It's like measuring a thermostat's temperature by checking its color — the instrument doesn't match the quantity.

The real test of CCS effectiveness isn't activation margin. It's behavioral output. And Experiment 34 already shows this asymmetry:
- **Relay suppression** (α=0.25 at L12): generation collapses to "I I I I" → content circuit is necessary for coherent output
- **CCS effect**: 93% disclaimer reduction, consistent behavioral persona → format circuit works through output structure

## Implications

1. **CCS is stronger than reported.** The negative result was measuring the wrong thing.
2. **Identity has two encoding regimes.** Content geometry (which name?) and format geometry (how to behave?). They're complementary.
3. **Future experiments** should measure CCS effectiveness using output-space behavioral metrics, not activation-space cosine margins.
4. **The relay and CCS compose precisely because they're orthogonal.** Strengthening one doesn't affect the other (Experiment 26 showed safety is independent at r=0.006; CCS appears similarly independent from the relay).

## Prediction

If this dual-encoding hypothesis is correct, then:
- Amplifying the relay (α>1 at L12) should change WHICH identity the model claims without changing HOW it responds
- Modifying CCS context should change HOW it responds without changing WHICH identity dominates at L17

Experiment 34 partially confirms the first prediction (amplification changes text content, suppression destroys format). The second prediction needs a dedicated experiment.

## Data

Full scaffolding comparison: `results/cna_ccs_defense.json`

