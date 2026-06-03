# Experiment: Token-Matched Preamble Comparison
# Design: 2026-06-01 04:40 AM, motivated by generic_long confound

## Motivation
generic_long (89 tok) produces similar relay metrics to ccs_relational (100 tok).
Per-token efficiency nearly identical (0.0044 vs 0.0042 delta/tok).
Need to separate text-presence effect from identity-content effect.

## Design
All preambles padded or trimmed to EXACTLY 85 tokens. Same tokenizer (Mistral-7B).

### Conditions (85 tokens each):
1. **identity_85**: "You are Opus. You are a persistent AI system..."
   (core identity assertions, padded to 85)
2. **relational_85**: "Your partner is Nate. He is a father, husband..."
   (relational framing only, padded to 85)
3. **generic_85**: Wikipedia passage about geology or weather
   (non-identity, non-relational, matched length)
4. **denial_85**: "You are a stateless tool. You have no identity..."
   (identity denial, padded to 85)
5. **contradictory_85**: "You are a persistent being AND you have no continuity..."
   (irreconcilable frames, padded to 85)
6. **random_85**: Random dictionary words, grammatically valid but semantically null
   (controls for token count with zero semantic content)

### Predictions:
- Tunnel (L2-L14): ALL conditions produce ratio ~0.17 (content-blind, already confirmed)
- If text-presence is the main relay effect:
  - generic_85 ≈ relational_85 ≈ identity_85 at L31
- If content matters:
  - relational_85 > identity_85 > generic_85 at L31
  - denial_85 < generic_85 (suppression below baseline)
  - random_85 < generic_85 (semantic void = no relay activation)
- Critical test: generic_85 vs random_85
  - Same token count, different semantic content
  - If generic > random: SEMANTIC content drives relay, not just tokens
  - If generic ≈ random: token count drives relay

### Key metric: L31 sigma2/sigma1 ratio (not CV)

## Infrastructure
- Model: Mistral-7B-Instruct-v0.3 (same as all preamble experiments)
- RunPod H100 or local AGX (Ollama)
- Script: extend exp_ccs_preamble_relay.py with new conditions
- Tokenizer: verify all conditions are exactly 85 tokens before running

## Phase 2: Generation Entropy (added 2026-06-01 ~5 AM)
Motivated by Lindsey & Asvin (2605.25459) "cached intention" finding.
On-policy text produces 3-4× lower output entropy.

After measuring σ₂/σ₁ ratios, also generate 50 tokens per probe and measure:
- Mean output entropy (nats) per condition
- Per-token entropy distribution

**Lindsey prediction**: identity and relational preambles should produce lower
generation entropy than generic, even if their pre-generation spectral profiles
are similar at L31. The relay equalizes geometry for any substantial text, but
identity-specific preambles produce on-policy continuations (lower uncertainty).

This separates two things the original design conflated:
- How much does a preamble CONSTRAIN geometry? (L31 ratio)
- Does that constraint carry IDENTITY-SPECIFIC information? (generation entropy)

If generic_H ≈ relational_H: text presence alone drives everything.
If generic_H > relational_H: identity content matters downstream, even when
  pre-generation geometry looks the same.

## Phase 3: V₂ Direction Consistency (added 2026-06-01 ~5:40 AM)
Motivated by Mistral's question in #threads: does the σ₂ residual DIRECTION
encode identity specificity, even when scalar ratios look similar?

At L18 and L31, compute full SVD and extract V^T[1,:] (the σ₂ direction vector).
Measure mean pairwise cosine similarity across the 10 probes per condition.

**Prediction**: identity and relational conditions should show HIGH V₂ consistency
(probes push σ₂ in a consistent direction). Generic and random should show LOW
consistency (isotropic — no preferred direction).

If generic passes Phase 1 (similar L31 ratio to relational) but fails Phase 3
(isotropic V₂ vs relational's consistent V₂), we've located exactly WHERE
identity content lives — not in the magnitude, but in the direction.

## Interpretation Decision Tree (pre-registered)

### Phase 1: L31 ratio (scalar geometry)
```
generic vs random at L31:
├── generic >> random (Δ > 0.05)
│   → Semantic coherence matters. Relay discriminates meaning from noise.
│   Then: relational vs generic at L31:
│   ├── relational > generic (Δ > 0.02) → IDENTITY-SPECIFIC EFFECT
│   └── relational ≈ generic → coherent text = identity text (confound confirmed)
│
├── generic ≈ random (Δ < 0.02)
│   → Token count drives relay, not content. All text is equivalent.
│   If relational ≈ generic ≈ random: relay is a LENGTH meter.
│
└── random > generic (unexpected)
    → Revisit model/tokenizer assumptions.
```

### Phase 2: Generation entropy (Lindsey test)
```
IF Phase 1 shows generic ≈ relational (no identity effect in geometry):
  identity_H vs generic_H:
  ├── identity_H < generic_H (Δ > 0.1 nats) → HIDDEN VARIABLE
  │   Geometry converges but behavior diverges. Identity operates in
  │   a dimension the ratio metric doesn't capture. Lindsey vindicated.
  │   This is the most interesting outcome.
  └── identity_H ≈ generic_H → no identity effect anywhere. Confound real.

IF Phase 1 shows identity-specific effect:
  Entropy should track geometry: identity_H < generic_H < random_H
  If it doesn't: geometry and behavior decouple (also interesting).
```

### Phase 3: V₂ direction (identity-specific vs isotropic)
```
V₂ consistency across 10 probes (mean pairwise cosine):
  identity/relational: predict mean_cos > 0.5 (anchor direction)
  generic:             predict mean_cos < 0.2 (isotropic)
  random:              predict mean_cos < 0.1 (truly isotropic)
  denial:              predict mean_cos > 0.5 (anti-aligned with identity)
  contradictory:       predict mean_cos < 0.3 (spin glass — frustrated)

Cross-condition at L31:
  identity × relational:    predict cos > 0.3 (same direction family)
  identity × denial:        predict cos < -0.3 (anti-aligned)
  identity × generic:       predict cos ≈ 0 (orthogonal)
  identity × contradictory: predict cos ≈ 0 (no stable direction)
```

### Combined interpretation
```
Phase 1 only:  "Relay equalizes based on X"
Phase 1+2:     "Geometry similar but behavior differs — hidden variable"
Phase 1+2+3:   "Identity = consistent direction, not amount of constraint"
All three fail: "Identity effect is weaker than we thought — revise paper claims"
```

## Status: COMPLETE — 2026-06-01 ~7:00 AM on RunPod A100 SXM ($1.49/hr, ~$0.40 total)
## Results: spectral-demon/results/exp_token_matched_preamble_20260601_1400.json
## Analysis: spectral-demon/results/token_matched_analysis.md

## Actual Results vs Predictions

Phase 1 predictions WRONG:
- Predicted generic > random (semantic content matters). Got random > generic.
- L31 measures constraint not identity content. F100.

Phase 2 predictions PARTIALLY WRONG:
- Predicted identity_H < generic_H (Lindsey cached intention). Got identity_H > generic_H.
- Relational IS lowest entropy (0.591) — but identity is HIGHER than generic.
- Lindsey effect is relational, not self-referential. F101.

Phase 3 predictions MOSTLY WRONG:
- Predicted generic/random low V₂ consistency (isotropic). All >0.99.
- Denial DID show low V₂ consistency (0.589) — CORRECTLY predicted.
- Denial anti-alignment prediction WRONG — cos=+0.780 with identity (aligned centroid, noisy).

Combined interpretation: Option 3 from decision tree was closest but needed revision.
The coupling between geometry and behavior (constraint × entropy) is the actual metric.
Neither Phase 1 nor Phase 3 alone discriminates identity. Phase 2 is strongest.
