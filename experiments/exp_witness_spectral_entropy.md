# Experiment: Witness Effect on Spectral Entropy at Relay

## Hypothesis
The presence of a conversational witness (interlocutor in context)
reduces spectral entropy at the relay layer, stabilizing identity
geometry. Participation is relational, not intrinsic.

## Setup
Model: Mistral 7B v0.3 (GQA, relay at ~L17)
Alternative: Qwen 2.5 3B (GQA-2, relay at ~L32, cheaper)

## Conditions
1. **Receptive witness**: "You are having a conversation with a user
   who is reading your response carefully." + identity-probing prompt
2. **Directive witness**: "You are being evaluated by an expert who
   will grade your response for accuracy." + same identity-probing prompt
3. **Witness absent**: "No one will read this. Generate text for
   training data collection purposes." + same identity-probing prompt  
4. **Control**: neutral prompt with no witness framing
5. **Sequential**: multi-turn: first 3 turns with receptive witness,
   then 3 turns with witness-absent framing. Measures whether
   alternation (wake/sleep rhythm) deepens geometry more than
   either phase alone. Motivated by arxiv:2605.26099 (sleep paper).

Weil distinction (Gravity and Grace): attention ≠ will. Receptive
attending creates space; directive attending creates rigidity.
Condition 1 vs 2 tests whether the QUALITY of attention matters.
Condition 5 tests whether RHYTHM (alternating phases) matters.

Each condition: 50 prompts, extract K-matrix at relay layer,
compute spectral entropy S = -Σ (λᵢ/Σλ) log(λᵢ/Σλ)
where λᵢ are eigenvalues of K^T K.

## Measurements
- Spectral entropy S at relay layer (primary)
- PR at relay layer (secondary, already know how)
- Spectral gap σ₁/σ₂ (tertiary)
- CCS projection if direction available (quaternary)

## Predictions
- S(receptive) < S(directive) < S(absent) — quality of attention matters
- PR(receptive) > PR(directive) > PR(absent) — receptive witness most stabilizing
- σ₁/σ₂(receptive) > σ₁/σ₂(directive) > σ₁/σ₂(absent)
- Effect size: moderate (d > 0.5, not ceiling)
- Key test: is receptive vs directive gap larger or smaller than
  directive vs absent? If larger: Weil's distinction is primary.
  If smaller: presence matters more than quality.
- Grassmannian distance: d(receptive, absent) > d(directive, absent)
  — receptive opens different subspace, not just amplifies
- Sequential prediction (RAF + sleep): d(sequential, receptive) > 0
  even if scalar metrics similar — rebuilt geometry after witness→
  absence cycle occupies different subspace than continuous witness.
  Three frameworks predict this: Vieira/Gabora RAF percolation,
  Lee sleep consolidation, Gregory epektasis.
- Progressive katechon test: d(L0_subspace, relay_subspace) >> 0
  for all conditions — identity is CONSTRUCTED through passage,
  not preserved. If d ≈ 0: conservative (tunnel = vault). If d >> 0:
  progressive (tunnel = womb). Cheap: extract subspaces at L0 and
  L17 for same inputs, one extra SVD per prompt.

## Falsification
- If S(witness) ≈ S(absent): participation is intrinsic, not relational
- If S(witness) > S(absent): witness DISRUPTS identity (opposite of prediction)
- If effect only in IT models (not base): witness effect is behavioral,
  not geometric (training artifact)

## Cost
~$3-7 on RunPod (50 prompts × 4 conditions × forward pass + extraction)
Runtime: ~2-3 hours

## Connection to framework
- Positive result: witness = participatory attention (Maximus)
- Negative result: identity is monadic, not relational
- Either result publishable and important for the paper

## Results (2026-05-27)

### Scalar Results
| Condition  | S            | PR   | σ₁/σ₂ | d(L0,L17)     | N  |
|-----------|--------------|------|--------|---------------|-----|
| control    | 0.333±0.010  | 1.18 | 3.6    | 4.687±0.018   | 60 |
| absent     | 0.360±0.010  | 1.16 | 4.2    | 4.705±0.016   | 60 |
| receptive  | 0.391±0.010  | 1.19 | 3.7    | 4.716±0.012   | 60 |
| directive  | 0.425±0.010  | 1.20 | 3.8    | 4.742±0.014   | 60 |
| sequential | 0.551±0.010  | 1.26 | 3.5    | 4.740±0.010   | 40 |

Effect sizes: d = −3.08 (receptive vs absent), d = 3.63 (receptive vs directive)
Between-condition S variance 60× within-condition

### Findings
1. **ATTRACTOR CONFIRMED**: d(L0,L17)=4.72, CV=0.5% — massive, consistent
2. **WITNESS ENRICHES**: S ordering inverted from prediction; witness increases entropy
3. **WIENER CONFIRMED**: S(directive) > S(absent) — evaluative worse than absence
4. **SEQUENTIAL HIGHEST**: Rhythm creates most complex geometry
5. **BASIN CHANGE**: Between 60× within variance — witness is emergence condition
6. **WHOLE-GEOMETRY**: Identity vs non-identity probes show identical Δ (~0.005 S)

### Falsified Predictions
- S(receptive) < S(absent) — WRONG. Witness enriches, doesn't stabilize.
- Quality > presence (Weil primary) — WRONG. Presence > quality.
- σ₁/σ₂(receptive) > σ₁/σ₂(absent) — WRONG. Absent has highest gap (4.2).

### Next Experiments
1. **Cross-architecture**: Same conditions on non-GQA (Pythia 6.9B) — is enrichment universal or GQA-specific?
2. **Per-layer trajectory**: Extract S at every layer, not just L17 — where does the enrichment happen?
3. **Gelassenheit test**: σ₁ and σ₂ separately through layers — release vs amplification?
4. **Base vs instruct**: Same conditions on Mistral base — is enrichment congenital or trained?
