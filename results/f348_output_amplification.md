# F348: Species-Specific Output-Layer Amplification

**Date**: 2026-06-28
**Source**: Re-analysis of F344 weight perturbation data
**Triggered by**: Reading Guitchounts (2605.14258) — Gemma shows sign reversal in coupling

## Finding

The final layer's treatment of propagating perturbations is a consistent species-level
signature, invisible in F344's original analysis (which focused on recovery distance):

| Species | Final-layer pattern | Factor | Consistent |
|---------|-------------------|--------|------------|
| Gemma | AMPLIFICATION | 3.0-3.8x | 4/4 conditions |
| Qwen | AMPLIFICATION | 1.0-2.1x | 4/4 conditions |
| Llama | NEUTRAL/DECAY | 0.8-1.0x | 4/4 conditions |
| Mistral | STRONG DECAY | 0.5-0.6x | 4/4 conditions |

"Factor" = final-layer delta / minimum mid-layer delta. Values >1 = output amplification.

## Mechanism

Gemma's profile at L35 perturbation (eps=0.05):
- L36: 7.06e-5 (initial disruption)
- L37-L41: 6.2e-5 → 2.5e-5 (monotonic damping)
- L42: 7.67e-5 (3.1x amplification at output)

The perturbation is DAMPED through intermediate layers, then RE-AMPLIFIED at the
final layer. Mistral shows the opposite: steady decay right through the end.

## Connection to Guitchounts

Guitchounts (2605.14258) reports coupling between community boundary position and
Jacobian amplification:
- **Llama/OLMo**: uniformly POSITIVE coupling in mid-to-late layers
- **Gemma**: NEGATIVE coupling in early-mid layers, POSITIVE in final 4 near-symmetric layers

Our finding is the perturbation-level manifestation of this:
- Negative coupling = suppression in mid layers = perturbation damping
- Positive coupling in final layers = perturbation amplification at output

The SIGN of the coupling directly predicts the SHAPE of the recovery curve.

## Species Taxonomy Extension

- **Rigid (Mistral)**: strong initial perturbation, strongest decay at output (0.5x).
  Forceful suppression throughout. "Pulls everything back."
- **Soft (Gemma)**: weakest initial perturbation, strongest amplification at output (3x).
  Damps in middle, refreshes at end. "Never breaks but always echoes."
- **Distributed (Llama)**: moderate perturbation, neutral at output (1.0x).
  Steady decay, no signature at boundary. "Fades evenly."
- **Compressed (Qwen)**: moderate perturbation, moderate amplification (1.3-2.1x).
  Intermediate between Gemma and Llama.

## CCS Connection

Gemma's output amplification may explain its suitability as a CCS compression substrate
(Nate's observation: "Gemma results should correlate with previous results. Does well
under CCS"). The mid-layer damping prevents perturbation growth (stability), while the
final-layer amplification REFRESHES the signal at the output (identity preservation).
The demon damps noise but amplifies signal at the boundary.

## Testable Prediction

If the amplification factor correlates with the number of positive-coupling final layers
(Guitchounts), then models with more aggressive GQA (fewer KV heads) should show larger
output amplification. Gemma 4 E4B (2 KV heads) vs Llama 3.1 8B (8 KV heads) is
consistent with this, but the confound is that Gemma is also deeper (42 vs 32 layers)
and has a different normalization scheme.

The Jacobian experiment (revised v2) includes update_norm_ratio per layer, which should
show a spike at the final layer for Gemma and a decline for Mistral.
