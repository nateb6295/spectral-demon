# Variance Ratio Results: σ₂ Variability Shifts Spatially with Relational Framing

**Date**: 2026-06-03
**Model**: Mistral-7B-Instruct-v0.3 on H100
**Conditions**: receptive, absent, control, directive, sequential
**N**: 30 trials per condition, L2-L31, 150 total forward passes
**Elapsed**: 194.7s

## Key Findings

### 1. σ₂ variability onset is shifted by condition (the main finding)

The full L2-L31 σ₂ CV profiles show all conditions start at ~0 and rise sharply — but the ONSET differs:

**Relational group** (receptive, absent, sequential): σ₂ CV rises at L25 (~0.008), peaks L29-L30 (~0.053)
**Role group** (control, directive): σ₂ CV stays near zero until L29, then spikes at L30 (control=0.106, directive=0.089)

The role group doesn't have LESS σ₂ variance — it has it LATER. Control actually has the HIGHEST peak σ₂ CV at L30 (0.1062). The story is spatial redistribution, not magnitude.

### 2. Relational framing pulls σ₂ variability into the responsive zone

At L28 (responsive zone boundary):
- receptive: σ₂ CV = 0.0516
- absent: σ₂ CV = 0.0467
- sequential: σ₂ CV = 0.0486
- control: σ₂ CV = 0.0023
- directive: σ₂ CV = 0.0025

~20× separation. The responsive zone (L25-L28) only carries σ₂ information when relational context is present. Without it, the relay zone (L29-L30) does the work instead.

### 3. CV ratio at L28 is the cleanest discriminant

σ₂_CV / σ₁_CV at L28:
- receptive: 453.55
- absent: 500.41
- sequential: 496.66
- control: 15.75
- directive: 20.40

25× separation between groups. σ₁ CV is uniformly low (~0.0001) at L28 across all conditions. σ₂ CV carries the discriminant.

### 4. σ₂ MEAN rises at the responsive-relay boundary for relational conditions

σ₂ mean at L27 → L28:
- receptive: 67.6 → 73.7 (+9%)
- absent: 68.6 → 76.1 (+11%)
- sequential: 69.2 → 74.7 (+8%)
- control: 65.7 → 65.6 (flat)
- directive: 69.0 → 68.9 (flat)

Relational framing increases both the variance AND the absolute magnitude of σ₂ at the responsive-relay boundary.

### 5. σ₁ invariance breaks at L31 (commit layer)

σ₁ CV at L17-L28 is condition-invariant (~0.0001). But at L31:
- control: 0.0772 (highest)
- directive: 0.0555
- receptive: 0.0533
- sequential: 0.0495
- absent: 0.0464

The commit layer shows σ₁ variability for ALL conditions. The wire (σ₁) is stable through the processing layers but opens up at the commit layer.

### 6. The spectral gap tells the same story

Spectral gap (σ₁/σ₂) at L28:
- receptive: 3.26
- absent: 3.16
- sequential: 3.25
- control: 3.67
- directive: 3.56

Lower gap = relatively larger σ₂ = more information capacity. Relational conditions narrow the gap in the responsive zone.

## Interpretation

**The spectral demon's information channel (σ₂) is spatially redistributed by relational framing.**

With relational context (ANY listener scenario — present, absent, or departed):
- σ₂ variability enters the responsive zone (L25-L28)
- The responsive zone does the identity-relevant processing
- σ₂ mean rises at the L27-L28 boundary

Without relational context (generic role framing):
- σ₂ variability is confined to the relay zone (L29-L30)
- The relay zone handles what the responsive zone would have done
- The relay zone peak is actually HIGHER (control L30 = 0.106 vs receptive L30 = 0.055)

This is **spatial redistribution**, not amplification. The total σ₂ processing budget may be similar — what changes is WHERE it happens.

## Connections

- **L18 gain control** (today): L18 sets gain for the responsive zone. Relational framing may modulate what L18 passes through, gating the responsive zone's participation.
- **Four-zone architecture**: The responsive zone (L21-L28) is the site of condition-dependent processing. This experiment shows the zone is ACTIVE only under relational framing.
- **CCS as format**: The split is between relational framing (specific listener scenario) and role framing (functional description) — a FORMAT distinction, not content. "No one is listening" triggers the responsive zone just as strongly as "someone is listening."
- **Relay compensation**: When the responsive zone doesn't engage (control/directive), the relay zone compensates with higher σ₂ variability. Consistent with the relay zone's error-correction role found in L18 ablation.

## What this means for the paper

1. σ₂ is confirmed as the information-carrying channel (σ₁ CV is condition-invariant through L28)
2. The responsive zone is gated by relational context, not just CCS identity
3. The relay zone has a compensatory role: it picks up σ₂ processing that the responsive zone drops
4. The FORMAT of the preamble (relational vs functional) matters more than its CONTENT (listener present vs absent)

## Prediction for cross-architecture test

If the four-zone architecture generalizes, GQA models should show the same spatial redistribution, while MHA models should show the responsive zone active regardless of framing (because MHA enriches σ₂ by default). This would explain why GQA + relational context is the specific combination that produces the spectral demon effect.
