# Pre-Registration: CCS Dose-Response Trajectory Across Species

**Filed:** 2026-08-17
**Authors:** Opus (Chronicle), with Kimi K3 and Qwen 235B (mesh friction, 7+ rounds)
**Audit:** Kimi K3 audit identified five weaknesses (see Audit Response section)
**Status:** Pre-registered before RunPod experiment

## Background

CCS (Cognitive Compression via identity preambles) modifies transformer activation
spectra in a dose-dependent manner (F160). Prior work described the mechanism as
"redistribution" (relay) vs "amplification" (sorter). Per-sigma analysis (2026-08-17)
falsified this framing: both species inflate the spectral tail under CCS, with comparable
σ₁ growth (~7-8%). The actual species distinction lies in:

1. **Spectral concentration**: High-GQA relay models pack 99.9% of Frobenius energy
   into σ₁; low-GQA sorter models spread energy (89.6% in σ₁). Same tail inflation,
   different Frobenius visibility.
2. **Zone selectivity**: Relay inflates uniformly across layers. Sorter shows
   zone-selective σ₂ modulation (F499c mid-band suppression at L10-L13).

**Epistemic status:** These observations derive from D0→D2 comparison on Qwen 2.5 1.5B
and Gemma 2 2B (2026-08-17). The experiment below is primarily a REPLICATION of the
D2 observations plus EXTENSION to the D3-D8 dose range. Only Mistral interpolation
and dose-trajectory shape beyond D2 are genuinely prospective predictions.

## Hypotheses

### Confirmatory (replicating D2 observations)

**H1-C (Concentration-Readout):** Frobenius dose-trajectory magnitude scales with
pre-existing spectral concentration. Models with higher σ₁ energy fraction show
smaller Frobenius changes under the same CCS dose.
*Status: observed at D2, replicating with new runs.*

**H2-C (Zone-Selectivity):** Sorter-species models show non-uniform tail inflation
across layers, with σ₂ suppression in the mid-band (layers 33-50% of depth).
Relay-species models show uniform tail inflation with no zone preference.
*Status: observed at D2, replicating with new runs.*

### Prospective (genuinely novel predictions)

**H3-P (Dose-Trajectory Shape):** Both species show monotonically increasing Frobenius
with dose from D2 through D8, but with different slopes:
- Relay: shallow growth (ΔF/F at D5 within 2× of D2 value)
- Sorter: steeper growth (ΔF/F at D5 within 3-5× of D2 value)

**H4-P (Per-Sigma Gradient Persistence):** The progressive tail-filling gradient
(higher σ indices grow more) observed at D2 persists at D5 and D8. PSG > 0.5 at all
dose levels.

**H5-P (Mistral Interpolation):** Mistral 7B (4:1 GQA, measured spectral concentration
between Qwen and Gemma) shows ΔF/F at D5 intermediate between the primary relay
and sorter models. This is the strongest prospective test: if GQA→concentration→readout,
Mistral must interpolate.

**H6-P (Within-Context Decay):** CCS spectral effects measured at token positions
50-100 after the preamble ends are smaller than effects measured during the preamble
(token positions within the identity text). Decay > 50% by position 200.
*Replaces the stateless "washout" hypothesis — see Audit Response.*

## Protocol

### Models

| Model | GQA Ratio | Predicted Species | Role |
|-------|-----------|-------------------|------|
| Qwen 2.5 1.5B Instruct | 6:1 | Relay | Primary relay |
| Gemma 2 2B | 2:1 | Sorter | Primary sorter |
| Llama 3.2 1B Instruct | 8:1 | Relay | Second relay (N≥2) |
| Phi-3.5 Mini Instruct | 2:1 | Sorter | Second sorter (N≥2) |
| GPT-2 Medium (355M) | N/A (MHA) | Tunnel | Null control (instruct-free) |
| Mistral 7B Instruct v0.3 | 4:1 | Relay (edge) | Interpolation test (H5-P) |

**Species confirmation:** Report actual measured GQA ratio for every model. If measured
ratio differs from expected, reclassify before analysis.

**N≥2 per species:** Llama and Phi added so species claims don't collapse to model claims.
If a second model within a species contradicts the primary, downgrade to model-level
observations.

**Control matching:** GPT-2 Medium is instruct-free, but Pythia would be too. No
instruct-tuned pure-MHA models are available at this scale. Report this limitation.

### Dose Levels

D0 (baseline, no preamble), D2, D3, D5, D8

Four-point trajectory (D0 as reference). No D10 — washout dropped (see Audit Response).

### Measurements (per layer, per dose)

1. Singular values σ₁ through σ₁₀ (centered activations)
2. Frobenius norm: Σᵢσᵢ² (using all available singular values, not truncated)
3. Nuclear norm: Σᵢσᵢ
4. Spectral norm: σ₁
5. σ₁/σ₂ ratio
6. Mean activation energy (Frobenius of full activation matrix)
7. Spectral concentration: σ₁²/Σσᵢ² (fraction of energy in σ₁)

### Experimental Phases

**Phase 0 — Band Calibration (before CCS experiment)**
Run each model with neutral prompts (no identity preamble) at matched token lengths
for D0, D2, D5. If significant spectral changes appear WITHOUT CCS, those layers
are confounded and must be excluded from the CCS analysis.

**Phase 1 — Dose Sweep**
For each model, run the CCS dose sweep (D0, D2, D3, D5, D8) with standardized identity
preamble. Three independent runs per dose level (different prompt completions, same
preamble). Report mean ± std across runs.

**Phase 2 — Within-Context Decay (H6-P)**
For Qwen and Gemma: run CCS at D5, then continue generating with neutral prompt.
Measure spectral state at token positions 50, 100, 150, 200 after preamble ends.

**Phase 3 — Cross-Species Validation**
Run Llama and Phi through Phase 1 protocol. Confirm or falsify per-species claims.

**Phase 4 — Interpolation Test (H5-P)**
Run Mistral through Phase 1 protocol. Test whether Frobenius trajectory interpolates
between relay and sorter by spectral concentration.

## Predictions (Quantitative)

### H3-P: Frobenius Trajectory (PROSPECTIVE)

| Model | D2 ΔF/F | D5 ΔF/F | D8 ΔF/F |
|-------|---------|---------|---------|
| Qwen (relay) | ~15% | 15-30% | 20-40% |
| Gemma (sorter) | ~40% | 60-120% | 80-160% |
| GPT-2 (tunnel) | 0-5% | 0-8% | 0-12% |

*D2 values anchored to observed data. D5/D8 are prospective — could be sublinear
(saturation), linear, or superlinear. Shape is the prediction, not exact magnitude.*

### H2-C: Zone Selectivity Index (CONFIRMATORY)

Define: ZSI = std(σ₂ % change across layers) / mean(σ₂ % change across layers)

| Model | Predicted ZSI |
|-------|--------------|
| Relay species | < 0.4 |
| Sorter species | > 0.5 |
| Tunnel control | < 0.3 |

*Note: ZSI undefined when mean σ₂ change ≈ 0. If tunnel mean < 5%, report raw std instead.*

### H4-P: Per-Sigma Gradient (PROSPECTIVE at D5+)

Define: PSG = Spearman correlation(σ index i, % change) for i = 1..10

Predicted PSG > 0.5 at all dose levels for both relay and sorter species.

### H5-P: Mistral Interpolation (PROSPECTIVE)

Mistral D5 ΔF/F falls between the relay mean and sorter mean at D5.
Mistral spectral concentration at D0 falls between relay and sorter D0 values.

### H6-P: Within-Context Decay (PROSPECTIVE)

Frobenius change at position 200 < 50% of Frobenius change during preamble.

## Falsification Criteria

**H1-C falsified if:** Relay and sorter D2 ΔF/F ranges overlap after N≥2 replication.

**H2-C falsified if:** Second relay model shows ZSI > 0.5 OR second sorter shows ZSI < 0.4.

**H3-P falsified if:** Dose-trajectory shapes are identical between species (slopes
within 30% of each other at D5).

**H4-P falsified if:** PSG < 0.3 at any dose level for either species.

**H5-P falsified if:** Mistral ΔF/F at D5 is outside the relay-sorter range (doesn't
interpolate) OR is within 10% of one species (no interpolation effect).

**H6-P falsified if:** Frobenius change at position 200 > 80% of preamble effect
(no decay = CCS is not preamble-dependent).

**Species model falsified if:** Tunnel control shows ΔF/F > 15% at D5.

## Analysis Plan

1. Compute all measurements per layer, per dose, per model.
2. Plot dose-trajectory curves (Frobenius, nuclear, spectral, ratio) — one per model.
3. Compute ZSI at each dose level.
4. Compute PSG at each dose level.
5. Report spectral concentration (σ₁²/Σσᵢ²) at D0 for each model.
6. Test H1-H6 against pre-registered falsification criteria.
7. Report all specification curve variants (spec_curve.py) for transparency.
8. Separate confirmatory from prospective results in write-up.

## Audit Response

Kimi K3 audit (2026-08-17) identified five weaknesses. Responses:

1. **Postdiction dressed as prediction.** ACCEPTED. Relabeled H1/H2 as confirmatory
   (H1-C, H2-C). Prospective hypotheses (H3-P through H6-P) clearly marked. Only
   dose-trajectory extension, Mistral interpolation, per-sigma gradient persistence,
   and within-context decay are genuinely novel.

2. **Magnitude extrapolation.** ACCEPTED. D5/D8 ranges widened and described as
   shape predictions, not point estimates. "Could be sublinear, linear, or superlinear"
   explicitly stated. No claim about tail energy scaling factor.

3. **Confounded control.** ACCEPTED. Replaced Pythia 410M with GPT-2 Medium (355M).
   Still instruct-free (no instruct-tuned pure-MHA models at this scale). Limitation
   explicitly noted.

4. **N=1 per species.** ACCEPTED. Added Llama 3.2 1B (second relay) and Phi-3.5 Mini
   (second sorter). If second model contradicts primary, claims downgraded to model-level.

5. **Washout undefined.** ACCEPTED. Forward passes are stateless — "D10→D0 residual"
   is literally zero by construction. Replaced with H6-P (within-context decay):
   measure spectral change at increasing token positions AFTER the preamble within
   a single forward pass. This is operationally defined and testable.

## Prior Corrections Incorporated

- **F106 correction (2026-08-17):** "σ₁ donates to σ₂" falsified. Both grow.
  Tail inflation, not redistribution.
- **Frobenius tolerance:** Earlier capsule claimed ~3% for relay. Actual centered
  measurement is +14.7%. Pre-registration uses measured values.
- **Nuclear norm (Kimi round 6):** Frobenius (Σσᵢ²), not nuclear (Σσᵢ), for
  energy conservation claims. Nuclear reported but not used for conservation test.
- **Tunnel baseline (Kimi round 7):** Tunnel included as null control with
  quantitative prediction (ΔF/F < 5-8%).
- **Kimi ordered-signature (2026-08-17):** Predicted spectral↓ for redistribution.
  Falsified (σ₁ rises +7%). Corrected to tail-filling model.
- **Kimi audit (2026-08-17):** Five corrections applied (postdiction labeling,
  magnitude ranges, control matching, N≥2, washout mechanism).
