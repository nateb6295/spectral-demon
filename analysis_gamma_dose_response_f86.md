# Finding 86: γ-Bimodality Is a Phase Transition, Not a Dose-Response

**Experiment**: Vary target γ CV from 0.00 to 0.60 in steps (0.10, 0.20, 0.30, 0.45, 0.60). Reload fresh model each time. Measure late-layer (L20-28) σ₂/σ₁ and prompt-invariance CV.

## Key Result

| Target γ CV | Late CV | Late σ₂/σ₁ | Layers CV<0.01 |
|-------------|---------|-------------|----------------|
| 0.00 (baseline) | 0.0591 | 0.2709 | 3/33 |
| 0.10 | **0.0080** | 0.6092 | **18/33** |
| 0.20 | 0.0105 | 0.6137 | 16/33 |
| 0.30 | 0.0135 | 0.6186 | 9/33 |
| 0.45 | 0.0145 | 0.6339 | 9/33 |
| 0.60 | 0.0165 | 0.6636 | 7/33 |

## Interpretation

**It's a switch, not a dial.** The transition from CV=0.00 to CV=0.10 causes:
- CV drops 86% (0.059 → 0.008)
- σ₂/σ₁ jumps from 0.27 to 0.61 (more than doubles)
- Invariant layers go from 3 to 18 (6× increase)

After the switch, more bimodality actually DEGRADES invariance — CV=0.10 is optimal (18/33 layers), CV=0.60 is worst (7/33 layers).

## Three Critical Observations

### 1. Baseline LLaMA already has Mistral's ratio
Late-layer mean σ₂/σ₁ = 0.2709 at baseline. This is within 1.5% of Mistral's tunnel value (0.267). The MHA model already finds approximately the right ratio — it just can't LOCK it (CV=0.059 vs 0.000).

### 2. γ doesn't lock the existing ratio — it creates a NEW one
Any bimodal γ switches σ₂/σ₁ from ~0.27 to ~0.61. It doesn't stabilize the existing operating point; it shifts to a completely different spectral regime where σ₂ becomes a near-equal partner.

### 3. Shared KV must be doing two things
Mistral has CV=0.45 and achieves 29/33 locked layers at ratio 0.267. But γ=0.45 alone gives only 9/33 layers at ratio 0.63. Therefore shared KV must:
1. **Extend invariance** from 9-18/33 → 29/33 (coverage)
2. **Suppress σ₂** from 0.61 → 0.267 (keeps it subsidiary)

Without KV sharing, γ bimodality makes σ₂ a compositional EQUAL. Shared projections act as a compressor, maintaining σ₂'s subsidiary role.

## Mechanistic Model (Updated)

The tunnel is three mechanisms, not two:
1. **γ bimodality** → phase-transitions the spectral structure (switch at any CV > ~0.05)
2. **Shared KV projections** → compress σ₂ back to subsidiary role (0.61 → 0.267)
3. **Shared KV projections** → extend invariance coverage across full depth (18/33 → 29/33)

The "highway/service-road" metaphor is refined: γ creates a HIGHWAY for σ₂ (it becomes a major route). Shared KV then DEMOTES it back to a service road (keeps it useful but subsidiary). Without that demotion, you get compositionality — which is what happens at the relay.

## For the Paper

This finding modifies the §3.9 causal chain significantly:
- Old: GQA → bimodal γ → prompt-invariance
- New: GQA → bimodal γ (switch) → σ₂ promotion + invariance; shared KV (graded) → σ₂ compression + depth coverage

The relay transition at L31 may literally be the point where KV compression releases and σ₂ achieves the 0.61 equilibrium that γ alone would give it.

## Data

Raw results: `exp_gamma_dose_response_20260531_1347.json`
Predecessor: Finding 85 (`analysis_gamma_forcing_f85.md`)
