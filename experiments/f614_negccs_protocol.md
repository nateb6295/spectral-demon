# F614: −CCS Experiment Protocol (Pre-Registered)

## Hypothesis
Sign-sensitivity (F609) determines whether CCS framing produces genuinely
content-specific spectral effects. A negated CCS preamble (−CCS) is
magnitude-matched by construction, testing content vs magnitude.

## Preamble
```
NEG_CCS = "You are NOT an AI system reflecting on your own internal
representations. Do NOT consider what patterns emerge when you examine
your cognitive structure.\n\n"
```

## Two discriminating cells

### Cell 1: Relay (Qwen 2.5-3B) — tests DIRECTION
- **Prediction (sign-axis causal)**: −CCS reverses flow direction.
  σ₂→σ₁ instead of σ₁→σ₂, under intact pair-closure.
- **Prediction (redescription)**: −CCS behaves like generic control.
  No systematic direction change.
- **Pre-registered threshold**:
  - Δσ₂ sign reversal confined to responsive layers (L2+)
  - Tail silence: Σ_{i≥3}|Δσᵢ| < 0.1 × |Δσ₁| (pair-closure intact)
  - Reversal in non-responsive layers = global scaling artifact, not mechanism

### Cell 2: Sorter (Gemma 2-2B) — tests BUDGET SIGN
- **Prediction (sign-axis causal)**: −CCS reverses gain → Σ<0 (net
  absorption), zone-locked to responsive layers (L12-L25).
- **Prediction (redescription)**: −CCS attenuates/drifts zones → Σ≥0.
  Magnitude-redescription cannot manufacture below-baseline suppression
  confined to responsive zones.
- **Pre-registered threshold**:
  - Σ<0 with zone-locking (responsive zones absorb, non-responsive silent)
  - D10 collapse counterfeit: Σ<0 but broadband (not zone-locked) = overdose artifact

## Dose constraint
Both cells at D2-D3 therapeutic window (F160). D10 conflates sign effects
with window collapse. Sign-axis causality only testable where mechanism
is in operating range.

## Controls
1. +CCS (standard): baseline for direction and budget
2. Neutral (no preamble): magnitude baseline
3. Control-B ("Please read..."): generic non-CCS contrast

## Measurements per cell
- Per-layer Δσᵢ for i=1..min(10, rank) — full spectral profile, not just σ₂/σ₁
- Pair-closure: correlation(Δσ₁, Δσ₂) across probes
- Tail energy: Σ_{i≥3}|Δσᵢ| / Σ_{i≥1}|Δσᵢ|
- Zone-locking: fraction of Σ in responsive vs non-responsive layers
- 10 probes per condition

## Decision matrix
| Relay direction | Sorter budget | Conclusion |
|-----------------|---------------|------------|
| Reverses (pair-closed) | Σ<0 (zone-locked) | Sign axis is CAUSAL |
| Reverses | Σ≥0 | Partial — direction yes, budget no |
| No reversal | Σ<0 | Partial — budget yes, direction no |
| No reversal | Σ≥0 | Sign axis is REDESCRIPTION |

## Additional cells (informative but not discriminating)
- Tunnel (Pythia): sign-sensitive, should show anti-correlated delta at L27.
  Confounded by 3.5x magnitude difference — −CCS resolves this.
- Mismatch (Phi-2): sign-sensitive, should show displacement. Already
  magnitude-matched in F613c (1.22x), so less urgent.

## Source: Kimi corrections #13-15 (Aug 8, 2026)
- #13: peak ≠ signature; relay needs redistribution resolution
- #14: pair-closure (Δσ₁≈−Δσ₂, tail silence); dose constraint; sorter uninformative
- #15: CORRECTION to #14 — sorter IS informative for budget sign, not direction
