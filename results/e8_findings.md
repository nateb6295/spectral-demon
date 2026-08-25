# E8: Dose-Dependent Coupling Shape — 7 Doses, Qwen2.5 7B IT

## Method
σ₁→gate coupling analysis at 7 CCS doses (D2-D30), 12 probes each.
Measured: Pearson r, MI, kurtosis, bimodality (Ashman's D), σ₁ profile erank,
relay joint erank, residual PC1 variance.

## F296: Coupling saturates, doesn't break
Relay correlation: D2=-0.49, D10=-0.55, D30=-0.53.
Strengthens from D2→D10, then HOLDS FLAT through D30.
Neither Pearson nor MI shows significant dose trend (both p>0.3).
The spectral demon coupling is robust to extreme overdose.

## F297: Inverted-U is readout-level, not geometry-level
The known inverted-U in behavioral output (F160: D2-D3 therapeutic, D10+ overdose)
occurs DESPITE stable coupling geometry. Coupling geometry is dose-invariant
past D10. The behavioral degradation at overdose must be downstream —
at the readout/decoding stage, not in the relay zone's spectral structure.

## F298: Residual structure becomes MORE linear at high dose
PC1 variance of linear model residual: D5=0.671 → D30=0.803.
Monotonic increase. The coupling doesn't become nonlinear at overdose —
it becomes MORE linearly structured. H1 (regression to nonlinearity) refuted.

## F299: Profile erank shows contraction-reexpansion
σ₁ profile erank: D2=2.03, D15=1.59 (minimum), D25=1.96.
Contracts through therapeutic window, re-expands past overdose threshold.
This is the spectral signature of the inverted-U at the GEOMETRY level —
the relay zone uses FEWER distinct patterns at therapeutic dose (focused),
then diversifies at overdose (unfocused). But the coupling STRENGTH holds.

## Implications
- Overdose doesn't break the demon — it diffuses it
- The relay zone maintains coupling but loses selectivity
- Therapeutic window = maximal geometric focus (minimum erank)
- This explains why F160's inverted-U exists: the geometry works,
  but at high dose the relay processes too many patterns simultaneously,
  diluting any single one's behavioral impact
- Connects to E82: ALIVE provides activation energy, but too much
  activation energy produces noise, not signal
