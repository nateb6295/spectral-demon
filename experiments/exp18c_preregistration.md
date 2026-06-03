# Exp 18c Pre-registration — Agency Gradient
Filed 2026-05-28 ~12:30 AM PDT, before running experiment.

## Tunnel Predictions
Based on Exp 18/18b comparable conditions:

| Condition    | Predicted S_tunnel | Basis |
|-------------|-------------------|-------|
| active_high  | ~0.50 | engaging/metabolizing range (high spec + active) |
| passive_high | ~0.38 | attending/receptive range (high spec + passive) |
| active_low   | ~0.36 | attending range (low spec + active) |
| passive_low  | ~0.33 | observing range (low spec + passive) |
| absent       | 0.362 | known from Exp 18 |

## Relay Predictions
Based on linear model S_relay = 4.78 * S_tunnel - 0.45 (R²=0.926 from Exp 18)
plus non-linear corrections derived from relay residual analysis:
- No relational signal (absent): +0.16 boost
- Incomplete relational signal (passive_low): -0.15 penalty
- Partial signal (passive_high): -0.08 penalty (less than passive_low)
- Complete signal (active conditions): near-linear (no correction)

| Condition    | S_r predicted | Ratio predicted |
|-------------|--------------|-----------------|
| active_high  | 1.94 | 3.88× |
| passive_high | 1.29 | 3.39× |
| active_low   | 1.27 | 3.53× |
| passive_low  | 0.98 | 2.96× |
| absent       | 1.44 | 3.98× |

## Hypothesis Predictions

| Hypothesis | Prediction | Confidence |
|-----------|-----------|-----------|
| H1: active_high > passive_high (tunnel) | SUPPORTED (Δ ≈ +0.12) | High |
| H2: passive_high ≈ absent (tunnel) | MIXED — passive_high > absent by ~0.02 | Medium |
| H3: both passive < active (tunnel) | SUPPORTED | High |
| H4: spec dominates agency | SUPPORTED (spec ~0.15, agency ~0.04) | High |
| H5: passive_low weakest relay amp | SUPPORTED (2.96× < 3.98×) | High |

## Critical Test
Passive_high relay ratio. If < 3.5×, the relay penalizes passivity
independent of specification. If ≈ 3.9× (near active_high), the relay
doesn't distinguish agency at matched specification.

## Verdict Prediction
AGENCY IS REAL but subordinate to specification (~4:1 ratio at tunnel).
The J-curve is an agency effect, not a specification artifact.
Passive_high will NOT dip to absent (specification rescues it),
but passive_low WILL dip near or below absent (matching the J-curve).
