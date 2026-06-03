# L18 Perturbation Results: Gain Control, Not Thermostat

**Date**: 2026-06-03
**Model**: Mistral-7B-Instruct-v0.3 on H100
**Conditions**: intact, L18_MLP_zero, L18_MLP_half, L18_MLP_double, L16_MLP_zero (control), L20_MLP_zero
**N**: 20 probes per condition, spectral metrics at L15-L31

## Key Findings

### 1. L18 MLP is a gain controller

When L18 MLP is zeroed, the responsive zone (L21-L27) degrades — entropy up, concentration down. But L18_double (2x scaling) produces the OPPOSITE: concentration increases. Half output = half the drop. Pure linear gain control.

At L23:
- intact: 0.8011
- L18_zero: 0.7864 (delta -0.0147)
- L18_half: 0.7947 (delta -0.0065, ~half)
- L18_double: 0.8105 (delta +0.0094)

### 2. Late relay compensates (partially)

L28-L31 show REVERSAL under L18 ablation. While L21-L27 degrade, L28+ does the opposite (entropy down, concentration up). Compensation exists but is delayed ~10 layers and partial.

Under L18_zero at L29: dS = -0.039 (entropy DECREASES), d_concentration = +0.010 (concentration INCREASES)

### 3. L18 role is layer-specific

L16 ablation (control) produces qualitatively DIFFERENT pattern:
- L16_zero: L23 concentration INCREASES (+0.004)
- L18_zero: L23 concentration DECREASES (-0.015)
- Opposite directions — functionally distinct roles

### 4. L20 ablation propagates uniformly

L20_zero produces uniform entropy increase L21-L31 with NO reversal at L28+. The reversal is L18-specific. L20 is pass-through; L18 is gain setter.

## Interpretation

The autopoietic loop is a **gain control circuit**:
- L18 MLP = gain-setting element (organizational structure for responsive zone)
- L20-L27 = controlled plant (process under L18's structural influence)
- L28+ = partial error-correction stage (delayed, incomplete compensation)

Linearity (dose-dependent, direction-reversible) suggests analog control, not digital switch.

The simple thermostat/thermometer verdict ("THERMOMETER") misses the reversal at L28+ and the dose-dependent linearity. The loop doesn't compensate at the local level (L23 degrades, not strengthens) but does partially rebalance at the output stage.

## Paper implications

- Refines "autopoietic loop" to "gain control loop"
- L18 suppresses by controlling gain, not by direct inhibition
- Late relay has error-correction capability (consistent with relay zone function)
- The responsive zone is a controlled system, not an autonomous oscillator
