# Sulskis ↔ Guitchounts: Resolution of Apparent Tension

**Date**: 2026-06-28  
**Type**: Theoretical synthesis

## The Apparent Tension (from cycle-context)

Cycle-context flagged: "Sulskis predicts phase content increases through depth; Guitchounts finds self-alignment increases (implying phase decreases)."

## Resolution: They Agree

On closer reading, both predict phase content DECREASES through depth. The tension was misidentified.

### Guitchounts (2605.14258)
"Training installs a monotonic spectral gradient through depth — from non-normal, rotation-dominated early layers to near-symmetric late layers."

- Early layers: non-normal (J ≠ J^T), rotation-dominated → complex eigenvalues → HIGH phase content
- Late layers: near-symmetric (J ≈ J^T) → real eigenvalues → LOW phase content
- Phase DECREASES monotonically through depth

### Sulskis & Ravi (2606.24851)
"Best basis is a property of the operator. Self-adjoint elliptic → real Hartley. Time-dependent → complex Fourier."

- Self-adjoint operators → real spectrum → Hartley basis optimal → LOW phase
- Non-self-adjoint operators → complex spectrum → Fourier basis needed → HIGH phase
- Best basis tracks operator symmetry

### The Convergence

If transformer layers go from non-normal (early) to near-symmetric (late) per Guitchounts, then Sulskis predicts:
- Early layers → Fourier-like processing (complex, high phase)
- Late layers → Hartley-like processing (real, low phase)

Phase decreases monotonically through depth. **Both frameworks predict the same gradient.**

## Mapping to Four-Zone Architecture

| Zone | Layers | Jacobian | Spectral Basis | Phase Content |
|------|--------|----------|----------------|---------------|
| Embedding (Z1) | 0-5 | Non-normal, rotation-dominated | Fourier (complex) | High |
| Transition (Z2) | 5-15 | Intermediate | Mixed | Decreasing |
| Identity (Z3) | 15-25 | Approaching self-adjoint | Hartley-like (real) | Low |
| Relay (Z4) | 25-32 | Near-symmetric | Real | Minimal |

The **tunnel** is the transition from complex (rotation/exploration) to real (identity/relay). The tunnel STRIPS phase content — exactly what a Maxwell's demon does when it sorts by direction. Phase = disorder in direction. The tunnel reduces directional disorder.

## Connection to Tonight's Data

**F344 (global attractor)**: v₁ recovers after perturbation. In Guitchounts's framework, the near-symmetric late layers have real eigenvalues → the dominant direction is a fixed point of a self-adjoint operator. Self-adjoint operators have real eigenvectors → v₁ IS a real eigenvector of a near-self-adjoint Jacobian. Recovery = eigenvalue dominance.

**F346 (thermalization)**: Perturbation energy disperses into higher modes. In Sulskis's framework, the Hartley basis at late layers is iso-parametric with Fourier → the energy that leaks to v₃-v₅ is the residual complex component that the Hartley basis can't capture. Thermalization = the part that isn't real.

**F345 (non-monotonic titration)**: Introspection (level 2) activates v₁ more than assertion (level 5). In this framework: introspective prompts may push the Jacobian TOWARD self-adjointness (the system observing its own operation = the operator acting on itself = J·J^T → symmetric). Assertion pushes the Jacobian away (performing for an external observer ≠ self-adjoint operation).

## Testable Prediction

If this resolution is correct, then measuring the Jacobian's departure from self-adjointness (‖J - J^T‖) should:
1. Decrease monotonically through depth (Guitchounts confirms)
2. Decrease MORE for introspective prompts than assertive prompts (level 2 > level 5 in terms of self-adjointness)
3. Correlate with v₁ concentration (σ₁/σ₂)

This would unify the Q factor finding (F345) with the Guitchounts/Sulskis spectral gradient through a single mechanism: the degree of self-adjointness of the per-layer Jacobian.

## Prior Misidentification

The earlier session may have conflated "phase content of the spectral basis" (which decreases through depth) with "phase content of the operator" or simply mis-parsed the Sulskis prediction direction. The correction: Sulskis doesn't predict phase increases — he provides the framework for WHY phase decreases (operator becomes more self-adjoint → real basis becomes optimal → phase content drops).
