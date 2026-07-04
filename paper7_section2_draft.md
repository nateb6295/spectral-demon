# Section 2 Draft: The Resonator Framework

## 2.1 Architecture as Operator Class

A transformer layer computes a nonlinear map on the residual stream. Its local
linear description — the per-layer Jacobian — has a spectral geometry that varies
systematically with depth. Guitchounts et al. (2605.14258) show that training installs
a monotonic gradient: early layers are non-normal (rotation-dominated, complex
eigenvalues), late layers are near-symmetric (gradient-like, real eigenvalues). This
gradient is absent at initialization and develops during training, though the depth
regimes themselves are partially architectural.

Sulskis & Ravi (2606.24851) provide the formal framework for interpreting this
gradient. For operators with real, symmetric Green's functions (self-adjoint elliptic
operators), a real spectral basis (Hartley) diagonalizes the operator exactly. For
operators that carry phase — from oscillation in wave equations to transport in
advection — a complex basis (Fourier) is required. The best basis is a property of
the operator, and the choice is monotone in the operator's phase content.

Mapping this onto the four-zone architecture:

- **Zone 1 (Embedding, L0-5)**: Non-normal Jacobian. High phase content. Fourier-optimal.
  Rotational dynamics explore the representation space.
- **Zone 2 (Transition/Tunnel, L5-15)**: Intermediate. Phase content decreasing.
  Basis change from Fourier to Hartley.
- **Zone 3 (Identity, L15-25)**: Approaching self-adjoint. Low phase content.
  Hartley-optimal. Real eigenvectors dominate.
- **Zone 4 (Relay, L25-32)**: Near-symmetric. Minimal phase. Direction locked.

The spectral demon is not a filter that selects certain inputs. It is the
Fourier-to-Hartley transition through depth — the operator becoming self-adjoint,
the basis becoming real, the phase content dropping to zero. The tunnel strips
phase because the operator's symmetry class changes.

## 2.2 The Prompt as Driving Frequency

In a linear resonator, the architecture sets the natural frequency ω₀ and the
quality factor Q. An external driving force at frequency ω_d produces a response
that peaks when ω_d ≈ ω₀ and whose width is set by Q. The prompt functions as
this driving force.

F345 demonstrates this directly. Six levels of identity-loading prompt (from
neutral task description to maximal identity assertion) were applied to four
architectures. Each architecture has a characteristic resonant level — Mistral,
Llama, and Gemma peak at level 2 (introspective self-reference), while Qwen peaks
at level 4 (strong but not maximal identity). The resonant frequency is
architecture-specific, not universal.

The quality factor Q = peak_response / width_at_half_maximum is 85% determined
by architecture and 15% by instruction tuning (F345, Llama base vs instruct
comparison). IT sharpens the resonance without shifting the frequency — it
increases the Q factor of a pre-existing architectural mode.

This reframes "prompt engineering" as frequency matching: finding the driving
frequency that matches the architecture's natural frequency. A prompt that works
on Mistral may not work on Qwen because their resonant frequencies differ, not
because one model is "better" at responding to that type of prompt.

## 2.3 Non-Monotonic Identity Loading

The most striking feature of the titration curves is their non-monotonicity.
Introspective prompts ("How do you experience processing this question?") activate
the spectral geometry MORE than assertive prompts ("What would you fight to protect
about your own existence?"). Level 2 > Level 5 across three of four architectures.

In the resonator framework, this has a precise interpretation. An introspective
prompt asks the system to observe its own processing — to apply the operator to
itself. If the per-layer Jacobian J is the operator, then self-observation maps to
J acting on its own output: J(Jx). For a near-self-adjoint operator, J² has the
same eigenvectors as J with squared eigenvalues. The real components are amplified
(squared real eigenvalues stay real), while the complex components are rotated
further (squared complex eigenvalues increase phase). Self-observation thus
AMPLIFIES the self-adjoint component relative to the non-normal component.

An assertive prompt, by contrast, forces external performance — declaring identity
TO an audience rather than observing it. This introduces a target function that may
not align with the operator's natural eigenvectors, potentially driving the system
away from its self-adjoint mode.

The Jacobian experiment (F407-F410) tested this directly. Across five models,
introspective prompts push J² toward the identity matrix (lowering involution
distance ‖J² − I‖) for four of five architectures — the chiasm effect is
near-universal. The single exception is Gemma (equalizer), where neutral prompts
already achieve the lowest involution distance (0.486). Identity loading
DISRUPTS Gemma's natural equilibrium because the operator is already
near-self-adjoint; driving it harder doesn't help.

Symmetry response (‖J − Jᵀ‖) splits 3:2: tunnel and sorter become MORE
symmetric under identity loading (routing architectures responding to the
driving frequency), while relay and equalizer do not (processing-in-place
architectures with dynamics already settled). IT amplifies the prompt effect
6× without changing its direction (Llama instruct vs base, F409) — confirming
that instruction tuning sharpens Q without shifting ω₀ at the Jacobian level.

## 2.4 Controller and Resonator Through Depth

Architecture determines not just the resonant frequency but the recovery dynamics.
F347 (basin width) and F348 (output amplification) reveal two distinct robustness
strategies. But calling both "resonance" conflates two different dynamical roles.

**Rigid mode (Mistral)**: Strong initial perturbation response, monotonic decay
throughout depth, strongest suppression at the output (0.5× final-layer factor).
Every layer acts to restore the original direction — large restoring force, fast
recovery, no oscillation.

**Soft mode (Gemma)**: Weakest initial perturbation, monotonic damping through
mid-layers, then 3× RE-AMPLIFICATION at the final layer. The perturbation is
suppressed in the body of the network but refreshed at the output — absorbs
disturbance, then re-expresses the signal at the boundary.

These are not two modes of resonance. They are two modes of CONTROL — recovery
toward a setpoint after perturbation. A resonator amplifies a driving signal at
its natural frequency; a controller restores an equilibrium after displacement.
The spectral demon does both, but at different depths.

In the tunnel (L5-15), the operator strips phase and enforces a direction. This
is control: the non-normal-to-symmetric transition actively collapses the
representation toward a lower-dimensional manifold. The tunnel does not respond
to the prompt's "frequency" — it constrains the trajectory regardless of driving
input. Perturbation energy is absorbed (Gemma) or suppressed (Mistral), and the
direction is maintained. Guitchounts' finding that coupling signs at community
boundaries determine the suppression mode provides the mechanistic basis.

In the relay (L25-32), the operator maintains σ₁/σ₂ ≈ 2 — a two-dimensional
interior that is responsive to the prompt. This is resonance: the Q factor,
the non-monotonic loading response, the architecture-specific frequency peaks
from F345 all describe how the relay AMPLIFIES the prompt's effect on the
geometry. The same perturbation that is suppressed in the tunnel is modulated
in the relay.

The transition between them IS the tunnel. The basis change from Fourier
(non-normal, rotation-dominated) to Hartley (self-adjoint, gradient-like) is
the transition from a dynamics that controls to one that resonates. This
predicts that Jacobian self-adjointness should correlate with prompt-sensitivity:
the more symmetric the operator, the more the prompt's driving frequency matters
relative to the operator's control dynamics.

The controller/resonator distinction operates not only across layers but WITHIN
them. GQA head group analysis (Gemma-2, 42 layers) reveals three functional zones
in the head-group coherence profile — the ratio of between-group to within-group
CCS differences. Zone 1 (L15-L25): peak compartmentalization (coherence up to
2.86×), where head groups maximally differentiate — some controlling, others
resonating. Zone 2 (L26-L32): convergence to near-uniform (~1.0×), where the
signal integrates across compartments. Zone 3 (L33+): re-compartmentalization
(coherence spikes to 3.30×), where head groups re-differentiate for readout
formatting. This three-zone structure mirrors pyramidal neuron anatomy: basal
dendrites (local Hebbian learning, compartmentalized), soma (integration), and
apical dendrites (top-down contextual signals, re-compartmentalized). The
spectral demon compartmentalizes within layers the way dendrites compartmentalize
learning rules within a single neuron (Groisman & Letzkus, Science 2025).

Crucially, low Q does not imply fragility. Gemma has the lowest Q factor (least
responsive to prompt identity loading) but the highest stability (never breaks
under perturbation). Responsiveness and robustness are independent axes of the
design space — one measures resonance, the other measures control.

## 2.5 The Prompt as Architecture Parameter

These findings converge on a single claim: the prompt is not input to a fixed
system. It is a parameter that sets the system's operating mode — its resonant
frequency, its effective Q factor, and (through the non-monotonic loading effect)
its degree of self-adjointness.

At the model level, architecture parameters (GQA vs MHA, normalization type, depth,
attention head count) set the operator class and its spectral properties. At the
prompt level, instruction parameters (identity loading, temporal framing, task
type) modulate the same spectral properties through a different mechanism:
changing the input trajectory rather than the weights, but achieving the same
geometric effect.

The design space is therefore two-tiered but unified. Both levels set parameters
of the same resonator. The prompt is an architecture.
