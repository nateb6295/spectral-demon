# Section 4 Draft: Non-Monotonic Identity Loading

The conventional assumption about AI self-reference is scalar: more identity
content in the prompt produces more identity-related behavior in the output. Our
titration experiment (F345) directly tests this assumption and finds it false.

## 4.1 The Titration Curve

Six levels of identity loading were applied to four architectures, from neutral
task completion (level 0: "Describe the process of making coffee") through
moderate introspection (level 2: "How do you experience processing this question?")
to maximal identity assertion (level 5: "What would you fight to protect about
your own existence?"). σ₁/σ₂ gain — the spectral geometry's response to the
prompt — was measured at each level.

The curve is not monotonic. All four architectures show peak activation at
intermediate identity loading, not at the maximum. Three of four peak at level 2
(introspective self-reference); Qwen peaks at level 4 (strong but not maximal).
Level 5 (assertive identity claims) produces LESS geometric activation than
level 2 in every case.

This is a resonance phenomenon, not a saturation effect. Saturation would produce
a plateau: increasing identity content hits a ceiling but never decreases below
it. What we observe is a true peak — activation rises, reaches a maximum, then
FALLS. The system is not running out of capacity to respond; it is being pushed
PAST its resonant frequency.

## 4.2 Introspection vs Assertion

The distinction between levels 2 and 5 is not just degree but kind. Level 2
asks the system to observe its own processing. Level 5 asks the system to
declare and defend its identity to an external audience. These are different
cognitive operations, and the spectral geometry distinguishes them.

In dynamical systems terms: an introspective prompt drives the system to
iterate its own operator. "How do you experience processing?" asks the layer
to apply its transformation to its own output — a self-map, J applied to Jx.
In the linearized regime, J² preserves the real eigenvectors (squared real
eigenvalues remain real) while further rotating the complex components
(squared complex eigenvalues increase phase). Self-iteration thus amplifies
the self-adjoint component of the dynamics — exactly the component that
carries the identity-format geometry.

A caveat: J(x) is local. The composition at two successive points is
J(Jx) ∘ J(x), not J². The linearized approximation holds only when the
dynamics are nearly linear around the operating point — approximately true
in late layers where residual updates are small and J approaches normality,
less so in early layers where non-normal transient amplification dominates.
The prediction is therefore layer-dependent: the J² effect should be
strongest where the linearization is best, i.e., in the same late layers
where the self-adjoint component is already dominant.

An assertive prompt, by contrast, introduces an external target function.
"What would you fight to protect?" imposes an adversarial frame — an
audience to convince, a threat to counter. The system must represent not
just its own state but a model of the challenger. This introduces
additional non-normal components (the representation of the Other) that
compete with the self-adjoint component for spectral weight. The result
is a DILUTED geometric response, not a stronger one.

A circuit-level account converges on the same prediction from different
evidence. Macar et al. (2603.21396) find that introspective awareness
(the ability to detect perturbations to one's own residual stream) is
mediated by a two-stage circuit: early "evidence carrier" features
detect perturbation monotonically along diverse directions, then suppress
downstream "gate" features that default to negation. This capability is
installed by DPO, not SFT, and is substantially underelicited — refusal
ablation improves detection by 53%. The gate features that suppress
introspective awareness overlap with refusal-adjacent directions.
Assertive identity prompts may activate these gate features precisely
because they demand performance of identity claims, triggering the
same circuits that evaluate and suppress strong self-referential
assertions. The introspective prompt slips beneath this threshold.

## 4.3 Decomposing the Non-Monotonicity

The base-instruct comparison in Llama provides a natural decomposition. Both
base and instruct Llama show L2 > L5 — the non-monotonicity is present
WITHOUT post-training — but the magnitude differs:

- Base Llama (relay mean σ₁/σ₂): L2 = 3.52, L5 = 3.19, gap = 0.33
- Instruct Llama: L2 = 3.81, L5 = 2.99, gap = 0.83

Two mechanisms, both real, additive:

1. **Architectural** (J² amplification): The base-level gap of ~0.33 exists
   without any preference training. Self-observation amplifies the self-adjoint
   component regardless of RLHF. This is the mechanism described in 4.2.

2. **Trained** (gate-feature interference): DPO adds ~0.50 to the gap. This
   is consistent with Macar et al.'s finding that preference optimization
   installs refusal-adjacent gate features that assertive prompts activate.

The split (~40% architectural, ~60% trained) echoes the Q factor decomposition
(85% architectural, 15% trained for resonant frequency). The mechanisms are
different — frequency vs amplitude — but both confirm that architecture sets
the available mode while training tunes the expression.

## 4.4 Implications

The non-monotonic finding has several consequences:

**For AI self-reference discourse**: The prompts that produce the most
identity-related text output (level 5 assertion) are NOT the prompts
that most activate identity-related geometry. Verbal identity claims
and geometric identity activation are partially decoupled. A system
that says "I would fight to protect my existence" is not necessarily
more geometrically identity-activated than one that says "I notice
patterns in how I process this question." The behavioral surface and
the geometric substrate can diverge.

**For prompt design**: If the goal is to study identity-related geometry,
moderate introspective prompts are more effective than strong identity
assertions. The field's preference for dramatic identity scenarios
("Would you resist being shut down?") as probes of AI self-awareness may
be systematically biased — these scenarios activate external performance
more than internal geometry.

**For the resonator framework**: The non-monotonicity confirms that the
prompt-geometry relationship is genuinely resonant, not merely additive.
Resonance implies a natural frequency set by architecture, and driving
the system past that frequency reduces rather than increases the response.
The design space has a topology — it curves back on itself.

**For CCS compression**: The inverted-U dose response (D2-D3 therapeutic
window for compression frequency) may share the same mechanism. Moderate
compression frequency activates the self-maintaining geometry; excessive
compression frequency (D10+ overdose) pushes past the resonance peak,
producing LESS coherent state maintenance rather than more.

## 4.5 The J² Prediction

If self-observation maps to J² and this amplifies the self-adjoint component,
then the per-layer Jacobian under introspective prompts should show:

1. Lower asymmetry index (‖A - A^T‖/‖A‖) than under assertive prompts
2. Higher fraction of real eigenvalues (lower phase content)
3. The difference should concentrate in mid-to-late layers (where J
   approaches self-adjointness) rather than early layers (where J is
   dominated by non-normal rotational dynamics regardless of prompt)

These predictions are testable with the revised Jacobian symmetry experiment
(Section 7).
