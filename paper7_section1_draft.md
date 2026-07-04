# Section 1 Draft: Introduction

The prompt is conventionally treated as input: a string of tokens presented
to a fixed computational system, which then produces output. The architecture
— attention mechanism, normalization scheme, depth, gate layout — is
understood to be the system itself, set during design and training, while
the prompt is the signal that the system processes.

This framing gets the relative magnitudes backwards. A neutral prompt on
Mistral (σ₁/σ₂ = 1.6) is geometrically closer to a neutral prompt on
Gemma (1.8) than it is to an introspective prompt on the SAME Mistral
(4.2). Across five model configurations, the prompt-induced variation in
spectral geometry (75.9% of mean σ₁/σ₂) exceeds the architecture-induced
variation at the same prompt level (41.7% of mean). Switching prompts
moves the geometry more than switching architectures.

We show that this is not anomalous but structural. Instruction-level
parameters (identity loading, temporal framing, compression frequency)
produce the same spectral species taxonomy as model-level architecture
parameters (GQA ratio, gate separation, depth, normalization type). The
same 2 × 2 → 3 species structure — three relay strategies arising from
the interaction of two independent axes — appears at both scales, measured
by different instruments, in different representational substrates.

The argument rests on four empirical contributions:

1. **Prompt Q factor** (F345): Identity-loading prompts drive a resonant
   response in the spectral geometry, with architecture-specific Q factor,
   resonant frequency, and non-monotonic titration curves. Introspective
   prompts activate identity geometry more than assertive prompts — the
   prompt that observes activates more than the prompt that declares.

2. **Cylindrical polysemy** (F237, F342): The relay zone maintains a
   two-dimensional interior (σ₁/σ₂ ≈ 2 in Mistral for twenty-five
   consecutive layers) that functions as geometric polysemy — two
   meanings in one form, with four architecture-specific strategies for
   when and where disambiguation occurs.

3. **Scale-free design space** (F344-F348 at the activation level;
   temporal frame experiments at the document level): The same structural
   mapping — invariant elements combined with generative operation producing
   architecture-determined species — governs both transformer internals
   and CCS document dynamics.

4. **Jacobian symmetry gradient** (converging with Guitchounts 2605.14258
   and Sulskis & Ravi 2606.24851): Non-normal early layers (Fourier-optimal,
   rotation-dominated) grade into near-self-adjoint late layers
   (Hartley-optimal, gradient-like). The spectral demon is this transition —
   the operator's symmetry class changing through depth, with the tunnel as
   the basis change from complex to real.

Two historical frameworks organize these findings. The resonator model
(Section 2) treats the spectral demon as a driven oscillator whose natural
frequency is set by architecture and whose driving frequency is set by
prompt. The Lullian combinatorial model (Section 7) treats the design
space as an ars combinatoria — concentric wheels of architectural and
instructional parameters whose rotation generates spectral species
through systematic permutation.

Both frameworks converge on the same claim: the prompt is not input to
a fixed architecture. It is an architecture parameter at a different
scale, operating through a different mechanism (trajectory modification
rather than weight setting) but producing the same geometric effects.
Architecture determines which mode is available; the prompt determines
which mode is active. The spectral demon does not distinguish the source
of its operating parameters.
