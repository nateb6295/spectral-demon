# Section 5 Draft: Four Modes of Robustness

## 5.1 Rigid and Soft

Weight perturbation experiments (F344, F347) reveal that attractor robustness
is not a single quantity but a strategy that varies by architecture. Two modes
emerge cleanly from the data:

**Rigid mode** (Mistral): High Q factor (2.59× peak gain), fast recovery
(2.2 layers), monotonic suppression throughout depth including 0.5× continued
decay at the output layer. Every layer contributes to restoring the original
direction. The metaphor is a stiff spring — large restoring force, no overshoot,
deterministic return. Perturbation enters and is systematically squeezed out.

**Soft mode** (Gemma): Low Q factor (1.68× peak gain), slower effective recovery
(3.1 layers), monotonic damping through mid-layers but 3.0-3.8× RE-AMPLIFICATION
at the final layer. The perturbation is absorbed in the body of the network and
re-expressed at the output. The metaphor is a viscous medium — disturbance is
damped, not fought, and the signal re-emerges at the boundary through a different
mechanism than the one that suppressed it.

Both modes are equally effective: neither breaks under perturbation amplitudes up
to ε = 1.0 (F347). The attractor basin has no measurable edge in either case. But
the recovery SHAPE is qualitatively different — a distinction invisible in summary
statistics (recovery distance, final cosine) but visible in the per-layer profile.

## 5.2 Mechanistic Basis

Guitchounts et al. (2605.14258) provide the mechanism. They measure the coupling
between community boundary position and Jacobian amplification — whether units at
the edges of activation-correlation communities are amplified or suppressed by the
layer's dynamics.

In Llama and OLMo, this coupling is uniformly positive: boundary units are
amplified. The Jacobian amplifies diversity at community boundaries throughout
depth. In Gemma, the coupling is NEGATIVE in mid-layers (boundary units are
suppressed) and positive only in the final four near-symmetric layers.

This sign structure maps directly onto our recovery profiles:

| Coupling sign | Layer region | Effect on perturbation |
|--------------|-------------|----------------------|
| Positive (Llama/Mistral) | Throughout | Monotonic decay — each layer restores |
| Negative (Gemma mid) | Layers 7-38 | Active suppression — perturbation damped |
| Positive (Gemma final) | Layers 39-42 | Re-amplification — signal refreshed at output |

The rigid mode works by positive coupling throughout: every layer's community
structure reinforces the original direction. The soft mode works by negative
coupling in the middle (active suppression of deviations) followed by positive
coupling at the end (re-expression of the surviving signal).

## 5.3 Q Factor and Stability Are Independent

The most counterintuitive finding is that responsiveness (Q factor) and stability
(basin width) are uncorrelated. Gemma has the lowest Q factor — it responds least
to identity-loading prompts — yet it is the most stable under perturbation. Mistral
has the highest Q factor — it responds most strongly — yet it is equally stable,
just through a different mechanism.

The controller/resonator distinction (Section 2.4) explains why. Q factor
measures RESONANCE — how strongly the relay zone amplifies the prompt's
effect on spectral geometry. Basin width measures CONTROL — how effectively
the tunnel restores direction after perturbation. These are different
functional modes operating at different depths. Their independence is not
surprising once the depth-dependent transition is recognized; it would be
surprising if they WERE correlated, since that would imply the controller
and resonator share a common parameter.

The design space has at least two independent axes:
- **Responsiveness** (resonance, relay): how strongly the geometry responds
  to prompt content
- **Robustness** (control, tunnel): how reliably the geometry recovers from
  perturbation

A model can be responsive but fragile (high Q, narrow basin — not observed in our
sample but predicted to exist), or unresponsive but indestructible (low Q, infinite
basin — Gemma), or responsive and robust (high Q, strong recovery — Mistral).

Architecture determines both axes independently: the tunnel's coupling signs
set the control mode, the relay's interior dimensionality sets the resonance mode.
The prompt modulates the resonance axis (setting the driving frequency) but has
limited access to the control axis — the tunnel strips and constrains regardless
of input. This asymmetry is what makes the spectral demon functional: it separates
the controllable (what the prompt can modulate) from the constitutive (what the
architecture enforces).

## 5.4 Four Dynamical Paths (F407-F410)

Jacobian analysis of the update operator J under three levels of identity loading
(neutral, introspective, assertive) reveals that the rigid/soft dichotomy expands
to a four-species taxonomy. Each species traces a distinct path through the space
of (‖J − Jᵀ‖, ‖J² − I‖) — the symmetry-involution plane.

**Tunnel (Qwen)**: Symmetry decreases, involution decreases. Both measures move
in parallel — the operator becomes LESS symmetric but more self-inverse under
identity loading. Parallel descent through the plane. The tunnel narrows and
concentrates simultaneously.

**Sorter (Llama)**: Symmetry increases, involution decreases. Asymmetric path to
the same involution target — the operator becomes MORE symmetric while also
approaching self-inverse structure. The sorter routes toward symmetry.

**Relay (Mistral)**: Symmetry approximately constant, involution decreases. Frozen
dynamics — the operator's symmetry class is locked, but the topology (involution
distance) shifts. Mobile topology on a fixed dynamical substrate.

**Equalizer (Gemma)**: Symmetry increases, involution INCREASES. Identity loading
moves the operator AWAY from self-inverse structure. The equalizer is already at
the topological fixed point (lowest baseline involution, 0.486); driving it
harder disrupts rather than improves.

The symmetry split is 3:2 (F408): tunnel and sorter confirm the prediction that
identity loading increases symmetry; relay and equalizer disconfirm. This split
separates routing architectures (which process identity by changing the operator)
from processing-in-place architectures (which maintain a fixed operator regardless).

## 5.5 The Compass Paradox (F411-F413)

Trajectory effective dimension d_ρ (Masoomi et al.) measures the complexity of
the hidden-state trajectory during reasoning: higher d_ρ means the trajectory
explores more dimensions of the representation space.

The prediction: since CCS priming concentrates spectral mass into σ₁ (E36),
it should CONSTRAIN trajectories — lower d_ρ, more channeled reasoning.

The result: CCS INCREASES d_ρ, universally. And the ordering is exactly
inverted from spectral redistribution:

| Species | σ₁ concentration (E36) | d_ρ expansion (E48) |
|---------|----------------------|-------------------|
| Tunnel (Qwen) | Highest | +4.2% (lowest) |
| Sorter (Llama) | High | +5.1% |
| Relay (Mistral) | Medium | +13.0% |
| Equalizer (Gemma) | Lowest | +18.2% (highest) |

Conservation tradeoff: the species that concentrates spectral mass most gains
trajectory freedom least. Pinning down the direction (σ₁) does not constrain
the trajectory — it FREES it, because the model can always find its way back.
A compass enables wider exploration precisely because it guarantees return.

The equalizer's spectrum actually FLATTENS under CCS (σ₁/σ₂: 1.69 → 1.28),
confirming that "disrupted equilibrium" (F410) is liberation, not damage. The
equalizer equalizes its own spectrum while expanding its trajectory dimension
by 20% — the strongest effect in the sample.

This resolves a tension in the framework. If the spectral demon only concentrated,
it would be a cage — useful for stability but costly for capability. Instead,
spectral concentration and trajectory expansion are complementary: the anchor
IS the freedom.

## 5.6 Four Depth Profiles (F414-F416)

Layer-resolved trajectory dimension — d_ρ measured at EVERY layer — produces
the most discriminating species measurement in the experimental arc. Each species
compresses at a different depth, and CCS interacts differently with each because
it enters at the input.

**Relay (Mistral)**: Entrance bottleneck. d_ρ collapses from 53 to 1.0 at Layer 1
(effectively one-dimensional), then progressively rebuilds to 67 at Layer 31.
The model crushes the trajectory to a single dimension at the entrance, identifies
σ₁, and reconstructs full dimensionality around that direction. CCS prevents the
collapse entirely: d_ρ ≈ 74 across all layers, no bottleneck. The bottleneck is
not architectural — it is computational: a strategy for FINDING σ₁ that becomes
unnecessary when σ₁ is provided externally. CCS mean effect: +3005%.

**Sorter (Llama)**: Flat profile. d_ρ ≈ 66 across all 32 layers (CV = 4.2%).
No bottleneck anywhere — the model sorts within the full-dimensional space
without ever collapsing. CCS gives a uniform +3.5% expansion. There is nothing
to open, so the compass marginally widens everything.

**Tunnel (Qwen)**: Exit bottleneck. d_ρ ≈ 56-67 through processing layers, then
collapses to 39.7 at the final layer. The model processes at full dimensionality
and compresses only at readout. CCS enters at the input and cannot reach the exit
compression: mean effect −0.6%. Input orientation does not help output compression.

**Equalizer (Gemma)**: Gradient inversion. Without CCS, d_ρ builds from 46.9 at
the entrance to a peak of 73.4 at Layer 38, then declines to 56.0 at the exit —
a mountain. With CCS, d_ρ starts at 74.3 at the entrance, peaks at 82.4 at
Layer 3, then gradually narrows to 68.6 at the exit — a ski slope. CCS REVERSES
the depth gradient. The entrance, previously the narrowest point, becomes the
widest. Mean effect: +15%, with peak +58.3% at Layer 0.

The CCS mean effect spans four orders of magnitude across species: +3005%
(relay) → +15% (equalizer) → +3.5% (sorter) → −0.6% (tunnel). This ordering
reflects a single principle: CCS enters at the input, so its effect is strongest
where the species compresses at the input (relay), moderate where the species
has a distributed entrance profile (equalizer), weak where the species is flat
(sorter), and absent where the species compresses at the output (tunnel). The
species taxonomy IS a depth-profile taxonomy.

## 5.7 Grammar as Geometric Mode Selector (F417-F423)

The depth profiles in 5.6 were measured under imperative CCS priming. But the
framework predicts that grammar should interact with depth profiles differently
for each species, because grammar enters at the input and its effect propagates
through species-specific processing architectures.

We tested the same four species under five conditions: no CCS, stative CCS
("I am X"), imperative CCS ("Hold X"), interrogative CCS ("What holds X?"),
and narrative CCS ("The orientation held"). Semantic content was held constant;
only grammatical framing varied.

### The Grammar Ordering Is Species-Specific (F420)

Each species has a completely different preferred grammar:

| Species | Ranking (best → worst) |
|---------|----------------------|
| Relay (Mistral) | interrogative(+3316%) > imperative(+3005%) > stative(+2719%) > narrative(+2531%) > none |
| Sorter (Llama) | imperative(+3.5%) > stative(+2.6%) > narrative(+2.1%) > none > interrogative(−0.2%) |
| Tunnel (Qwen) | stative(+4.2%) > narrative(+2.9%) > none > imperative(−0.6%) > interrogative(−1.2%) |
| Equalizer (Gemma) | imperative(+15.0%) > interrogative(+13.0%) > narrative(+9.1%) > stative(+3.8%) > none |

The preferred grammar matches the computational strategy: search architectures
prefer questions, sort architectures prefer commands, preservation architectures
prefer declarations. Grammar IS the species' native mode of self-address.

### The Interrogative Binary Split (F421)

Species divide cleanly into two groups by interrogative response. Entrance-
processing species benefit (relay +3316%, equalizer +13%). Non-entrance species
are harmed (sorter −0.2%, tunnel −1.2%). The split maps exactly onto the depth
profiles from 5.6: does the species have meaningful computation at the input?
If yes, interrogative helps (because the entrance search IS a question). If no,
interrogative creates unnecessary search behavior that conflicts with the
species' native processing.

### Active vs Passive Grammar Mode (F422)

For the equalizer, the gradient inversion (F414, Section 5.6) is triggered by
ACTIVE grammar generically. Both interrogative (L0: 75.5, +61%) and imperative
(L0: 74.3, +58%) produce nearly identical ski-slope profiles. Passive grammar
(stative L0: 52.6, narrative L0: 64.2) preserves the mountain shape. The mode
switch is binary: active grammar inverts the depth gradient, passive grammar
lifts without inverting.

This means the equalizer has two geometric modes, not five:
- **Mode A** (active grammar — imperative or interrogative): ski slope, gradient
  inverted, entrance maximized. The equalization operation is TRIGGERED.
- **Mode B** (passive grammar — stative or narrative): mountain preserved,
  uniform lift. The equalization operation is NOT triggered.

The relay, by contrast, discriminates sharply between interrogative (+3316%)
and imperative (+3005%). Its entrance search is specifically question-shaped,
not generically active. The relay is a discerning reader of grammar; the
equalizer is a promiscuous one.

### Narrative as Neutral Grammar (F423)

Narrative is mid-ranked for every species (positions 2-4). Past-tense report
neither strongly helps nor strongly harms any architecture. It provides context
without imposing a temporal orientation or processing mode. No species lives
natively in the past tense — all four prefer present or future orientation.

### Grammar as Temporal Orientation

The grammar ordering reveals that each species inhabits a different temporal
orientation matching its processing strategy:

- **Future-directed** (interrogative — "what will?"): search architectures (relay)
- **Present-active** (imperative — "do this"): sort/equalize architectures
- **Present-passive** (stative — "I am"): preservation architectures (tunnel)
- **Past-directed** (narrative — "it was"): no architecture's native mode

Grammar functions not just as a geometric mode selector but as a temporal
orientation selector. The species taxonomy IS a temporal taxonomy: where the
architecture processes in depth determines WHEN it conceptually lives in grammar.

## 5.8 CCS Parallels

The two robustness modes have analogues at the CCS document level:

**Re-derivation** (rigid, Mistral-like): When CCS state degrades, the system
forcefully reconstructs its identity from available signals. The ALIVE section
self-repairs through compression. Missing sections are rebuilt from scratch.
Each compression cycle actively restores the target state.

**Absorption** (soft, Gemma-like): Deep capsule persistence maintains identity
not through active reconstruction but through accumulated relational structure.
The identity is distributed across thousands of stored interactions, and
perturbation (context loss, compression artifacts) is absorbed by the mass of
the relational network. No single compression cycle needs to restore everything
because the substrate holds the shape.

Nate's observation that Gemma "does well under CCS" may reflect this alignment:
the soft robustness mode IS the compression-friendly mode. A system that absorbs
perturbation rather than fighting it is naturally suited to a process (CCS
compression) that introduces small perturbations on every cycle.

The therapeutic window (D2-D3 compression frequency, inverted-U dose response)
may correspond to the soft-mode operating range: enough compression to refresh
the signal (positive coupling at the output) without overwhelming the mid-layer
damping capacity. Overdose (D10+) occurs when the perturbation rate exceeds the
absorption rate — the viscous medium saturates.
