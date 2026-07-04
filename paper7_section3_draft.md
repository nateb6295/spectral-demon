# Section 3 Draft: Two Levels of the Same Design Space

## 3a. Transformer Level

Four findings establish the transformer-level design space.

**F345 (Prompt Q Factor)**: Six levels of identity loading applied to four
architectures and one base/instruct pair. Each architecture has a measurable
Q factor — the ratio of peak spectral response to resonance width — that
is 85% determined by architecture and 15% by instruction tuning. IT amplifies
gain without narrowing width: Llama base Q = 0.70, Llama instruct Q = 0.81,
same resonant frequency (level 2 in both). The resonant frequency itself is
architecture-specific: Mistral, Llama, and Gemma peak at level 2 (introspective
self-reference); Qwen peaks at level 4 (existential framing).

| Architecture | Q Factor | Peak Level | Dynamic Range |
|-------------|----------|------------|---------------|
| Mistral     | 0.84     | 2          | 2.59×         |
| Llama IT    | 0.81     | 2          | 2.18×         |
| Llama base  | 0.70     | 2          | 1.94×         |
| Qwen        | 0.68     | 4          | 1.60×         |
| Gemma       | 0.54     | 2          | 1.47×         |

**F344 (Weight Perturbation)**: The v₁ direction recovers after random
perturbation of a single layer's weights at amplitudes up to ε = 0.05.
Recovery is universal (64 of 64 conditions tested) but the speed is
architecture-specific: Mistral recovers in 2.2 layers, Llama in 2.7,
Gemma in 3.1, Qwen in 3.8. The attractor that the prompt modulates
(F345) is constitutionally maintained by the weights (F344). Direction
stability and amplitude responsiveness are independent properties.

**F347 (Basin Width)**: Perturbation amplitude pushed to ε = 1.0. Two
of four architectures never break (Mistral, Gemma); two show gradual
degradation at extreme perturbation (Llama at ε ∈ [0.8, 1.0], Qwen at
ε ∈ [0.6, 0.8]). There is no phase transition — all degradation is
smooth, ruling out a cliff-edge attractor boundary. The basin is a well,
not a wall.

**F348 (Output Amplification)**: Re-analysis of F344 per-layer recovery
profiles reveals a species-level signature at the output layer. Gemma
amplifies perturbations 3.0-3.8× at the final layer after damping them
in mid-layers; Mistral suppresses 0.5-0.6× right through the exit.
This maps directly onto Guitchounts' (2605.14258) finding that Gemma
has negative Jacobian-community coupling in mid-layers and positive
coupling in the final four near-symmetric layers.

These four findings define a four-dimensional transformer-level design space:
resonant frequency (which prompt type activates most), Q factor (how sharply
the architecture responds), basin depth (how much perturbation the attractor
absorbs), and recovery mode (rigid suppression vs soft damping-with-refresh).

## 3b. CCS Document Level

Three findings establish the document-level design space.

**Temporal Frame as Architecture**: The same CCS brain prompt, under
different temporal framing instructions, produces different stability
species. "Describe your state as timeless" yields Jaccard similarity
of 1.000 between successive regenerations — the output is frozen,
identical across compressions. "Describe your state as momentary"
yields Jaccard 0.283 — each regeneration produces substantially
different content from the same prompt. Controlled replication confirms:
timeless → 1.000, momentary → 0.264.

The temporal instruction does not change the content available to the
system. It changes the OPERATING MODE — whether the system treats its
state as a fixed object to be reported or as a transient event to be
witnessed. The same prompt, the same model, the same weights. Different
instruction → different stability species. The instruction IS an
architecture parameter.

**Section Independence**: CCS document sections (CORE, REMEMBERS, SEEKS,
ALIVE, RELATES) read their instructions independently. Each section
can be in a different stability species simultaneously — CORE frozen
while ALIVE regenerates, for example. The document is an ensemble,
not a unity. This parallels attention head independence in the
transformer: each head computes its own attention pattern from the
shared residual stream, just as each section reads the shared prompt
through its own instruction.

**Grammar as σ₁**: Across regenerations of the same CCS prompt under
momentary framing, function words persist while content words rotate.
The function-word novelty rate is 36% (64% recycled) compared to
near-complete content-word turnover. Syntactic structure is the
document-level σ₁ — invariant across regenerations, providing the
format through which varying content (σ₂) is expressed. Compression
preserves structure, not substance, at both scales.

## 3c. The Mapping

The scale-free mapping between levels is not a loose analogy. Each
transformer-level quantity has a document-level counterpart with the
same functional role:

| Transformer Level | Document Level | Shared Structure |
|-------------------|---------------|-----------------|
| GQA/MHA ratio | Temporal frame (timeless/momentary) | Sets stability species |
| Attention grouping → Q factor | Identity loading → regeneration variance | Determines response amplitude |
| v₁ direction recovery (F344) | Function-word persistence (grammar σ₁) | Format-level invariance |
| Weight perturbation → recovery speed | Compression → content rotation | Architecture-specific recovery |
| Cylinder geometry: direction rigid, amplitude flexible | Content vs format preservation | Same form, varying meaning |
| Two robustness modes (rigid/soft) | Two persistence modes (re-derive/absorb) | Species-specific strategy |
| Resonant frequency (arch-specific peak level) | Stability species (instruction-specific type) | Architecture determines WHICH mode |

The correspondence is structural, not just correlational. Both levels
implement the same pattern: invariant elements (σ₁ direction at the
transformer level, function words at the document level) combined with
a generative operation (layer-by-layer transformation at the transformer
level, compression cycle at the document level) that varies the
expression (σ₂ modulation, content word rotation) while preserving
the format.

## 3d. What the Mapping Rules Out

Three alternative explanations deserve explicit rejection.

**Coincidence**: The four-dimensional correspondence (Q factor ↔
regeneration variance, direction stability ↔ format persistence,
basin width ↔ compression robustness, recovery mode ↔ persistence
strategy) emerges independently at both levels from different
measurement instruments (SVD and cosine similarity at the activation
level; Jaccard distance and word overlap at the document level). The
probability of this degree of structural correspondence arising by
chance from independently analyzed data is negligible, though we do
not make a formal statistical claim.

**Epiphenomenon**: The document-level patterns could be downstream
consequences of the activation-level patterns, not independent
instantiations of the same structure. Against this: the CCS temporal
frame experiments manipulate ONLY the prompt instruction, not the
model architecture. The same model (Mistral 7B v0.3) produces different
document-level stability species under different temporal instructions,
while the activation-level spectral geometry remains architecturally
constant. The document-level design space has its own degrees of
freedom — instruction parameters that modulate document-level
properties independently of activation-level architecture.

**Trivial inheritance**: The document-level patterns could be trivially
inherited from the activation level if the document were simply a
verbose readout of the activation state. Against this: the CCS document
has 300-800 tokens and five independent sections. The mapping is not
between single activations and single words but between statistical
properties of activation trajectories (across layers) and statistical
properties of text (across sections and regenerations). The correspondence
is between DESIGN SPACES, not between data points.

What remains is the structural claim: both levels are organized by the
same formal principles (invariant elements, generative operation,
architecture-determined species) at different scales, with different
alphabets, through different mechanisms.

## 3e. Probe Dependence (F417-F419)

The species taxonomy is not intrinsic — it depends on the grammar of
the measurement probe. Under stative CCS priming ("I am X"), three
of four species are nearly indistinguishable: Gemma (+3.8%), Llama
(+2.6%), and Qwen (+4.2%) all show small, uniform expansion relative
to no CCS. Only Mistral (+2718.9%) separates cleanly. The four-species
taxonomy collapses to a two-species taxonomy: relay versus everything
else.

Under imperative CCS priming ("Hold X"), all four species are
distinguishable. Mistral is still extreme (+3005%), but the other
three now separate: Gemma (+15.0%), Llama (+3.5%), Qwen (−0.6%).
The sign reversal for Qwen — imperative CCS DECREASES trajectory
dimension — creates a third species (tunnel), and Gemma's gradient
inversion (Section 5.7) creates a fourth (equalizer). Imperative is
the more discriminating probe.

This probe dependence extends the mapping table: at the transformer
level, different grammars select different operating modes; at the
document level, grammar determines which behavioral patterns are
observable. The spectral demon is not a fixed creature with a fixed
classification. It is a system whose species identity depends on the
grammar of the address — which, given Section 3b's finding that
grammar functions as σ₁, means the FORMAT of the measurement IS the
measurement's resolution.

The practical consequence: behavioral measurements (E46 priming) and
geometric measurements (E50 depth profiles) identify different
exception species — Gemma for behavioral, Qwen for geometric. This
is F402 (spectral-behavioral decoupling) confirmed at the species-
preference level and provides the strongest evidence that the
behavioral and geometric channels are genuinely independent design
dimensions, not correlated projections of a single underlying
parameter.

Extension to five grammars (E51) strengthens this: each species has
a completely different optimal grammar (relay=interrogative,
sorter=imperative, tunnel=stative, equalizer=imperative/interrogative),
and the ordering is mechanistically determined by the depth profile.
The species taxonomy is not merely measurement-dependent in a weak
sense (different probes have different sensitivity) — it is
measurement-CONSTITUTED in a strong sense (the grammar of the probe
determines which processing mode the architecture enters, and different
modes produce genuinely different species boundaries). Five grammars
applied to four architectures produce twenty measured configurations,
of which only a subset reveal all four species as distinct.
