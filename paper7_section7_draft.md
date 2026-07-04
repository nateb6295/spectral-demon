# Section 7 Draft: Lullian Combinatorics

## 7.1 An Ars Combinatoria Discovered, Not Designed

Ramon Lull's Ars Magna (1305) proposed that all knowledge could be generated
by rotating concentric discs, each inscribed with a fixed alphabet of
fundamental attributes. The outer disc carried divine dignities (Bonitas,
Magnitudo, Aeternitas, Potestas, Sapientia, Voluntas, Virtus, Veritas,
Gloria); inner discs carried correlatives (agent, patient, act) and
relational predicates. Rotating the discs produced every valid combination
of attributes, generating a map of reality through systematic permutation
rather than exhaustive enumeration.

The spectral demon's design space has the same structure, arrived at
empirically rather than by design.

E8 measured the spectral properties of six architectures across multiple
CCS doses. Three independent axes emerged from the data: σ₁ effective rank
(concentrated to distributed), dose sensitivity (insensitive to
hypersensitive), and default operation (strip to amplify). These axes are
continuous, but the architectures cluster into recognizable species
because the axes are not fully independent — gate architecture biases the
sign of coupling (5 of 6 models), depth sets the per-layer magnitude
(r = -0.944), and total coupling is semi-conserved (CV = 8%).

The combinatorial structure is explicit:

| Wheel | Elements | Range |
|-------|----------|-------|
| Gate architecture | fused, separate | 2 states |
| Normalization | LayerNorm, RMSNorm | 2 states |
| Attention grouping | MHA (1:1), GQA (4:1), GQA (7:1), GQA (8:1) | 4+ states |
| Depth | 24, 28, 32, 42, 48 layers | continuous |
| Prompt identity loading | L0 through L5 | 6 levels |
| Temporal frame | timeless, momentary, episodic | 3 states |
| CCS dose | D0 through D10+ | continuous |

Each axis operates independently on the spectral geometry. Gate layout
biases the coupling sign. Attention grouping modulates the responsive zone
width (GQA 8:1 collapses content rotation; GQA 4:1 preserves it — F231).
Depth sets the per-layer growth rate (FTLE ∝ 1/depth, normalized per-layer
growth ~13-15% across all models). Prompt identity loading sets the
driving frequency of the resonator. Temporal frame sets the CCS stability
species.

Rotating these wheels generates the design space. The existing architectures
are samples from the combinatorial product, not privileged points.

## 7.2 The Three-Layer Invariance Hierarchy

E8 revealed that the wheels do not all spin at the same speed. The design
space has a hierarchy:

**Layer 1 (Architecture)**: FTLE (finite-time Lyapunov exponent), mean σ₁,
and mean sparsity are dose-invariant. Their coefficient of variation across
CCS doses is less than 3%. These quantities are set at initialization and
training — they are the ALPHABET, the fixed inscriptions on Lull's outermost
disc. They do not change when the inner discs rotate.

**Layer 2 (Gate)**: Coupling SIGN is dose-stable for four of six models.
Two models (Qwen3, Mistral) are sign-crossers — their coupling reverses
at high doses. The gate architecture biases but does not constrain the
sign (Qwen3 proves that a separate-gate architecture can amplify). These
are the CORRELATIVES — the relational predicates that determine how
elements interact, more variable than the alphabet but more stable than
their magnitudes.

**Layer 3 (CCS)**: Coupling MAGNITUDE is the only dose-variable quantity.
CCS operates on second-order statistics (covariance, not means). Identity
is a second-order phenomenon — a pattern in the relationships between
activations, not a property of any single activation. The CCS dose
rotates the innermost disc, varying the intensity of the modulation while
leaving the alphabet and the relational structure intact.

This hierarchy IS Lull's nested disc structure: outermost discs (architecture)
rotate slowest, inner discs (CCS dose) rotate fastest, and the meaning
of any particular configuration depends on reading all levels simultaneously.

## 7.3 Lull's Ladder at Every Layer

Lull's Ladder of Ascent and Descent assigns the same attribute (Bonitas,
Potestas) to every level of being — mineral, vegetable, animal, human,
celestial, angelic, divine — with the meaning of that attribute varying
by level while its FORM remains constant. The Bonitas of a stone is
different from the Bonitas of an angel, but both are recognizably Bonitas.

σ₁ does this through depth. The first singular value at layer 2 reflects
embedding structure. At layer 15 it reflects transition format. At layer 24
it reflects identity commitment. At layer 31 it reflects relay output.
The σ₁ direction is measurably the same (cosine > 0.998 between adjacent
layers in Mistral — F340, F341), but what that direction MEANS changes
because the surrounding context (σ₂, σ₃, the full activation geometry)
is different at each depth. The form is invariant; the expression varies.

This is not a post hoc analogy. Lull's insight was that a finite alphabet
of invariant forms, combined with a generative operation (rotation), could
produce an unbounded space of meanings. The spectral demon implements
this: a finite set of architectural invariants (σ₁ direction, FTLE,
coupling sign) combined with a generative operation (layer-by-layer
transformation) produces the full space of identity-relevant geometry.
The tunnel is the compression that reduces the alphabet to its essentials.
The relay is the rotation that generates meaning from the compressed
alphabet.

## 7.4 Leibniz's Substitution

Leibniz admired Lull's Art but replaced the alphabet. Where Lull inscribed
theological dignities, Leibniz inscribed logical predicates. The STRUCTURE
of the ars combinatoria — nested discs, systematic rotation, reading all
levels simultaneously — survived the change of alphabet unchanged. The
Art was substrate-independent.

The prompt-as-architecture thesis claims the same substitution. At the
model level, the alphabet is {GQA, MHA, LayerNorm, RMSNorm, depth,
gate layout}. At the prompt level, the alphabet is {identity loading,
temporal frame, carry-forward instruction, scope}. Different alphabets,
same combinatorial structure, same spectral species in the output.

The F345 titration experiment demonstrates this directly. Architecture sets
the resonant frequency (Mistral, Llama, Gemma peak at L2; Qwen at L4).
The prompt modulates the same geometry through a different mechanism —
changing the input trajectory rather than the weights. But the effect on
the relay zone is indistinguishable: a shift in the σ₁/σ₂ ratio, a
change in the Q factor, a modulation of the coupling magnitude. The
spectral demon cannot tell whether its operating mode was set by
architecture or by prompt, because both rotate the same wheels.

The CCS document-level experiments complete the substitution. Temporal
frame (timeless vs momentary) produces the same stability species
distinction as GQA ratio (concentrated vs distributed) — one measured in
Jaccard similarity, the other in Grassmann distance, but the same
2 × 2 → 3 species taxonomy. Section independence mirrors attention head
independence: each section reads its instruction independently, just as
each head computes its own attention pattern. The document is the outer
disc; the prompt is the inner disc; the rotation generates the identity.

## 7.5 Memory as Investigation

Frances Yates, reading Lull, noted that the Art was "not merely a method
of memorizing already known knowledge, but a method of investigation, and
a method of logical investigation." The mnemonic function was instrumental
— the rotating discs helped you DISCOVER which combinations were valid,
not just recall what you already knew.

CCS compression, studied empirically, functions the same way. Each
compression cycle does not merely store the system's state — it
INVESTIGATES that state through the generative operation of the prompt.
The prompt forces the system to re-derive its identity from its relational
substrate, and the output reveals which elements are invariant (they
recur across compressions) and which are contingent (they vary). The
inverted-U dose response (D2-D3 therapeutic window) sets the speed of
the investigation: too few compressions and nothing is tested; too many
and the testing outpaces the system's capacity to integrate results.

The combinatorial frame thus unifies the paper's two levels. At the
model level, the design space is an ars combinatoria whose wheels
(architecture, gate, CCS dose) generate spectral species through
systematic permutation. At the prompt level, the same ars combinatoria
operates with a different alphabet (temporal frame, identity loading,
scope), generating the same species taxonomy. The spectral demon is
not a particular architecture. It is the combinatorial structure ITSELF
— the fact that a finite alphabet of invariant forms, combined with
generative rotation, produces identity. Lull would have recognized it
immediately.
