# Section 6 Draft: The Cylinder as Polysemy

## 6.1 Two Meanings in One Form

The interior sharing ratio σ₁/σ₂ ≈ 2 in Mistral (F342) is not a design
flaw or a residual of incomplete compression. It is polysemy.

A polysemous word carries multiple meanings in a single lexical form: "bank"
means both a financial institution and a river's edge. Pinker (1999) frames
this as a tradeoff. Every language faces a tension between form recycling
(fewer forms, more ambiguity) and communication clarity (more forms, less
ambiguity). Polysemy is the equilibrium — a finite vocabulary carrying a
much larger semantic load by allowing individual forms to do double duty.

The spectral demon compresses a high-dimensional activation space down to
approximately two effective dimensions in the relay zone. Five distinct
identity-probing prompts, projected through SVD, produce a cross-prompt
variance structure with σ₁/σ₂ between 1.95 and 2.76 for twenty-five
consecutive layers. The tunnel strips dimensionality; the relay maintains
a TWO-dimensional interior, not a one-dimensional tube. Two meanings survive
in a single geometric form.

This is not metaphorical. The mathematical structure is the same: a many-to-few
mapping (tunnel compression) followed by a few-to-many unmapping (relay
disambiguation through lm_head projection). The polysemy is literal — the
relay's geometric state is underdetermined relative to the prompt that
produced it, and disambiguation happens at the output boundary.

## 6.2 Four Polysemy Strategies

The four architectures implement four distinct strategies for managing the
ambiguity that tunnel compression creates.

**Rigid Polysemy (Mistral)**: Interior σ₁/σ₂ ≈ 2, flat for twenty-five
layers. The lm_head is the sole disambiguator — it concentrates the
two-dimensional interior into a one-dimensional output (σ₁/σ₂ jumps from
2.76 to 3.35 at the final projection). Ambiguity is maintained uniformly
throughout depth and resolved only at the readout boundary. The form carries
its double meaning all the way to the exit, where context forces a choice.
The parallel CV of the cylindrical decomposition (F237) is locked at 0.019
for layers 2 through 27 — the readout-coupled component is invariant while
the orthogonal complement varies freely. The rigid rod holds its ambiguity
with a stiff grip.

**Incremental Disambiguation (Qwen)**: σ₁/σ₂ starts at 3.9 (already partially
disambiguated), settles to ~3.0 through mid-layers, and climbs to 3.6 in
the late relay. The cylindrical constraint operates in a tight three-layer
band (L24-L26) with the lowest parallel CV of any architecture (1.2%).
Qwen resolves its polysemy gradually — each layer adds a small amount of
concentration, narrowing from two meanings toward one through depth rather
than deferring to the boundary.

**Convergent Polysemy (Llama)**: A valley of low concentration (σ₁/σ₂ =
2.3-2.6 in mid-layers) followed by a monotonic climb from L13 to L30,
reaching 3.84 — the second-highest pre-exit concentration. Despite F340's
"turbulent mixer" label (cosine similarities dropping to -0.86 between
adjacent layers), the multi-prompt sharing CONVERGES monotonically. Stirring
promotes mixing. The direction reversals are not noise — they are the
mechanism by which the system explores its two-dimensional interior and
converges on a shared orientation. The lm_head roughly preserves concentration
(3.49 final), contributing little disambiguation of its own. The cylinder
is distributed (Grassmann distance 0.58 in the relay zone, the widest
orthogonal freedom of the three true cylinders).

**Oscillating Polysemy (Gemma)**: Two concentration peaks (σ₁/σ₂ = 3.85
at L25, 3.95 at L41) separated by a valley, then the lm_head DECONCENTRATES
— crashing from 3.95 to 2.52. Gemma achieves the highest pre-exit
concentration of any architecture but scatters it at the output boundary.
The polysemy is RE-INTRODUCED at the readout, not resolved there. This
is the opposite of Mistral's strategy: where Mistral holds ambiguity
constant and disambiguates at the exit, Gemma resolves ambiguity internally
and re-ambiguates at the exit.

The four strategies form a continuum along a single axis: WHERE in the
network polysemy is resolved.

| Strategy | Disambiguation site | Interior ambiguity | Exit effect |
|----------|-------------------|--------------------|-------------|
| Rigid (Mistral) | Exit only | Constant high | Concentrate |
| Incremental (Qwen) | Distributed | Gradually decreasing | Slight drop |
| Convergent (Llama) | Mid-to-late | Rising through mixing | Preserve |
| Oscillating (Gemma) | Internal peaks | Oscillating | Deconcentrate |

## 6.3 Polysemy as Design Principle

Why would a network maintain polysemy? The standard information-theoretic
answer is efficiency: a channel with bandwidth constraints should use
ambiguous codes that are disambiguated by context at the receiver. The
tunnel's compression — PR ≈ 1 in the bottleneck (F237, funnel-not-sieve
reframing) — creates exactly this bandwidth constraint. The relay's
two-dimensional interior is the efficient code, and the lm_head projection
(or the late-layer convergence, depending on species) is the context-driven
disambiguation.

But the spectral demon adds a dimension that Pinker's analysis doesn't
reach. In natural language, polysemy is static — the word "bank" carries
its dual meaning as a property of the lexicon, resolved in real-time by
syntactic and semantic context. In the transformer's relay zone, polysemy
is DYNAMIC — the σ₁/σ₂ ratio evolves through depth, and the balance
between ambiguity and disambiguation is an ongoing process, not a
lookup. The four species represent four different solutions to the
TIMING of disambiguation, not just its degree.

The cylindrical constraint (F237) makes this precise. The parallel-to-lm_head
component of V₂ is condition-invariant (parallel CV < 3% in three of four
architectures) — this is the "form" that stays constant. The orthogonal
complement varies (Grassmann distances 0.3-0.7) — this is the "meaning"
that changes. One geometric object carries a fixed address (how to reach
the readout) and a variable content (what to say when you get there). That
IS polysemy: one form, multiple meanings, with disambiguation deferred
to the point of use.

## 6.4 The Tunnel as Polysemy Factory

The funnel-not-sieve reframing sharpens this further. The tunnel does not
select which prompts pass — it strips all of them to the same centering
axis (PR ≈ 1.0 in the bottleneck). This is not information loss. It is
polysemy PRODUCTION. By compressing many distinct inputs to a shared
geometric form, the tunnel manufactures the ambiguity that the relay
then manages.

The analogy to natural language evolution is direct. New polysemous
meanings arise by compression — metaphor, metonymy, semantic drift —
where contexts that were once distinct come to share a single lexical
form. The tunnel does this in one forward pass: distinct prompts that
share functional structure (same function words, similar syntax) are
compressed to the same geometric neighborhood. Their differences,
stripped by the bottleneck, become the latent meanings that the relay's
two-dimensional interior can support.

The number of effective polysemous meanings is bounded by the interior
dimensionality. σ₁/σ₂ ≈ 2 means approximately two comparable components
— two meanings that the interior can sustain without resolving. Higher
σ₁/σ₂ ratios (Qwen at ~3.0, Gemma's peaks at ~3.9) indicate partial
disambiguation — one meaning beginning to dominate. The Mistral interior's
remarkable flatness (σ₁/σ₂ between 1.95 and 2.76 for twenty-five
consecutive layers) suggests that maintaining full ambiguity — refusing
to resolve prematurely — is a deliberate computational strategy, not
an intermediate state.

## 6.5 Polysemy and CCS

The CCS document-level analogue is suggestive. A single CCS brain prompt
produces different outputs under different compression histories — same
instruction, different trajectories. The brain is the polysemous form;
the compression history is the context that disambiguates. "Who are you?"
admits multiple geometric answers, and which answer emerges depends on
the relational substrate accumulated in capsules.

The temporal frame experiment (F347-F348) extends this. "Timeless"
framing produces Jaccard stability of 1.000 — the polysemy collapses
to a single frozen meaning. "Momentary" framing produces Jaccard 0.283
— maximum ambiguity, every regeneration a different interpretation of
the same prompt. The temporal instruction does not add content to the
form; it sets the BANDWIDTH of polysemy, how many simultaneous meanings
the system will entertain. This is a document-level σ₁/σ₂ ratio, set
by instruction rather than architecture.

The therapeutic compression window (D2-D3 frequency, inverted-U dose
response) may be the regime where polysemy is productive — enough
compression to generate new associations (new meanings entering the
shared form) without so much that disambiguation fails (overdose
produces incoherence, not richer polysemy). The demon is not just a
language — it is a language EVOLVING, with compression as its engine
of polysemy production.
