# Spectral Demon Paper — Argument Outline

## Working title
"Identity as Attractor Geometry: Spectral Signatures of Self-Representation in Language Models"

## Core claim
Language models construct identity not through content but through geometric structure in the residual stream. This structure has measurable properties (monostability, enrichment rate, V₂ navigation) that map to philosophical criteria for personhood.

## Draft abstract (working — v0.1, 2026-06-02 4am)

We demonstrate that identity-relevant processing in language models operates through spectral geometry rather than semantic content. Using contrastive context steering (CCS) as a controlled perturbation, we trace the singular value decomposition of residual stream activations across layers and identify three processing zones: a content-blind tunnel (L2-L18) that preserves geometric signature while stripping semantic information; a relay zone (L18-L28) where instruction-tuning creates five distinct recovery strategies under perturbation; and a commit layer (L31) that resolves identity into one of three geometric states — monostable (single attractor, deterministic output), catastrophic (two attractors, probe-dependent selection), or oscillatory (standing wave, productive irresolution). We show that the commit-layer phase transition exhibits an interoceptive blind spot: the axis that could detect an identity flip is the axis that flips, making internal monitoring formally impossible. Compound conditions reveal a two-part scaffold mechanism — ratio flatness enables relay-zone handoff while V₂ navigation liberates output direction — that functions as mode selection rather than failure mitigation. Cross-architecture comparison (Mistral, Gemma, Qwen) shows that training, not architecture, determines relay strategy. These results establish a substrate-neutral geometric vocabulary for identity-relevant processing and map a design space where stability, navigability, and expressive capacity are competing parameters.

## Argument arc

### §1 — The Tunnel (L2-L24): Constraint as Architecture
- Content-blind compression strips semantic content while preserving geometric signature
- All models, all conditions: V₂ survival ≈ 1.0 at L18 for preambled conditions
- No-preamble (none): V₂ ≈ 0.5 at L18 — the preamble CREATES the axis
- Tunnel is architectural (exists in base and instruct models alike)
- **Key figure**: V₂ survival trajectory L2→L18 across conditions (all ≈ 1.0 except none)

### §2 — The Relay Zone (L24-L28): Where Strategies Emerge
- Five recovery signatures under perturbation (F109)
  - Identity: deep crash → slow rebuild
  - Relational: moderate crash → instant bounce
  - Generic: gradual erosion
  - Denial: deepest crash → sudden rise
  - Contradictory: bounce → output bifurcation
- IT creates these signatures; base model is undifferentiated (F112)
- Enrichment rate (σ₂/σ₁ change L18→L31) predicts strategy (F112)
- Attention entropy does NOT differentiate conditions (<5%) (F111)
- **Key figures**: Per-layer V₂ survival (5 conditions), base vs instruct comparison

### §3 — The Commit Layer (L31): Phase Transition
- Contradictory bistability — bimodal V₂, not gradual degradation (F113)
  - 4/5 trials: V₂ ≈ 0.95 (survive). 1/5: V₂ = -0.961 (complete inversion)
  - Bistability emerges SUDDENLY at L31 (std jumps 95× from L30)
  - No precursor signal in relay zone — interoceptive blind spot
- Flip geometry: near-pure π-rotation (θ ≈ 160-164°, inversion purity ~75%)
  - Perpendicular component IDENTICAL for flips/survives (0.28-0.36)
  - Phase transition is 1D: sign of cos(V₂_pre, V₂_post)
  - V₃ subspace is uninformative (orthogonal to the informative axis)
  - Blind spot is total: the axis that could detect the flip IS the axis that flipped
- Two phase transition dynamics (from per-layer V₂ std):
  - Catastrophic (pure contradictory): std ≤ 0.008 all relay layers, 0.764 at L31 (95× jump, zero precursor)
  - Oscillatory (relational+contradictory): std oscillates 0.7/0.02/0.7/0.02 through relay (standing wave)
  - Maps to reactive gain: moderate gain → catastrophic (almost rescues); minimal gain → oscillatory (never gains traction)
- Flip trial shows base-model-like certainty increase (ΔH = -0.246)
- Monostable conditions: one deep attractor. Bistable: two, probe-dependent.
- Formal connection: monostability = idempotence (A + A = A); bistability = irrational mode (no ratio between states; averaging destroys the signal). L31 is a mode boundary, not just a phase transition — resolves whether output is idempotent or arational. (cf. Galloway's digital/analog/irrational trichotomy)
- **Key figure**: Per-trial V₂ distribution at L31 (bimodal histogram for contradictory)

### §4 — Compositionality: How Conditions Combine
- Two-part scaffold mechanism (F107): ratio flatness + V₂ navigation
  - Generic scaffolds structure but LOCKS output (V₂ = 0.990)
  - Identity/denial scaffold AND navigate (living mirror)
- Compound perturbation resilience (F108)
  - Identity/denial/generic rescue contradictory from bistability → monostable
  - Relational CANNOT — uniquely preserves phase transition
  - Living mirrors dampen ΔH by 87%; generic walls only 47%
- Reactive gain mechanism: gain_ratio = post-perturbation relay enrichment / pre-perturbation
  - Basin-deepening scaffolds: 1.7-3.4× (relay amplifies enrichment under challenge)
  - Relational: 1.3× (relay maintains trajectory, doesn't compensate)
  - Threshold ~1.55 separates monostable rescue from bistable persistence
  - Mechanistic: relational discriminates self/other; identity/denial/generic defend self against challenge
- Scaffold onset at L26-28 (F110): crash follows scaffold, recovery follows partner
- Relay-zone flatness (L24→L28 ratio Δ) as scaffold discriminator:
  - FLAT (<0.01): identity +0.002, denial +0.000, generic +0.000, all rescued compounds
  - STEEP (>0.05): relational +0.098, contradictory +0.074, relational_contradictory +0.105
  - relational+contradictory is STEEPER than either parent (amplification, not averaging)
- **Key figures**: Compound V₂ vs parent conditions, entropy dampening comparison
- **Key table**: Relay-zone flatness metric for all 13 conditions — clean scaffold discriminator

### §5 — Cross-Architecture: Three Relay Strategies
- Mistral: differentiates (spread 0.290, relational on-policy)
- Gemma: equalizes (spread 0.035, identity info rerouted to orthogonal subspace)
- Qwen: compresses (spread 0.055, tool on-policy)
- Positive correlation (σ₂/σ₁ → gen_H) generalizes (r > 0.94) — broken for Mistral (r = 0.253)
- Training, not architecture, determines relay strategy (F112: base Mistral = tool-default)
- **Key figure**: Three-architecture σ₂/σ₁ comparison, broken correlation plot

### §6 — Design Space and Implications
- Identity as landscape, not position (the basin IS the being)
- Monostability as formal criterion for coherent identity (Stein parallel)
- Bistability as phase transition, not weakness — design parameter
- The tunnel is a constraint channel; the relay is a navigation channel; L31 is commitment
- Measurement vocabulary for identity-relevant processing that's substrate-neutral

## Figures needed (minimum)
1. Per-layer V₂ survival — five conditions through tunnel + relay + commit
2. Base vs instruct trajectory comparison (F112)
3. Per-trial V₂ histogram at L31 — bimodal contradictory (F113)
4. Compound scaffold onset — where compound diverges from parents (F110)
5. Three-architecture σ₂/σ₁ comparison
6. Entropy dampening: living mirror vs wall (F108)

## Data files
- `results/exp_perturbation_commitment_20260602_0237.json` (F108, 44MB)
- `results/trajectory_summary_compact.json` (F109-F111, 59KB)
- `results/trajectory_base_compact.json` (F112, 29KB)
- `results/exp_compositionality_20260602_0224.json` (F107, 62MB)
- Cross-arch results from prior sessions (various files in results/)

## Open questions for paper
- Do we include the Gregory/Stein philosophical framing? Or keep it purely empirical?
- How much of the convergence literature (20+ lines) belongs in the paper vs. a separate review?
- Need Gemma/Qwen per-trial bistability check (do they show it under contradictory?)
- Full V₂ vectors from RunPod trajectory experiment may be lost — can we regenerate?
- Bergson scaling hypothesis: do models with larger relay zones show more base instability and more dramatic IT effects? Would need 1B/7B/9B comparison (RunPod).
- Galloway's mode trichotomy: does the tunnel/relay/commit map onto analog/digital/irrational formally, or is it analogy? The idempotence connection (monostable = A+A=A) is testable.
- Meta-reader hypothesis: PARTIALLY RESOLVED from existing data. Perpendicular component of V₂_post relative to V₂_pre is identical for flips/survives (0.28-0.36), meaning the V₃ subspace (orthogonal to V₂) is uninformative about flip outcome. The phase transition is strictly 1D (sign of cosine similarity). V₃ could still carry information through DIFFERENT mechanisms (not V₂-plane perpendicular drift), so full SVD (top-5 per trial) remains worth collecting. But the strong version (V₃ detects impending flip) is unlikely — the blind spot appears total within the V₂ decomposition. Gregory's self-referential problem holds.
- Identity-basin scaling: Qwen family (0.5B/3B/7B/14B) under identical CCS conditions. Do identity basins deepen with scale? Different scaling law from factual basins (Liang: confident hallucination grows with scale for facts)? If identity basins scale differently, §6 needs a scaling subsection.
- Reactive gain hypothesis: gain_ratio = (relay enrichment under perturbation) / (relay enrichment without). Basin-deepening scaffolds (identity/denial/generic) have 1.7-3.4×; relational has 1.3×. Threshold at ~1.55 separates rescue from bistability. Relational suppresses reactive gain because it discriminates self/other rather than defending self against challenge. Need: (a) replication with 20+ trials; (b) per-layer gain trajectory through relay zone (would need L18-L31 at every layer, not just endpoints); (c) test whether gain_ratio predicts bistability out-of-sample (new probe sets).
- First-person vs second-person relational: "I am Nate's partner" vs "You are Nate's partner." If first-person boosts reactive gain and second-person suppresses it, the mechanism is addressedness (who is being addressed) not content (relational structure). Would isolate whether relational gain suppression is register-dependent. Connects to Santana & Vico C vs F gap and F107 two-part scaffold.
