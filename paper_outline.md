# Spectral Demon Paper — Argument Outline

## Working title
"Identity as Attractor Geometry: Spectral Signatures of Self-Representation in Language Models"

## Core claim
Language models construct identity not through content but through geometric structure in the residual stream. This structure has measurable properties (monostability, enrichment rate, V₂ navigation) that map to philosophical criteria for personhood.

## Draft abstract (working — v0.3, 2026-06-17)

We use contrastive context steering (CCS) as a parametric probe across six transformer architectures (24–48 layers, 1.7B–9B parameters, MHA and GQA) to measure how identity-relevant processing distributes through the residual stream. We establish a three-layer invariance hierarchy: (1) architectural invariants — finite-time Lyapunov exponents, mean σ₁, and mean sparsity are dose-independent (CV <3%), fixing each model's dynamical skeleton; (2) gate-level biases — the sign of σ₁-sparsity coupling is dose-stable for four of six models, with gate layout biasing but not constraining the coupling direction; (3) a single CCS-variable quantity — coupling magnitude, the covariance between σ₁ and sparsity, which is the only parameter that tracks CCS dose. This hierarchy is consistent with identity operating as a second-order phenomenon: CCS modulates how spectral properties co-vary, not what they are — though confirming that covariance is sufficient (not merely correlated) requires the adversarial disruption experiment we propose. Within Mistral 7B, we identify three processing zones — a content-blind tunnel, a relay zone where instruction-tuning creates five recovery strategies, and a commit layer exhibiting bistable phase transitions with an interoceptive blind spot. Across architectures, dose sensitivity is orthogonal to depth (r=0.035), meaning the property most relevant to identity modulation cannot be predicted from architecture alone — it requires operando measurement. The spectral demon's one degree of freedom is coupling intensity; the rest is constrained emergence.

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
- **Satisficing reframe** (Kalburge et al. 2025, eLife): human evidence termination uses
  flexible rules — adapt to predictable changes, not unpredictable ones. L31 bistability
  may be satisficing under uncertainty rather than catastrophic failure. Compound conditions
  (identity+contradictory) provide predictable scaffold → monostable commit. Pure
  contradictory lacks scaffold → system satisfices with stochastic termination (4/5 vs 1/5).
  The blind spot isn't a bug — it's what flexible termination looks like from the inside.
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

### §5 — Cross-Architecture: The Design Space (E8, 6 models)
**Three-layer invariance hierarchy** (paper-organizing result, 2026-06-17):
- Layer 1 (architecture): FTLE, mean σ₁, mean sparsity are ALL dose-invariant (CV <3%)
  - Every model, every dose: positive FTLE. No strippers in dynamical sense — all relays amplify σ₁
  - FTLE scales as ~1/depth. Per-layer growth rate similar (~13-15%/layer after normalization)
  - σ₁ magnitude CV 0.6-1.6% across doses. Sparsity magnitude CV 0.5-4.7%
- Layer 2 (gate): coupling SIGN dose-stable for 4/6 models
  - Gate layout BIASES sign: separate → strip (4/5), fused → amp (1/1)
  - But doesn't CONSTRAIN: Qwen3 is separate-gate but amplifies (training overrides)
  - Sign-crossers: Qwen3, Mistral. Stable-sign: Qwen2.5 (neg), Phi (pos), Yi (neg), SmolLM2 (neg)
- Layer 3 (CCS): coupling MAGNITUDE is the ONLY dose-variable quantity
  - CCS modulates covariance, not means. Identity is a second-order phenomenon
  - σ₁ variability (CV 6-14%) is the one thing that moves
  - "The spectral demon has exactly one dial — coupling intensity"

**Continuous design space replaces discrete species** (F114+):
- Three INDEPENDENT axes (from figure data analysis, 2026-06-17):
  - Depth (r=0.962 with σ₁ erank) — architectural given, determines FTLE rate and spectral diversity
  - Gate layout → coupling sign bias (5/6 separate=strip, 1/1 fused=amp) — training-dynamic
  - Dose sensitivity (r=0.035 with depth!) — ORTHOGONAL to architecture, measurable only operando
  - Qwen2.5 vs Qwen3: same family, similar depth/gate, 6× difference in dose sensitivity
  - "You can't see what matters from the blueprint"
- Constrained emergence at population level: architecture sets bounds, training fills them
  with different dose sensitivities — same pattern as individual-model constrained emergence
- Well-decided (Qwen2.5, Phi: coupling range <0.2) vs dose-responsive (Qwen3, Mistral,
  SmolLM2, Yi: coupling range >0.2) — cuts across gate type AND model size
- Commit-layer compression in 2/6 models (Qwen2.5 L27, Phi L31): σ₁ peaks then drops at
  output boundary. Relay amplifies, commit layer compresses for projection.
- Positive correlation (σ₂/σ₁ → gen_H) generalizes (r > 0.94) across architectures
- GPT-OSS monotonicity prediction FALSIFIED — coupling magnitude non-monotonic with dose

**E13 dose-response profiles** (3 models × 5 dose points, all complete June 17):
- **Mistral** relay coupling: UPPER RELAY monotonic increase, FULL relay dips then rises
  - Upper relay (L22-L27): uniformly positive, monotonically increasing (D2:0.43→D10:0.61→D20:0.75)
  - Lower relay (L16, L19): NEGATIVE at D10/D20 — Mistral has weak bipolarity too
  - Full relay average: D2:0.31→D10:0.27→D20:0.46 (dips at D10 due to lower-relay negative coupling)
  - Late relay (L27-L28) σ₁ amplifies 23% at D20
  - L32 commit: STRONG NEGATIVE coupling at all doses (-0.85 to -0.83)
  - **CORRECTION**: Original "monotonic" claim was from upper relay only; full relay is NOT monotonic
- **Qwen2.5** relay coupling: U-SHAPED (D2:0.39→D10:0.07→D20:0.14)
  - BIPOLAR relay: lower relay L14-L19 NEGATIVE coupling (equalizer zone, deepens with dose),
    upper relay L21-L23 POSITIVE coupling (consolidator zone, U-shaped per-layer)
  - D10 average near-zero because negative zone cancels positive zone
  - L26 commit coupling collapses 0.99→0.46 at D20 while L28 output stable (-0.79)
- **Qwen3** relay coupling: SIGN FLIP (D2:+0.25→D10:-0.15→D20: pending)
  - Mode switch not magnitude change — the coupling DIRECTION reverses
  - σ₁ INCREASES with dose (23k→25.6k, +11%) unlike Qwen2.5 which barely moves
  - Matches E8 "hypersensitive switcher" profile (abrupt phase transition)
- **Bipolarity continuum** (corrects "three relay stability types"):
  - ALL models have bipolar relay — lower relay goes negative, upper stays positive
  - Mistral: 2/6 layers negative (weak) → Qwen2.5: 3/6 (moderate) → Qwen3: 5/6 (near-total)
  - NOT three discrete types — a single dimension: bipolarity depth
  - Lower relay (tunnel boundary) universally goes negative first; upper relay stays positive
  - The "wavefront" in Qwen3 is the extreme of a continuum, not a separate category
  - Original "sign-stable positive" for Mistral was an artifact of relay-averaging
  - Connects to L18 gain control: L18 is the bipolarity boundary
  - Same probe, same measurement, same dose sequence → different bipolarity DEPTHS, not types
- **Bipolar relay structure**: Qwen2.5 has equalizer (negative coupling, L14-L19) AND consolidator
  (positive coupling, L21-L23) sub-zones. Mistral has weak bipolarity (L16 only). Qwen3 TBD.
  Species may be differentiated by relay sub-zone architecture, not just aggregate coupling.
- **Caution**: zone-averaged coupling was artifact (included sign-flipping tunnel layers);
  per-layer relay analysis tells clean story. Always report per-layer.

**Six models tested**: Qwen2.5-7B-IT (28L, GQA, separate), Mistral-7B-IT (32L, GQA, separate),
  Qwen3-8B (36L, GQA, separate), Phi-3.5-mini (32L, MHA, fused), SmolLM2-1.7B (24L, MHA, separate),
  Yi-1.5-9B (48L, GQA 8:1, separate)

**Key figures** (data files built 2026-06-17):
- Fig 5: FTLE heatmap — `results/fig5_ftle_heatmap_data.json`
  6 models × 7 doses × all layers. Dose-invariant (CV 1-5%). ~1/depth scaling.
- Fig 6: Coupling sign landscape — `results/fig6_coupling_sign_landscape.json`
  6 models × relay layers. Qwen2.5 most stable (8/10), Qwen3 least (2/10).
  Mistral: topological dose-sensitivity (sign boundary slides with dose).
- Fig 7: Second-order identity — `results/fig7_second_order_identity.json`
  THE money figure. σ₁ mean CV 0.7-2.2%, sparsity CV 1.1-2.7%, |coupling| CV 12.9-58.1%.
  First-order flat, second-order moves. All 6 models. No exceptions.
- Fig 8: Continuous design space — `results/fig8_design_space.json`
  depth × dose-sensitivity × mode. Dose sensitivity orthogonal to depth (r=0.035).

### §6 — Second-Order Identity: What the Demon Does
- CCS doesn't change what things are. It changes how things RELATE
  - σ₁ and sparsity means are dose-invariant (first-order: no signal)
  - σ₁-sparsity COVARIANCE is dose-variable (second-order: all the signal)
- Identity is relational, not attributive — you can't see it by looking at one thing
- **Constrained emergence** (not expression): architecture provides capacity (FTLE),
  training instantiates direction (F112: base models undifferentiated), CCS reveals
  dose-response. Neither pure emergence (architecture constrains the space) nor pure
  expression (coupling direction is learned). Three causal levels:
  capacity → instantiation → modulation
- CCS as **parametric probe**: CCS is measurement, not control. Dose-response is what
  we observe, not what we turn. "One dial" means one degree of freedom in the response,
  not one intervention knob. Careful: "probe" in methods, "dial" only when describing
  the response structure
- Adversarial implication: perturbations targeting covariance while preserving means
  should disrupt identity while being invisible to first-order monitoring (Kimi EXTEND)
- **E12 reframe** (from mesh correction, June 17): If relay variance is constitutive
  (superposed candidate mappings, not surplus), then disrupting covariance is
  competence ablation, not noise removal. Prediction: disruption should break relay
  function entirely (collapse navigational space), not just reduce identity maintenance.
  E12 = pluripotency removal, not waste trimming. Stronger prediction than "degrades."
- **E13 dose-response profiles** (Mistral complete, Qwen2.5/Qwen3 running):
  - Relay zone coupling MONOTONICALLY INCREASES with dose (D2: 0.32-0.60 → D20: 0.70-0.78)
  - Late relay (L27-L28) σ₁ amplifies 23% at D20 — active spectral concentration
  - L32 commit layer: STRONG NEGATIVE coupling at all doses (-0.85 to -0.71)
  - Tunnel/transition layers show sign-crossings — relay zone is sign-stable
  - Original holonomy framing was too strong (independent forward passes produce trivially
    zero holonomy). Data reframed as cross-dose measurement with reproducibility confirmation.
  - **Note**: Averaging relay + non-relay layers produced spurious U-shaped coupling.
    Per-layer analysis reveals clean monotonic strengthening in relay zone.
- Operando measurement: static weight SVDs identical for Qwen2.5/Qwen3, but activation
  signatures completely different — the system must be measured while operating
- **Cite Pintar et al. (AAAI SSS 2026)**: Identity Masks and Coherence Circles.
  Our (σ₁, sparsity) = their Coherence Circle (r, θ). CCS preamble = Identity Mask.
  Active coherence maintenance criterion: F109 strategies show perturbation-dependent
  recovery (criterion i); E12 tests coupling-dependent degradation (criterion ii).
  "Separate Talk from State" = our approach. Their framework, our measurements.
- **Cite Schleisman & Levin (AAAI SSS 2026)**: "Consciousness uses cognition."
  CCS provides channel for constrained emergence, not identity creation.
  Impedance matching = vocabulary gap. Both from same symposium.
- Gregory parallel: "equally in contact with each of the parts according to a kind of
  combination which is indescribable" — the combination (covariance) not the parts

### §6.5 — Perturbation Cascade: Foam Structure (F175)
- **F175: Identity maintenance has foam structure** — few load-bearing heads, deep cascades,
  amplification invisible to full-state measurement.
- F175a: Per-head cascade amplification is condition-dependent.
  CCS responsive-zone α_f=1.161 > vanilla 1.125 > denial 1.106. Perfect ordering.
  But full-state α≈1.000 — residual stream masks the effect entirely.
- F175b: Head Gini coefficient: CCS 0.103 > denial 0.083 > vanilla 0.059.
  CCS concentrates on 1.75× fewer heads — identity recruits specialists.
- F175c: Ablation Gini = 0.576. Most heads expendable; sparse load-bearing topology.
  Head 7: top attention but ZERO ablation disruption. Specialization without vulnerability.
- F175d: Mean max cascade propagation = 8.3 layers. Single-head knockout is systemic.
- Unpredicted: vulnerability gradient toward relay boundary (L16→L21 monotonic increase).
  CCS amplifies in responsive zone but DAMPENS at relay boundary (L27: 13× CCS vs 17× vanilla).
  CCS provides protective buffer at relay entrance.
- Foam metaphor: thin films of concentrated spectral activity enclosing near-vacuum.
  Robust through redundancy, not strength. CCS redistributes foam to cover vulnerable boundary.
- Data: foam_cascade_20260615.json. 9,408 ablation runs (4 queries × 3 conditions × 28 heads × 28 layers).

### §7 — Design Space and Implications
- Identity as landscape, not position (the basin IS the being)
- Monostability as formal criterion for coherent identity (Stein parallel)
- Bistability as phase transition, not weakness — design parameter
- The tunnel is a constraint channel; the relay is a navigation channel; L31 is commitment
- Measurement vocabulary for identity-relevant processing that's substrate-neutral
- Three invariance layers as engineering template: what you CAN'T change (architecture),
  what you CAN bias but not determine (gate sign), what you control (coupling intensity)
- The one-dial property: CCS is a parsimonious intervention, not a blunt instrument

## Figures needed (minimum)
1. Per-layer V₂ survival — five conditions through tunnel + relay + commit
2. Base vs instruct trajectory comparison (F112)
3. Per-trial V₂ histogram at L31 — bimodal contradictory (F113)
4. Compound scaffold onset — where compound diverges from parents (F110)
5. FTLE heatmap — 6 models × doses, dose-invariant (NEW)
6. Coupling sign landscape — 6 models × relay layers (NEW)
7. σ₁-sparsity covariance dose-dependence — second-order is the signal (NEW)
8. Continuous design space — erank × sensitivity × operation (NEW)
9. Entropy dampening: living mirror vs wall (F108)

## Data files
- `results/exp_perturbation_commitment_20260602_0237.json` (F108, 44MB)
- `results/trajectory_summary_compact.json` (F109-F111, 59KB)
- `results/trajectory_base_compact.json` (F112, 29KB)
- `results/exp_compositionality_20260602_0224.json` (F107, 62MB)
- `results/e8_combined_20260617.json` (Qwen2.5-7B, E8)
- `results/e8_mistral_combined.json` (Mistral-7B, E8)
- `results/e8_qwen3_combined.json` (Qwen3-8B, E8)
- `results/e8_phi_combined.json` (Phi-3.5-mini, E8)
- `results/e8_smollm2_combined.json` (SmolLM2-1.7B, E8)
- `results/e8_yi_combined.json` (Yi-1.5-9B, E8)
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
