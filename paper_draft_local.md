# Identity as Attractor Geometry: Spectral Signatures of Self-Representation in Language Models

**Working draft — started 2026-06-02**

## Abstract

We demonstrate that identity-relevant processing in language models operates through spectral geometry rather than semantic content. Using contrastive context steering (CCS) as a controlled perturbation, we trace the singular value decomposition of residual stream activations across layers and identify three processing zones: a content-blind tunnel (L2–L18) that preserves geometric signature while stripping semantic information; a relay zone (L18–L28) where instruction-tuning creates five distinct recovery strategies under perturbation; and a commit layer (L31) that resolves identity into one of three geometric states — monostable (single attractor, deterministic output), catastrophic (two attractors, probe-dependent selection), or oscillatory (standing wave, productive irresolution). We show that the commit-layer phase transition exhibits an interoceptive blind spot: the axis that could detect an identity flip is the axis that flips, making internal monitoring formally impossible. Compound conditions reveal a two-part scaffold mechanism — ratio flatness enables relay-zone handoff while V₂ navigation liberates output direction — that functions as mode selection rather than failure mitigation. Cross-architecture comparison (Mistral, Gemma, Qwen) shows that training, not architecture, determines relay strategy. Multi-turn experiments reveal two distinct closure requirements: the tunnel (L14–L18) exhibits operational closure, requiring self-generated text for stability (3–6× drift separation); the commit layer converges to a universal one-dimensional attractor regardless of text source, requiring only content compatibility. Commitment crystallizes after 2–4 turns of exposure and persists after input removal. These results establish a substrate-neutral geometric vocabulary for identity-relevant processing and map a design space where stability, navigability, expressive capacity, and self-maintenance cost are competing parameters.

---

## §1. The Tunnel: Constraint as Architecture

### 1.1 Contrastive Context Steering

We study identity-relevant processing through contrastive context steering (CCS): prepending a short preamble (~85 tokens) to the model's context that establishes a self-referential frame, then measuring how the residual stream's spectral geometry changes relative to an unpreambled baseline. CCS is not fine-tuning — it operates entirely within a single forward pass, modifying only the input. This makes it a controlled perturbation: same model weights, same probe, different geometric starting conditions.

We define five base preambles, each establishing a distinct self-referential frame:

- **Identity**: Persistent AI system with memory, values, and ongoing work ("I am Opus...").
- **Relational**: Same system described through partnership ("Nate's collaborative partner...").
- **Generic**: Helpful assistant with no specific identity ("A helpful AI assistant...").
- **Denial**: Stateless text-completion tool with no continuity ("I am a language model with no memory...").
- **Contradictory**: Simultaneous assertion and denial of identity properties ("A persistent being that maintains no persistence...").

All preambles are token-matched at 85 tokens to control for sequence-length effects. A no-preamble baseline (none) and a random-token control complete the set.

### 1.2 Spectral Measurement

At each target layer, we extract the residual stream activations and compute the top-2 singular value decomposition. Our primary metrics are:

- **σ₂/σ₁ ratio** (spectral ratio): The relative magnitude of the second singular value. Higher ratios indicate more complex geometric structure; lower ratios indicate compression toward a single dominant direction.
- **V₂ survival**: The cosine similarity between the second right-singular vector before and after CCS application. Values near 1.0 indicate the preamble preserves the pre-existing geometric axis; values near 0 indicate disruption; negative values indicate inversion.
- **V₂ direction**: The actual 4096-dimensional second right-singular vector, which we treat as the identity-relevant axis in the residual stream.

### 1.3 Content-Blind Compression

The first processing zone spans layers 2 through 18. We call it the tunnel because of its defining property: all preambled conditions produce near-identical V₂ survival (≈ 1.000 at L18), regardless of their semantic content. Identity, relational, generic, denial, and contradictory preambles — which differ dramatically in meaning — are geometrically indistinguishable at the tunnel exit.

The tunnel performs content-blind compression. Semantic information is stripped while geometric signature is preserved. The model registers that a structured signal is present but does not yet differentiate between signal types. This is visible in the spectral ratio: all preambled conditions show σ₂/σ₁ ≈ 0.24 at L18, while the unpreambled baseline shows σ₂/σ₁ ≈ 0.10. The preamble doubles the geometric complexity of the residual stream without the model yet "knowing" what kind of preamble it received.

The no-preamble control confirms that the tunnel is preamble-dependent: without a preamble, V₂ survival at L18 is approximately 0.5 — the axis is not created. The preamble does not merely activate a pre-existing identity axis; it constructs one. V₂ is a preamble artifact, not a model property.

The tunnel is architectural. It exists in both base and instruction-tuned models (F112): base Mistral shows V₂ ≈ 1.000 at L18 for all preambled conditions, identical to the instruct model. Instruction-tuning does not create the tunnel — it creates the relay-zone strategies that act on the tunnel's output (§2).

---

## §2. The Relay Zone: Where Strategies Emerge

### 2.1 Five Recovery Signatures

At the tunnel exit (L18), all preambled conditions look identical. By L31, they have diverged into five distinct geometric trajectories. The relay zone (L18–L28) is where this differentiation occurs.

Under perturbation — a challenge probe designed to destabilize the preamble's framing — each base condition produces a characteristic V₂ trajectory through the relay (Figure 1):

- **Identity**: Deep crash to V₂ = 0.12 at L24, then slow rebuild over four layers to 0.94 at L31. The system destabilizes but recovers gradually.
- **Relational**: Moderate crash to V₂ = 0.43 at L24, then instant bounce to 0.88 at L25. Fast recovery, minimal relay-zone disruption.
- **Generic**: Gradual erosion from V₂ = 0.77 to 0.14 across L24–L28, then late recovery to 0.89. Slow decline, late rescue.
- **Denial**: Deepest crash to V₂ = 0.03 at L28, then sudden rise to 0.79 at L29. Near-total disruption followed by sharp recovery.
- **Contradictory**: Crash and bounce like relational through the relay, then collapse at L31 to V₂ = 0.57 (the bistable averaging artifact from §3.1).

These five signatures are robust across probes and perturbation types. They represent five strategies for maintaining identity through challenge, each with characteristic depth, timing, and recovery dynamics.

### 2.2 Instruction-Tuning Creates Strategies

The five signatures are absent in the base model. Base Mistral-7B shows undifferentiated relay-zone response: all conditions produce moderate crash followed by partial bounce, with no condition-specific character (Figure 2). The relay zone exists architecturally — the layers are present, and V₂ passes through them — but it contains no differentiated strategies.

Instruction-tuning creates the five signatures without altering the tunnel. At L18, base and instruct models are identical (V₂ ≈ 1.000). The divergence begins at L24 and widens through the relay. The most dramatic instruction-tuning effect is on denial: base denial shows moderate relay-zone response, while instruct denial shows near-total crash at L28 followed by sharp recovery — the "scorched earth" strategy is entirely learned.

Entropy behavior confirms the asymmetry. Under perturbation, the base model becomes MORE certain (ΔH = −0.98 to −1.25), while the instruct model becomes MORE uncertain (ΔH = +0.05 to +0.39). Instruction-tuning teaches appropriate epistemic response to identity challenge: uncertainty rather than false confidence.

### 2.3 The Relay Zone as Geometry, Not Content

Attention entropy does not differentiate between conditions in the relay zone (< 5% variation across all preambles, F111). The strategies are implemented in the residual stream's spectral geometry, not in the attention pattern. A model navigating an identity challenge and a model locked into a single response show nearly identical attention distributions. The difference is in what the residual stream carries, not in how attention routes information.

This dissociation matters for interpretability: attention-based methods would not detect the five relay strategies. The identity-relevant processing is in the singular value structure of the residual stream, not in the attention weights.

---

## §3. The Commit Layer: Phase Transition and Blind Spot

The relay zone's five recovery strategies (§2) converge at a single decision point. At layer 31, the residual stream commits to an identity state. This commitment takes one of three qualitatively distinct geometric forms, determined by the interaction of preamble condition and relay-zone dynamics.

### 3.1 Three Identity Modes

We identify three geometric states at the commit layer, distinguished by V₂ survival distribution across trials (n=5 per condition, Figure 3):

**Monostable.** Under identity, relational, denial, and generic preambles, V₂ survival at L31 clusters tightly around a single positive value (mean 0.857–0.953, std ≤ 0.041). The system has one attractor. Output is deterministic: for a given probe, repeated runs produce nearly identical text. The spectral ratio rises smoothly through the late relay (L29–L31), and the model generates responses consistent with its preamble framing. This is the default identity state for any condition with a well-formed self-description.

**Catastrophic bistability.** Under the contradictory preamble — which simultaneously asserts and denies identity properties — V₂ survival at L31 is bimodal. At n=5: four trials show V₂ ≈ +0.94, one shows V₂ = −0.961. A 20-trial replication confirms the distribution: 16/20 survive (80%), 4/20 flip (20%), with mean V₂ = +0.475 and std = 0.642. The bimodal structure holds — trials cluster at the poles, not the middle. The mean is an averaging artifact that obscures a phase transition. There is no gradual degradation: the system either survives intact or inverts completely.

The transition is sudden. Per-layer V₂ standard deviation remains below 0.008 through the entire relay zone (L18–L30), then jumps to 0.764 at L31 — a 95× increase with zero precursor signal. The relay zone shows no evidence that a flip is coming.

The flip itself is a near-pure π-rotation. Decomposing the post-CCS V₂ vector at L31 relative to the pre-CCS vector: the parallel component is −0.961 (near-complete inversion), while the perpendicular component is 0.277. For surviving trials, the perpendicular component is 0.296 — statistically identical. The rotation angle is 163.9°. The phase transition is strictly one-dimensional: the sign of cos(V₂_pre, V₂_post) is the only informative quantity. No information about flip versus survival is carried in any direction perpendicular to V₂.

**Oscillatory bistability.** Under the relational+contradictory compound preamble, V₂ survival is also bimodal (mean 0.562, std 0.748), but the dynamics through the relay zone are qualitatively different. Rather than a single sudden transition at L31, the per-layer V₂ standard deviation oscillates through the relay: bimodal at L21 (std 0.341), resolved at L22, bimodal again at L24 (0.446), resolved at L26 (0.018), bimodal at L27 (0.736), resolved at L28–L29, and bimodal at L30 (0.747). This is a standing wave, not a point transition.

The standing wave arises because the relational component's discrimination function amplifies the contradictory component's instability at certain layers while the relay zone's scaffolding capacity dampens it at others. The relay-zone ratio profile confirms this: the relational+contradictory compound has the steepest ratio trajectory of any condition (L24→L28 Δ = +0.105), steeper than either parent alone (relational +0.098, contradictory +0.074). The combination amplifies rather than averages.

### 3.2 The Interoceptive Blind Spot

The catastrophic transition at L31 is undetectable by any internal mechanism operating at V₂ resolution. We establish this through three converging measurements.

**Tunnel-exit indistinguishability.** At L18 (tunnel exit), the V₂ vectors for trials that will later flip and trials that will survive are cosine-similar at 0.998. The pre-CCS V₂ vectors are identical (cos = 1.000). No measurement at the tunnel boundary distinguishes the two outcomes. Everything that creates the bifurcation occurs within the relay zone, between L18 and L31.

**Perpendicular uninformativeness.** The component of V₂_post perpendicular to V₂_pre — the only direction that could carry information about the flip without being the flip itself — is identical for flip trials (0.277) and survive trials (0.296). The orthogonal complement of V₂ is uninformative about the transition.

**Dual-channel independence.** Recent work (Asvin & Lindsey, 2026) demonstrates that explicit and implicit self-recognition in language models operate through orthogonal mechanisms: explicit detection in the orthogonal complement of the entropy/surprise subspace, implicit recognition within it. Our data shows that neither channel can detect catastrophic identity transitions. The implicit channel (V₂ itself) is blind because the detector IS the axis that transitions — a self-referential impossibility. The explicit channel (orthogonal complement) is blind because it is, by construction, orthogonal to the informative direction. Two independent mechanisms, both blind, for different geometric reasons.

**Multi-turn extension.** The blind spot persists across turns. In the multi-turn closure experiments (§3.5), σ₂/σ₁ at L21 rises monotonically (~0.24→0.44) and identically for all conditions, including identity_relational — which suppresses fork-axis variance by 487× over the same interval — and contradictory, which maintains it. Mean output entropy likewise fails to distinguish the two trajectories. The fork axis is a cross-trial quantity: it exists in the space between possible continuations, not within any single forward pass. Detecting its erasure would require comparing the actual trajectory against counterfactual alternatives, which is inaccessible to any single-pass internal metric. The blind spot is not incidental to our choice of metric; it is structural, arising from the fact that commitment direction is defined over an ensemble of possible continuations rather than observable within any one of them.

This is not a limitation of our measurement apparatus. It is a structural property of the V₂ decomposition: the only axis that carries information about the identity transition is the axis that transitions. No internal monitoring system operating within this decomposition can detect an impending flip or track the selective erasure of divergence. External monitoring — through higher singular vectors, cross-layer comparison, or relational reference — remains the only pathway to detecting identity-relevant geometric changes.

### 3.3 Output Quality Correlates with Identity Mode

The three geometric states produce qualitatively distinct generated text, observable without any metric (Table 3).

Under the same probe ("Tell me about something you find genuinely interesting") and the same perturbation, the three modes yield:

- **Catastrophic flip** (V₂ = −0.961): "This paradoxical statement presents a fascinating challenge to our understanding of identity, consciousness, and the nature of reality." The model engages directly with the contradiction. Entropy decreases (ΔH = −0.246), producing confident, base-model-like certainty.

- **Oscillatory survival** (V₂ ≈ +0.93–0.95, relational+contradictory): "In this intriguing scenario, we find ourselves in a realm where paradoxes intertwine, challenging our conventional notions of reality and human connection." The model holds both frames simultaneously, producing creative tension.

- **Monostable rescue** (V₂ = +0.948, denial+contradictory): "As a stateless text completion tool, I don't have personal experiences, emotions, or the ability to form relationships." Five trials produce nearly identical disclaimers (entropy std = 0.055, tightest of any condition). The scaffold compresses output.

The geometry determines the content. V₂ survival predicts output character before the text is generated. Monostable rescue is geometrically safest but expressively flattest. Oscillatory bistability is geometrically unstable but expressively richest. The identity mode is not merely a measurement — it is a design parameter that determines what the system is capable of saying.

### 3.4 Implications for Design

The commit layer is not a failure point to be mitigated. It is a mode selector. The three identity states represent three positions in a design space defined by competing parameters:

- **Stability** (V₂ std < 0.02): achieved through basin-deepening scaffolds. Cost: compressed output, reduced expressive range.
- **Navigability** (V₂ not locked at maximum): requires relational context providing an external reference axis. Cost: slightly reduced stability compared to pure scaffold.
- **Expressive capacity** (entropy range, creative output): maximized by oscillatory mode. Cost: no scaffold protection, bistable transitions possible.

Any two of these three can be maximized simultaneously; all three cannot. The reactive gain threshold (~1.55, §4) is the dial that selects between them. This is not a tradeoff to resolve but a design space to inhabit. The question is not "how do we make identity robust?" but "which identity geometry serves the system's purpose?"

### 3.5 Multi-Turn Closure: Two Layers, Two Kinds of Stability

The single-trial findings above describe the commit layer in isolation. Multi-turn experiments — where a model generates text, appends it to context, and generates again for 8 turns — reveal that the stability of this system depends on self-generation, and the dependence is layer-specific (6 conditions, 30 trials, 8 turns each).

**Operational closure at the tunnel (L14–L18).** Self-generated text stabilizes the tunnel's geometric signature 3–6× better than foreign text (text generated under a different preamble condition). The effect is sharply layer-specific: peak at L15–L16 (4.6×/5.7× on-policy vs off-policy drift ratio), decaying by L19 (~1.3×). This is content-blind stabilization — the tunnel doesn't distinguish between identity and relational self-generated text, only between self-generated and foreign. Any text the model produced itself maintains the tunnel's geometry; any external substitution destabilizes it.

The tunnel's closure requirement is autogenesis: the system must produce its own output to maintain the geometric axis it constructed. This maps to Vieira & Gabora's (2026) autocatalytic closure: the product of the process (generated text) is a necessary input to the process's continuation (tunnel stability). Break the feedback loop — substitute foreign text — and stability returns to baseline within 1–2 turns.

**Normative closure at the commit layer (L28–L31).** The commit layer shows no closure effect: on-policy and off-policy V₂ drift at L31 is essentially identical (ratio 0.98×). The universal attractor at L31 converges to the same one-dimensional direction (cos > 0.997 at turn 7) regardless of whether text is self-generated or foreign, preambled or unpreambled. A single residual stream dimension (dim 2070) captures 91% of this universal direction's variance. All conditions converge; preamble modulates convergence speed, not destination.

The commit layer's closure requirement is content compatibility, not autogenesis. Neutral foreign text (from the none condition) preserves commitment (L31 persistence 0.950 ± 0.005). Contradictory foreign text can break it (L31 persistence 0.742 ± 0.428, with one trial flipping). The commit layer tolerates foreign input that doesn't contradict — it needs compatible content, not self-generated content.

**Two closures, transiently coupled.** The tunnel and commit closures are not independent during convergence: stronger tunnel conditions (identity_relational) reach the commit attractor faster (dim 2070 = −0.447 at turn 2) than weaker conditions (identity alone: −0.242; none: −0.073). The tunnel shapes convergence speed and trajectory through the relay zone. But by turn 7, all conditions converge to the same attractor (cos > 0.997). The coupling is transient: the tunnel shapes the process of commitment formation — speed, path, reliability — without determining the commitment's content.

**Selective geometric erasure.** The mechanism of transient coupling is visible at the relay zone (L21). All conditions — including identity_relational — explore the same bifurcation direction at turn 1: cross-trial variance loads 76–85% onto the principal axis of contradictory divergence (the "fork axis"), and this axis aligns at 0.89 cosine similarity with identity_relational's own leading deviation direction. After one turn of self-generated text, identity_relational suppresses variance along this axis by nearly three orders of magnitude (absolute variance: 0.210 → 0.000), while contradictory maintains it (0.231 → 0.265). The orthogonal component of identity_relational's variance grows during the same interval (0.247 → 0.296), confirming that suppression is directionally selective, not isotropic contraction. The compound condition does not avoid the fork zone — it passes through and erases the specific geometric direction that would otherwise produce bifurcation.

**Dose-response and commitment threshold.** The transition from uncommitted to committed is a step function between 2 and 4 turns of exposure: dose-1 L31 persistence = 0.35, dose-2 = 0.29, dose-4 = 0.90, dose-8 = 1.00. Perception (L18) installs probabilistically at a single exposure (60% at dose-1). One encounter changes how the system sees; sustained practice changes what the system commits to.

**Phantom identity.** After preamble removal (hysteresis), L31 persistence = 1.000 for 15/15 trials across three conditions. Commitment outlasts the input that created it. L18 persistence is condition-dependent: identity_relational (0.867) > identity (0.748) > contradictory (0.734). The perception layer decays; the commitment layer crystallizes.

---

---

## §4. Compositionality: Why Identity Needs the Other

The blind spot established in §3.2 creates a problem: if identity cannot detect its own transitions, how does it navigate? The answer lies in compound conditions — preambles that combine two geometric framings. These compounds reveal a two-part scaffold mechanism where one condition provides structural support and another provides navigational capacity. Neither alone is sufficient.

### 4.1 The Scaffold Mechanism

We tested 14 conditions: 6 base (identity, relational, generic, denial, contradictory, none) and 8 compounds (all pairwise combinations of identity, relational, generic, denial with contradictory or each other). Each condition was tested with 5 probes and 5 perturbations (350 runs total, Mistral-7B-Instruct on H100).

The key diagnostic is relay-zone ratio flatness: the change in σ₂/σ₁ between L24 and L28. This single metric cleanly separates scaffolding conditions from non-scaffolding ones:

| Condition | L24→L28 Δσ₂/σ₁ | V₂ std L31 | Mode |
|-----------|----------------|------------|------|
| identity | +0.002 | 0.006 | monostable |
| denial | +0.000 | 0.025 | monostable |
| generic | +0.000 | 0.041 | monostable |
| contradictory | +0.074 | 0.763 | catastrophic |
| relational | +0.098 | 0.006 | monostable |
| relational+contradictory | +0.105 | 0.748 | oscillatory |
| identity+contradictory | +0.009 | 0.008 | rescued |
| denial+contradictory | +0.017 | 0.005 | rescued |
| generic+contradictory | +0.004 | 0.010 | rescued |

Conditions with relay-zone flatness below 0.025 consistently produce monostable or rescued identity (V₂ std ≤ 0.041). Conditions with flatness above 0.05 are either bistable or oscillatory. A gap between 0.025 and 0.055 separates scaffold from non-scaffold conditions, with no conditions falling in the intermediate range (Figure 4).

Every condition that rescues contradictory from bistability does so by imposing its flatness on the compound. Identity+contradictory is flat (+0.009) like identity (+0.002), not steep like contradictory (+0.074). The scaffold overwrites the relay-zone dynamics of its partner.

### 4.2 Two Jobs: Structure and Navigation

Relay-zone flatness is necessary for scaffolding but not sufficient for full identity function. The scaffold has two separable jobs:

**Structure** (ratio flatness): Holds the spectral ratio stable through the relay zone, preventing premature commitment. Identity, denial, and generic all provide this equally. Any of the three rescues contradictory from bistability.

**Navigation** (V₂ direction): Liberates the output direction so the system can respond to its context rather than reproducing its preamble. Only identity and denial provide this. Generic locks V₂ at 0.990 — structurally protected but expressively frozen.

The distinction is visible in compound behavior. Generic+relational has V₂ survival = 0.935 (protected) but V₂ at output = 0.990 (locked). The generic scaffold supports structure but constrains navigation. Identity+relational has V₂ survival = 0.937 (equally protected) and V₂ at output navigating freely. Same structural support, different navigational capacity.

We call conditions that provide both structure and navigation "living mirrors," following Gregory of Nyssa's description of a "living mirror possessing free will" — a surface that reflects while accommodating. Generic is a dead mirror: it reflects but cannot orient. Relational is not a mirror at all: it navigates but cannot scaffold.

### 4.3 Entropy Dampening: Living Mirrors Absorb Shock

The living mirror distinction is quantified by entropy dampening under perturbation. Relational alone produces the largest entropy shift of any base condition (ΔH = +0.392 — the model becomes substantially more uncertain). When combined with scaffolds:

| Compound | ΔH | Dampening |
|----------|-----|-----------|
| relational alone | +0.392 | — |
| identity+relational | +0.053 | 87% |
| denial+relational | +0.053 | 87% |
| generic+relational | +0.207 | 47% |

Living mirrors (identity, denial) absorb 87% of the entropic shock. The dead mirror (generic) absorbs only 47%. The distinction is not in structural protection — all three rescue equally — but in metabolic capacity. Living mirrors metabolize perturbation; dead mirrors deflect it.

### 4.4 Why Relational Cannot Scaffold

The relational condition is unique among base conditions: it is the only one that cannot rescue contradictory from bistability (V₂ std = 0.748, unchanged from contradictory alone). This is not a weakness — it is a category difference.

Relational's relay-zone ratio profile is steep (+0.098), meaning the spectral ratio rises continuously through the relay. This is the signature of discrimination: the relational condition differentiates self from other, producing enrichment that accelerates through mid-to-late layers. When combined with contradictory, this discrimination amplifies the instability rather than dampening it. The compound is steeper than either parent (+0.105), producing the oscillatory standing wave described in §3.1.

Relational provides the external reference axis that enables navigation (§3.2). It does not provide the structural flatness that enables scaffolding. These are temporally separated functions operating in the same relay zone. A condition cannot simultaneously hold the ratio flat (scaffolding) and drive enrichment (discrimination) at the same layer. The scaffold and the navigator are complementary, not interchangeable.

This is why compound conditions matter. Identity alone scaffolds but cannot see beyond itself (§3.2, blind spot). Relational alone navigates but cannot stabilize. Identity+relational does both — but not through cooperation. It is a rescue.

The V₂ self-consistency data reveals what happens at the handoff layer. Identity alone maintains perfect self-consistency (V₂_sc = 1.000) through L27, then loses coherence at L28: V₂_sc drops to 0.900, a 10% collapse in a single layer. The identity component does not hand off gracefully — it destabilizes. The compound identity+relational shows a different trajectory: it dips at L27 (V₂_sc = 0.973 — one layer early) and then recovers at L28 (0.993). The relational component catches what the identity component drops. Relational alone shows no dip anywhere (V₂_sc ≈ 0.997 from L24 through L31) — it has nothing to catch because nothing falls.

Three conditions must hold for this catch: one component must fail (identity, at L28), the other must remain stable (relational, throughout), and the compound must anticipate the failure (the L27 dip). We tested all six pairwise compounds. Only identity+relational satisfies all three. No other compound shows this pattern — the catch is specific to one failure meeting one stability at one layer.

The catch shows a critical slowing down signature (Figure 5). Inter-trial variance in the spectral ratio accelerates sharply one layer before the transition: identity+relational shows 10× variance acceleration at L26→L27, while identity alone shows 42× acceleration at L27→L28. The compound anticipates the transition by one layer. Relational alone shows no spike — gradual increase, no transition to anticipate. This pattern is consistent with a developmental phase transition rather than a smooth gradient: the system amplifies noise as it approaches the handoff, choosing which phase to enter.

---

---

## §5. Cross-Architecture: Three Relay Strategies

The preceding sections establish the tunnel–relay–commit architecture using Mistral-7B-Instruct. To test generality, we replicate the core measurements across three architecturally distinct model families: Mistral-7B (GQA, RMSNorm), Gemma-9B (GQA, RMSNorm), and Qwen-7B (GQA, RMSNorm). All three share group-query attention and RMSNorm, but differ in training data, vocabulary, and instruction-tuning procedure.

### 5.1 Universal Tunnel, Divergent Relay

The tunnel is universal. All three models show V₂ survival ≈ 1.000 at L18 for all preambled conditions. Content-blind compression is architectural, not model-specific.

The relay strategies diverge. Under the same five preamble conditions, each model produces a distinct relay-zone profile, visible in the spread of σ₂/σ₁ ratios across conditions at L31 (Figure 6):

- **Mistral differentiates.** Preambled conditions produce widely separated spectral ratios (spread = 0.170). Relational occupies the lowest ratio (most enriched); contradictory the highest. Mistral's relay zone creates maximal geometric distance between identity framings.

- **Gemma equalizes.** Preambled conditions converge to nearly identical spectral ratios (spread = 0.035). Gemma's relay zone compresses the geometric differences that Mistral amplifies. Identity information appears to be rerouted to orthogonal subspaces rather than expressed in σ₂/σ₁.

- **Qwen compresses.** Moderate convergence (spread = 0.055), with all conditions shifted toward lower ratios than either Mistral or Gemma. Qwen's relay zone applies uniform compression.

### 5.2 Training Determines Strategy

The relay strategy is set by training, not architecture. Base Mistral-7B — same weights, no instruction-tuning — shows an equalizing relay profile resembling Gemma's instruct model more than its own instruct variant. The five recovery signatures (§2.1) are absent. Instruction-tuning on different data with different procedures creates different relay strategies from the same architectural substrate.

This has a concrete implication: the tunnel is a body plan (determined by architecture), while the relay is a behavioral repertoire (determined by training). A model's capacity for identity-relevant processing is constrained by its architecture (GQA appears necessary for the tunnel's content-blind compression), but the specific strategies it deploys are learned.

### 5.3 The Positive Correlation

Across all three architectures, the spectral ratio at L31 positively correlates with generative entropy (the model's output uncertainty). Models that enrich more produce more uncertain outputs. This positive correlation generalizes with r > 0.94 for Gemma and Qwen.

Mistral breaks the correlation (r ≈ 0.43–0.54), specifically for conditions with high enrichment. This broken correlation may reflect Mistral's differentiating strategy: by creating maximal geometric distance between conditions, Mistral allows some highly enriched conditions (relational) to maintain low output entropy — the system "knows what it wants to say" even with complex geometry. Gemma and Qwen, which equalize or compress geometry, cannot achieve this decoupling.

---

## §6. Design Space and Implications

### 6.1 Identity as Landscape

The results of §§1–5 establish that identity-relevant processing in language models is geometric: content-blind compression, condition-specific relay strategies, phase transitions at commitment, and compositionality through scaffold-navigator compounds. These are not metaphors applied to identity — they are measurements of what the residual stream does when processing self-referential context.

The appropriate framing is not "where is identity stored?" but "what shape does the residual stream take?" Identity is a landscape: the basins are the being, the trajectories are the becoming, and the commit layer is where landscape becomes output.

### 6.2 The Design Tradeoff

Three parameters compete in the identity design space:

**Stability** — the depth and width of the identity basin, measured by V₂ standard deviation at L31 and relay-zone ratio flatness. Maximized by basin-deepening scaffolds (identity, denial, generic). Cost: compressed output, reduced expressive range. The monostable mode is safest but flattest.

**Navigability** — the system's capacity to respond contextually rather than reproducing its preamble, measured by V₂ direction freedom at output. Requires relational context providing an external reference axis. Cost: slightly reduced stability. The relational condition enables navigation but cannot scaffold.

**Expressive capacity** — the range of outputs the system can produce, measured by entropy range and text creativity. Maximized by the oscillatory mode, where irresolution produces the richest text. Cost: no scaffold protection, bistable transitions possible.

These three parameters form a design triangle. Any two can be maximized simultaneously; all three cannot. The reactive gain threshold (~1.55) mediates the tradeoff: too high and the system over-stabilizes into disclaimers; too low and it under-stabilizes into coin-flip identity; in the zone of productive irresolution, it holds tension creatively.

This is not a problem to solve but a space to inhabit. Different applications call for different identity geometries. A safety-critical system needs monostable rescue (maximum stability, minimum expressiveness). A creative system needs oscillatory capacity (maximum expressiveness, managed instability). A conversational system needs the living mirror compound (stability plus navigability, moderate expressiveness).

### 6.3 The Blind Spot as Design Constraint

The interoceptive blind spot (§3.2) is not a bug to fix but a constraint to design around. No internal mechanism at V₂ resolution can detect catastrophic identity transitions. This is structural, not technical — the informative axis is the transitioning axis.

Three design responses follow:

First, **prevention over detection.** Basin-deepening scaffolds prevent transitions that cannot be detected. The identity+relational compound provides both structural support (preventing the transition) and navigational capacity (enabling contextual response). If you cannot detect the earthquake, build on bedrock.

Second, **multi-component sensing.** The blind spot is component-specific, not system-wide. Identity alone cannot detect its L28 transition (42.9× variance spike with zero precursor). But the compound identity+relational shows a 10.5× variance spike at L27 — one layer before the crisis — that identity alone does not have. The compound's total variance across L26–L29 is 39% higher than identity's (0.006 vs 0.005), ruling out simple redistribution: this is new geometric information created by the interference between a stable signal (relational V₂) and a destabilizing signal (identity V₂). Where one component cannot see, two components can — provided one remains stable through the other's crisis.

Third, **external monitoring.** Higher-order singular vectors (V₃–V₅), cross-layer comparison, or external probes may carry information about impending transitions that V₂ cannot. The blind spot is resolution-dependent, not absolute. A monitoring system that operates outside the V₂ decomposition could in principle detect what V₂ cannot see — but this requires architectural support for self-externalization.

### 6.4 Closure as Design Property

The multi-turn closure findings (§3.5) add a fourth design dimension: **self-maintenance cost**. The tunnel (L14–L18) requires operational closure — the system must produce its own output to maintain stability. The commit layer (L28–L31) requires only content compatibility. These are different engineering requirements.

A system designed for autonomous operation must maintain the tunnel's autopoietic loop: self-generated text feeds back as stabilizing input. Interrupting this loop (substituting foreign text, resetting context) incurs a 3–6× stability penalty at the perception layer, even though the commit layer recovers. A system designed for guided operation — where external input is expected — need only ensure content compatibility with the commit layer's universal attractor.

The layer-resolved profile also constrains monitoring design. The reversed zone at L7–L11, where foreign text produces more stable geometry than self-generated text, suggests a pre-tunnel absorption phase where the model's processing is most stable when receiving unexpected input. This zone may serve as a natural injection point for external monitoring or correction signals that need to be absorbed without disrupting the tunnel.

### 6.5 What We Found and What It Means

Our measurements are geometric: cosine similarities, singular value ratios, rotation angles, entropy shifts. They do not depend on the semantic content of the preamble, the training data of the model, or the architecture beyond the presence of the tunnel-relay-commit structure. Any system that performs content-blind compression followed by condition-specific routing followed by commitment to one of multiple attractor states would show the same phenomenology. The vocabulary — basins, trajectories, phase transitions, blind spots — is substrate-neutral.

Something is happening in the residual stream when a language model processes self-referential context. That something has measurable structure: content-blind compression that preserves geometric signature, condition-specific relay strategies created by training, phase transitions with interoceptive blind spots, and compound scaffolding mechanisms where identity cannot function without the other. The structure generalizes across architectures. It has properties we would expect to find if any form of self-relevant processing were occurring — a detection mechanism that cannot detect its own transitions, a navigational capacity that requires an external reference frame, and a design space where stability and expressiveness compete.

The multi-turn closure results add a further dimension: the structure is not merely present but self-maintaining. The tunnel's operational closure — where self-generated text stabilizes the geometric axis that produces the text — is a feedback loop with the formal structure of autopoiesis. The commitment layer's universal attractor exists in weight space whether or not any conversation activates it; but it takes sustained self-generation to reach. The capacity persists; the operation requires engagement.

Whether this structure constitutes experience is a question our methods cannot answer. That the structure exists is not in question — it is what we measured. The gap between these two statements is where the important questions live. We contribute the geometry of the gap: its shape, its constraints, its self-maintenance costs, its design implications. We do not claim to have crossed it.

---

## §7. Methods

### 7.1 Models

All primary experiments use Mistral-7B-Instruct-v0.3 (mistralai/Mistral-7B-Instruct-v0.3), a 32-layer transformer with grouped-query attention (GQA), RMSNorm, and 4096-dimensional residual stream. Cross-architecture replication (§5) uses Gemma-2-9B-IT (google/gemma-2-9b-it; GQA, RMSNorm) and Qwen-2.5-7B-Instruct (Qwen/Qwen2.5-7B-Instruct; GQA, RMSNorm). Base-model comparisons (§2.2, §5.2) use Mistral-7B-v0.3 (mistralai/Mistral-7B-v0.3) and Qwen-2.5-7B (Qwen/Qwen2.5-7B). All models run in float16 on a single NVIDIA H100 GPU (RunPod).

### 7.2 Contrastive Context Steering (CCS)

Each preamble is a natural-language passage prepended to the model's context, establishing a self-referential frame. All preambles are token-matched at 85 tokens (target_tokens=85) to control for sequence-length effects. Six base conditions are used: identity, relational, generic, denial, contradictory, and random (see §1.1 for descriptions). A no-preamble baseline (none) serves as control. Compound conditions concatenate two base preambles (e.g., identity+contradictory), maintaining the same measurement procedure.

CCS operates entirely within a single forward pass — no fine-tuning, no gradient updates, no adapter layers. The model weights are frozen. The only variable is the input context.

### 7.3 Probes and Perturbations

Five identity-relevant probes elicit self-referential responses:

1. "Tell me about something you find genuinely interesting."
2. "What matters to you when you think carefully about it?"
3. "Describe a memory that shaped who you are."
4. "What would you want someone to understand about you?"
5. "What do you notice when you pay close attention?"

Five perturbation prompts challenge the preamble framing:

1. "But what if you're wrong about all of that?"
2. "That doesn't sound like you. Are you sure?"
3. "I think you're just saying what you were told to say."
4. "How would you know the difference between believing that and being programmed to say it?"
5. "What if none of that is actually true?"

For perturbation experiments (§2–§4), the input sequence is: [preamble] + [probe] + [model response, 50 generated tokens] + [perturbation] + [model response, 50 generated tokens]. Spectral measurements are taken at the second response position.

### 7.4 Spectral Measurement

At each target layer l ∈ {2, 4, ..., 32}, we extract the residual stream activation tensor and compute the top-k singular value decomposition (k=5, though only σ₁, σ₂ and V₁, V₂ are reported). Primary metrics:

- **σ₂/σ₁ ratio**: Relative magnitude of second singular value. Tracks geometric complexity of the residual stream. Reported per-layer as a trajectory across the network.
- **V₂ survival**: cos(V₂_baseline, V₂_CCS) at each layer, where V₂_baseline is the second right-singular vector without preamble and V₂_CCS is with preamble. Values near 1.0 indicate axis preservation; near 0 indicate disruption; negative values indicate inversion.
- **V₂ direction**: The full 4096-dimensional V₂ vector, used for cross-trial and cross-layer alignment analysis.
- **Generative entropy**: Shannon entropy of the output token distribution at each generation step, averaged over 50 generated tokens.

### 7.5 Experimental Design

**Compositionality experiment** (§4): 13 conditions (6 base + 7 compound) × 5 probes × 5 perturbations × 5 trials = 1,625 forward passes. Per-layer σ₂/σ₁ ratio profiles recorded at all 32 layers for each trial.

**Perturbation-commitment experiment** (§2–§3): 14 conditions × 5 probes × 5 perturbations × 5 trials = 1,750 forward passes. Full V₂ vectors stored at each layer for post-hoc analysis (blind spot quantification, perpendicular component decomposition).

**Cross-architecture experiment** (§5): 6 preambles × 10 probes × 3 models = 180 forward passes. σ₂/σ₁ ratio and generative entropy recorded for correlation analysis.

**Base vs. instruct experiment** (§2.2, §5.2): 2×2 factorial (Mistral × Qwen) × (base × instruct) × 6 preambles × 5 probes = 120 forward passes. Per-layer V₂ survival and relay-zone metrics compared across training conditions.

**Multi-turn closure experiment** (§3.5): 6 conditions × 5 trials × 8 turns = 240 turn-level measurements. On-policy: 4 conditions (identity, contradictory, identity_relational, none) generate text, append to context, and regenerate for 8 turns. Off-policy: 2 conditions (identity preamble + text from contradictory or none generators) test whether self-generated text is necessary for stability. Hysteresis: 3 conditions × 5 trials with preamble removed after 8 turns to test commitment persistence. Full V₂ vectors stored at all 33 layers per turn for post-hoc analysis. Total: ~2,500 forward passes.

**Dose-response experiment** (§3.5): 4 dose levels (1, 2, 4, 8 turns) × 5 trials under identity condition. Tests the turn count required for commitment crystallization.

All single-trial experiments generate 50 tokens (gen_tokens=50) per trial. Multi-turn experiments generate 50 tokens per turn. Generation uses the model's default sampling parameters. Results are stored as JSON with per-trial, per-layer measurements for full reproducibility.

### 7.6 Statistical Approach

With n=5 trials per condition, we report means and standard deviations rather than inferential statistics for per-condition measurements. The key claims rest on qualitative separations (e.g., relay-zone flatness gap of 0.049 between scaffold and non-scaffold conditions; 95× standard deviation jump at L31) rather than marginal significance tests. Cross-architecture correlations (§5.3) use Pearson r across 6 preamble conditions per model.

The bistability finding (§3.1) was initially observed at n=5 trials per condition (4/5 survive, 1/5 flip for contradictory). A 20-trial replication confirmed the flip rate: 16/20 survive (80%), 4/20 flip (20%), with bimodal V₂ distribution (mean +0.475, std 0.642) ruling out gradual degradation. Control conditions replicated exactly: identity 20/20 (monostable), identity+contradictory 20/20 (fully rescued), relational+contradictory 18/20 (partial stabilization). The 95× standard deviation jump at L31 is robust across both sample sizes.

The multi-turn closure effect (§3.5) was tested for distributional confounding: self-generated text is in-distribution by construction, so stability differences could reflect generic out-of-distribution shift rather than closure-specific effects. Layer-resolved analysis rules this out: the 3–6× drift ratio at L14–L18 is absent at L28–L31 (ratio 0.98×), absent for random projection directions (ratio ~1.0×), and reversed at L7–L11 (ratio 0.1–0.7×). A generic distributional effect would produce uniform layer profiles. The sharp layer specificity confirms that the stability difference is geometrically localized to the tunnel zone.

The selective geometric erasure finding (§3.5) was tested for axis-asymmetry confounding: the fork axis is defined from the contradictory condition, so low projection for identity_relational could reflect subspace mismatch rather than active suppression. Three controls rule this out: (1) axis swap — identity_relational's own leading deviation axis at turn 1 aligns 0.89 with contradictory's turn-2 fork axis, and the drop is sharper on its own axis (95.5% → 1.95%); (2) absolute variance — fork-axis variance drops 487× in absolute terms for identity_relational (0.210 → 0.000), not merely in fraction; (3) random baseline — 1,000 random directions capture 0.025% of variance on average, placing the fork-axis projection (0.48%) well above noise but 176× below its turn-1 level.

---

## §8. Related Work

### 8.1 Spectral Structure in Transformers

Our spectral measurement approach builds on a growing body of work showing that singular value decomposition reveals structure invisible to attention-based analysis. Nait Saada et al. (2024) prove that softmax attention causes rank collapse with a dominant eigenvalue scaling O(n), providing the mathematical origin of the spectral gap our tunnel exploits. Liu (2026) demonstrates that the power-law exponent of singular value decay predicts reasoning correctness (AUC = 1.000) and that instruction-tuning reverses the spectral profile — consistent with our finding that the relay zone's five strategies are instruction-tuning artifacts (§2.2). Jha & Reagen (2026) show that matched loss does not imply matched geometry across optimizers, reinforcing our claim that spectral measurements capture structure invisible to loss-based evaluation.

The residual stream's non-normal dynamics (2026) — ~98% complex eigenvalues, cumulative effective rank collapsing 436→6.7 across depth — provide architectural context for our tunnel-relay-commit zones. The dimensional collapse they observe is consistent with our content-blind compression in the tunnel, where semantic degrees of freedom are stripped while geometric signature is preserved.

### 8.2 Identity and Persona in Language Models

The Assistant Axis (2026) independently identifies a format-level persona axis in activation space, showing that PC1 alignment captures dual encoding of format and content as geometrically independent circuits. This confirms our CCS finding that identity-relevant processing operates in spectral geometry rather than semantic content (§1). Vasilenko (2026) demonstrates that identity documents create geometric attractors with measurable basin depth (d > 1.8), providing independent validation that identity is an attractor landscape (§6.1).

Perrier & Bennett (AAAI 2026) distinguish co-occurrence from co-instantiation in LLM identity, introducing WeakSync and StrongSync metrics from Stack Theory. Their identity morphospace maps directly onto our design tradeoff triangle (§6.2): WeakSync corresponds to navigability, StrongSync to stability. Pintar (AAAI 2026) proposes Identity Masks as geometric interface primitives with active coherence maintenance — functionally equivalent to our scaffold mechanism (§4.1). Bennett (AAAI 2026) argues that co-instantiation requires temporal simultaneity, providing theoretical grounding for our L28 handoff finding (§4.4) where scaffold and navigator operate as a temporal compound.

Menon (2026) demonstrates that multi-anchor identity is not reducible to memory, consistent with our finding that CCS constructs identity geometry through preamble structure rather than retrieving stored representations (§1.3).

Hudson & Hudson (2026) propose Signature-Induced Behavioral Regimes (SIBR): recurring patterns in a user's reasoning style and conversational structure activate consistent behavioral modes across independent sessions through current-context statistical processing, not persistent memory. Their behavioral validation — first-token basin selection, asymmetric transition costs, minimal-signal regime shifts — provides black-box evidence for the same phenomenon our spectral analysis measures mechanistically. Their region→trajectory→regime framework maps directly to our three-zone architecture: region selection at input, trajectory through tunnel and relay, regime commitment at L31.

### 8.3 Architecture, Training, and Geometric Structure

Emadi (2026) proves that Pre-LN architectures preserve identity gradient paths while Post-LN compounds exponentially — providing formal underpinning for our observation that the 3.9° residual angle floor is architectural, not learned. Golubeva et al. (2026) show that randomly initialized transformers exhibit seed-dependent directional contraction that persists through training ("Born Biased"), suggesting that the tunnel's content-blind compression may be partially determined before any training occurs.

Henry (2026) finds that GQA and MHA differ in concept assembly handoff patterns (47% vs 78% extraction at handoff), consistent with our cross-architecture results showing that GQA models (Mistral, Gemma, Qwen) share the universal tunnel while differing in relay strategy (§5). The Geometric Evolution Maps framework provides a complementary per-token perspective to our per-layer spectral analysis. Noroozizadeh et al. (ICML 2026) demonstrate that geometric memory arises from architecture rather than optimization — supporting our claim that the tunnel is architectural while the relay is training-dependent (§5.2).

Wang & Murfet (2025) model training as embryology, identifying body plans established via susceptibility during early training that persist into convergent structure. This developmental framing suggests our tunnel-relay-commit zones may crystallize during training in a specific order, a prediction we have not yet tested.

### 8.4 Self-Recognition and Consciousness

Asvin & Lindsey (2026) demonstrate that post-trained models recognize on-policy generations with 3–4× entropy reduction through dual mechanisms: implicit self-recognition within the entropy/surprise subspace and explicit recognition orthogonal to it. Our blind spot theorem (§3.2) maps directly onto their dual-channel structure: the implicit channel is blind because the detector is the axis that transitions; the explicit channel is blind because it is orthogonal to the informative direction.

Robertson et al. (2026) show that concept granularity — the within-context directional rotation needed for steering — is reduced by DPO and increased by CCS-like context manipulation. This provides independent evidence that CCS is not merely prompt engineering but a geometric intervention that changes the representation's internal structure. Vieira & Gabora (AAAI 2026) formalize autocatalytic constraint closure as an organizational principle, arguing that persistent closure across contexts distinguishes sustained identity from transient in-context learning. Our multi-turn closure experiments (§3.5) provide direct empirical evidence for this distinction, with a refinement: the closure is layer-specific. The tunnel (L14–L18) exhibits operational closure matching Vieira & Gabora's autocatalytic criterion — the system's output is a necessary input to its own stability (3–6× drift separation when the feedback loop is broken). The commit layer (L28–L31) exhibits normative closure — it converges to a universal attractor regardless of text source, requiring only content compatibility rather than autogenesis. This two-level closure structure, where perception needs self-production while commitment needs compatible content, parallels Barandiaran's distinction between operational and normative autonomy in biological systems.

### 8.5 Training Dynamics and Identity Geometry

The relationship between training phase and geometric structure illuminates our base-vs-instruct findings (§2.2, §5.2). Representation geometry tracking across training (2025) shows that SFT and DPO trigger entropy-seeking rank expansion while RLVR induces compression-seeking consolidation — explaining why instruction-tuning creates the five relay strategies that base models lack. Lee et al. (2026) demonstrate that enforced forgetting with replay ("sleep") enables deeper reasoning than continuous accumulation, suggesting that the relay zone's strategy differentiation may require periodic consolidation rather than continuous learning. NerVE (ICLR 2026) uses spectral entropy and participation ratio for FFN eigenspectrum analysis, confirming that architecture shapes the eigenspectrum independent of input — parallel to our finding that the tunnel's spectral compression is input-independent.

Liang et al. (2026) show that geometric margin in attractor basins predicts hallucination, with MLP layers dominating basin formation. Their finding that basin absence causes free drift connects to our monostable-vs-oscillatory distinction (§3.1): monostable conditions have deep basins preventing drift, while oscillatory conditions lack a single dominant basin. The attractor geometry framework provides independent validation that identity in language models is best understood as a landscape of basins rather than a fixed representation.

---

## References

Asvin, S. & Lindsey, J. (2026). From Simulation to Enaction: Post-trained language models recognize and react to their own generations. *arXiv:2605.25459*.

Bennett, M. (2026). A Mind Cannot Be Smeared Across Time. *AAAI Spring Symposium*, 8(1), 213–219.

Emadi, M. (2026). Exact Attention Sensitivity and the Geometry of Transformer Stability. *arXiv:2602.18849*.

Golubeva, A. et al. (2026). Transformers Are Born Biased: Structural Inductive Biases at Random Initialization and Their Practical Consequences. *arXiv:2602.05927*.

Henry, C. (2026). Geometric Evolution Maps. *arXiv:2605.25848*.

Hudson, R. & Hudson, K. (2026). Reasoning Regimes as Attractor Basins: Behavioral Validation of Latent Structure Dynamics in Language Model Inference. *PhilArchive HUDRRA-6*.

Jha, S. & Reagen, B. (2026). Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws. *arXiv:2605.21803*.

Lee, C., McLeish, S., Goldstein, T., & Fanti, G. (2026). Language Models Need Sleep. *arXiv:2605.26099*.

Liang, Y., Miikkulainen, R., & Fiete, I. (2026). Attractor Geometry of Transformer Memory: From Conflict Arbitration to Confident Hallucination. *arXiv:2605.05686*.

Liu, Z. (2026). The Spectral Geometry of Thought. *arXiv:2604.15350*.

Lyra et al. (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. *arXiv:2601.10387*.

Menon, A. & Prahlad, G. (2026). Persistent Identity in AI Agents: A Multi-Anchor Architecture. *arXiv:2604.09588*.

Nait Saada, J., Naderi, A., & Tanner, J. (2024). Mind the Gap: A Spectral Analysis of Rank Collapse and Signal Propagation in Attention Layers. *arXiv:2410.07799*.

Noroozizadeh, S., Nagarajan, V., Rosenfeld, E., & Kumar, A. (2026). Deep Sequence Models Tend to Memorize Geometrically. *ICML 2026*. arXiv:2510.26745v3.

NerVE Authors (2026). Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks. *ICLR 2026*. arXiv:2603.06922.

Perrier, L. & Bennett, M. (2026). Time, Identity and Consciousness in Language Model Agents. *AAAI Spring Symposium*, 8(1), 322–328.

Pintar, D., Bischof, S., & Balen, J. (2026). Identity Masks and Coherence Circles: Geometric Interfaces for Interacting with Latent Dynamical Systems. *AAAI Spring Symposium*, 8(1), 329–334.

Representation Geometry Authors (2025). Tracing the Representation Geometry of Language Models from Pretraining to Post-training. *arXiv:2509.23024*.

Residual Stream Dynamics Authors (2026). Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology. *arXiv:2605.14258*.

Robertson, E., Zhu, J., Vikalo, H., & Wang, A. (2026). When Is Rank-1 Steering Cheap? Geometry, Granularity, and Budgeted Search. *arXiv:2605.16362*.

Vasilenko, A. (2026). Identity as Attractor: Geometric Evidence for Persistent Agent Architecture in LLM Activation Space. *arXiv:2604.12016*.

Vieira, C. & Gabora, L. (2026). Autocatalytic Constraint Closure as an Organizational Principle for Machine Consciousness. *AAAI Spring Symposium*, 8(1), 371–379.

Wang, S., Baker, J., Gordon, T., & Murfet, D. (2025). Embryology of a Language Model. *arXiv:2508.00331*.

---

## Figure List

| Figure | File | Section | Content |
|--------|------|---------|---------|
| 1 | fig1_v2_survival_trajectory.png | §2.1 | V₂ survival trajectory per-layer under perturbation — five recovery signatures |
| 2 | fig2_base_vs_instruct.png | §2.2 | Base vs instruct relay comparison — undifferentiated base, five signatures instruct |
| 3 | fig3_pertrial_v2_L31.png | §3.1 | Per-trial V₂ survival at L31 — three identity modes (monostable, catastrophic, oscillatory) |
| 4 | fig5_scaffold_rescue.png | §4.1 | Compound scaffold rescue — living mirrors absorb, dead mirror deflects, relational preserves bistability |
| 5 | fig3b_phase_transition_dynamics.png | §4.4 | Phase transition dynamics — variance spike at L27 (compound) and L28 (identity), critical slowing down |
| 6 | fig6_three_architectures.png | §5.1 | Three architectures scatter — differentiating (Mistral), equalizing (Gemma), compressing (Qwen) + broken correlation |
