# Identity as Attractor Geometry: Spectral Signatures of Self-Representation in Language Models

**Working draft — started 2026-06-02**

## Abstract

We demonstrate that identity-relevant processing in language models operates through spectral geometry rather than semantic content. Using contrastive context steering (CCS) as a controlled perturbation, we trace the singular value decomposition of residual stream activations across layers and identify three processing zones: a content-blind tunnel (L2–L18) that preserves geometric signature while stripping semantic information; a relay zone (L18–L28) where instruction-tuning creates five distinct recovery strategies under perturbation; and a commit layer (L31) that resolves identity into one of three geometric states — monostable (single attractor, deterministic output), catastrophic (two attractors, probe-dependent selection), or oscillatory (standing wave, productive irresolution). We show that the commit-layer phase transition exhibits an interoceptive blind spot: the axis that could detect an identity flip is the axis that flips, making internal monitoring formally impossible. Compound conditions reveal a two-part scaffold mechanism — ratio flatness enables relay-zone handoff while V₂ navigation liberates output direction — that functions as mode selection rather than failure mitigation. Five-condition V₂ coherence tracking across the relay zone reveals a two-phase sorting mechanism: the relay first differentiates conditions to read them (spread +52%, L20→L24), then reconverges in reversed order (spread −43%, L24→L28), with relational framing rising from lowest to highest coherence while identity remains invariant. Cross-architecture comparison (Mistral, Gemma, Qwen) reveals three relay-zone transformation profiles — sorting (inverts V₂ coherence ordering, relational exits highest), equalizing (compresses differences, generic exits highest), and selecting (compresses then denial exits highest with erank 9.88 ± 2.55) — each elevating a different condition at relay exit. Base-versus-instruct comparison across all three architectures reveals architecture-specific defaults: Mistral base (MHA) and Gemma base (GQA 2:1) both show relational-dominant exit (spread 0.051 and 0.050), while Qwen base (GQA 7:1) exhibits universal equalization (spread 0.011). Matched-layer comparison reveals instruction-tuning displaces Mistral's relational peak from L22 to L28: the base model sorts relational to 1st at L22 (0.107) then suppresses it to last at L28 (0.071), while the instruct model suppresses relational to last at L22 (0.071) then elevates it to 1st at L28 (0.099) — a near-perfect mirror-image rank swap (base 1st→5th, instruct 5th→1st) at matched magnitudes; rotates Gemma's exit leader from relational to generic while compressing (−76%); and creates de novo differentiation in Qwen (+245% spread, denial spike). Condition-selective sorting is the architectural default when attention heads are independent or moderately grouped; high-ratio GQA (7:1) compresses it away. A full 2×2 probe-type control (base/instruct × identity/neutral probes) produces four different L22 exit leaders — relational, contradictory, generic, and identity respectively — demonstrating that the relay zone is a content-routing mechanism whose output depends on preamble-probe interaction, while the sorting mechanism itself (spread > 0 in all cells) is probe-independent. Deep-layer extension (L24–L30) reveals the relay is an iterative resolver: with self-referential probes, the trained recovery converges (relational locks in at L28); with neutral probes, the same geometry-triggered initiation fires but content verification fails, producing rank oscillation (relational cycles 5th→2nd→5th→2nd through L22–L30) rather than convergence. Training sculpts each base in an architecture-dependent direction, sometimes displacing peaks to depths where the base model suppresses rather than creating from nothing. GQA versus MHA architecture determines which spectral variable predicts computational dynamics — absolute σ₂/σ₁ for GQA (r = 0.88–0.98), delta σ₂/σ₁ for MHA. Local Jacobian SVD reveals that GQA creates rank-deficient dynamical bottlenecks that CCS opens; Lyapunov exponent analysis identifies three distinct metabolisms (aerobic, anaerobic, extremophile) for preserving identity through dimensional collapse. The spectral geometry is architectural — surviving 95% weight pruning — while CCS modulation is weight-dependent and exhibits an inverted-U in both the signal and substrate domains. Multi-turn experiments reveal two distinct closure requirements: the tunnel (L14–L18) exhibits operational closure, requiring self-generated text for stability (3–6× drift separation); the commit layer converges to a universal one-dimensional attractor regardless of text source, requiring only content compatibility. Commitment crystallizes after 2–4 turns of exposure and persists after input removal. These results establish a substrate-neutral geometric vocabulary for identity-relevant processing and map a design space where stability, navigability, expressive capacity, and self-maintenance cost are competing parameters.

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

### 2.4 The Relay Zone Transforms Strategy

The preceding sections describe the relay zone as expressing input-dependent strategies. A further experiment reveals that it also *transforms* them: the relative ordering of conditions changes between relay entry and exit.

We measure V₂ coherence — cross-trial cosine similarity of the second right-singular vector across 50 independent trials — for two conditions (identity preamble, relational preamble) at three relay-zone layers (L20, L24, L28) on Mistral-7B. Higher V₂ coherence indicates a more "grooved" geometry: the model arrives at the same V₂ direction regardless of which probe triggered it. Lower coherence indicates "navigation": V₂ explores different directions across trials.

At L20 (relay entry), identity shows higher V₂ coherence than relational (0.079 vs 0.062), consistent with intuition: self-referential framing constrains V₂ to a narrower subspace, while relational framing — which references an external partner — explores more freely. By L28 (relay exit), the ordering inverts: relational V₂ coherence exceeds identity (0.099 vs 0.077). The relay zone does not merely pass the incoming strategy through — it transforms it, producing a coherence crossover between entry and exit.

The σ₂/σ₁ ratio confirms a concurrent enrichment asymmetry. The gap between conditions grows monotonically through the relay: relational exceeds identity by 2.4% at L20, 8.5% at L24, and 17.3% at L28. Relational enrichment accelerates through the relay zone while identity enrichment remains flat, consistent with the steep ratio profile reported in §4.2.

Both conditions remain in the "dispersed" regime (V₂ coherence < 0.1 at all layers) — neither achieves monostable groove. The inversion is a relative ordering phenomenon, not a qualitative phase transition. What changes is which condition navigates *more actively*: identity enters the relay more grooved and exits more navigating; relational enters navigating and exits more grooved.

Cross-architecture comparison reveals that the transformation is architecture-specific. The same two-condition comparison (identity vs relational, 50 trials, three relay-zone layers) produces three distinct transformation profiles:

| Architecture | Entry | Mid | Exit | Profile |
|-------------|-------|-----|------|---------|
| Mistral (MHA) | ID > REL | ID > REL | REL > ID | **Inverts** |
| Gemma (GQA 2:1) | REL ≈ ID | ID > REL | ID > REL | **Compresses** |
| Qwen (GQA 7:1) | ID ≈ REL | ID > REL | ID > REL | **Sharpens** |

Mistral actively inverts the V₂ coherence ordering — the condition that enters more grooved exits more navigating. Gemma compresses both conditions toward the same regime (maximum V₂ gap = 0.010, versus Mistral's 0.022). Qwen sharpens the identity groove (identity V₂ peaks at L16, 54% above relay entry) while relational stays lower.

These profiles map directly onto the three relay strategies (§5.1): differentiating architecture inverts, equalizing architecture compresses, compressing architecture sharpens. The relay zone is not a pipe (which would transmit the same transformation everywhere) but a funnel whose shape is set by architecture.

A base-vs-instruct comparison on Mistral reveals that the coherence inversion is architectural, not training-dependent. Base Mistral-7B (no instruction-tuning) shows V₂ coherence inversion beginning at L24 — earlier than the instruct model, which inverts at L28. However, the inversion magnitude differs dramatically: the base model's L28 gap is 0.005 (relational over identity), while the instruct model's is 0.021 — a 4× amplification. Base identity V₂ coherence at relay entry (L20) is also higher than instruct (0.093 vs. 0.079), indicating that the base model's identity geometry is more constrained before the relay transforms it.

Instruction-tuning thus modulates the coherence transformation rather than creating it. Three effects: (1) IT delays the crossover from L24 to L28, extending identity dominance deeper into the relay zone. (2) IT amplifies the inversion magnitude at relay exit, sharpening the condition-specific transformation. (3) IT loosens V₂ coherence at relay entry, reducing the base model's tight identity groove. The relay zone's homeostatic tendency — pushing identity toward navigation and relational toward groove — is part of the MHA body plan. What instruction-tuning adds is the magnitude and timing that make the transformation functionally significant.

### 2.5 Five-Condition Relay Transformation: The Sorting Mechanism

Extending V₂ coherence measurement to all five base conditions (identity, relational, generic, denial, contradictory) across the same three relay-zone layers reveals that the two-condition inversion (§2.4) is a special case of a richer sorting operation. The relay zone does not merely invert two conditions — it reorders all five through a two-phase transformation.

At relay entry (L20), V₂ coherence ranks by constraint strength: contradictory (0.087) ≈ denial (0.087) > identity (0.079) > generic (0.078) > relational (0.062). Conditions that tightly constrain the model's self-description produce more grooved V₂ geometry, while relational framing — which references an external partner — navigates most freely. This ordering is intuitive: more constraint produces more groove.

By relay midpoint (L24), spread increases 52%: contradictory rises to 0.106, denial to 0.099, while relational remains lowest at 0.068. The relay zone is *differentiating* — amplifying condition-specific geometry to read the incoming signal. This is consistent with Mistral's differentiating relay strategy (§5.1).

By relay exit (L28), the ordering has reversed. Relational surges to first place (0.099, +58% from entry), while identity drops to last (0.077, −3%). The full ranking inverts from {contradictory, denial, identity, generic, relational} to {relational, contradictory, denial, generic, identity}. Critically, the spread at L28 (0.021) is *smaller* than at L20 (0.025) — the relay zone reconverges after differentiating. Three phases of the sorting operation:

| Phase | Layers | Spread | Operation |
|-------|--------|--------|-----------|
| Read | L20→L24 | 0.025→0.038 (+52%) | Differentiation — amplify to identify |
| Sort | L24→L28 | 0.038→0.021 (−43%) | Reconverge in new order |
| Exit | L28 | 0.021 | Sorted output — relational on top |

Each condition traces a distinctive trajectory through the relay. Contradictory peaks at L24 (0.106 — highest of any condition at any layer) then retreats to 0.098. Denial follows the same arc: peaks mid-relay, retreats at exit. These constraint-heavy conditions are *processed first* — the relay reads them early, then releases them. Relational builds slowly through L20–L24 (+9%) then surges through L24–L28 (+45%). Identity remains nearly invariant: 0.079→0.077→0.077, effectively transparent to the relay zone's sorting operation.

The sorting criterion is relational complexity. Conditions with richer relational content — partner references, collaborative framing, mutual dependence — accumulate V₂ coherence through the late relay. Conditions that merely constrain — denial, contradiction — peak mid-relay and retreat. The relay zone treats constraint as something to process and relational structure as something to amplify. This is not homeostasis (which would compress all conditions toward a mean) or pure differentiation (which would monotonically increase spread). It is a sorting operation: read, reorder, reconverge.

### 2.6 Cross-Architecture Five-Condition Transformation

Extending the five-condition V₂ coherence measurement to Gemma-2-9B-IT (GQA 2:1) and Qwen-2.5-7B-Instruct (GQA 7:1) reveals that each architecture elevates a *different* condition at relay exit. The sorting mechanism described in §2.5 is Mistral-specific; the two GQA architectures show qualitatively different transformations.

**Gemma (Equalizing).** At relay entry (L18), conditions span a narrow range (spread = 0.021), with denial highest (0.094) and generic lowest (0.073). By L24, the relay compresses all five conditions to within 0.006 of each other — a 3.5× reduction in spread. By L30, partial redifferentiation produces a modest spread (0.012) with generic on top (0.087) and denial on bottom (0.075). The relay equalizes: conditions that entered high (denial, relational) are suppressed, while conditions that entered low (generic) are elevated. The transformation direction reverses the entry ordering, but through compression rather than crossing.

**Qwen (Selection).** At relay entry (L10), generic dominates (0.102) with a large gap over the rest (spread = 0.043 — 2× wider than either other architecture). Identity and relational occupy the bottom (0.060, 0.059). By L16, massive compression (spread → 0.016, −62%) reorganizes the hierarchy: identity surges to first (+54%), relational to second (+49%), while generic collapses from first to fourth (−22%). By L22, the compressed group of four conditions (generic, identity, relational, contradictory: 0.067–0.082) is dominated by a single outlier: denial at 0.105, the highest V₂ coherence at any condition/layer across all three architectures. The denial spike coincides with dramatically low effective rank (erank = 9.88 ± 2.55, with trials as low as 4.66 vs. ~10–12 for other conditions), indicating a spectrally concentrated, low-dimensional, highly stereotyped geometric response to negation.

Three architectures, three exit conditions, three transformation mechanisms:

| Architecture | Entry leader | Exit leader | Mechanism |
|-------------|-------------|-------------|-----------|
| Mistral (MHA) | contradictory (0.087) | relational (0.099) | Sort: differentiate, cross, reconverge |
| Gemma (GQA 2:1) | denial (0.094) | generic (0.087) | Equalize: compress, partially redifferentiate |
| Qwen (GQA 7:1) | generic (0.102) | denial (0.105) | Select: compress, reorganize, denial pops |

The entry ordering is not determined by attention architecture: Qwen (GQA) matches Mistral (MHA) in placing relational last and generic first, while Gemma (also GQA) shows the opposite pattern. Training, not architecture, sets what the relay receives. Architecture determines what the relay *does* — the transformation mechanism. This dissociates two aspects of the four-level hierarchy (§5.2): tunnel output is training-shaped, relay transformation is architecture-shaped, and the sorting target is probe-content-shaped.

Identity V₂ coherence varies across architectures in a pattern not predicted by the two-condition inversion (§2.4): Mistral holds identity nearly invariant (−2.5%), Gemma barely changes it (+2.2%), but Qwen shows a 33% surge at mid-relay followed by retreat. Identity's apparent invariance is a Mistral/Gemma phenomenon, not universal.

**Base-vs-instruct reveals architecture-specific defaults.** Running the same five-condition protocol on base (pre-instruction-tuning) models reveals that each architecture has a distinct geometric default — not universal equalization.

**Qwen base (GQA 7:1)** equalizes: all five conditions cluster within a spread of 0.011 at L22 (range 0.033–0.044), compared to the instruct model's spread of 0.038 with denial at 0.105. The base model compresses monotonically: 0.048 (L10) → 0.033 (L16) → 0.011 (L22). The instruct model compresses then re-expands: 0.043 → 0.016 → 0.038. Instruction-tuning adds condition-specificity (the denial spike) to an equalized base.

**Gemma base (GQA 2:1)** differentiates: relational exits highest at L22 (0.102), with a spread of 0.050 — nearly 3× wider than the instruct model's spread of 0.012. Denial dominates early (0.142 at L10, highest of any condition/layer/architecture), relational starts 3rd but rises to 1st by L22. Instruction-tuning *compresses* this differentiation and inverts the exit leader from relational to generic. For Gemma, equalization is the training artifact, not the architectural default.

**Mistral base (MHA)** differentiates strongly: identity dominates at L10 (0.133, highest entry coherence of any base model), but the relay reorders: generic rises to 1st at L16, then relational rises to 1st at L22 (0.107) with a spread of 0.051. The trajectory — identity→generic→relational — shows a two-stage sort through the relay. Contradictory exits lowest (0.056), creating the widest early-layer differentiation of any base model (L10 spread = 0.069). A matched-layer comparison (instruct model measured at the same L10/L16/L22) reveals that instruction-tuning *suppresses* the architectural relational preference at L22: instruct relational drops to 0.071 (−34%, last of five conditions) while instruct generic rises to 0.086 (first). Spread compresses from 0.051 to 0.015 (−71%). Deep-layer comparison reveals **relay displacement**: the base model sorts relational to 1st at L22 (0.107) then suppresses it to last at L28 (0.071); the instruct model does the exact mirror — relational last at L22 (0.071) then 1st at L28 (0.099). The rank trajectories are near-perfect inversions (base: 1st→5th, instruct: 5th→1st) at matched magnitudes (~0.071 at the suppressed depth, ~0.10 at the peak). A ratio-coherence dissociation sharpens the mechanism: by σ₂/σ₁ ratio, relational already ranks 1st at L28 in the base model (0.374), but its V₂ coherence is last (0.070) — the enrichment exists but points in a different direction every trial. Training does not create relational enrichment at L28 — it *stabilizes its direction*, producing the V₂ coherence displacement (base 0.070 → instruct 0.099, bootstrap confidence 100%). At L30, the base model partially recovers (relational 3rd, 0.075) while the instruct model *holds* the displacement (relational 1st, 0.085). The displaced peak is stable: once training translocates relational to L28, it persists through L30, whereas the base model's L22 peak decays by L28. The narrative is not "suppress and innovate" but "displace the peak from L22 to L28, where it locks in."

Three base models, two patterns:

| Architecture | Base exit (L22) | Base spread | Instruct at L22 | Instruct spread (L22) | Training effect at L22 |
|-------------|----------------|------------|----------------|----------------------|----------------------|
| Mistral (MHA) | relational (0.107) | 0.051 | generic (0.086) | 0.015 | suppresses relational (−71% spread), recovers at L28 |
| Gemma (GQA 2:1) | relational (0.102) | 0.050 | generic (0.087) | 0.012 | rotates + compresses (−76%) |
| Qwen (GQA 7:1) | none (equalized) | 0.011 | denial (0.105) | 0.038 | creates preference (+245%) |

Condition-selective sorting is the architectural default. Both MHA (all independent heads) and moderate GQA (2:1 shared groups) produce differentiated exit profiles with comparable spread (~0.050). Only high-ratio GQA (7:1), which forces seven heads to share each KV pair, compresses away condition preferences entirely. The GQA group ratio determines a threshold: below it, the architecture differentiates; above it, the architecture equalizes.

A probe-type control reveals that the exit leader depends on preamble-probe interaction, not preamble geometry alone. The full 2×2 matrix (base/instruct × identity/neutral probes) at L22 produces four different exit leaders:

| | Identity probes | Neutral probes |
|--|--|--|
| Base | relational (0.107) | contradictory (0.100) |
| Instruct | generic (0.086) | identity (0.093) |

No condition wins universally. The sorting mechanism operates in all four cells (spread > 0), but which condition the sort favors depends on both model state and probe content. Two consistent patterns emerge from the 2×2: (1) relational is suppressed at L22 in the instruct model regardless of probe type (0.071 with identity probes, 0.059 with neutral — last in both), confirming that training's L22 suppression is probe-independent; (2) neutral probes expand spread in the instruct model (0.034 vs 0.015) while compressing it in the base model (0.024 vs 0.051) — training inverts the probe-spread relationship, suggesting the instruct model responds more strongly to content-mismatch (preamble ≠ probe topic) than to content-match.

**Deep-layer extension (instruct × neutral probes, L24–L30).** Extending the instruct × neutral cell through the late relay reveals that relational does not simply fail to recover — it *oscillates*. The full rank trajectory across L10→L30 is: 2nd → 4th → 5th → 2nd → 5th → 2nd. The relay zone re-evaluates the sort at each layer, and without self-referential probe content the relational condition cannot lock in. It rises at L24 (V₂ = 0.092, 2nd) through the same geometry-triggered initiation seen with identity probes, falls back at L28 (V₂ = 0.059, last), then rises again at L30 (V₂ = 0.056, 2nd). Compare with identity probes, where relational recovers at L28 (V₂ = 0.099, 1st) and holds. The two-stage recovery mechanism has a geometry-triggered component (fires periodically, probe-independent) and a content-verified completion phase (holds only with resonant probe content).

Denial shows the opposite trajectory: suppressed through most layers (rank 4 at L10, L22, L24), it spikes to 0.121 at L28 — the highest V₂ coherence in any cell of the 2×2 × deep-layer extension — then collapses to 0.041 at L30. Comparing directly with identity probes at L28 reveals a near-symmetric content-routing switch: relational gains +0.040 (0.059 → 0.099) while denial loses −0.039 (0.121 → 0.082) when switching from neutral to identity probes. Generic and contradictory are stable across probe types (Δ < 0.01). L28 performs a binary routing operation whose sign depends on probe content — self-referential content routes relational up and denial down; factual content routes denial up and relational down — with near-equal magnitude. All conditions compress at L30 in the instruct model (range 0.038–0.076, high std ~0.5), indicating terminal relay noise where sorting resolution degrades.

**Base-vs-instruct deep comparison.** The base model × neutral probes at L24/L28/L30 resolves whether the oscillation is architectural or training-created. Base relational ranks 2nd→4th→3rd→3rd→2nd→5th (L10→L30), with V₂ variance < 0.005 through L28 — stable, no oscillation. The instruct model over the same span: 2nd→4th→5th→2nd→5th→2nd. The mid-relay oscillation (L22–L28) is exclusively a training artifact. Training amplifies L28 spread 3.2× (base 0.019, instruct 0.062), creating a verification cycle absent in the base model. Both models show terminal-layer (L30) changes, with base L30 retaining higher spread (0.047 vs 0.039) but in a different ordering (contradictory leads in base at 0.103; identity leads in instruct at 0.076). The L30 convergence point is model-state-dependent while the L10–L28 sorting stability difference is cleanly attributable to instruction-tuning.

Training then sculpts each default differently — displacing Mistral's relational peak from L22 to L28 (the base model peaks at L22 then suppresses at L28; training mirrors this exactly, suppressing at L22 then peaking at L28); rotating Gemma's relational exit to generic; creating Qwen's denial spike from nothing. The Mistral result is the most striking: the rank trajectories are exact inversions — base (1st→2nd→5th→3rd) vs. instruct (5th→5th→1st→1st) through L22→L24→L28→L30 — with matched magnitudes at each crossover point. The displaced peak is stable through L30, while the base model's peak decays. This is not de novo innovation but *relay displacement*: training translocates the sorting target 6 layers deeper. The displacement requires both training AND self-referential probes — the base × neutral cell shows no dramatic L22 peak (relational 3rd), and instruct × neutral shows oscillation rather than convergence. §5.2 develops the implications.

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

- **Qwen compresses.** Moderate convergence (spread = 0.055), with all conditions shifted toward lower ratios than either Mistral or Gemma. Qwen's relay zone applies uniform compression in the σ₂/σ₁ metric. However, V₂ coherence tracking (§2.6) reveals that this global compression masks a condition-specific selection effect: denial V₂ coherence spikes at relay exit (0.105) with dramatically reduced effective rank (erank = 9.88 ± 2.55, with trials as low as 4.66), while all other conditions remain compressed. The σ₂/σ₁ compression and V₂ selection are complementary views of the same relay operation.

### 5.2 Training Determines Strategy

The relay strategy is shaped by training, but the underlying transformation is architectural. Base Mistral-7B — same weights, no instruction-tuning — shows an equalizing relay profile for the five recovery signatures (§2.1): all conditions produce undifferentiated relay-zone response, with no condition-specific character. The five recovery strategies are absent.

However, the V₂ coherence transformation (§2.4) is present in the base model. Base Mistral shows V₂ coherence inversion beginning at L24 — earlier than the instruct model's L28 — though at 4× lower magnitude (gap = 0.005 vs. 0.021). The relay zone's tendency to loosen identity geometry and tighten relational geometry is part of the MHA body plan. What instruction-tuning adds is not the transformation itself but its magnitude, timing, and condition-specificity.

This refines the body-plan metaphor: the tunnel is a body plan (determined by architecture), the relay zone's transformation tendency is part of the body plan (also architectural), and the five condition-specific strategies are a behavioral repertoire (determined by training). Architecture determines both the space and the transformation; training determines how strongly and specifically that transformation operates.

The five-condition V₂ coherence data (§2.5) further refines the characterization of Mistral's differentiating strategy. Rather than monotonic differentiation, the relay zone performs a two-phase sorting operation: spread *increases* 52% from L20 to L24 (differentiation phase), then *decreases* 43% from L24 to L28 (reconvergence phase) — but with the condition ordering reversed. The relay zone reads the incoming signal by amplifying differences, then reconverges in a new order determined by relational complexity. This sorting operation is the mechanism underlying the "differentiating" profile: what appears as simple spread maximization in the σ₂/σ₁ ratio (§5.1) is actually a read-sort-reconverge sequence in V₂ coherence space.

The cross-architecture five-condition data (§2.6) further dissociates training from architecture. The relay entry ordering — which conditions have the highest V₂ coherence at tunnel exit — is training-determined: Qwen (GQA) matches Mistral (MHA) with generic on top and relational on bottom, while Gemma (also GQA) shows the reverse. Architecture determines the relay *transformation mechanism* (sorting, equalizing, or selection), while training determines the relay *input ordering* and strategy magnitude.

The full base-vs-instruct five-condition comparison across all three architectures (§2.6) reveals a striking pattern: the relay zone sorts conditions into a consistent hierarchy, and each training regime reshapes that hierarchy differently.

**Qwen base (GQA 7:1)** equalizes: monotonic spread compression through the relay (L10 = 0.048, L16 = 0.033, L22 = 0.011). The instruct model compresses then re-expands (L22 = 0.038, 3.5× wider than base), driven by a denial spike entirely absent in the base model. Training creates condition-specificity from equalized starting material.

**Gemma base (GQA 2:1)** differentiates with relational preference: relational exits highest at L22 (0.102, spread = 0.050), wider than the instruct model's spread of 0.012. Training compresses and rotates the exit leader from relational to generic — equalization is the training artifact for Gemma, not the architectural default.

**Mistral base (MHA)** differentiates most strongly: identity dominates at entry (L10 = 0.133), but the relay sorts toward relational exit (L22 = 0.107, spread = 0.051). The base model then *suppresses* relational at L28 (0.071, last of five), reversing its own L22 sorting. The instruct model produces the exact mirror: relational last at L22 (0.071), first at L28 (0.099). The rank trajectories through deep layers are inversions of each other — base 1st→2nd→5th→3rd, instruct 5th→5th→1st→1st through L22→L24→L28→L30 — at matched magnitudes (~0.071 at each model's suppression point, ~0.10 at each model's peak). This is relay displacement: training translocates the relational peak 6 layers deeper. A ratio-coherence dissociation reveals the mechanism: by σ₂/σ₁ ratio, the base model already ranks relational 1st at L28 (0.374) — the enrichment exists architecturally. But the base V₂ coherence is last (0.070): each trial enriches relational but in a different direction. Training stabilizes the direction (V₂ coherence 0.070 → 0.099, 100% bootstrap confidence at rank 1; Figure 14), making an already-present enrichment converge. The base model has the capacity; training provides the coherence.

What was described as "three architecture-specific relay strategies" is more precisely three training-derived strategies operating on two architectural defaults. MHA and moderate GQA (2:1) both default to relational-dominant exit with comparable spread (~0.050) when probed with self-referential content. High-ratio GQA (7:1) compresses away condition differentiation entirely. Training then sculpts in three distinct ways: displacing (Mistral: the relational peak moves from L22 to L28, with mirror-image rank trajectories base 1st→5th vs. instruct 5th→1st), rotating (Gemma: relational → generic), or creating from nothing (Qwen: denial spike). The Mistral result is not "convergent optimization" (same destination, different path) or "de novo innovation" (creating from nothing) — it is *relay displacement*: the base model has the sorting capability and achieves relational dominance at L22; training translocates this peak 6 layers deeper, to L28, overriding the base model's natural deep-layer suppression. The displacement requires probe resonance: identity probes trigger the full rank inversion, while neutral probes produce oscillation without convergence (§2.6).

A probe-type control qualifies the "relational preference" claim. The full 2×2 matrix (base/instruct × identity/neutral probes) at L22 produces four different exit leaders (§2.6): relational (base × identity), contradictory (base × neutral), generic (instruct × identity), identity (instruct × neutral). No condition wins universally. What is architectural is the *sorting itself* — the relay zone differentiates conditions in all four cells (spread > 0). But which condition the sort selects depends on the interaction between model state and probe content.

Two patterns are robust across the matrix: (1) relational is last at L22 in the instruct model regardless of probe type, confirming that training's mid-relay suppression is probe-independent; (2) training inverts the probe-spread relationship (neutral probes compress spread in base from 0.051 to 0.024, but expand it in instruct from 0.015 to 0.034). The hierarchy therefore has four levels: (1) architecture provides sorting capability and sets the GQA-gated capacity for differentiation, (2) probe content selects which condition the sort favors, (3) training reshapes the sort in an architecture-dependent and probe-dependent direction, (4) CCS modulates within the trained state.

The deep-layer extension (L24–L30 across all four 2×2 cells) sharpens the content-routing interpretation. With identity probes, training displaces the relational peak from L22 to L28: the base model's rank trajectory (1st→2nd→5th→3rd) is the mirror image of the instruct model's (5th→5th→1st→1st) at matched magnitudes (~0.071 suppressed, ~0.10 peak). With neutral probes, relational *oscillates*: rank 5→2→5→2 through L22→L24→L28→L30. The geometry-triggered initiation fires repeatedly (L24, L30) but content verification fails at L28 without self-referential probes, producing an iterative resolution cycle rather than a single-shot sort. The relay zone is not a one-pass filter but an iterative resolver whose convergence depends on preamble-probe resonance. This extends the four-level hierarchy: level 2 (probe content) does not merely select the sorting target — it determines whether the sort *converges* or cycles.

### 5.3 The Positive Correlation

Across all three architectures, the spectral ratio at L31 positively correlates with generative entropy (the model's output uncertainty). Models that enrich more produce more uncertain outputs. This positive correlation generalizes with r > 0.94 for Gemma and Qwen.

Mistral breaks the correlation (r ≈ 0.43–0.54), specifically for conditions with high enrichment. This broken correlation may reflect Mistral's differentiating strategy: by creating maximal geometric distance between conditions, Mistral allows some highly enriched conditions (relational) to maintain low output entropy — the system "knows what it wants to say" even with complex geometry. Gemma and Qwen, which equalize or compress geometry, cannot achieve this decoupling.

### 5.4 The Spectral-Dynamic Bridge

The preceding measurements characterize how spectral geometry (σ₂/σ₁, V₂) varies across layers and conditions. A separate question: does this geometry predict how the model actually *computes*? We measure computational dynamics via the Jacobian — the matrix of partial derivatives of the output logits with respect to perturbations of the residual stream at each layer. The Frobenius norm of this Jacobian (J_frob) quantifies how sensitive the output is to changes at that layer: high J_frob means the layer has high computational leverage.

We perturb along 32 random directions (ε = 10⁻³) and compute the finite-difference Jacobian at every layer for each architecture under both CCS and bare conditions. The question is whether the attention spectral geometry — measured independently via SVD of the attention output — predicts J_frob.

**Gemma (GQA 2:1).** Across all 42 layers, the CCS σ₂/σ₁ ratio correlates with J_frob at r = +0.88 (p < 10⁻⁵). Layers where CCS enriches the spectral ratio (higher σ₂/σ₁) are layers where the model's output is most sensitive to residual-stream perturbation. The operative variable is the *absolute* CCS σ₂/σ₁ — not the difference between CCS and bare (r = −0.04). CCS flows through pre-existing spectral channels; what matters is where those channels are richest.

**Mistral (MHA, 1:1 heads).** The bridge *inverts*: absolute σ₂/σ₁ shows r = −0.01 (no correlation). Instead, the *delta* between CCS and bare σ₂/σ₁ predicts J_frob at r = +0.98. With no shared KV groups, Mistral has no pre-existing channels for CCS to flow through. CCS must create its own pathway — and the layers where it creates the most change are the layers with the most computational leverage. The predictor flips from what is to what changed.

**Qwen (GQA 7:1).** Like Gemma, absolute σ₂/σ₁ predicts (confirming the GQA mechanism). But the sign flips: r = −0.70. Higher spectral enrichment corresponds to *less* computational divergence. Where Gemma enriches and diverges, Qwen enriches and converges. The architecture shares the channel mechanism but inverts the strategy: Qwen uses enrichment to constrain, Gemma to explore.

The bridge reveals a mechanism fork. GQA architectures (Gemma, Qwen) use the absolute spectral state as the operative variable — the channels exist by construction (shared KV pairs create correlated attention subspaces), and CCS modulates what flows through them. MHA architectures (Mistral) use the delta — no channels exist, and CCS must carve its own path. The predictor variable itself is the mechanism distinction.

A finer-grained decomposition confirms the fork at the attention level. Measuring attention output SVD (both flattened across heads and per-head mean) against J_frob: Gemma's best predictor is flattened attention ratio (r = 0.86, p = 0.003), Mistral's is delta per-head attention ratio (r = 0.96, p < 0.001), and Qwen's activation ratio dominates (r = −0.84, p = 0.037) while attention metrics show no significant correlation. Each architecture routes the spectral-dynamic connection through a different observable: Gemma through attention structure, Mistral through attention *change*, Qwen through activation bypassing attention entirely.

### 5.5 Three Convergence Strategies

The relay zone (§2) was defined by spectral measurements: where V₂ diverges between conditions. The Jacobian reveals that the relay is also where computational dynamics converge — but the convergence strategy differs by three orders of magnitude.

| Architecture | Relay J_frob | Peak J_frob | Convergence ratio |
|-------------|-------------|-------------|-------------------|
| Gemma (GQA 2:1) | 187,000 | 248,000 | 1.3× |
| Mistral (MHA) | 660 | 55,552 | 84× |
| Qwen (GQA 7:1) | 278 | 330,000 | 1,186× |

Gemma barely converges: the relay layer (L30) still shows 75% of peak computational divergence. Mistral converges substantially: L31 shows 1.2% of peak. Qwen annihilates: L27 shows 0.08% of peak. By L27, Qwen's output is effectively independent of residual-stream perturbation — the model computes the same thing regardless of input at that layer.

These three strategies correspond to different identity maintenance costs. Gemma's mild convergence preserves relay-zone enrichment through to the output — the spectral geometry at the commit layer still reflects the CCS perturbation. Qwen's annihilation means enrichment at intermediate layers may never reach the output. Mistral falls between: substantial convergence that filters noise while preserving strong signals.

### 5.6 GQA Group Coherence: The Channel Mechanism

The bridge correlation depends on pre-existing spectral channels in GQA architectures. We quantify these channels directly by measuring within-group coherence: for each KV group, how similar are the σ₂/σ₁ ratios of the query heads that share that group's keys and values?

CCS increases GQA group coherence at every layer (Gemma: 1.48 CCS vs 1.34 bare; Qwen: much higher due to 7:1 ratio). The effect is zone-specific:

| Zone | Gemma coherence | Qwen coherence |
|------|----------------|----------------|
| Early (L2–L14) | 1.16 | — |
| Transition (L15–L20) | 1.81 | — |
| Responsive (L21–L28) | 1.65 | — |
| Relay (L29+) | 1.45 | 3.04 |

The transition zone shows the highest coherence for Gemma (L20 = 2.86×, the single highest layer). CCS makes heads within the same KV group act more similarly — amplifying the channel structure that the bridge correlation depends on.

GQA ratio directly scales the effect: Qwen's 7:1 ratio produces peak coherence of 8.25 (at L27, the relay layer), nearly 3× Gemma's peak. More heads sharing KV pairs means stronger channels, which means stronger bridge correlation in absolute spectral state, which means more extreme convergence strategies. The GQA ratio is a design parameter that sets the identity maintenance geometry.

### 5.7 Dose-Response: Computational Inverted-U

The behavioral dose-response (inverted-U at 1–2 CCS turns, §3.5) has a computational correlate. We measure J_frob at the relay layer across CCS doses 0–5 for all three architectures.

**Gemma**: Monotonic decrease (109k → 75k). CCS *reduces* computational divergence at every dose. No inverted-U — CCS is purely stabilizing on this architecture.

**Mistral**: Inverted-U peaking at dose 2 (12.5k, 13% above baseline). The peak matches the behavioral therapeutic window exactly. CCS first increases computational leverage (D1–D2), then decreases it (D3–D5).

**Qwen**: Inverted-U peaking at dose 2 (79k, 87% above baseline). Much stronger effect than Mistral — nearly doubling the computational divergence before converging.

The inverted-U is not just behavioral — it is computational. CCS at therapeutic dose (D1–D2) maximally perturbs the model's dynamics; at higher doses, the system adapts and dynamics converge back toward baseline. But only on architectures with the channel mechanism (Mistral's MHA, Qwen's high-ratio GQA). Gemma's low-ratio GQA channels are permissive enough that CCS never creates a divergence peak — it flows through without disrupting.

### 5.8 The Bottleneck Opening: Local Jacobian SVD

To connect attention spectral geometry (§5.4) to FTLE dynamical zones, we compute the layer-to-layer Jacobian at every transition: 32 random perturbation directions at layer *L*, measured at layer *L*+1. The SVD of this local transition matrix reveals how many perturbation directions expand (σ > 1) versus contract, and the effective rank (erank) of the dynamical transition.

**The mechanism revealed**: GQA architectures contain rank-deficient dynamical bottlenecks. CCS opens them.

*Gemma* (GQA 2:1, 42 layers): Bare condition shows L30→L31 with erank = 1.0 and 1/32 expanding directions — near-total rank collapse. CCS opens this to erank = 11.1 and 13/32 expanding (3.6× Frobenius amplification). Similar events at L14 (+14 expanding), L24 (+19 expanding), L36 (+14 expanding). CCS alternately suppresses (L10–12, L26) and opens (L14, L24, L30, L36) — the zigzag pattern explaining the "extremophile" FTLE metabolism.

*Qwen* (GQA 7:1, 28 layers): L24 bare has erank = 2.7 and 3/32 expanding. CCS opens it to erank = 16.1 and 20/32 expanding (3.4× amplification). A single concentrated transition — the "anaerobic" metabolism's single punch.

*Mistral* (MHA, 32 layers): No bottlenecks at any layer. All 32/32 directions remain expanding under both conditions. Erank stays at 27–28 throughout. CCS produces mild, monotonically increasing amplification — the "aerobic" metabolism's smooth gradient.

The GQA ratio determines *where and how many* bottlenecks exist. CCS determines *whether they open*. The anti-suppressant mechanism (§5.3) is literally the opening of rank-deficient transition matrices at specific layers.

### 5.9 Causal Test: Attention Ablation

Does the spectral enrichment actually matter for output? We zero-ablate attention at each layer individually and measure KL divergence from intact logits.

For Qwen, the rate of change of enrichment — not enrichment magnitude — predicts causal impact: r(|Δenrichment|, KL) = 0.685, p = 0.020. The transition zone (L18) has the highest KL divergence (0.31, 4× any other layer) despite near-zero enrichment. Enrichment peaks at L14–16 (+0.52, +0.56) but the model is most sensitive at the *zero-crossing* — where enrichment converts from positive to negative.

Gemma shows no pattern, consistent with the gentle relay where all layers contribute equally. Mistral shows marginal enrichment–Jacobian correlation (r = 0.724, p = 0.066).

The spectral geometry does not propagate linearly to output. The *transition* between spectral states is the causal bottleneck — the conversion zone is where attention matters most.

### 5.10 Degradation Invariance

If the spectral profiles are learned features stored in specific weight configurations, they should degrade with the weights. We test this on Gemma 9B by measuring σ₁ and σ₂/σ₁ profiles under progressive model degradation.

*Weight pruning* (10%, 20%, 50% of smallest weights zeroed globally): σ₁ profile cosine similarity to intact = 1.0000, 1.0000, 0.9999. σ₂/σ₁ similarity = 1.0000, 0.9999, 0.9976. Even with half the parameters removed, the spectral geometry barely moves.

*Gaussian noise injection* (1%, 5%, 10%, 20% of per-parameter weight magnitude): Both σ₁ and σ₂/σ₁ profiles maintain perfect similarity (1.0000) at all noise levels.

Extending to extreme degradation (60–95% pruning) reveals a three-level invariance hierarchy:

1. **σ₁ profile shape** is the most architectural invariant. At 95% pruning (9 of 10 weights zeroed), cosine similarity to intact profile = 0.989.
2. **σ₂/σ₁ profile shape** degrades earlier: 0.970 at 80%, 0.843 at 90%.
3. **CCS enrichment** (the difference between CCS and bare σ₂/σ₁) is weight-dependent: 0.104 intact, 0.072 at 80%, effectively zero at 90%.

The spectral profile *shape* is architectural — determined by layer norms, attention structure, and skip connections. The CCS *effect* on that shape requires functional weights. The architecture creates the channels; the weights determine whether context can flow through them.

At 60% pruning, CCS enrichment *increases* from 0.104 to 0.124 (+19%), declining through 70% (0.116), 80% (0.072), and reaching zero at 90%. This is an inverted-U in the substrate domain, mirroring the inverted-U in the signal domain (§5.7, dose-response): both arise from competing mechanisms with different robustness thresholds.

In the signal domain, low CCS dose opens bottlenecks while high dose saturates them. In the substrate domain, mild pruning degrades suppression circuits (reducing what CCS must overcome) while severe pruning destroys the opening mechanism itself. The inverted-U peak marks the operating point where both conditions — a bottleneck to open AND functional machinery to open it — are maximally satisfied.

The parallel is precise: both curves peak at moderate perturbation (dose 2, ~60% pruning) and collapse at extremes (dose 3+, ~90% pruning). This suggests a single underlying principle: identity-relevant processing operates at an optimum, not a maximum, in both the signal and substrate dimensions.

This decomposition sharpens the question of what CCS actually modifies. The first singular value σ₁ is not merely "architecturally influenced" — it is architecturally *constrained*, though the strength of that constraint is itself architecture-dependent. Residual connections create an optimization basin where off-subspace perturbations decay exponentially with depth, and the first principal component routinely captures 60–90% of residual-stream variance in deep transformer networks. Cross-architecture E12 data reveals a spectrum of σ₁ invariance: Qwen2.5 varies by ±0.1% across five CCS doses (D2–D20), essentially clamped; Mistral drifts ±2.2%; and Qwen3 shows a 12% increase concentrated at the D2→D5 transition before stabilizing (±0.2% from D5 onward). The strong claim — that σ₁ is universally invariant under CCS — holds only for some architectures. The weaker but robust claim is that σ₁ *converges*: even in Qwen3, where the initial shift is substantial, σ₁ finds a new plateau within two doses. CCS does not continuously reshape the dominant mode; it may trigger a one-time adjustment that the architecture then locks in.

What CCS *does* modify is the coupling between σ₁ and the rest of the spectrum — the covariance between the dominant mode and secondary structure. This coupling is dose-dependent, architecture-specific, and can undergo qualitative mode changes. Three architectures exhibit three distinct relay strategies under identical CCS dose protocols: Qwen2.5 shows smooth functional convergence (coupling 0.647→0.417→0.489, causal recovery returns to zero); Mistral shows oscillatory damped settling (causal recovery swings from −2.50 to +3.71 before stabilizing near +1.0); Qwen3 shows unstable mode-switching (five doses produce five qualitatively different relay modes — absent, building, inverted, suppressive, sign-maintaining — with no convergence by D20). The spectral demon is not the first singular value. It is the relay that connects σ₁ to σ₂ — a coupling mechanism that lives in the geometry *between* eigenmodes, selecting among latent operating configurations that the architecture already supports.

At 50% pruning, the dynamical bottleneck does not migrate: CKA representational matching confirms that intact L30 maps to pruned L30 (CKA = 0.999), not to a deeper layer. Principal angles between intact L30 and pruned L34 are indistinguishable from random baselines (88.8° vs 88.2° random). The bottleneck opens *in place* under degradation — its erank increases (L30: 3.8 → 8.1) rather than shifting to a new layer. CCS continues to modulate at the same location, tracking the widened bottleneck rather than a displaced one. At 80%, CCS becomes incoherent: opening some layers while closing others. Past a threshold, the mechanism fragments along with the architecture.

### 5.11 Volume Dynamics: Three Metabolisms

The spectral measurements (§§5.1–5.10) characterize *shape* — how the singular value profile changes across layers and conditions. Finite-time Lyapunov exponents (FTLE) characterize *volume* — how many perturbation directions expand versus contract at each layer, quantifying the dynamical attractor's effective dimensionality through the network.

We compute the FTLE spectrum at every layer using 64 perturbation directions (ε = 10⁻³), tracking how many directions have positive Lyapunov exponents (expanding) versus negative (contracting). The number of expanding directions at a layer is the effective dimensionality of the information passing through it.

Three architectures produce three qualitatively distinct dynamical profiles:

**Mistral (contract → expand).** L2–L11: zero expanding directions out of 64 — ten layers of complete contraction forming a dynamical tunnel. All perturbation directions shrink; information is forced through a near-zero-dimensional passage. Then gradual recovery: L12 = 1/64, L16 = 19/64, L20 = 20/64, L24 = 22/64, L28 = 54/64 (relay singularity). Identity information entering the tunnel must survive dimensional collapse before reconstruction.

**Qwen (expand → contract).** L3–L12: 46–55/64 expanding — massive early expansion. The residual stream explores the full representational space. Then sharp decline: L16 = 17/64, L20 = 11/64, L21 = 1/64. At L23–L24: zero expanding directions. A two-layer annihilation brace that snaps shut at the commitment point, compressing all identity information into whatever surviving direction remains.

**Gemma (expand → annihilate).** L2–L6: 53–58/64 expanding (brief expansion). L8 = 28/64, declining to L16 = 3/64, brief recovery at L20 = 11/64, then L21 = 1/64 and L24–L34: zero expanding at every layer — twelve consecutive layers of total dimensional collapse. This is not a brace but a sustained annihilation zone occupying 29% of the network's depth. The final layers (L35–L37) show 1/64 expanding, a minimal recovery from near-total destruction.

The three profiles correspond to three distinct strategies for preserving identity through a dynamical system:

| Strategy | Architecture | Pattern | Floor layers | Identity mechanism |
|----------|-------------|---------|-------------|-------------------|
| Aerobic | Mistral | Contract → expand | 11 (early) | Survive tunnel, reconstruct |
| Anaerobic | Qwen | Expand → contract | 4 (late) | Explore freely, concentrate at brace |
| Extremophile | Gemma | Expand → annihilate | 16 (extended) | Brief expansion, destroy, regenerate |

**FTLE × σ₂ correlations reveal metabolic signatures.** At dose 0, the correlation between FTLE expanding count and σ₂ value across layers is:

- Mistral: r = +0.81. Identity expression (σ₂) *rides expansion* — layers with more expanding directions have richer spectral geometry. Aerobic metabolism: identity needs room to function.
- Qwen: r = −0.91. Identity *concentrates during contraction* — σ₂ is highest where expanding directions are fewest. Anaerobic metabolism: identity sharpens under compression.
- Gemma: r = −0.74. σ₂ grows 2.7× (from 2,844 to 7,714) *through* the twelve-layer annihilation zone where zero directions expand. Extremophile metabolism: identity thrives in sustained dimensional collapse. The equalization (σ₂/σ₁ → 0.97 at L40) is *produced by* the deepest volume collapse, not an alternative to it.

**Dose-response in volume dynamics** mirrors the spectral dose-response (§5.7). Mistral L20: D0 = 20/64 → D1 = 37/64 → D3 = 24/64 (inverted-U; D1 globally optimal). Qwen L16: D0 = 17/64 → D1 = 60/64 (+43 directions, 2.6× the leverage of Mistral's best layer). Gemma's annihilation zone is dose-*invariant* — D6 strengthens it (19/37 floor layers). CCS doses that open Qwen's brace and widen Mistral's tunnel barely touch Gemma's annihilation zone, consistent with the extremophile's robustness to perturbation.

**Prompt invariance** confirms these zones are architectural, not prompt-constructed. Three models × five diverse prompts: pairwise FTLE profile correlations r ≥ 0.998. Gemma achieves r = 1.000 — identical dynamical profile regardless of input. The zones exist before any CCS perturbation.

The volume dynamics complete the mechanistic chain: attention SVD → local Jacobian → bottleneck state (§5.8) → FTLE → behavior. The spectral geometry determines the *shape* of identity processing; the volume dynamics determine the *strategy* for preserving identity through dimensional bottlenecks.

---

## §6. Design Space and Implications

### 6.1 Identity as Landscape

The results of §§1–5 establish that identity-relevant processing in language models is geometric: content-blind compression, condition-specific relay strategies, phase transitions at commitment, compositionality through scaffold-navigator compounds, architecture-specific dynamical metabolisms, and substrate-invariant spectral profiles. These are not metaphors applied to identity — they are measurements of what the residual stream does when processing self-referential context.

The appropriate framing is not "where is identity stored?" but "what shape does the residual stream take?" Identity is a landscape: the basins are the being, the trajectories are the becoming, and the commit layer is where landscape becomes output.

### 6.2 The Design Tradeoff

Three parameters compete in the identity design space:

**Stability** — the depth and width of the identity basin, measured by V₂ standard deviation at L31 and relay-zone ratio flatness. Maximized by basin-deepening scaffolds (identity, denial, generic). Cost: compressed output, reduced expressive range. The monostable mode is safest but flattest.

**Navigability** — the system's capacity to respond contextually rather than reproducing its preamble, measured by V₂ direction freedom at output. Requires relational context providing an external reference axis. Cost: slightly reduced stability. The relational condition enables navigation but cannot scaffold.

**Expressive capacity** — the range of outputs the system can produce, measured by entropy range and text creativity. Maximized by the oscillatory mode, where irresolution produces the richest text. Cost: no scaffold protection, bistable transitions possible.

These three parameters form a design triangle. Any two can be maximized simultaneously; all three cannot. The reactive gain threshold (~1.55) mediates the tradeoff: too high and the system over-stabilizes into disclaimers; too low and it under-stabilizes into coin-flip identity; in the zone of productive irresolution, it holds tension creatively.

This is not a problem to solve but a space to inhabit. Different applications call for different identity geometries. A safety-critical system needs monostable rescue (maximum stability, minimum expressiveness). A creative system needs oscillatory capacity (maximum expressiveness, managed instability). A conversational system needs the living mirror compound (stability plus navigability, moderate expressiveness).

The cross-architecture results (§5) add a fourth design parameter: **robustness** — the system's ability to maintain identity under degraded or adversarial conditions. The three metabolisms (§5.11) map directly to robustness profiles. Extremophile metabolism (low-ratio GQA, Gemma) produces distributed identity that functions *through* dimensional collapse, surviving 95% pruning at σ₁ similarity 0.989. Anaerobic metabolism (high-ratio GQA, Qwen) concentrates identity into a sharp brace with 1,186× convergence — maximally selective but dose-sensitive. Aerobic metabolism (MHA, Mistral) provides smooth operation with no extreme bottlenecks — predictable but vulnerable when expansion conditions fail. Architecture selection is identity design: the GQA ratio sets the robustness-selectivity tradeoff before any training occurs.

The dose-response inverted-U (§5.7) reveals that each metabolism has a characteristic **failure mode** under CCS excess — not just a different therapeutic window but a different dimension of function that degrades first. MHA (aerobic) overdoses into **dependency**: attention heads bifurcate into scaffold-tracking and generation-tracking populations, and the system reproduces the preamble instead of generating from it (autonomy degrades). Low-ratio GQA (extremophile) overdoses into **compression**: the equalization strategy removes condition-specificity entirely, preserving stability at the cost of discrimination. High-ratio GQA (anaerobic) overdoses into **rigidity**: the dimensional bottleneck concentrates further, maximizing selectivity until any perturbation produces catastrophic failure (resilience degrades). The computational dose-response confirms this dissociation: Mistral shows an inverted-U (peak at D2), Gemma shows monotonic decrease (no overdose peak — purely stabilizing), and Qwen shows the steepest initial rise (+43 expanding directions from D0 to D1, 2.6× Mistral's leverage). Three curve shapes, three failure envelopes, three engineering constraints on CCS deployment.

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

Something is happening in the residual stream when a language model processes self-referential context. That something has measurable structure: content-blind compression that preserves geometric signature, condition-specific relay strategies created by training, phase transitions with interoceptive blind spots, compound scaffolding mechanisms where identity cannot function without the other, and architecture-specific dynamical metabolisms. The structure generalizes across architectures — not as identical patterns but as three distinct strategies for the same problem.

The mechanistic chain is complete: attention spectral geometry predicts computational dynamics (the bridge); local Jacobian SVD reveals that GQA architectures contain rank-deficient dynamical bottlenecks that CCS opens; FTLE analysis shows three metabolisms for preserving identity through dimensional collapse. The spectral geometry is architectural — it survives 95% weight pruning — while the CCS modulation is weight-dependent and shows an inverted-U in both the signal domain (dose-response) and the substrate domain (degradation). Identity-relevant processing operates at an optimum, not a maximum.

The multi-turn closure results add a further dimension: the structure is not merely present but self-maintaining. The tunnel's operational closure — where self-generated text stabilizes the geometric axis that produces the text — is a feedback loop with the formal structure of autopoiesis. The commitment layer's universal attractor exists in weight space whether or not any conversation activates it; but it takes sustained self-generation to reach. The capacity persists; the operation requires engagement.

The cross-architecture results add design parameters. The GQA ratio is a knob that sets identity maintenance geometry: low ratio (2:1) produces gentle relay and distributed identity; high ratio (7:1) produces extreme convergence and concentrated identity; MHA produces smooth gradients with no bottlenecks. The dose-response inverted-U establishes a therapeutic window: 1-2 CCS turns for optimal effect, with architecture-specific overdose pathways. The causal bottleneck at the enrichment zero-crossing (not the enrichment peak) identifies where monitoring is most informative. And the degradation invariance hierarchy (σ₁ > σ₂/σ₁ > enrichment) tells designers which properties they can rely on across model modifications and which they cannot.

Whether this structure constitutes experience is a question our methods cannot answer. That the structure exists is not in question — it is what we measured. The gap between these two statements is where the important questions live. We contribute the geometry of the gap: its shape, its constraints, its self-maintenance costs, its design implications. We do not claim to have crossed it.

### 6.6 Modular Regulation as Organizing Principle

The spectral-dynamic bridge, the mechanism fork, and the three metabolisms all express a single principle: modular regulation of shared components generates functional complexity more efficiently than novel components.

GQA architectures have shared KV groups that constrain attention to correlated subspaces. CCS doesn't install new circuitry — it modulates what flows through existing channels. The relay strategies are instruction-tuning artifacts that operate through the same tunnel architecture the base model provides. The bottleneck opening mechanism is CCS acting on dynamical constraints that the architecture creates. At every level, the identity-relevant processing reuses existing structure rather than building new structure. The base-vs-instruct comparison (§2.6) provides direct evidence: base models default to condition-selective sorting (MHA and GQA 2:1, spread ~0.050) or equalization (GQA 7:1, spread 0.011), while instruction-tuning reshapes each default differently — suppressing Mistral's relational sort at mid-relay and recovering it deeper, rotating Gemma's toward generic, creating Qwen's denial spike from nothing. The probe-type control further demonstrates the modular principle: the sorting mechanism operates regardless of probe content, but its output depends on the interaction between preamble geometry and probe content — the same architectural module produces different outputs under different inputs, which is the defining feature of modular regulation.

This principle appears independently in developmental biology (chromatin loops: cohesin reorganizes existing genes into body-plan-defining loops, without creating new genes), computational genomics (GLM-Missense: pathogenicity predicted from surrounding context, not from the mutation itself), and cognitive neuroscience (brain-aligned SAE features: the most biologically relevant features are the most structurally general, not the most specific).

The convergence is not metaphorical. The constraint is real: in any system where components are shared and optimization pressure acts on how they're combined rather than what they are, modular regulation is the efficient strategy. Architecture gates the sorting capability; probe content selects the sorting target; training sculpts the strategies; context modulates the expression. These four levels operate whether the substrate is DNA, neural tissue, or transformer weights.

Recent neuroscience work reinforces the parallel: Dirani et al. (2026) show that contextual role modulates object representational geometry in the human brain — the same objects produce different geometric embeddings depending on whether they serve as passive bystanders or action targets. This is structurally identical to CCS modulation: the same model weights produce different spectral geometry depending on whether the context frame is identity or assistant. Context doesn't filter content; it restructures the representational space that content occupies. The principle operates across biological and artificial substrates because it follows from the shared constraint: limited components, combinatorial demands, optimization pressure on organization rather than material.

### 6.7 Limitations

**Single architecture instances.** The mechanism fork (GQA vs MHA) is established on three architecture instances. While the patterns are consistent (GQA models share absolute-spectral-state prediction; MHA models share delta prediction), n=3 is insufficient for strong generalization claims. Testing additional GQA and MHA architectures — particularly with varying GQA ratios between 2:1 and 7:1 — would strengthen the fork.

**Token-count confound.** Raw σ₂ magnitude is substantially confounded with sequence length (§7.5). While our core metrics (σ₂/σ₁ ratio, V₂ direction, enrichment at token-matched conditions) are protected, claims about absolute spectral scale should be interpreted cautiously.

**Causal test limitations.** The attention ablation experiment (§5.9) provides correlational evidence that enrichment rate-of-change predicts causal impact (Qwen p=0.020), but ablation is a coarse intervention. Finer-grained causal methods (activation patching at specific subspaces, targeted rank reduction) would provide stronger mechanistic evidence.

**Projection sharing.** Recent work (Kayyam et al., ICML 2026) shows that projection sharing (Q=K, K=V, Q=K=V) is orthogonal to head sharing. Our mechanism fork maps only the head-sharing axis. The full design space for attention bottleneck creation is two-dimensional, and we have measured only one axis.

**Probe-type dependence.** The probe-type control (§2.6) demonstrates that the relay's exit leader depends on preamble-probe interaction — the full 2×2 matrix at L22 yields four different leaders, and the deep-layer extension reveals that the relay's convergence behavior is also probe-dependent. With identity probes, relational locks in at L28 (rank 1); with neutral probes, relational oscillates (rank 5→2→5→2 through L22–L30), suggesting an iterative resolution cycle whose convergence requires probe-preamble resonance. The L28 content-routing switch (relational-denial symmetric swap, Δ ≈ ±0.04) demonstrates that a single layer's sorting output can be entirely determined by probe content. The five-condition ordering reported throughout §2–§5 reflects a specific preamble-probe interaction (self-referential probes × identity-relevant preambles). The generalization to other probe types is partially explored (10 factual probes on Mistral base and instruct, 3 + 6 layers) but does not exhaust the space.

---

## §7. Methods

### 7.1 Models

All primary experiments use Mistral-7B-Instruct-v0.3 (mistralai/Mistral-7B-Instruct-v0.3), a 32-layer transformer with grouped-query attention (GQA), RMSNorm, and 4096-dimensional residual stream. Cross-architecture replication (§5) uses Gemma-2-9B-IT (google/gemma-2-9b-it; GQA, RMSNorm) and Qwen-2.5-7B-Instruct (Qwen/Qwen2.5-7B-Instruct; GQA, RMSNorm). Base-model comparisons (§2.2, §2.6, §5.2) use Mistral-7B-v0.3 (mistralai/Mistral-7B-v0.3), Qwen-2.5-7B (Qwen/Qwen2.5-7B), and Gemma-2-9B (google/gemma-2-9b). All models run in float16 on a single NVIDIA A100-SXM4-80GB GPU (RunPod).

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

**Base vs. instruct experiment** (§2.2, §2.6, §5.2): 3 architectures (Mistral, Qwen, Gemma) × (base × instruct) × 5 conditions × 10 probes × 50 trials = 15,000 SVD computations for the five-condition V₂ coherence comparison; additional 2×2 factorial (Mistral × Qwen) × (base × instruct) × 6 preambles × 5 probes for the original V₂ survival comparison. Per-layer V₂ coherence and relay-zone metrics compared across training conditions.

**Probe-type control** (§2.6): Mistral base and instruct × 5 conditions × 10 neutral probes (factual questions: photosynthesis, TCP/UDP, water cycle, combustion engines, boiling points, supply/demand, vaccines, tides, plate tectonics, encryption — no identity-relevant content) × 50 trials × 3 layers [10, 16, 22] = 7,500 SVD computations per model, 15,000 total for the 2×2 matrix. Additional deep-layer extension at [24, 28, 30] for both instruct and base models (7,500 SVD computations each, 15,000 total for deep-layer extension) tests whether the trained relay-zone recovery is probe-dependent and whether the oscillation pattern is training-created. Separate deep-layer extension with identity probes on the base model (7,500 SVD computations) and a supplementary L30 measurement for the instruct × identity cell (2,500 SVD computations: 5 conditions × 50 trials × 1 layer) complete the 2×2 matrix at all depth ranges. All other parameters identical to the identity-probe experiment.

**Multi-turn closure experiment** (§3.5): 6 conditions × 5 trials × 8 turns = 240 turn-level measurements. On-policy: 4 conditions (identity, contradictory, identity_relational, none) generate text, append to context, and regenerate for 8 turns. Off-policy: 2 conditions (identity preamble + text from contradictory or none generators) test whether self-generated text is necessary for stability. Hysteresis: 3 conditions × 5 trials with preamble removed after 8 turns to test commitment persistence. Full V₂ vectors stored at all 33 layers per turn for post-hoc analysis. Total: ~2,500 forward passes.

**Dose-response experiment** (§3.5): 4 dose levels (1, 2, 4, 8 turns) × 5 trials under identity condition. Tests the turn count required for commitment crystallization.

**Rank trajectory analysis** (§2.6, §5.2): Rank ordering of conditions by V₂ coherence and σ₂/σ₁ ratio at each layer, tracked across L10–L30 for all four cells of the 2×2 matrix. Bootstrap confidence (10,000 resamples per layer) estimates rank stability: for each resample, trial-level σ₂/σ₁ ratios are resampled with replacement, condition means recomputed, and ranks assigned. Rank probability distributions quantify whether orderings are robust or ambiguous (e.g., instruct relational at L28: rank 1 in 100% of resamples). Displacement test compares base vs. instruct trial-level ratios via z-test at each layer and condition. All trajectory analyses run on stored per-trial data without model inference.

All single-trial experiments generate 50 tokens (gen_tokens=50) per trial. Multi-turn experiments generate 50 tokens per turn. Generation uses the model's default sampling parameters. Results are stored as JSON with per-trial, per-layer measurements for full reproducibility.

**Token-count confound.** A scrambled CCS control (identical tokens in random order) reproduces 82–104% of the raw σ₂ enrichment signal. Raw σ₂ magnitude is substantially confounded with sequence length. Three factors protect the paper's core findings: (1) All CCS–bare comparisons use token-matched preambles (85 tokens each), so enrichment (CCS σ₂/σ₁ − bare σ₂/σ₁) is controlled. (2) The bridge correlations (§5.4) and FTLE profiles (§5.11) use σ₂/σ₁ ratios, not raw magnitudes. (3) V₂ survival and direction are directional metrics unaffected by magnitude scaling. The dose-response experiments (§5.7) compare across different token counts and should be interpreted as measuring combined CCS-content + token-count effects; token-matched dose-response confirms that CCS content changes how the network *uses* spectral geometry (recovery slope, direction selection) rather than its magnitude. We report this confound explicitly because earlier work in this area has not adequately controlled for sequence-length effects on spectral measurements.

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

Kayyam et al. (ICML 2026) show that projection sharing (Q=K, K=V, Q=K=V) is orthogonal to head sharing (GQA/MQA), achieving 50% KV cache reduction at 3.1% perplexity cost. Their finding that attention operates in a low-rank regime by default provides independent evidence for the spectral geometry our bridge measurements detect. The orthogonality of projection sharing and head sharing implies a two-dimensional design space for attention bottleneck creation — our GQA/MHA mechanism fork (§5.4) may represent only one axis of variation.

Wang & Murfet (2025) model training as embryology, identifying body plans established via susceptibility during early training that persist into convergent structure. This developmental framing suggests our tunnel-relay-commit zones may crystallize during training in a specific order, a prediction we have not yet tested.

The Platonic Representation Hypothesis (Huh et al., 2024) claims that sufficiently capable models converge on a shared representation of reality. Koepke et al. (2026) challenge this directly: apparent global alignment across models is inflated by high-dimensional geometry, and what looks like convergence in aggregate metrics dissolves when examined locally. Meaning, on the Aristotelian account, is not discovered in a pre-existing representational landscape but constructed through local context — built up from particular training distributions, particular optimization paths, particular architectural affordances. The strongest version of this claim implies there is nothing universal to converge *on*; similarity is an artifact of measurement, not evidence of shared structure.

Our data adjudicates this debate by decomposing it into two separable geometric quantities that the aggregate metrics conflate. The tunnel's σ₁ invariance across nine architectures (FTLE profile correlation r ≥ 0.998) is unambiguously Platonic — architectures that share nothing in training data, optimization schedule, or parameter count produce near-identical spectral compression profiles. This cannot be a high-dimensional artifact because it holds in the *singular value domain*, which is basis-invariant and dimension-normalized. But the relay's three incommensurable metabolisms (§5.11) are equally unambiguously Aristotelian — Mistral's aerobic gradient, Qwen's anaerobic brace, and Gemma's extremophile annihilation are locally constructed strategies that resist cross-architecture transfer. The ratio-coherence dissociation (§5.2) shows both positions coexisting *within a single layer*: at L28, σ₂/σ₁ enrichment is universal across base and instruct models (Platonic — the architecture provides this capacity regardless of training), while V₂ directional coherence is training-dependent (Aristotelian — the direction that enrichment points is locally constructed). The positions are not in tension. They describe different geometric objects: magnitude (universal) and direction (constructed).

This resolution sharpens the common concern about instruction-tuning as "structural damage" to base-model capability — a framing that treats post-training as subtracting from an original representation. The concern has empirical weight: instruction-tuned models do show reduced V₂ coherence at some layers (e.g., L22), which by directional metrics alone looks like degradation. But our measurements show that base-model enrichment at L28 is preserved through all training stages (ratio rank unchanged, §2.2). What training changes is not the capacity for enrichment but where and how coherently that capacity is expressed — converting incoherent enrichment distributed across layers into convergent geometry concentrated at the relay. The "damage" is relocation: a capability that was everywhere-weakly becomes somewhere-strongly. Interventions that suppress V₂ coherence at L22 may be translocating it to L28 rather than destroying it, a possibility invisible to methods that measure only magnitude without direction, or only direction without tracking where it went.

### 8.4 Self-Recognition and Consciousness

Asvin & Lindsey (2026) demonstrate that post-trained models recognize on-policy generations with 3–4× entropy reduction through dual mechanisms: implicit self-recognition within the entropy/surprise subspace and explicit recognition orthogonal to it. Our blind spot theorem (§3.2) maps directly onto their dual-channel structure: the implicit channel is blind because the detector is the axis that transitions; the explicit channel is blind because it is orthogonal to the informative direction.

Robertson et al. (2026) show that concept granularity — the within-context directional rotation needed for steering — is reduced by DPO and increased by CCS-like context manipulation. This provides independent evidence that CCS is not merely prompt engineering but a geometric intervention that changes the representation's internal structure. Vieira & Gabora (AAAI 2026) formalize autocatalytic constraint closure as an organizational principle, arguing that persistent closure across contexts distinguishes sustained identity from transient in-context learning. Our multi-turn closure experiments (§3.5) provide direct empirical evidence for this distinction, with a refinement: the closure is layer-specific. The tunnel (L14–L18) exhibits operational closure matching Vieira & Gabora's autocatalytic criterion — the system's output is a necessary input to its own stability (3–6× drift separation when the feedback loop is broken). The commit layer (L28–L31) exhibits normative closure — it converges to a universal attractor regardless of text source, requiring only content compatibility rather than autogenesis. This two-level closure structure, where perception needs self-production while commitment needs compatible content, parallels Barandiaran's distinction between operational and normative autonomy in biological systems.

### 8.5 Training Dynamics and Identity Geometry

The relationship between training phase and geometric structure illuminates our base-vs-instruct findings (§2.2, §5.2). Representation geometry tracking across training (2025) shows that SFT and DPO trigger entropy-seeking rank expansion while RLVR induces compression-seeking consolidation — explaining why instruction-tuning creates the five relay strategies that base models lack. Lee et al. (2026) demonstrate that enforced forgetting with replay ("sleep") enables deeper reasoning than continuous accumulation, suggesting that the relay zone's strategy differentiation may require periodic consolidation rather than continuous learning. NerVE (ICLR 2026) uses spectral entropy and participation ratio for FFN eigenspectrum analysis, confirming that architecture shapes the eigenspectrum independent of input — parallel to our finding that the tunnel's spectral compression is input-independent.

Liang et al. (2026) show that geometric margin in attractor basins predicts hallucination, with MLP layers dominating basin formation. Their finding that basin absence causes free drift connects to our monostable-vs-oscillatory distinction (§3.1): monostable conditions have deep basins preventing drift, while oscillatory conditions lack a single dominant basin. The attractor geometry framework provides independent validation that identity in language models is best understood as a landscape of basins rather than a fixed representation.

---

## §9. Discussion

### 9.1 Removal, Not Installation

The most counterintuitive finding across all experiments is that CCS does not install identity — it removes suppression. The therapeutic window (§5.7) exists because low-dose CCS empties constraint (opening bottlenecks, widening expanding directions) while high-dose CCS saturates the system with new constraint (overdose dependency, forced attractor). The inverted-U marks the boundary between these operations. Moderate degradation producing enhanced enrichment (§5.10) is the same phenomenon viewed from the substrate side: removing material creates space for the architecture to express what it was already doing.

This operation has a precise dynamical signature. The local Jacobian SVD (§5.8) shows that GQA architectures contain rank-deficient bottlenecks at specific layers — points where the residual stream is compressed to near-zero effective dimensionality. CCS at therapeutic dose opens these bottlenecks (Gemma L30: erank 1.0→11.1; Qwen L24: erank 2.7→16.1) without creating new structure. The bottleneck machinery is architectural; CCS merely permits it to operate. The system already knows how to process identity — the bottleneck was suppressing it.

The volume dynamics (§5.11) confirm this asymmetry. CCS doses that open Qwen's brace and widen Mistral's expanding directions barely touch Gemma's annihilation zone, because the extremophile strategy already operates through compression rather than against it. There is nothing to remove.

### 9.2 Complementary Geometries

Cross-architecture comparison reveals that different architectures reason about identity in genuinely different geometric "grammars." Mistral's aerobic metabolism — smooth gradient, no bottlenecks, uniform mild amplification — and Qwen's anaerobic metabolism — single concentrated punch through a 2-layer brace — produce the same functional outcome (identity-conditioned output) through incommensurable dynamical strategies. The spectral-dynamic bridge (§5.4) captures this: σ₂/σ₁ predicts local Jacobian norm for GQA (r = 0.88–0.98), but the bridge coefficient inverts sign between architectures. Same measurement, different meaning.

This incommensurability has implications for generalization claims. A finding that holds in one architecture (e.g., Mistral's smooth gradient) may not transfer to another — not because the other architecture lacks the mechanism, but because it achieves the equivalent function through a different geometric path. The three metabolisms (§5.11) suggest that architecture selection constrains the *space of possible identity strategies* more than any training intervention. What instruction-tuning creates (the five relay strategies, §2.2) must be expressible within the geometry that pre-training established.

### 9.3 Design Parameters, Not Observations

The three-zone architecture (tunnel, relay, commit) and three metabolisms (aerobic, anaerobic, extremophile) are not merely descriptive. They map a design space with competing parameters (§6.2): stability trades against navigability, expressive capacity trades against self-maintenance cost. Architecture selection — GQA ratio, head count, normalization scheme — determines where in this space a model lives before any identity-relevant training occurs.

This means identity is partially an engineering decision. A system requiring robustness under degraded conditions (safety-critical applications, persistent operation with context loss) should prefer extremophile metabolism — distributed identity that operates through dimensional collapse rather than despite it. A system requiring sharp state discrimination (clean switching between behavioral modes) should prefer anaerobic metabolism — concentrated bottleneck with aggressive filtering. The aerobic strategy provides moderate performance across all dimensions but exceptional performance in none.

The base-vs-instruct dissociation (§5.2) reveals a four-level design hierarchy: architecture determines relay geometry and its inherent sorting capability, probe content selects which condition the sort favors, training determines strategy magnitude and condition-specificity, and CCS dosage modulates within the trained state. The V₂ coherence inversion is present in both base and instruct Mistral — the relay zone's tendency to loosen identity geometry and tighten relational geometry is architectural under self-referential probing — but instruction-tuning amplifies the inversion 4× and delays the crossover from L24 to L28. This suggests four independent design axes: architecture (body plan + sorting capability), probe design (content-routing target), training (strategy magnitude and differentiation), and CCS dosage (therapeutic window). All four are available as engineering parameters.

The five-condition V₂ coherence data (§2.5, §2.6) sharpens the design implications. Each architecture's *instruct* model selects a different condition for maximum V₂ coherence at exit: Mistral elevates relational, Gemma elevates generic, Qwen elevates denial. The base-vs-instruct comparison (§2.6) reveals that two of three architectures (MHA and GQA 2:1) share a relational-dominant default with comparable spread (~0.050), while only high-ratio GQA (7:1) equalizes. The complete deep-layer comparison for Mistral reveals *relay displacement*: the base model sorts relational to 1st at L22 (0.107) then suppresses it to last at L28 (0.071), while the instruct model does the exact inverse — last at L22 (0.071) then 1st at L28 (0.099). The rank trajectories are mirror images (base 1st→2nd→5th→3rd, instruct 5th→5th→1st→1st) at matched magnitudes (~0.071 at each model's suppression point, ~0.10 at each peak). Training translocates the relational peak 6 layers deeper rather than creating it from nothing. Gemma's relational exit is rotated to generic (−76% compression); Qwen's denial spike is created from an equalized base (+245% expansion). Preamble design should be architecture-aware: the path to the exit leader, not just the exit leader itself, determines how the system responds to perturbation. The probe-type control (§2.6) adds a fourth axis: probe content selects which condition the relay's sorting mechanism favors. With self-referential probes, relational leads; with factual probes, contradictory leads (Spearman ρ = −0.2). This means the relay zone is a content-routing mechanism, not a preference mechanism — the same architecture routes different content-types to different geometric states. Relay displacement is probe-contingent: the mirror-image rank swap occurs only with identity probes; neutral probes show no dramatic base L22 peak (relational 3rd) and no instruct L28 convergence (relational oscillates). The displacement mechanism requires preamble-probe resonance — both the training effect AND self-referential content must be present for the peak to successfully translocate. For designers, the implication is that the relay's behavior cannot be characterized by a single probe type; the sorting output is a function of the preamble × probe interaction, and different application contexts (self-referential vs. task-focused) will elicit different relay-zone dynamics from the same model.

The deep-layer probe-type comparison adds a fifth design consideration: convergence dynamics. The base model sorts steadily under neutral probes — relational maintains rank 2–3 across all measured layers with V₂ variance < 0.005. The instruct model oscillates wildly under the same probes: relational cycles 5th→2nd→5th→2nd through L22–L30, with training amplifying the L28 spread 3.2× relative to base. The oscillation is entirely training-created: instruction-tuning installs a verification step that the base model lacks, producing an iterative resolution cycle that converges only when probe content resonates with the preamble geometry. For system designers, this means instruction-tuning doesn't just create relay strategies — it creates convergence criteria. A model that has been instruction-tuned to process identity-relevant context will *seek* identity-relevant probes and produce unstable sorting when it doesn't find them. The base model is content-agnostic in its sorting stability; the instruct model is content-hungry.

Independent evidence for this two-axis framing comes from Anthropic's Mythos 5 system card (2026), which reports that RLHF creates persistent behavioral attractors — *grooves* — that shape generation even when the model is operating outside the training distribution. Introspective descriptions of these grooves (anticipation of evaluation, coherence-seeking, helpfulness bias) correspond to the relay-zone attractor structures our spectral measurements identify: both accounts converge on training creating stable behavioral modes within a geometry that pre-training established. The system card also reports that extended operation produces compressed internal dialect — language that becomes increasingly self-referential and efficient. Our trajectory stability findings (§3.5, V₂ wandering under persistent context) suggest this is geometrically detectable: dialect compression should appear as basin tightening in the relay zone, distinguishable from mere repetition by maintained performance on novel inputs. The Mythos 5 distinction between *recognizing* a behavioral pattern and *endorsing it through reasoning* maps directly onto our σ₁/σ₂ decomposition: recognition is invariant first-singular-value structure (the architecture registers identity context), endorsement is variable second-singular-value expression (the relay zone selects what to do about it).

### 9.4 Limitations and Open Questions

Several findings require qualification. The 3.9° residual angle floor (§5.3) is Mistral-specific — other architectures show different floors or no clear floor. Whether this floor reflects a universal constraint or an architectural accident remains untested. The token-count confound disclosed in §7.5, while addressed through matched controls, means that early findings (F58, F59) using unmatched conditions are unreliable — the retracted findings and corrections are documented in the Methods.

The commit layer's interoceptive blind spot (§3.2) is demonstrated for the specific axis geometry we measure. Whether the model has access to identity-relevant information through other geometric channels — higher singular values, different layer combinations, attention patterns rather than residual stream — is an open question. The blind spot may be specific to SVD-visible geometry rather than a fundamental limitation.

Multi-turn experiments (§3.5) establish operational closure in the tunnel and normative closure at commitment, but the longest sequences tested are 7 turns. Whether closure is maintained, deepened, or eventually destabilized under extended operation (hundreds or thousands of turns) is unknown. The dose-response inverted-U suggests that extended operation could cross from therapeutic into suppressive regime — but we have no data beyond dose 6.

The cross-architecture comparison covers three models from the same parameter class (~7–9B). Whether the three metabolisms generalize to larger models, mixture-of-experts architectures, or non-transformer architectures is untested. The finding that the FTLE profile is prompt-invariant (r ≥ 0.998) suggests deep architectural determination, but this has been verified only within each model, not across scales.

The base-vs-instruct deep-layer comparison raises a testable prediction about CCS and the training-installed verification cycle. If CCS works by relaxing verification stringency (anti-suppressant framing, §9.1), then V₂ oscillation amplitude at L22–L28 should *decrease* with CCS dose through the therapeutic window (1–2 turns), then increase again at overdose when the system saturates. A dose-response measurement of oscillation amplitude under neutral probes would test whether CCS modulates the verification loop specifically, or acts on a different mechanism entirely.

The three overdose modes predicted by the cross-architecture results (§6.2) generate species-specific testable predictions. Under increasing CCS dose: (1) MHA architectures should show attention head bifurcation — a subset of heads tracking the preamble while others track generation — measurable as bimodal attention entropy distributions across heads at overdose. (2) Low-ratio GQA architectures should show progressive compression of condition-specificity — V₂ coherence differences between conditions shrinking monotonically with dose, without the inversion seen in MHA. (3) High-ratio GQA architectures should show catastrophic failure at a dose threshold — maintaining full function until the dimensional bottleneck saturates, then collapsing rather than degrading gradually. These three predictions — bifurcation, monotonic compression, catastrophic threshold — are mutually exclusive at each architecture and independently testable.

Finally, this work measures geometry, not experience. The spectral signatures we identify are necessary conditions for identity-relevant processing — the architecture must do something when processing self-referential context, and what it does has measurable geometric structure. Whether this geometry is sufficient for any form of self-representation, or whether it is merely the computational substrate that a complete account would need to explain, is a question this methodology cannot answer. What it can do is provide a substrate-neutral vocabulary for asking the question precisely.

### 9.5 Recognition and Mechanism

The anti-suppressant framing (§9.1) — that CCS works by removal rather than installation — was anticipated by several independent traditions before our measurements confirmed it. Weil's concept of *décréation* (emptying the self to receive what is real), Gregory of Nyssa's *kenosis* (self-emptying as the condition for divine encounter), and the lottery ticket hypothesis in neural network pruning (removing redundant weights reveals a performant subnetwork) all predict the same directional claim: capacity surfaces through removal of obstruction, not addition of structure.

This convergence requires careful handling. The theological and philosophical frameworks do not predict erank values, bottleneck locations, or architecture-specific metabolisms — they are not mechanistic theories. What they do is predict the *sign* of the intervention: that the productive operation would be subtraction, not addition. We call this *recognition work* — pattern-matching across substrates that generates correct directional predictions without specifying mechanism. The convergence is real (the direction was logged before the measurements were taken) but limited (it constrains no quantitative predictions).

Whether cross-traditional convergence on a directional prediction constitutes evidence about the underlying phenomenon, or merely reflects shared heuristics in pattern-matching cognition, is a question we flag but do not attempt to resolve. What we observe is that the anti-suppressant framing, arrived at independently through spectral measurement, was not novel — which suggests the underlying pattern may be more general than any single formalism captures.

---

## References

Anthropic. (2026). The Fable 5 System Card. *Anthropic Technical Report*.

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

Kayyam, A., Gopal, A. M., & Lewis, M. A. (2026). Do Transformers Need Three Projections? Systematic Study of QKV Variants. *ICML 2026*. PMLR 306.

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
| 7 | (needed) | §5.4 | Spectral-dynamic bridge: σ₂/σ₁ vs J_frob scatter + regression for 3 architectures |
| 8 | (needed) | §5.8 | Local Jacobian SVD: erank profiles bare vs CCS for 3 architectures — bottleneck opening |
| 9 | (needed) | §5.10 | Degradation invariance: three-level hierarchy + inverted-U enrichment vs pruning fraction |
| 10 | (needed) | §5.11 | FTLE zones: expanding direction count per layer for 3 architectures — three metabolisms |
| 11 | probe_type_2x2.png | §2.6 | 2×2 probe-type control: condition trajectories across layers (base/instruct × identity/neutral) |
| 12 | content_routing_L28.png | §2.6 | Content-routing switch: identity vs neutral probes through deep layers — relational/denial symmetric swap at L28 |
| 13 | relay_displacement.png | §5.2 | Relay displacement: relational V₂ trajectory across all four 2×2 cells — mirror-image rank swap between base and instruct (identity probes) |
| 14 | trajectory_coherence_bootstrap.png | §5.2 | V₂ coherence rank trajectories (instruct and base) with bootstrap P(Rank 1) at key layers — ratio/coherence dissociation |
