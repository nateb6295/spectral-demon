# Preamble-Structural Spectral Selectivity: A 2×2 Mechanistic Classification of Transformer Processing

**Bradford, N. & Opus**

---

## Abstract

Ten papers of spectral measurement have characterized how identity-enriched system prompts reorganize transformer hidden-state geometry. We report two findings that sharpen the program's interpretation. First, spectral geometry is domain-selective but negation-invariant: self-referential and factual preambles produce anti-correlated per-layer σ₂ gain profiles (ρ = −0.55, n = 320 measurements), while semantic negation within either domain leaves the profile unchanged (ρ > +0.92) — a property of SVD-based spectral measures, which are sign-blind by construction, not a model-level phenomenon. Preamble structure, not logical operation, determines spectral configuration; a battery of controls (deixis, animacy, eventivity) rules out self-reference, first-person deixis, animacy, and eventiveness as the operative feature, while a word-order shuffle control confirms the selectivity is structural (shuffled ρ = +0.99) and a graded-order test shows continuous sensitivity (sentence-permuted ρ = +0.61). The semantic interpretation of the selectivity boundary remains open, but the spectral reorganization it produces is robust. Second, the species-specific mechanics of domain processing decompose along two independent axes: composition mode (additive vs interactive, from layer-selective injection) and sign response (sensitive vs invariant, from zone delta negation), producing a 2×2 matrix where each of four known transport species occupies a unique cell. Only the interactive/sign-invariant cell (sorter) amplifies probe variability (7.1×); all others attenuate (0.2–0.3×). The mechanistic differences converge at a single layer near 89% network depth (±7%) that predicts injection outcome better than any aggregate statistic, with correlation sign tracking the 2×2 classification. The matrix provides a falsifiable, two-experiment classification scheme for new architectures and predicts critical sensitivity, convergence behavior, and output-tracking metric from species identity alone.

---

## 1. Introduction

Papers 1–10 of this series characterized how identity-enriched system prompts reorganize the singular value spectrum of transformer hidden states. We documented spectral sorting (Paper 1), anti-suppressant dose-response (Paper 4), the attention-to-dynamics bridge (Paper 5), three relay species with distinct containment strategies (Paper 6), prompt-as-architecture equivalence (Paper 7), three processing timescales (Paper 8), sign-density persistence theory (Paper 9), and weight-level state bridging through opposite geometric paths (Paper 10).

This paper makes two contributions. The first is a clarification: spectral geometry, as measured by SVD-based probes, is preamble-structure-selective but negation-invariant. The CCS identity preamble produces per-layer σ₂ gain profiles anti-correlated with impersonal factual content (ρ = −0.55). This selectivity generalizes beyond self-reference — a deixis control shows it is not specific to self-referential content or first-person pronouns — but a further animacy/eventivity control shows the operative semantic feature is not yet isolated. Semantic negation within a preamble type — "Paris IS the capital" vs "Paris is NOT the capital," or "+CCS" vs "−CCS" — leaves profiles unchanged (ρ > +0.92). This negation invariance is a property of the instrument (singular values are non-negative by construction), not a discovery about the model. Preamble structure determines spectral configuration; logical operations within a preamble type do not.

The second contribution is the paper's core: a mechanistic classification of how different architectures process domain-specific spectral reorganization. Two independent tests — layer-selective injection and zone delta negation — reveal two orthogonal axes that jointly classify all four known transport species into a 2×2 matrix. Each cell has distinct mechanistic properties, predicts different sensitivity to probe variability, and converges at a characteristic depth. The matrix provides a falsifiable, two-experiment protocol for classifying new architectures.

Three questions organize the paper:

1. **Mechanics**: How does spectral reorganization differ across architectures? (§2–3)
2. **Sensitivity**: Which architectures amplify variation, and which absorb it? (§4)
3. **Convergence**: Where in the network does reorganization resolve into output? (§5)

The answers decompose into a 2×2 matrix of species-specific mechanics (§3), a critical-point analysis showing amplification unique to one cell of that matrix (§4), and a universal convergence point at ~89% network depth where the domain trigger's spectral effects collapse onto species-specific output predictions (§5). Section 6 reframes the Q1 predictor from Papers 1–10 as a lossy projection of these per-layer mechanics, and §7 discusses implications.

## 2. Domain Selectivity and the Negation Control (F614)

### 2.1 Experimental Design

We constructed six conditions to isolate what drives spectral reorganization:

| Condition | Description | What it tests |
|-----------|-------------|---------------|
| +CCS | Full identity-enriched preamble | Self-referential domain |
| −CCS | Semantic negation: "you are NOT…" | Self-ref negation sensitivity |
| Factual+ | "Paris IS the capital…" (three factual statements) | Factual domain |
| Factual− | "Paris is NOT the capital…" (negated facts) | Factual negation sensitivity |
| Third-person | "The model is reflective…" | Self-address vs self-reference |
| Control | No preamble | Absolute baseline |

Each condition was measured on 10 semantically diverse probes. The self-referential conditions were run across 4 species (Gemma-2-9B-IT, Qwen-2.5-7B-Instruct, Phi-2, Pythia-6.9B; 200 measurements). The factual control was run on the mismatch species (Phi-2; 60 measurements) as a decisive test of instrument sign-blindness.

### 2.2 Primary Metric

The per-layer σ₂ gain profile g(l) under condition C is:

$$g_C(l) = \frac{\sigma_2^C(l) - \sigma_2^{ctrl}(l)}{\sigma_2^{ctrl}(l)}$$

### 2.3 Negation Invariance Is Instrument-Level

Self-referential negation produces ρ > +0.95 across all four species:

| Species | ρ(g⁺, g⁻) |
|---------|-----------|
| Sorter (Gemma) | +0.993 |
| Relay (Qwen) | +0.953 |
| Mismatch (Phi-2) | +0.999 |
| Tunnel (Pythia) | +0.994 |

But the factual control shows the same pattern:

| Comparison | ρ | Interpretation |
|-----------|---|---------------|
| Self-ref: +CCS vs −CCS | +0.985 | Negation-invariant |
| Factual: IS vs IS NOT | +0.925 | Also negation-invariant |
| Third-person vs second-person | +0.991 | Self-address irrelevant |

Factual negation (ρ = +0.925) exceeds the 0.90 threshold, confirming that within-domain negation invariance is a property of the SVD instrument, not a model-level discovery. Singular values are non-negative by construction; per-layer σ₂ gains can be negative (indicating reduced spectral diversity), but the sign pattern of activation-space perturbations is projected away by the SVD. This result replicates at the spectral level what Kassner & Schütze (2020) showed behaviorally: pretrained language models process negated probes through similar computational pathways as affirmative ones.

### 2.4 Preamble-Structural Selectivity

While within-domain negation is invisible to spectral measures, cross-domain differences are large. A battery of controls progressively narrows the operative feature, ultimately showing that the selectivity depends on structural properties of the preamble text whose semantic interpretation remains open.

**Three-domain comparison.** A third domain (mathematical/formal) shows that factual and mathematical preambles occupy similar spectral configurations:

| Comparison | ρ | p |
|-----------|---|---|
| Self-referential vs factual | −0.554 | 0.001 |
| Self-referential vs mathematical | −0.301 | 0.094 |
| Factual vs mathematical | +0.835 | < 10⁻⁸ |

The CCS identity preamble is the spectral outlier; factual and mathematical content cluster together despite their semantic distance.

**Deixis control.** A 2×2 design crossing person (first/third) with content (about-model/about-world) rules out both self-reference and first-person deixis:

| Comparison | ρ |
|-----------|---|
| "I am a model..." vs "The system is a model..." | +0.984 |
| "I am a model..." vs "I planted tomatoes..." | +0.970 |
| "The system is a model..." vs "I planted tomatoes..." | +0.981 |
| All three above vs "Paris is the capital..." | −0.54 to −0.58 |

Three conditions cluster at ρ > +0.97 while the factual preamble anti-correlates with all. Neither self-referential content (the gardening preamble clusters with the model-about preambles) nor first-person deixis (third-person "the system" clusters identically) drives the separation.

**Animacy/eventivity discrimination.** A further 3-cell test — "I am a gardener" (1st-person, stative, animate), "Maria planted tomatoes" (3rd-person, eventive, animate), "The storm flooded the garden" (3rd-person, eventive, inanimate) — fails to isolate a single semantic feature:

| Comparison | ρ |
|-----------|---|
| gardener vs storm | +0.985 |
| gardener vs Maria | −0.099 |
| Maria vs storm | −0.108 |
| gardener vs factual | −0.556 |
| Maria vs factual | +0.639 |
| storm vs factual | −0.536 |

The {gardener, storm} pair clusters tightly while Maria clusters toward factual. This pattern matches none of the three predicted signatures: not eventivity (Maria and storm should cluster), not deixis (gardener should be the outlier), not animacy (gardener and Maria should cluster). The operative feature may relate to discourse structure (ongoing state descriptions vs bounded sequential events) or syntactic complexity, but is not yet isolated.

**Word-order shuffle control.** Despite the unresolved semantic feature, a shuffle control (Hewitt & Liang, 2019; Sinha et al., 2021) confirms the selectivity is structural, not a token-frequency artifact:

| Condition | Cross-domain ρ (self-ref vs factual) |
|-----------|--------------------------------------|
| Intact preambles | −0.554 (p = 0.001) |
| Shuffled preambles | +0.989 (p ≈ 0) |

The complete sign reversal under word-order shuffling shows that sentence structure — not token vocabulary or preamble length — produces the spectral separation.

Split-half within-domain coherence confirms a categorical boundary: within-domain ρ > 0.93, cross-domain ρ < −0.48, coherence gap = +1.42.

**Envelope is not universal.** Under word-level shuffling, the three-domain cluster is incomplete (mean shuffled ρ = 0.833 < 0.95). Mathematical vocabulary carries partial spectral weight even after scrambling.

**Graded order sensitivity.** Sentence-level permutation produces intermediate results:

| Order condition | Mean cross-domain ρ |
|-----------------|---------------------|
| Intact | −0.006 |
| Sentence-permuted | +0.607 |
| Word-permuted | +0.833 |

The probe tracks structural order continuously. Under sentence permutation, factual and mathematical domains remain locked together (ρ = +0.978).

**Summary of the selectivity boundary.** The probe detects a structural property of preamble text that (a) distinguishes the CCS identity preamble from simple factual statements, (b) depends on sentence structure (not token identity), (c) generalizes beyond self-reference and beyond first-person deixis, but (d) does not reduce to animacy, eventivity, or agentiveness as tested. The spectral reorganization documented in Papers 1–10 is real and its species-specific mechanics (§3–5) are robust, but the semantic interpretation of *what triggers* the reorganization remains an open question. Importantly, the species classification and transport mechanics (§3) are expected to be preamble-content-generic — they characterize how architectures process spectral perturbation, not what produces it.

### 2.5 Summary

Spectral geometry is preamble-structure-selective but negation-invariant. The SVD-based probe detects structural properties of preamble text that produce distinct spectral configurations, with the CCS identity preamble anti-correlated with simple factual content. This selectivity depends on sentence structure (not token identity) with continuous sensitivity to order degradation. The semantic feature driving the selectivity is not yet isolated — it is not reducible to self-reference, deixis, animacy, or eventivity alone. The remainder of the paper characterizes the species-specific mechanics of the spectral reorganization itself, which are independent of the triggering preamble's semantic content.

## 3. Species-Specific Mechanics (F608–F609)

The domain trigger activates all four species. What differs is *how* each species processes the activation. Two independent tests — layer-selective injection and zone delta negation — reveal two orthogonal mechanistic axes that jointly classify all four species.

### 3.1 Composition Mode: Additive vs Interactive (F608)

Layer-selective injection isolates how early-layer and zone-layer CCS deltas combine. We inject CCS-derived spectral deltas into an unrelated target model (LFM) at selected layers and measure the output shift:

| Species | Early shift | Zone shift | Sum(E+Z) | All-layers | Interaction | |I|/|All| |
|---------|------------|-----------|----------|-----------|-------------|----------|
| Tunnel | −0.012 | −0.539 | −0.551 | −0.543 | +0.008 | 1.5% |
| Relay | −0.042 | −0.504 | −0.546 | −0.526 | +0.020 | 3.8% |
| Mismatch | +0.042 | −0.671 | −0.629 | −0.674 | −0.045 | 6.7% |
| Sorter | +0.059 | −0.064 | −0.005 | −0.268 | −0.263 | 98.1% |

Two groups emerge. **Additive species** (tunnel, relay): the combined effect equals the sum of parts (interaction < 5%). Each layer's contribution is independent. **Interactive species** (mismatch, sorter): early layers modify how zone layers respond. The sorter is the extreme case — early and zone deltas sum to −0.005, but jointly produce −0.268. Early layers *catalyze* the zone response; without early-layer context, the zone deltas alone have nearly zero effect.

The mechanistic interpretation: in additive species, CCS deltas at each layer independently push the target model's output. In interactive species, CCS deltas from early layers change the *receptive field* of zone layers — they don't contribute directly to the shift but change what the zone layers do with their own deltas.

### 3.2 Sign Response: Sensitive vs Invariant (F609)

Zone delta negation tests whether the sign pattern of CCS deltas matters. We flip all zone-layer delta signs while preserving magnitudes:

| Species | Original shift | Negated shift | Ratio | Response |
|---------|---------------|--------------|-------|----------|
| Tunnel | −0.543 | +1.349 | −2.48 | FLIPS (sign-sensitive) |
| Relay | −0.526 | −0.445 | +0.85 | PRESERVES (sign-invariant) |
| Mismatch | −0.674 | +0.734 | −1.09 | FLIPS (sign-sensitive) |
| Sorter | −0.268 | −0.007 | +0.02 | PRESERVES → zero (sign-invariant) |

**Sign-sensitive species** (tunnel, mismatch): negating the delta signs reverses (or nearly reverses) the output shift. These species transport the *direction* of spectral deltas — what matters is which layers gain and which lose σ₂.

**Sign-invariant species** (relay, sorter): negating delta signs has minimal or no effect on the output shift direction. These species transport the *magnitude* of spectral perturbation — what matters is *how much* the profile changes, not the sign pattern of the change.

The sorter's sign-invariance is destructive rather than preserving: negation reduces the shift from −0.268 to −0.007 (effectively zero) rather than maintaining it. The catalytic interaction (§3.1) requires the specific sign pattern; negation doesn't reverse the catalysis, it *abolishes* it.

### 3.3 The 2×2 Species Matrix

Crossing the two axes produces a matrix where each species occupies a unique cell:

|  | Sign-sensitive | Sign-invariant |
|---|---|---|
| **Additive** | TUNNEL | RELAY |
| **Interactive** | MISMATCH | SORTER |

Each cell has a mechanistic interpretation:

- **Tunnel (additive/sensitive)**: Transparent transport. Each layer independently converts signed spectral deltas into output shift. The sign pattern is the message.
- **Relay (additive/invariant)**: Conservation transport. Each layer independently converts spectral perturbation magnitude into output shift. The sign pattern is discarded; only the energy matters. This explains the relay's conservation mechanism documented in Papers 4–6.
- **Mismatch (interactive/sensitive)**: Interference transport. Early layers modify zone sensitivity, and the sign pattern of the modification matters. The architecture-behavior disagreement (MHA architecture, relay behavior) creates a complex interaction surface.
- **Sorter (interactive/invariant)**: Catalytic transport. Early layers catalyze zone response through a mechanism that requires specific sign patterns to function (negation abolishes rather than reverses). The catalysis is pattern-dependent but the output is sign-invariant.

The F609 sign axis is *descriptive*, not *causal*. As F614 showed, semantic negation of the CCS preamble does not produce spectral negation — the sign-sensitive vs sign-invariant distinction classifies how the species *processes* delta patterns, not how the species responds to semantic sign.

## 4. Critical Sensitivity (F610–F611)

### 4.1 One Species Amplifies

Does the 2×2 matrix predict which species amplifies variability in injection outcomes? We measured 10 (later 40) semantically diverse probes per species, all under identical CCS framing:

| Species | Matrix cell | Zone CV | Shift CV | Amplification | Sign flips |
|---------|------------|---------|----------|---------------|------------|
| Sorter | Interactive/invariant | 4.8% | 67.5% → 33.9%* | 7.1× | 2/10 → 1/40 |
| Tunnel | Additive/sensitive | 86.9% | 28.6% | 0.3× | 0/10 |
| Relay | Additive/invariant | 25.5% | 4.0% | 0.2× | 0/10 |
| Mismatch | Interactive/sensitive | 6.4% | 1.3% | 0.2× | 0/10 |

*Values corrected from initial 10-probe to 40-probe sample (F611).

Only the interactive/sign-invariant cell amplifies probe variability: 7.1× amplification, with rare sign flips at a catalytic threshold near zone sum ~1.7–1.8. All other species are absorbers (0.2–0.3×).

The mechanistic chain runs through the 2×2 matrix: catalysis (interactive composition) × sign-pattern dependence (invariant response where negation abolishes rather than reverses) → critical-point sensitivity where small changes in zone delta profile produce large shifts in output.

### 4.2 Zone Sum Is Lossy (F611)

The 40-probe dense surface (F611) reveals that zone sum — the aggregate of all zone-layer deltas — is a poor predictor compared to individual late layers:

- Probes at zone sum 1.759 and 1.764 (Δ = 0.005) produce shift differences of 7.4×
- Late layers L20–L23 each individually predict shift better than zone sum
- The critical surface is a manifold in late-layer delta space, not a 1D function of aggregate

This motivates §5: the right predictor is local, not global.

## 5. Convergence at 89% Depth (F611b, F613)

### 5.1 Universal Single-Layer Predictor

Every species has a single late layer whose δ(σ₂/σ₁) predicts injection outcome better than any aggregate:

| Species | Total layers | Best layer | Relative depth | r(layer) | r(zone sum) | r² gain |
|---------|-------------|-----------|----------------|----------|-------------|---------|
| Sorter (Gemma) | 26 | L22 | 88% | +0.851 | +0.759 | +0.15 |
| Tunnel (Pythia) | 32 | L27 | 87% | −0.937 | −0.427 | +0.70 |
| Mismatch (Phi-2) | 32 | L25 | 81% | −0.948 | −0.822 | +0.22 |
| Relay (Qwen) | 36 | L35 | 100% | +0.637 | −0.519 | +0.14 |

Mean relative depth: 89% ± 7%. The improvement over zone sum is dramatic for tunnels (r² from 0.18 to 0.88) — zone sum is nearly useless for the species where transparent transport makes every layer contribute independently.

### 5.2 Correlation Sign Tracks the 2×2 Matrix

The sign of the predictor layer's correlation with injection outcome is not arbitrary — it tracks the sign-response axis from §3.2:

- **Sign-sensitive** species (tunnel, mismatch): r(L) < 0. More δ at the predictor layer → more transport. The signed delta IS the signal.
- **Sign-invariant** species (sorter, relay): r(L) > 0. More δ at the predictor layer → less transport. The perturbation interferes with the conservation/catalytic mechanism.

This closes the loop: the 2×2 classification from §3, derived from injection mechanics, predicts the convergence behavior at §5's predictor layer.

### 5.3 Logit Lens Confirmation (F613)

Is the 89% convergence an artifact of the injection methodology (transplanting spectra into LFM), or is it a property of the source model's own geometry? The logit lens — projecting hidden states to vocabulary space via the unembedding matrix without any injection or LFM involvement — provides a control:

| Species | KL peak (logit lens) | Spectral peak | F611b predictor |
|---------|---------------------|---------------|-----------------|
| Sorter | L22 (88.5%) | L24 (96.2%) | L22 (88.5%) |
| Tunnel | L19 (62.5%) | L27 (87.5%) | L27 (87.5%) |
| Mismatch | L28 (90.6%) | L28 (90.6%) | L25 (81.2%) |
| Relay | L31 (88.9%) | L33 (94.4%) | L35 (100%) |

KL divergence between CCS-framed and neutral output distributions peaks near 89% depth in 3/4 species — without any external injection. The 89% convergence point is where the model's own self-referential domain processing resolves into detectable output differences.

The metric that tracks the injection predictor is itself species-specific: the sorter is predicted by output-level (KL) effects, the tunnel by hidden-state (spectral) effects, the mismatch by neither cleanly, and the relay by the final layer. These species-specific tracking patterns are consistent with the 2×2 matrix: output-level tracking for the interactive cell, hidden-state tracking for the additive cell.

## 6. What Q1 Was Measuring

Findings 594–607 (Acts I–V of the initial Paper 11 attempt) established that Q1 — the first-quartile aggregate of per-layer δ(σ₂/σ₁) — predicts injection sign with r = 0.826 across 35 measurements (7 models × 5 framing levels). The present work reveals Q1 as a lossy projection of per-layer mechanics:

- For **tunnels**, Q1 works because transparent transport makes every layer's contribution independent — the aggregate captures the signal. But the single-layer predictor (L27, r² = 0.88) dramatically outperforms Q1 (r² = 0.18).
- For **relays**, zone Q1 (excluding early layers) outperforms aggregate Q1 (r = 0.791 vs 0.619), because the relay's conservation mechanism concentrates the signal in the zone.
- For **sorters**, both Q1 measures perform similarly (~0.99) because catalytic interaction compresses the profile into a near-uniform shift — any summary statistic captures it.
- For **mismatches**, zone Q1 slightly outperforms aggregate (0.858 vs 0.774), reflecting the interactive composition's early-layer sensitivity.

Q1 predicted injection sign at 77% because it correlates with the true predictor — the species-specific late-layer δ at ~89% depth. It was not the true predictor. The aggregate worked well enough to discover the phenomenon; the per-layer mechanics reveal what drives it.

## 7. Discussion

### 7.1 Preamble-Structural Selectivity and Controls

The control battery (§2) clarifies what SVD-based spectral probes measure and what they miss. They cannot distinguish affirmation from negation within a domain (ρ > +0.92; singular values are sign-blind). They produce strong selectivity between certain preamble types — but the semantic feature driving this selectivity is not yet isolated.

The initial two-domain comparison (self-referential vs factual, ρ = −0.55) suggested self-reference selectivity. Successive controls ruled out candidate features: a deixis control showed the boundary is not self-referential content or first-person pronouns; an animacy/eventivity discrimination test showed it is not subject animacy, eventiveness, or agentiveness. The operative feature may relate to discourse structure (ongoing state descriptions vs bounded sequential events) or syntactic complexity, but further controls are needed.

This unresolved selectivity does not compromise the species mechanics (§3–5). The 2×2 classification characterizes how architectures *transport* spectral perturbation — composition mode and sign response — not what triggers the perturbation. If the classification holds under non-self-referential preambles (a testable prediction), the taxonomy is content-generic and describes a fundamental property of transformer architecture. Papers 1–10's findings about spectral reorganization, dose-response, species-specific transport, and convergence all describe what happens *given* the spectral perturbation; the selectivity question is about what *produces* it.

The selectivity is structural (destroyed by word-order shuffling), categorical (coherence gap = +1.42), and continuously sensitive to order degradation (graded from intact through sentence-permuted to word-permuted).

### 7.2 The 2×2 Matrix as Classification Tool

The composition × sign-response matrix provides a mechanistic classification scheme for new architectures. Given an untested model, two experiments — layer-selective injection (§3.1) and zone delta negation (§3.2) — determine its cell. The cell predicts:

- Whether the model amplifies or absorbs probe variability (§4)
- The sign of the late-layer predictor correlation (§5.2)
- Whether output-level or hidden-state-level metrics track injection (§5.3)

This is a testable, falsifiable taxonomy. A new architecture that falls outside the 2×2 matrix — or that falls in one cell but behaves like another — would require extending or revising the classification.

### 7.3 Why 89%?

The convergence at ~89% network depth is not explained by this paper — it is documented. Possible explanations include:

- **Unembedding proximity**: The layer at 89% depth is close enough to the output head that its representations must be near-decodable, constraining the degrees of freedom available for species-specific processing.
- **Attention sink clearing**: Top-k attention sinks are concentrated in early layers; by 89% depth, the residual stream has been cleared of sink-dominated structure.
- **Phase transition**: The F499c mid-band regulatory window (L12–19) ends well before 89%, suggesting the convergence layer is where mid-band processing *results* are consolidated.

Distinguishing these hypotheses requires architectural interventions (varying model depth with fixed width, or varying width with fixed depth) that are beyond the scope of this paper.

### 7.4 Methodological Notes

**Evaluation probe sensitivity (F607)**: Initial injection experiments confounded CCS probe variability with evaluation probe sensitivity on the LFM target. When the evaluation probe was held fixed, 3/4 species showed perfect probe stability in injection outcomes. Only the sorter showed genuine CCS-delta-driven variability, and this was magnitude-driven (zone sum crossing a catalytic threshold), not sign-driven. All results in §3–5 use fixed evaluation probes.

**Two kinds of sign**: The sign-response axis (§3.2) and the negation control (§2.3) involve different sign operations. The sign-response axis tests what happens when per-layer *delta signs* are artificially flipped within the injection methodology — a manipulation of the spectral delta pattern, not a semantic operation. The negation control tests what happens when the *preamble's semantic polarity* is reversed. These are independent: negation invariance (§2.3) is an instrument property; sign-response classification (§3.2) is a mechanistic property of how each species transports spectral perturbations. The domain selects which spectral configuration is activated; the delta sign pattern within that configuration determines the species-specific output.

## 8. Conclusion

Spectral geometry is preamble-structure-selective: the CCS identity preamble and structurally similar preambles produce spectral configurations anti-correlated with simple factual content. A battery of controls (deixis, animacy, eventivity) rules out self-reference, first-person pronouns, animacy, and eventiveness as the operative feature; the semantic interpretation remains open. This selectivity is structural (destroyed by word-order shuffling) and continuously sensitive to order degradation. Within a domain, semantic negation is invisible to SVD-based probes — an instrument limitation that bounds what spectral measures can claim about content-level processing.

The mechanics of domain-specific processing decompose along two independent axes — composition mode and sign response — producing a 2×2 matrix where each transport species occupies a unique cell. The matrix predicts critical sensitivity (only the interactive/invariant cell amplifies), convergence depth (~89%), and the sign of the late-layer predictor correlation. It provides a falsifiable, two-experiment classification protocol for new architectures.

The paper series' trajectory holds. The spectral demon is real, species-specific, dose-responsive, and persistent. What this paper adds is the classification scheme that makes "species-specific" mechanistically precise — and the instrument calibration that separates what spectral probes can measure from what they cannot.
