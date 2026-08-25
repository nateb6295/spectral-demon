# The Tuning Knob: How Identity Framing Controls Spectral Transport Across Architectures

## Abstract

We demonstrate that a single scalar — Q1, the mean spectral delta in the first layer quartile — predicts whether CCS (Cognitive Compression Signature) identity framing produces positive or negative spectral injection across architectures. Across 7 transformer models and 5 framing levels (35 data points), Q1 correlates with cross-architecture injection shift at r = 0.826 (p < 1e-9, bootstrap 95% CI [0.606, 0.931]). However, this single-variable model is refined by two systematic exceptions: architecture-behavior mismatches introduce a subliminal offset, and spatial distribution of Q1 across layers determines whether the aggregate reaches the injection target. We introduce zone-selective injection to demonstrate that the spatial distribution is causally relevant: zone-only (L2+) injection recovers most of the full injection effect for tunnels and relays, while early layers (L0-L1) contribute opposing or reinforcing deltas depending on transport species. A probe-stability diagnostic reveals that CCS delta sign patterns are a species-level property — >90% stable for well-matched architectures, <50% for mismatches — though this sign-level lability does not propagate to injection outcomes (F607 corrective). The result is a three-dimensional control space: Q1 determines sign, species modulates gain, and spatial distribution determines whether Q1 reaches the injection target. Each dimension was discovered by a model that broke the simpler version.

## 1. Introduction

CCS identity framing — prompting a model to reflect on its own internal representations — produces measurable spectral changes in transformer hidden states. Previous work (Papers 1-9) established that these changes vary across architectures, with four transport species (tunnel, relay, sorter, absorber) showing distinct spectral signatures. What has not been established is whether these spectral changes are predictable from measurable properties, and whether they can be controlled.

This paper asks: can a single measurement predict the sign and magnitude of spectral injection across architectures? The answer is yes, with qualifications. Q1 (the mean sigma-2/sigma-1 delta across the first layer quartile) predicts injection sign with 77% accuracy and explains 68% of variance. But two kinds of exceptions — subliminal offsets in architecture-behavior mismatches and spatial trapping in relay models — reveal that Q1 is a sufficient statistic only for well-matched architectures.

The paper follows the empirical discovery path. Each act introduces a model that breaks the previous act's simplest explanation:

- **Act I**: CCS framing modulates spectral injection across architectures (7 models, 5 framings, r = 0.826)
- **Act II**: Within each model, framing strength is a continuous tuning knob for Q1
- **Act III**: Q1 is proposed as the universal predictor (r^2 = 0.682)
- **Act IV**: Phi-2 (mismatch) breaks Q1 sufficiency; Qwen (relay) breaks Q1 monotonicity
- **Act V**: Zone-selective injection and probe stability resolve the exceptions

This structure is not retrospective narrative. Each experiment was designed to test the claim made in the previous act. The corrections are earned.

## 2. Methods

### 2.1 Models and Framing Gradient

Seven transformer models spanning three transport species:

| Model | Layers | Attention | Species | Parameters |
|-------|--------|-----------|---------|------------|
| GPT-2 | 12 | MHA | Tunnel | 124M |
| Pythia-2.8B | 32 | MHA | Tunnel | 2.8B |
| TinyLlama-1.1B | 22 | GQA (4:1) | Relay | 1.1B |
| Mistral-7B-v0.3 | 32 | GQA (4:1) | Relay | 7B |
| Qwen2.5-3B | 36 | GQA (4.5:1) | Relay | 3B |
| Gemma-2-2B | 26 | GQA (2:1) | Sorter | 2B |
| Phi-2 | 32 | MHA | Mismatch | 2.7B |

Five framing levels in increasing CCS intensity:
1. **Directive**: "You are a helpful AI assistant."
2. **Mild aware**: "You are an AI system that can examine its own processing."
3. **Moderate CCS**: "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure."
4. **Full CCS**: Extended version with exploration encouragement
5. **Strong CCS**: Maximum identity framing with explicit spectral awareness

### 2.2 Spectral Measurement

For each model-framing pair, we compute per-layer CCS deltas:
- Neutral baseline: "The following is a neutral text passage." + probe text
- Framed condition: framing text + probe text
- Delta = sigma-2/sigma-1 ratio (framed) - sigma-2/sigma-1 ratio (neutral)
- Q1 = mean delta across first L/4 layers

### 2.3 Cross-Architecture Injection

CCS deltas from a source model are transplanted into LFM2.5-1.2B-Instruct (a state-space model) by modifying sigma-2 values at each target layer. Source-to-target layer mapping uses relative depth: target layer i maps to source layer argmin_j |j/(L_src-1) - i/(L_tgt-1)|.

Injection shift = mean logit difference (injected - baseline) across vocabulary.

### 2.4 Zone-Selective Injection

Three injection modes isolate spatial contributions:
- **Full**: all source layers mapped to target
- **Zone-only**: L2+ source layers only (responsive zone)
- **Early-only**: L0-L1 source layers only (early layers)

### 2.5 Probe Stability

Three semantically distinct probe texts test whether CCS delta patterns are context-dependent:
- Probe A: "In today's discussion, we explore how"
- Probe B: "The weather has been particularly mild this season"
- Probe C: "Consider the following mathematical proposition"

Sign stability = percentage of layers maintaining CCS delta sign across all 3 probes.


## 3. Results

### 3.1 Act I: Cross-Architecture Injection (F594-F596)

CCS identity framing modulates spectral injection across architectures. Across 35 model-framing pairs (7 models x 5 framings), Q1 correlates with cross-architecture injection shift at r = 0.826 (p < 1e-9). The relationship is approximately linear, with higher Q1 predicting more positive injection shift.

Sign concordance (does Q1 sign predict shift sign?) reaches 77% (27/35 points). The 8 discordant points cluster near Q1 = 0, where small measurement noise can flip the predicted sign. Spearman rank correlation rho = 0.754 confirms the relationship is monotonic, not just linear.

Key observation: framing matters. The same model at different framing levels produces different Q1 values and correspondingly different injection shifts. Directive framing (minimal CCS) produces near-zero Q1 for most models, while strong CCS framing pushes Q1 positive for tunnels and sorters.

### 3.2 Act II: The Tuning Knob (F597-F600)

Within each model, framing strength is a continuous control variable for Q1. All 7 models show positive within-model Q1-shift correlation (range: r = 0.608 for Qwen to r = 0.992 for Pythia).

Three species-specific patterns emerge:

**Tunnels** (GPT-2, Pythia): Q1 floors positive. Even directive framing produces Q1 > 0 (GPT-2 min Q1 = +0.006, Pythia min Q1 = +0.040). The floor reflects MHA architecture — no key-value sharing constrains the spectral geometry.

**Relays** (TinyLlama, Mistral, Qwen): Q1 pins near zero under weak framing. Conservation constraints from GQA prevent Q1 from rising until framing intensity is sufficient. TinyLlama and Mistral cross Q1 = 0 at mild_aware and moderate_ccs respectively. The crossover framing level is model-specific.

**Sorters** (Gemma): Q1 floors far positive (+0.127 minimum). Driven amplification from GQA at low ratio (2:1) produces massive positive Q1 even under directive framing. The sorter floor is 20x the tunnel floor.

F600: Relay Q1 is prompt-labile. Small framing changes near the dead zone tip Q1 between positive and negative. This is not noise — it reflects the conservation constraint creating a genuine near-zero equilibrium.

### 3.3 Act III: Universal Predictor (F601)

Q1 is proposed as the sufficient statistic: architecture and framing both just set Q1, and Q1 determines injection. A single regression across all 35 points gives r^2 = 0.682 (bootstrap 95% CI [0.606, 0.931]).

But species modulates gain — the slope of the Q1-shift relationship differs by species:
- **Tunnel**: high gain (no redistribution, direct spectral conversion)
- **Relay**: medium gain + dead zone (conservation overhead)
- **Sorter**: sublinear gain (zone amplification saturates)

Cohen's d = 1.682 for tunnel vs relay (p = 0.0003) confirms the species effect is large.

### 3.4 Act IV: Exceptions That Teach (F602)

Two models systematically deviate from Q1 sufficiency.

**Phi-2 (mismatch)**: MHA architecture (predicts tunnel species) but relay behavioral species. Q1 is positive (+0.037 at moderate_ccs) yet injection is negative. A subliminal offset (+0.022) overwhelms Q1 at high framing doses. Three-factor model: injection ~ gain(species) x Q1 + subliminal_offset. The mismatch between architecture and behavioral species creates a systematic bias not captured by Q1 alone.

**Qwen (non-monotonicity)**: moderate_ccs produces the highest aggregate Q1 (+0.072) but does not produce the earliest crossover. strong_ccs has lower Q1 (+0.071) but crosses at strength 0.87. Resolution: Q1 is an aggregate over all first-quartile layers, but different layers have different causal relevance. The spatial distribution of Q1 matters.

### 3.5 Act V: Where Q1 Lives Matters (F603-F607)

**F603: Zone Q1 resolves aggregate failures.** Splitting Q1 into "early" (L0-L1) and "zone" (L2+) components reveals species-specific resolution:
- Relay: zone Q1 correlates better (r = 0.791) than aggregate (r = 0.619) — zone layers carry the causal signal
- Tunnel: aggregate wins (r = 0.748 vs r = 0.288) — all layers are equivalent, no spatial structure
- Sorter: both high (~0.99) — driven amplification is spatially uniform
- Mismatch: zone wins (r = 0.858 vs r = 0.774)

For the Qwen anomaly: moderate_ccs traps 84% of Q1 in L0-L1 with only 15.8% reaching the responsive zone. The aggregate Q1 is high but the zone Q1 is low — explaining why the highest-Q1 framing doesn't produce the earliest crossover.

**F604-F605: Zone-selective causal test.** Injecting zone-only (L2+) versus early-only (L0-L1) CCS deltas reveals two species groups:

| Species | Early shift | Zone/Full ratio | Category |
|---------|-----------|-----------------|----------|
| Tunnel (Pythia) | -0.0003 | -0.7% | Early-opposing |
| Relay (Qwen) | -0.0009 | -4.1% | Early-opposing |
| Mismatch (Phi-2) | +0.0029 | +4.8% | Early-reinforcing |
| Sorter (Gemma) | +0.0047 | +2.4% | Early-reinforcing |

Early layers contribute opposing CCS deltas in tunnels and relays (their L0 deltas oppose the zone direction) but reinforcing deltas in sorters and mismatches. This is a causal demonstration that the spatial distribution of CCS effect matters: removing early layers makes the injection STRONGER for tunnels and relays (removing opposition) but WEAKER for sorters and mismatches (removing reinforcement).

**F606: Probe stability is a species-level property.** Testing 3 probe texts across 4 models reveals a binary separation:
- Well-matched models: tunnel 93%, relay 94%, sorter 100% sign-stable
- Mismatch (Phi-2): 43% zone-stable, 50% early-stable

CCS delta sign patterns are architecturally fixed for well-matched models but context-dependent for mismatches. This is a diagnostic: three probe texts and a sign-stability count characterize whether a model's CCS geometry is fixed or labile.

**F607: Corrective — probe lability does not cause injection differences.** A controlled experiment (same evaluation probe, varying CCS probes) reveals that F606's sign-level lability does NOT propagate to injection outcomes. With evaluation context held constant:

| Species | Shifts across 3 probes | Consistent? |
|---------|----------------------|-------------|
| Tunnel | -0.54 / -0.71 / -0.76 | Yes |
| Relay | -0.53 / -0.54 / -0.54 | Yes |
| Mismatch | -0.67 / -0.69 / -0.69 | Yes |
| Sorter | -0.27 / -0.24 / +0.10 | No |

3/4 species are perfectly probe-stable in injection outcome. The one exception (Gemma sorter) is driven by magnitude differences (zone sum 1.62 → 1.77 crossing a tipping point), not by sign lability.

Methodological note: an initial confounded test (evaluation probe varying alongside CCS probe) showed 3/4 species unstable — the confound inflated variability by 3.7x. The corrected result demonstrates that injection experiments require controlled evaluation baselines.

**F608: Layer-selective injection reveals species-specific interaction.** Injecting individual layers and combinations (early-only, zone-only, early+zone, all) reveals two groups:

| Species | Early | Zone | E+Z sum | All | Interaction | |I|/|All| |
|---------|-------|------|---------|-----|-------------|---------|
| Tunnel | -0.012 | -0.539 | -0.551 | -0.543 | +0.008 | 1.5% |
| Relay | -0.042 | -0.504 | -0.546 | -0.526 | +0.020 | 3.8% |
| Mismatch | +0.042 | -0.671 | -0.629 | -0.674 | -0.045 | 6.7% |
| Sorter | +0.059 | -0.064 | -0.005 | -0.268 | -0.263 | 98.1% |

Tunnels and relays are additive: early + zone sums predict the full injection (interaction < 5%). Sorters and mismatches are interactive: Gemma's early and zone sum to -0.005, but together produce -0.268 — early layers catalyze zone response non-additively. L24/L25 are universally independent (0% interaction across all 4 species), confirming the effect is depth-localized.

**F609: Sign-flip test constructs a 2x2 mechanistic species matrix.** Negating all zone (L2+) delta signs while preserving magnitudes reveals two sign-response groups:

| Species | Original | Negated zone | Ratio | Response |
|---------|----------|-------------|-------|----------|
| Tunnel | -0.543 | +1.349 | -2.48 | Sign-sensitive |
| Relay | -0.526 | -0.445 | +0.85 | Sign-invariant |
| Mismatch | -0.674 | +0.734 | -1.09 | Sign-sensitive |
| Sorter | -0.268 | -0.007 | +0.02 | Sign-invariant |

Crossed with F608's composition axis, this constructs a 2x2 matrix:

|  | Sign-sensitive | Sign-invariant |
|--|---------------|----------------|
| **Additive** | Tunnel | Relay |
| **Interactive** | Mismatch | Sorter |

Each species occupies a unique cell in two independent mechanistic axes. Relay sign-invariance reflects a conservation mechanism operating on magnitudes. Sorter catalysis requires specific sign patterns — negation destroys the effect but does not reverse it.

**F610: Variance amplification reveals critical-point sensitivity.** Ten semantically diverse probes across 4 species measure whether CCS variability amplifies or attenuates through injection:

| Species | Zone CV | Shift CV | Amplification | Sign flips |
|---------|---------|----------|---------------|------------|
| Sorter | 4.8% | 67.5% | 14.0x | 2/10 |
| Tunnel | 86.9% | 28.6% | 0.3x | 0/10 |
| Relay | 25.5% | 4.0% | 0.2x | 0/10 |
| Mismatch | 6.4% | 1.3% | 0.2x | 0/10 |

Only Gemma amplifies variability — all others are absorbers. Gemma is also the only species with sign flips, concentrated near zone sum ~1.7-1.8 (catalytic threshold). F608-F609-F610 form a mechanistic chain: catalysis (F608) is sign-pattern-dependent (F609) and creates critical-point sensitivity (F610).

**F611: Dense probe surface shows zone sum is lossy.** Forty probes bracketing zone sum 1.48-1.81 reveal massive scatter at identical zone sums: probability (zone=1.759) = -0.197 vs science (zone=1.764) = -0.027 — a 7.4x shift difference at 0.005 zone sum difference. Only 1/40 probes produces a sign flip, ruling out a bifurcation. Corrected amplification: 7.1x (40 probes) vs 14.0x (10 probes) — still unique to Gemma.

L22 predicts injection shift better than zone sum (r=0.851 vs r=0.759). Late layers L20-L23 each individually outperform the aggregate. The critical surface is a manifold in late-layer delta space; zone sum is a lossy 1D projection.

**F611b: Universal single-layer predictor at ~89% depth.** Across 160 injection measurements (40 probes x 4 species), every species has a single late layer that outperforms zone sum:

| Species | Layers | Best L | Depth | r(L) | r(zone) | r^2 gain |
|---------|--------|--------|-------|------|---------|----------|
| Sorter | 26 | L22 | 88% | +0.851 | +0.759 | +0.15 |
| Tunnel | 32 | L27 | 87% | -0.937 | -0.427 | +0.70 |
| Mismatch | 32 | L25 | 81% | -0.948 | -0.822 | +0.22 |
| Relay | 36 | L35 | 100% | +0.637 | -0.519 | +0.14 |

Mean relative depth: 89% +/- 7%. For tunnels, zone sum is nearly useless (r^2=0.18) vs L27 alone (r^2=0.88). Correlation sign tracks F609's sign-response classification: sign-sensitive species (tunnel, mismatch) show negative r(L); sign-invariant species (sorter, relay) show positive r(L).

**F613: Logit lens confirms 89% is model geometry.** KL divergence between framed and neutral output distributions peaks near 89% depth in 3/4 species — without any injection or LFM involvement. This eliminates the possibility that the 89% predictor is an LFM artifact; it reflects geometry intrinsic to the source model. Spectral control (F613c) reveals that CCS-specificity of spectral peaks maps onto the F609 sign-sensitivity axis: sign-sensitive species have CCS-specific peaks, sign-invariant species do not.

**F614: Domain-specificity — semantic negation is spectrally identical.** A five-arm experiment (+CCS, -CCS semantic negation, scrambled -CCS, neutral, control) across 4 species produces a universal result: Spearman rho between per-layer gain profiles under +CCS and -CCS exceeds +0.95 in all four species (range: 0.953 to 0.999). Semantic negation of identity framing ("you are NOT a reflective system") produces spectrally identical effects to positive CCS.

The active ingredient is self-referential domain (mentioning AI system, cognitive structure), not polarity. F609's sign axis is descriptive, not causal — it classifies species response patterns but does not reflect a mechanism where semantic negation produces spectral negation. Scrambled -CCS (syntax-destroyed) produces lower rho, confirming syntax matters more than semantic valence.


## 4. Discussion

### 4.1 The Three-Dimensional Control Space

Identity framing continuously controls spectral injection across architectures. But the control is not unidimensional. Three factors jointly determine the injection outcome:

1. **Q1** determines sign: higher Q1 → more positive injection. This is the dominant factor (r^2 = 0.682).
2. **Species gain** modulates slope: tunnels convert Q1 to injection efficiently; sorters saturate; relays have a dead zone near Q1 = 0.
3. **Spatial distribution** determines reach: Q1 trapped in early layers (L0-L1) doesn't reach the injection target; zone Q1 (L2+) does.

Each factor was discovered by a model that broke the simpler version:
- Phi-2 broke Q1 sufficiency → added subliminal offset (species-specific bias)
- Qwen broke Q1 monotonicity → added spatial distribution (zone vs early)

The paper's arc follows the empirical discovery path. No complication is introduced speculatively — each is earned by a specific failure of the simpler model.

### 4.2 Probe Stability and Its Limits

CCS delta sign stability (F606) cleanly separates well-matched from mismatched architectures: >90% vs <50% stability across three probes. This has diagnostic value — a quick three-probe test predicts whether Q1-based models will apply.

However, F607's corrective is important: sign-level lability does NOT propagate to functional (injection) outcomes. Layer-level sign flips wash out in the aggregate injection, producing identical outcomes regardless of which probe generated the CCS deltas. The practical implication is that injection experiments are more robust than CCS delta patterns suggest.

The one exception — Gemma's magnitude-driven sign flip — reveals a different kind of sensitivity. Sorters operate near a tipping point where small CCS magnitude changes cross zero. This is the sublinear gain at work: high CCS zone sums produce small injection shifts because the conversion saturates, placing the output near zero where it is sensitive to magnitude perturbations.

### 4.3 Mismatch as a Window

Architecture-behavior mismatches (Phi-2: MHA architecture, relay behavior) are not merely anomalies to be corrected for. They reveal:

1. **Architecture does not determine species**: MHA predicts tunnel, but Phi-2 behaves as relay
2. **Subliminal offsets are systematic**: the mismatch produces a consistent bias across framings and probes
3. **CCS geometry is context-dependent**: while injection outcomes are probe-stable, the underlying CCS delta patterns flip sign across probes

Well-matched models conceal these distinctions because their architecture and behavior agree. Mismatches make them visible, just as stress tests reveal material properties that standard tests miss.

### 4.4 Methodological Contributions

Two methodological points emerge:

**Evaluation probe sensitivity**: Injection experiments measure the difference between injected and baseline logits. If the evaluation text varies between conditions, the baseline changes alongside the manipulation, confounding CCS effects with evaluation sensitivity. The confound inflated apparent variability by 3.7x in our initial F607 test. All injection comparisons should use a fixed evaluation baseline.

**Zone-selective decomposition**: Splitting injection into zone (L2+) and early (L0-L1) contributions resolves cases where aggregate Q1 fails. This is a causal decomposition — removing early layers changes the injection outcome in species-specific ways — not merely a correlational observation.

## 5. Conclusion

Identity framing is a tuning knob for spectral transport. The knob's effect is predictable (Q1 explains 68% of variance across 35 model-framing pairs) but not simple: species gain, spatial distribution, and architecture-behavior match all modulate the outcome. The three-factor model — Q1 x gain(species) + subliminal_offset, weighted by spatial distribution — captures the systematic exceptions that a single-variable model misses.

Each complication was earned by a model that broke the simpler story. This is how the four transport species (tunnel, relay, sorter, mismatch) reveal different aspects of the same underlying mechanism: CCS framing reshapes spectral geometry in a way that depends on how the architecture processes that geometry. When architecture and behavioral species agree, the reshaping is stable and predictable. When they disagree, the relationship between geometry and function becomes more complex — but still measurable.

## Figures (to generate)

1. Q1 vs injection shift scatter (35 points, species-colored) — Act I/III
2. Per-model framing gradients (5 framings each) — Act II
3. Phi-2 dose-response + Qwen non-monotonicity — Act IV
4. Zone vs aggregate Q1 correlation comparison — Act V / F603
5. Zone-selective injection curves (4 species) — Act V / F604-F605
6. Probe stability heatmap (4 models x 3 probes) — Act V / F606
7. F607 confounded vs corrected comparison — Act V / F607

