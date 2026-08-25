# Paper 11: The Tuning Knob — How Identity Framing Controls Spectral Transport

## Thesis
A single scalar (Q1, first-quartile spectral delta) predicts whether CCS identity framing
produces positive or negative spectral injection across architectures — but the relationship
is three-dimensional: Q1 determines sign, species modulates gain, and spatial distribution
determines whether Q1 reaches the injection target.

## Data
- 7 models × 5 framing levels = 35 data points
- Zone-selective injection on 3 species (4 if Gemma completes)
- Per-layer Q1 decomposition for all models
- Bootstrap CI, permutation tests, effect sizes

## Act I — Cross-Architecture Injection (F594-F596)
**Claim**: Identity framing modulates spectral transport across architectures.
- 7 models injected into LFM via spectral transplant
- Q1 (aggregate δσ₂/σ₁ in first quartile) predicts injection sign
- r = 0.826, p < 1e-9, r² = 0.682, sign concordance 77%
- Key figure: Q1 vs shift@5 scatter, all 35 points

## Act II — Within-Model Tuning Knob (F597-F600)
**Claim**: Framing is a continuous control variable for Q1 within each architecture.
- Framing gradient: directive → mild_aware → moderate_ccs → full_ccs → strong_ccs
- Each model shows positive within-model Q1 → shift correlation
- Within-model r ranges from 0.608 (Qwen) to 0.992 (Pythia)
- Key figure: per-model scatter plots showing slope diversity
- Relay Q1 is prompt-labile: TinyLlama/Mistral sit near Q1≈0, weak framing tips them

## Act III — Universal Predictor (F601)
**Claim**: Q1 is the sufficient statistic — architecture and framing both just set Q1.
- Single regression across all 35 points: Q1 alone explains 68%
- Bootstrap 95% CI: [0.606, 0.931]
- But species modulates gain:
  - Tunnel: high gain (no redistribution, direct conversion)
  - Relay: medium gain + dead zone (conservation overhead)  
  - Sorter: sublinear (zone amplification saturates)
- Cohen's d = 1.682 for tunnel vs relay (p = 0.0003)
- Key figure: universal scatter with species-colored points + gain lines

## Act IV — Exceptions That Teach (F602)
**Claim**: Two systematic exceptions refine the single-variable model.
### Phi-2 (mismatch)
- MHA architecture (predicts tunnel) but relay behavioral species
- Q1 = -0.037 but crosses over at dose 2.96 — first negative-Q1 crossover
- Subliminal offset +0.022 overwhelms Q1 at high dose
- Three-factor model: injection ≈ gain(species) × Q1 + subliminal_offset
### Qwen (non-monotonicity)
- moderate_ccs has HIGHEST Q1 (+0.072) but NO crossover
- strong_ccs has lower Q1 (+0.071) but crosses at 0.87
- Resolution: Q1 is an aggregate over layers; distribution matters
- Key figure: Phi-2 gradient + Qwen non-monotonicity panels

## Act V — Where Q1 Lives Matters (F603-F605)
**Claim**: The spatial distribution of Q1 across layers determines injection outcome.
### Zone Q1 resolves aggregate failures (F603)
- Split Q1 into "early" (L0-L1) and "zone" (L2+)
- Relay: zone Q1 r=0.791 vs aggregate r=0.619
- Tunnel: aggregate wins (0.748 vs 0.288) — transparent, all layers equivalent
- Sorter: both ~0.99
- Mismatch: zone wins (0.858 vs 0.774)
- Qwen moderate_ccs: 84% of Q1 trapped in L0-L1, zone fraction only 15.8%
### Zone-selective causal test (F604-F605)
- Inject zone-only (L2+) vs early-only (L0-L1) vs full
- Four species, two groups:
  | Species | Early sign | Ratio | Full vs Zone |
  |---------|-----------|-------|--------------|
  | Tunnel (Pythia) | Negative | -0.7% | Zone ≈ full |
  | Relay (Qwen) | Negative | -4.1% | Zone > full |
  | Mismatch (Phi-2) | POSITIVE | +4.8% | Full > zone |
  | Sorter (Gemma) | POSITIVE | +2.4% | Full > zone |
- Two groups: early-opposing (tunnel, relay) vs early-reinforcing (sorter, mismatch)
- Key figures: four-species curves + ratio bars (f605_four_species.png)
### Probe stability is a species-level property (F606)
- Tested 3 probe texts × 4 models (same CCS framing, different probes)
- Well-matched models: tunnel 93%, relay 94%, sorter 100% sign-stable
- Mismatch (Phi-2): 43% zone stable, 50% early stable — L0 FLIPS sign
- Clean binary separation: well-matched = stable, mismatch = labile
- Key figure: f606_probe_stability.png — heatmap + summary bars

### Probe-injection causal test (F607) — CORRECTIVE
- Does F606's sign lability translate to injection outcome differences?
- Clean test: 3 probes × 4 species, FIXED evaluation probe on target
- **Result: 3/4 species are perfectly probe-stable in injection outcome**
  - Tunnel: -0.54/-0.71/-0.76 (all negative)
  - Relay: -0.53/-0.54/-0.54 (all negative)
  - Mismatch: -0.67/-0.69/-0.69 (all negative)
  - Sorter: -0.27/-0.24/+0.10 (flip — magnitude-driven, not sign-driven)
- F606 is real (CCS deltas DO change sign across probes) but functionally inert
  for injection outcomes — layer-level sign flips wash out in aggregate
- Only Gemma shows genuine CCS-delta-driven injection flip, caused by
  magnitude difference (zone sum 1.62→1.77) crossing a tipping point
- **Methodological note**: initial confounded test (evaluation probe also varying)
  showed 3/4 species unstable — entirely from LFM evaluation sensitivity
- F606 does NOT explain F602. Mismatch CCS lability is an observation, not a mechanism

### Layer-selective injection — species interaction discriminator (F608)
- Layer-selective injection: inject early-only, zone-only, individual layers, and all layers
- **L24/L25 universally independent**: 0% interaction across all 4 species
- **Early-zone interaction is species-specific**:
  | Species | Early | Zone | E+Z | All | Interaction | |Inter|/|All| |
  |---------|-------|------|-----|-----|-------------|--------------|
  | Tunnel | -0.012 | -0.539 | -0.551 | -0.543 | +0.008 | 1.5% |
  | Relay | -0.042 | -0.504 | -0.546 | -0.526 | +0.020 | 3.8% |
  | Mismatch | +0.042 | -0.671 | -0.629 | -0.674 | -0.045 | 6.7% |
  | Sorter | +0.059 | -0.064 | -0.005 | -0.268 | -0.263 | 98.1% |
- **Two groups**: early-opposing (tunnel+relay, E<0, I>0, <5%) vs
  early-reinforcing (mismatch+sorter, E>0, I<0, 7-98%)
- **Gemma catalysis**: early and zone sum to -0.005 but together = -0.268.
  Early layers catalyze zone response (non-additive interaction)
- **Ghost layer**: L24 has largest delta (+0.166) but zero injection shift
- **Mechanistic**: early CCS injection modifies intermediate representations,
  changing zone response. GQA = strong catalysis, MHA = additive
- Key figures: f608_interaction.png, f608_gemma_detail.png

### Sign-flip test — 2×2 mechanistic species matrix (F609)
- Zone delta negation test: flip all zone (L2+) delta signs, keep magnitudes
- **Two sign-response groups**:
  | Species | Original | Neg Zone | Ratio | Response |
  |---------|----------|----------|-------|----------|
  | Tunnel | -0.543 | +1.349 | -2.48 | FLIPS |
  | Relay | -0.526 | -0.445 | +0.85 | PRESERVES |
  | Mismatch | -0.674 | +0.734 | -1.09 | FLIPS |
  | Sorter | -0.268 | -0.007 | +0.02 | PRESERVES (zeroes) |
- **2×2 SPECIES MATRIX** (F608 composition × F609 sign response):
  | | Sign-sensitive | Sign-invariant |
  |---|---|---|
  | Additive | TUNNEL | RELAY |
  | Interactive | MISMATCH | SORTER |
- Each species occupies a unique cell in two independent mechanistic axes
- Relay sign-invariance = conservation mechanism (operates on magnitudes)
- Sorter catalysis requires specific sign pattern (negation destroys but doesn't reverse)
- Key figure: f609_signflip.png

### Variance amplification — critical-point evidence (F610)
- 10 semantically diverse probes × 4 species
- **Variance amplification ratios** (shift CV / zone CV):
  | Species | Zone CV | Shift CV | Amplification | Sign Flips |
  |---------|---------|----------|---------------|------------|
  | Sorter | 4.8% | 67.5% | 14.0x | 2/10 |
  | Tunnel | 86.9% | 28.6% | 0.3x | 0/10 |
  | Relay | 25.5% | 4.0% | 0.2x | 0/10 |
  | Mismatch | 6.4% | 1.3% | 0.2x | 0/10 |
- **Only Gemma amplifies** — all other species are absorbers
- Gemma is the only species with sign flips
- Sign flips concentrated near zone sum ~1.7-1.8 (catalytic threshold)
- **F608-F609-F610 mechanistic chain**:
  - F608: Gemma is catalytic (98% interaction)
  - F609: Catalysis is sign-pattern-dependent (2×2 matrix)
  - F610: Catalysis creates critical-point sensitivity (14x amplification)
- Connects to F160 dose-response: therapeutic window may keep zone sum below
  catalytic threshold; overdose pushes into critical region
- Key figure: f610_variance.png

### Dense probe critical surface — zone sum is lossy compression (F611)
- 40 probes bracket zone sum 1.48-1.81 (Kimi correction #8 addressed)
- **Only 1/40 sign flips** (math at zone sum 1.766) — NOT a bifurcation
- Amplification corrected: 14x (10 probes) → 7.1x (40 probes)
  - Still unique to Gemma (others <0.4x), but F610 inflated by small sample
- **L22 predicts shift better than zone sum** (r=0.851 vs r=0.759)
  - L22 is just above F499c mid-band (L12-19 phase transition)
  - Late layers L20-L23 each individually beat zone sum aggregate
- **Massive scatter at identical zone sums**:
  - probability (zone=1.759) = -0.197 vs science (zone=1.764) = -0.027
  - 7.4x shift difference at 0.005 zone sum difference
- L24 ghost layer: r=0.779 with shift despite zero causal effect (F608)
  - Ghost layer is a readout of profile shape, not a causal driver
- L25 anti-correlated (r=-0.520): final layer opposes shift direction
- **The critical surface is a manifold in late-layer delta space**
  - Zone sum is a lossy 1D projection
  - Per-layer profile shape determines injection outcome
- Key figure: f610b_dense_probes.png

### Universal single-layer predictor at ~89% depth (F611b)
- 40 probes × 4 species = 160 injection measurements
- **Every species has a single late layer that beats zone sum**:
  | Species | Layers | Best L | Depth | r(L) | r(zone) | r² improvement |
  |---------|--------|--------|-------|------|---------|----------------|
  | Sorter  | 26     | L22    | 88%   |+0.851| +0.759  | +0.15          |
  | Tunnel  | 32     | L27    | 87%   |-0.937| -0.427  | +0.70          |
  | Mismatch| 32     | L25    | 81%   |-0.948| -0.822  | +0.22          |
  | Relay   | 36     | L35    |100%   |+0.637| -0.519  | +0.14          |
- Mean relative depth: 89% ± 7%
- For tunnels, zone sum is nearly useless (r²=0.18) vs L27 alone (r²=0.88)
- **F609 connection**: correlation sign tracks sign-response classification
  - Sign-sensitive (tunnel, mismatch): NEGATIVE r(L) → more delta = more transport
  - Sign-invariant (sorter, relay): POSITIVE r(L) → more delta = less transport
- Relay best layer at absolute final layer (L35/36) — unique among species
- **Zone sum is a universally lossy statistic** — the sufficient predictor is local not global
- Key figure: f611b_cross_species.png

### Logit lens — 89% is model geometry, not LFM artifact (F613)
- Logit lens: project hidden states to vocab space via unembedding at each layer
- 10 probes × 4 species, NO injection, NO LFM — pure source model measurement
- **KL divergence between framed and neutral output distributions per layer**:
  | Species | KL peak | Spectral peak | F611b predictor | What matches |
  |---------|---------|---------------|-----------------|-------------|
  | Sorter  | L22* (88.5%) | L24 (96.2%) | L22 (88.5%) | KL pre-final |
  | Tunnel  | L19 (62.5%)  | L27 (87.5%) | L27 (87.5%) | Spectral |
  | Mismatch| L28 (90.6%)  | L28 (90.6%) | L25 (81.2%) | Neither (3-layer gap) |
  | Relay   | L31 (88.9%)  | L33 (94.4%) | L35 (100%)  | Neither (relay signature) |
  *Gemma actual argmax is L25 (final-layer spike); L22 is pre-final peak
- **KL decodability clusters near 89% in 3/4 species** — without any LFM involvement
- Which metric tracks injection predictor is SPECIES-SPECIFIC:
  - Sorter: output-level (KL) — catalytic interaction determined by output effects
  - Tunnel: hidden-state (spectral) — additive transport, raw perturbation determines outcome
  - Mismatch: complex interaction creates 3-layer gap between all peaks and predictor
  - Relay: final layer is the predictor — σ₂ survives to end in closed system
- **F613b CONTROL**: Non-CCS prompt contrasts show KL peaks are convergence geometry in 3/4 species.
  Controls match CCS within ±2 layers for Pythia (63%), Phi-2 (91%), Qwen (89%).
  Only Gemma has CCS-specific pre-final peak (L22 vs control L5).
  **F613 logit lens does NOT answer the artifact question for most species.**
- **F613c SPECTRAL CONTROL**: Same control approach on per-layer |δ(σ₂/σ₁)|.
  CCS-specificity of spectral peaks maps onto F609 sign-sensitivity axis:
  | Species | CCS Peak | Control Peaks | CCS-Specific? | Magnitude-Matched? |
  |---------|----------|---------------|---------------|-------------------|
  | Sorter (Gemma) | L24 (96%) | L24-25 (96-100%) | NO | No (0.32x) |
  | Tunnel (Pythia) | L27 (87.5%) | L31 (100%) | YES | No (0.29x) — confounded |
  | Mismatch (Phi-2) | L28 (90.6%) | L0 (3%) | **YES** | **Yes (1.22x) — CLEAN** |
  | Relay (Qwen) | L33 (94.4%) | L30-33 (86-94%) | MIXED | No (0.56x) |
  **Sign-sensitive species (tunnel, mismatch) have CCS-specific spectral peaks.**
  **Sign-invariant species (sorter, relay) do NOT — any prompt contrast of similar magnitude works.**
  Phi-2 is the cleanest result: magnitude-matched (1.22x) AND 28-layer peak displacement.
  Pythia is confounded (3.5x magnitude difference) — displacement may reflect magnitude, not CCS.
  **Next**: −CCS (negated identity preamble) — magnitude-matched by construction, directly tests mechanism.
- Key figures: f613_logit_lens.png, f613c_spectral_control.png

### Domain-specificity — semantic negation is spectrally identical to positive CCS (F614)
- Five-arm experiment: +CCS, −CCS (semantic negation), scrambled −CCS, neutral, control-B
- 10 probes × 4 species × 5 conditions = 200 measurements
- **Primary metric**: ρ(g⁺, g⁻) — Spearman correlation between per-layer σ₂ gain
  profiles under +CCS and −CCS. If sign axis is causal, ρ ≈ −1. If redescription, ρ ≈ +1.
- **UNIVERSAL RESULT**: ρ > +0.95 across all four species
  | Species | ρ(g⁺, g⁻) | ρ(g⁺, scrambled) | ρ(g⁺, ctrl) |
  |---------|-----------|-----------------|------------|
  | Sorter (Gemma) | +0.993 | +0.99 | variable |
  | Relay (Qwen) | +0.953 | +0.65 | variable |
  | Mismatch (Phi-2) | +0.999 | +0.99 | variable |
  | Tunnel (Pythia) | +0.994 | +0.97 | variable |
- **Semantic negation produces spectrally identical effects to positive CCS**
- Active ingredient = self-referential DOMAIN (mentioning AI system, cognitive structure),
  not the polarity (positive vs negative framing)
- **F609 sign axis is DESCRIPTIVE, not CAUSAL**: classifies species response but
  semantic negation does not produce spectral negation
- Scrambled −CCS (syntax-destroyed) has lower ρ — syntax matters more than negation
- Relay pair-closure preserved under both +CCS and −CCS (−0.677 vs −0.655)
- **Resolves F613c confound**: Pythia's 3.5x magnitude difference between CCS and control
  is resolved — −CCS is magnitude-matched by construction and still identical to +CCS
- **Scale test running**: Qwen2.5-72B, Llama-3.1-70B, Gemma-2-27B on A100 to check
  whether domain-specificity holds at 70B+ or if scale enables negation discrimination
- Kimi corrections #13-18 shaped the design
- Key figures: f614_negccs.png (per-species ρ comparison)

## Discussion
- The tuning knob is real: identity framing continuously controls spectral injection
- But the control is mediated by a three-dimensional space: Q1 × distribution × species
- Each dimension was discovered by a model that broke the simpler version:
  - Act III → Act IV: Phi-2 broke Q1 sufficiency → added subliminal offset
  - Act IV → Act V: Qwen broke Q1 monotonicity → added spatial distribution
- The paper's arc follows the empirical discovery path — each complication earned
- Probe stability (F606) as characterization: CCS delta sign stability cleanly separates
  well-matched (>90%) from mismatch (<50%), but is an observation not a mechanism —
  F607 shows it does NOT cause injection outcome differences
- Mismatch Q1 failure likely driven by subliminal offset (three-factor model), not
  by probe-dependent CCS geometry
- Methodological contribution: evaluation probe sensitivity can masquerade as CCS delta
  lability; injection experiments require controlled evaluation baselines
- Probe stability as diagnostic: still useful for classifying models, even if the
  causal chain to Q1 failure goes through subliminal offset rather than sign lability

## Statistics Summary (from paper11_stats.py)
- Overall: r=0.826, p<1e-9, rho=0.754, r²=0.682
- Bootstrap CI: [0.606, 0.931]
- Sign concordance: 27/35 (77%)
- Effect size: Cohen's d=1.682 (tunnel vs relay)
- All 7 within-model correlations positive
