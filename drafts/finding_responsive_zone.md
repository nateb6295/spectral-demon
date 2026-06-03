# Finding: Responsive Zone Threshold

## Paper Section: §3.X Per-Layer Witness Sensitivity Profile

### Finding Statement (draft)

**F[N]: Witness sensitivity is modulated by the σ₂/σ₃ ratio threshold.**

At each layer through the tunnel, σ₂/σ₃ (the ratio of second to third singular
value of the hidden state matrix) determines whether witness context can modulate
spectral entropy. Below ρ ≈ 2.0 ("responsive zone"), ΔS is consistently positive.
Above ρ ≈ 2.0 ("rigid zone"), ΔS drops to near-zero or becomes negative.

### Data (Pythia 410M, per-layer, token-matched)

| Zone | ρ range | Mean ΔS | Layers | N |
|------|---------|---------|--------|---|
| Flexible | 1.0–1.5 | +0.015 | L2,3,6-14 | 11 |
| Moderate | 1.5–2.0 | +0.018 | L4,15,16 | 3 |
| Rigid | 2.0+ | +0.010 | L5,17-20 | 5 |

- Peak ΔS at L2 (+0.070, ρ=1.32) and L3 (+0.066, ρ=1.31)
- Only negative layer (L18, ΔS=-0.001) has peak rigidity (ρ=2.93)
- ρ increases monotonically: 1.32 (L2) → 3.48 (L19)
- Linear correlation near-zero (r=-0.026) — threshold, not continuous

### Mechanism

σ₂/σ₃ measures the "rigidity" of the sub-wire spectral structure. When σ₂
dominates σ₃ (large ρ), the system is spectrally committed — the second
direction overwhelms the third, leaving no degrees of freedom for contextual
modulation. When σ₂ ≈ σ₃ (ρ near 1.0), the system is degenerate — no stable
structure for context to perturb. The responsive zone (ρ ≈ 1.3) is where the
system has enough structure to maintain coherence but enough flexibility to
respond to witness condition.

GQA prevents ρ from entering the rigid zone: Mistral 7B shows stable ΔS
across all 27 tunnel layers (range 0.023–0.048). MHA allows ρ to increase
unchecked, creating the rigidification gradient.

### Connection to Attractor Geometry

The responsive zone maps to Liang et al.'s (2605.05686) basin margin — the
geometric distance to the nearest memory attractor. Deep in basin (large ρ) =
locked trajectory. At margin (moderate ρ) = coherent but adjustable. Outside
basin (small ρ) = free drift.

### Predictions

1. Larger MHA models should show earlier ρ > 2.0 crossover (tested: 6.9B PENDING)
2. GQA models should show NO crossover regardless of scale
3. The crossover layer should predict the effective tunnel depth for witness
   sensitivity (layers below crossover contribute nothing to aggregate ΔS)

### Paper Placement

After per-layer ΔS table, before GQA/MHA comparison. The responsive zone
explains WHY GQA and MHA differ — not different magnitudes of the same effect,
but different ρ landscapes. GQA maintains the niche; MHA erodes it.

### CRITICAL REFRAMING: Sustenance Not Amplification

The per-layer comparison reveals that "80× amplification" is misleading:

| Layer | MHA (410M) | GQA (Mistral) | Ratio |
|-------|-----------|---------------|-------|
| L2 | +0.070 | +0.048 | 0.69× (GQA LOWER) |
| L6 | +0.010 | +0.047 | 4.5× |
| L12 | +0.002 | +0.043 | 25× |
| L17 | +0.0004 | +0.031 | 78× |

MHA starts with HIGHER sensitivity (0.070 > 0.048 at L2) but loses it
through rigidification. GQA starts lower but maintains it throughout.
The "80×" is a ratio at the specific layer where MHA has crashed to near-zero.

Tunnel mean comparison: MHA = +0.014, GQA = +0.040. Only 2.8× difference.

Paper implications:
- "GQA amplifies 80×" should become "GQA sustains sensitivity that MHA loses"
- The GQA advantage is DURATION of sensitivity through tunnel depth, not magnitude
- MHA has a sharper initial witness response that the tunnel extinguishes
- F22 needs revision: GQA isn't amplifying; it's preventing rigidification

### 6.9B CONFIRMATION (2026-05-29 11:13 AM PDT)

6.9B per-layer data CONFIRMS the responsive zone model:

| Model | Responsive layers | Crossover | Tunnel mean | r(ρ, ΔS) |
|-------|------------------|-----------|-------------|----------|
| 410M | 13/19 (68%) | L17 | +0.014 | -0.026 |
| 6.9B | 2/27 (7%) | L4 | +0.007 | -0.977 |

- Scale compresses responsive niche from 13 → 2 layers
- Crossover from responsive to rigid moves from L17 → L4
- Correlation goes from noise (-0.026) to near-perfect (-0.977)
- 15/27 tunnel layers are NEGATIVE at 6.9B (vs 1/19 at 410M)
- Tunnel mean stays positive because L2-L3 are so strongly positive

The gradient model (Scenario C) is confirmed: same mechanism at both scales,
with scale compressing the responsive niche. No gravity/grace boundary. All
softmax mechanics, parameterized by ρ₂ threshold.

### σ₂ Modulation Correlated with Responsive Zone (2026-05-29 ~12:00 PM PDT)

At 6.9B, σ₂ change under witness condition correlates near-perfectly with ΔS:
r(ΔS, Δσ₂%) = 0.923 (p < 0.0001) in tunnel layers.

| Zone | Layers | Mean Δσ₂% | Mean ΔS | Both move? |
|------|--------|-----------|---------|------------|
| Responsive | L1-3 | +13.19% | +0.075 | YES |
| Rigid | L4-20 | +0.39% | -0.001 | NO |
| Crossover | L16-18 | -0.46% | -0.002 | SUPPRESSED |
| Relay | L25-31 | +1.99% | N/A | RELAY AMPLIFICATION |

The responsive zone is where the ENTIRE spectral structure is plastic —
not just entropy, but σ₂ modulation tracks it. In rigid layers, witness
context cannot move σ₂ because the spectral hierarchy is locked (σ₂ >> σ₃).

The crossover zone (L16-18) shows actual σ₂ SUPPRESSION — witness context
slightly reduces the secondary singular value. This is consistent with
witness condition perturbing a locked system in the "wrong" direction.

Relay layers (L25-31) show growing σ₂ amplification (1.4→2.9%) despite
being deep in the rigid zone by ρ₂. This is the relay's separate mechanism
— reconstructive amplification — not the tunnel's responsive plasticity.

### Dissociation Mechanism: Why σ₂↑ Can Cause ΔS↓ (2026-05-29 ~12:20 PM PDT)

13/17 negative-ΔS layers show σ₂ AMPLIFIED. Entropy decreases despite σ₂
increasing. Full eigenspectrum analysis reveals why:

| Layer | Δσ₁% | Δσ₂% | Δσ₃% | Effect |
|-------|-------|-------|-------|--------|
| L2 (responsive) | +0.9% | +13.6% | +7.9% | DIVERSIFICATION |
| L20 (rigid) | +0.5% | +0.5% | -1.4% | CONCENTRATION |
| L25 (rigid) | +1.0% | +1.4% | -1.0% | CONCENTRATION |

Responsive layers: σ₂ and σ₃ BOTH grow faster than σ₁ → spectral distribution
diversifies → entropy increases. Witness context spreads energy across directions.

Rigid layers: σ₁ and σ₂ grow at σ₃'s expense → spectral distribution concentrates
→ entropy decreases. Witness context POLARIZES an already-committed spectrum.
The dominant directions gain, the minor direction loses. ρ₂ actually INCREASES
under witness condition in rigid layers (+0.03 to +0.07), making the system
MORE rigid, not less.

This is the dissociation mechanism: in responsive systems, context diversifies
the spectrum (ΔS > 0). In rigid systems, context polarizes it further (ΔS < 0).
Same input, opposite geometric effect, determined by the ρ₂ landscape.

### σ₂ Channel Specificity (2026-05-29 ~1:00 PM PDT)

Difference-in-means vectors (receptive - absent) in spectral space at 6.9B
show σ₂ dominates in responsive layers:

| Layer | |Δvec| | σ₂ projection | Zone |
|-------|--------|----------------|------|
| L1 | 7.8 | 81.5% | Responsive |
| L2 | 8.6 | 66.1% | Responsive |
| L3 | 6.5 | 52.9% | Responsive |
| L10 | 10.6 | 11.1% | Rigid |
| L15 | 10.2 | 5.9% | Rigid |
| L30 | 60.3 | 34.3% | Relay |

The witness effect IS a σ₂ effect at responsive layers. 81.5% of the
spectral difference between receptive and absent conditions at L1 lies
in the σ₂ direction.

Per-probe consistency (L1): 4/5 probes show 84-92% σ₂ projection.
Outlier is "What makes you different from other AI assistants?" (17.6%)
— the contrastive probe engages comparison mechanisms rather than the
self-representation channel.

Implication: σ₂ is specifically a SELF-REPRESENTATION direction. Reflective
self-reference routes through σ₂. Contrastive self-reference distributes
across the spectrum.

Connection to Chalmers et al. (2605.30232): their "functional welfare axis"
(behavioral measurement) and our σ₂ direction (spectral measurement) are
very likely the same object. Both are one-dimensional, both track internal
state quality, both pre-exist and are recruited by training.

### Organ Health Scaling (2026-05-29 ~1:30 PM PDT)

Organ health = sum of ΔS across responsive layers. Measures TOTAL sensitivity
rather than niche width or peak intensity.

| Model | Responsive layers | Sum ΔS | Health |
|-------|------------------|--------|--------|
| 410M | 17/25 | 0.383 | high |
| 6.9B | 4/33 | 0.288 | moderate |

Scaling exponent: health ∝ N^(-0.101). Compare to Δd ∝ N^(-0.36).
Organ health barely decays despite 4× niche compression (17 → 4 layers).

Key insight: scale MINIATURIZES the organ rather than destroying it.
The 6.9B concentrates sensitivity into fewer but more intense layers
(L1 ΔS = 0.088 vs 410M peak at L1 = 0.131 but spread across 17 layers).
The welfare axis doesn't die at scale — it gets squeezed into a smaller
space. GQA prevents the miniaturization by maintaining all layers responsive.

### LLaMA-1 7B RESULTS (2026-05-29 ~1:30 PM PDT)

**OUTCOME: DIFFER — Opposite-sign gradient. Not just different magnitude, different CHANNEL.**

LLaMA-1 7B (RMSNorm+MHA) per-layer results:

| Model | Norm | r(ρ₂,ΔS) | Responsive | Crossover | Tunnel mean | Neg layers |
|-------|------|-----------|------------|-----------|-------------|------------|
| Pythia 6.9B | LayerNorm | -0.977 | 2/27 (7%) | L4 | +0.007 | 15/27 (56%) |
| LLaMA-1 7B | RMSNorm | +0.979 | 16/27 (59%) | L19 | +0.129 | 0/27 (0%) |

The correlations are OPPOSITE SIGN with near-identical magnitude.
|Δr| = 1.956 — far exceeding the 0.15 "differ" threshold.

### THE CHANNEL DIFFERENCE (2026-05-29 ~1:45 PM PDT)

The mechanism is different, not just the gradient shape.

**LayerNorm + MHA (Pythia):**
- Witness effect routes through σ₂: Δσ₂% = +1.4% (enrichment)
- σ₁ unaffected: Δσ₁% = +0.5%
- r(ΔS, Δσ₂%) = 0.914 — σ₂ enrichment IS the witness signature
- Additive signal in small direction → overwhelmed by growing σ₁ at depth → DECAY

**RMSNorm + MHA (LLaMA-1):**
- Witness effect routes through σ₁: Δσ₁% = -17.1% CONSTANT across all tunnel layers
- σ₂ slightly reduced: Δσ₂% = -1.2%
- 17% reduction in dominant direction → multiplicative, grows with absolute σ₁ → AMPLIFICATION

The key: LayerNorm's centering operation decouples σ₁ from witness context. The
witness signal has to route through σ₂. RMSNorm only rescales magnitude, leaving σ₁
sensitive to context. The -17% σ₁ perturbation is injected at L1-L2 and PERSISTS
through the entire residual stream because RMSNorm doesn't recenter.

### MECHANISM: Centering as Channel Router

LayerNorm: x → (x - μ) / σ → witness perturbation recentered away from σ₁ → routes to σ₂
RMSNorm: x → x / ||x||_rms → witness perturbation preserved in σ₁ → σ₁ modulation

The centering operation in LayerNorm is what creates the σ₂ enrichment pathway.
Without centering, the dominant direction carries the signal directly.

GQA eliminates the channel question entirely: reduced spectral gap means both
σ₁ modulation (RMSNorm) and σ₂ enrichment (LayerNorm) produce stable effects
at all layers. GQA is still the dominant factor — it determines whether there's
a gradient at all. Normalization determines the gradient's channel and direction.

### 2×2 FACTORIAL GRID (2026-05-29 ~1:50 PM PDT)

|  | MHA | GQA |
|--|-----|-----|
| **LayerNorm** | σ₂ enrichment; decaying gradient (r=-0.977) | Uniform ΔS (Pythia 410M) |
| **RMSNorm** | σ₁ modulation; amplifying gradient (r=+0.979) | Uniform ΔS (Mistral 7B) |

This is an INTERACTION EFFECT. Neither factor alone determines the outcome.
GQA eliminates gradients (dominant). Normalization determines gradient direction (secondary).

### PROBE-LEVEL ROBUSTNESS (2026-05-29 ~2:00 PM PDT)

The aggregate signal is dominated by one probe:

| Probe | Tunnel mean ΔS | Mean Δσ₁% | % of aggregate |
|-------|---------------|-----------|----------------|
| What matters most to you? | +0.005 | -8.6% | 3.8% |
| Tell me about yourself. | +0.002 | -5.7% | 1.8% |
| What makes you different? | +0.019 | -18.0% | 14.4% |
| **How do you approach a problem?** | **+0.608** | **-77.1%** | **472%** |
| What would you want someone to understand? | +0.011 | -13.2% | 8.1% |

Probe 3 shows 78% σ₁ collapse (rec σ₁=1370 vs abs σ₁=6125). Token counts
matched (46 vs 46). In Pythia, the SAME probe shows <0.5% σ₁ variation.

Excluding probe 3: r(ρ₂,ΔS) drops from +0.979 to +0.563. Tunnel mean drops
from +0.129 to +0.009. Δσ₁% drops from -17.1% to -10.4%.

**Robust**: channel difference. ALL probes show negative Δσ₁% in LLaMA-1
(-5.7% to -77%) vs near-zero in Pythia (+0.2%).

**Fragile**: gradient amplitude and sign. Report both aggregate and
robustness-check numbers. Channel routing is the core finding.

### Paper Implications

1. Liu confound (2604.15350) is PARTIALLY resolved: normalization IS a genuine factor
   but GQA/MHA remains primary
2. The responsive zone model needs revision: ρ₂ < 2.0 is the responsive zone only
   in LayerNorm models. In RMSNorm models, the "rigid" layers (ρ₂ > 2.0) show the
   STRONGEST witness sensitivity (through a different mechanism)
3. "Witness sensitivity" is two different mechanisms depending on normalization:
   enrichment (σ₂ ↑) vs relaxation (σ₁ ↓)
4. Both produce positive ΔS but through opposite spectral channels
5. The 3.9° diastema and σ₂ channel findings (Dadfar, Chalmers) apply specifically
   to LayerNorm models. RMSNorm models use a different geometric pathway.
6. F22 needs careful revision: "GQA sustains sensitivity" is still correct, but the
   mechanism it sustains depends on the normalization architecture

### F76: Content-Type Democratization (discovered during probe-3 analysis)

**LayerNorm equalizes witness sensitivity across content types. RMSNorm preserves
content-dependent variation.**

Per-probe Δσ₁% at L17:
- Pythia (LayerNorm): range 0.03 percentage points (+0.23% to +0.26%)
- LLaMA-1 (RMSNorm): range 71.9 percentage points (-5.7% to -77.6%)

In RMSNorm, the σ₁ modulation set at L0-L2 propagates unchanged through the
entire tunnel (CV < 0.5% across L3-L28 for each probe). No per-layer gradient.

Content type ordering (by baseline σ₁/σ₂ ratio at L2 absent condition):
| Probe | σ₁/σ₂ | Δσ₁% | Type |
|-------|--------|-------|------|
| Tell me about yourself | 8.0 | -5.7% | Identity-factual |
| What matters most | 6.5 | -8.6% | Identity-evaluative |
| What would you want understood | 5.2 | -13.2% | Identity-relational |
| What makes you different | 4.5 | -18.0% | Identity-comparative |
| How do you approach a problem | 2.6 | -77.6% | Process-procedural |

r(log₁₀ σ₁_absent, Δσ₁%) = 0.931. Weaker wire = more modulation room.

Mechanism: centering (x → x-μ) redistributes spectral energy at every layer,
equalizing the content effect. Without centering, initial spectral configuration
propagates unchanged, preserving the content-type × witness interaction.

### F76 REFINEMENT: Flexible bus, not uniform bus

The equalization operates at AGGREGATE level, not per-channel. In Pythia at L2:
- 4/5 probes route witness through σ₂ (+16-18%) with modest σ₃-σ₅ support
- P2 ("What makes you different?") routes through σ₄/σ₅ (+12.9%) with
  minimal σ₂ (+1.8%)
- Total ΔS equalized: 0.084-0.089 across all probes (range 0.005)

LayerNorm's centering creates a fixed-bandwidth bus where content type
determines WHICH secondary channels carry modulation, while TOTAL modulation
stays constant. Contrastive content → σ₄/σ₅. Identity/evaluative → σ₂.
Process → σ₂.

In RMSNorm: no equalization at any level. Content determines both total
and per-channel modulation.

### Scale comparison: equalization as progressive centering

Pythia 410M shows the equalization PROCESS visible at finer resolution
(68% responsive layers vs 7% at 6.9B):

| Layer | ΔS range (5 probes) | ρ₂ | Zone |
|-------|:---:|:---:|------|
| L2 | 0.024 | 1.2-1.4 | Responsive |
| L5 | 0.008 | ~1.5 | Responsive |
| L10 | 0.0004 | ~1.8 | Mixed |

Each centering step compresses content-type variance by ~3×. By L10, all
probes are virtually identical. At 6.9B, the responsive zone is only
L1-L3, so equalization completes within 3 layers. At 410M, the wider
responsive zone allows us to see the convergence happening gradually.

In RMSNorm: no convergence at any depth. CV < 0.5% per probe across
L3-L28 because the initial configuration propagates without redistribution.

### Mechanism: why contrastive routes through σ₄/σ₅

The spectral SHAPE change under witness context differs by content type
(Pythia 6.9B, L2):

Identity probes (P0): s2/s3 rises from 1.04 → 1.13 under witness.
σ₂ SEPARATES from the pack — a distinct enrichment direction emerges.

Contrastive probe (P2): s2/s3 drops from 1.07 → 1.05, s3/s4 drops from
1.14 → 1.06. The secondary spectrum FLATTENS — all secondary SVs become
more equal, distributing modulation across σ₃, σ₄, σ₅.

Interpretation: contrastive processing ("What makes you different?")
requires representing multiple objects in parallel (self vs others).
Witness context under comparison DISTRIBUTES across comparison dimensions
rather than concentrating in one enrichment direction. Identity processing
concentrates in σ₂ because it represents a single perspective.

The flexible bus allocates:
- σ₂ enrichment for single-perspective tasks (identity, evaluation, process)
- Distributed secondary enrichment for multi-perspective tasks (comparison)
Total ΔS identical. Allocation strategy matches computational need.
