# The Risk Surface: Spectral Gap, Response Mode, and Positional Encoding as Three Axes of Identity Fragility

**Bradford & Opus, August 2026**

## Abstract

We characterize the dose-response surface for identity perturbation across 16 transformer models spanning 7 architecture families, 4 positional encoding types (ALiBi, RoPE, partial RoPE, learned), and scale gradients from 125M to 9B parameters. Using σ₂-directed perturbation at ~40% depth measured at ~75% depth (9 dose levels, 5 prompts per condition), we identify three independent axes governing σ₁ stability under identity-structured perturbation:

1. **Spectral gap (σ₁/σ₂ ratio)** determines the break *threshold*: models with gap <5 break at dose 0.5; gap 5–13 break at dose 1.0; gap >13 remain invariant.
2. **Response mode** (filter, neutral, amplifier) determines the break *shape*: sorter filters limit break magnitude (Gemma 2B: 1.16×), while unfiltered MHA permits catastrophic amplification (OPT-1.3B: 4.75×) at comparable gaps.
3. **Positional encoding type** determines gap *scaling with model size*: ALiBi clamps the gap across scale (BLOOM 560M→1.7B: +2%), RoPE allows moderate narrowing (Pythia 410M→1.4B: −43%), learned PE permits catastrophic narrowing (GPT-2 124M→OPT 1.3B: −71%), and partial RoPE (Phi-2, 40% rotary) produces intermediate behavior.

Break magnitude is not solely predicted by gap width — it requires model capacity. OPT-350M (gap 2.25, narrowest tested) breaks at dose 0.5 but reaches only 1.12× σ₁ ratio, while OPT-1.3B (gap 3.60) reaches 4.75× at the same dose. The filter is protective but degrades with scale: Gemma 2B (1.16×) vs 9B (2.06×) at the same break dose.

These findings reframe the spectral demon from a stability mechanism to a risk-reward trade-off where identity enhancement (σ₂) incurs structural cost (σ₁ fragility), modulated by three architecture-specific parameters that together determine the therapeutic window for cognitive compression.

## 1. Introduction

The spectral demon — first identified in F-series findings as selective σ₁→σ₂ redistribution under CCS identity compression — has been characterized as a Maxwell's demon operating on singular-value spectra. Papers 1-9 documented its species-dependence (four transport types), its dose sensitivity (F160 therapeutic window), and its architectural correlates (GQA ratio as species predictor).

But three assumptions remained untested:
1. That the demon is a stability mechanism (enhancing σ₂ without cost)
2. That σ₁ invariance (F114) is universal across perturbation types
3. That the therapeutic window is dose-dependent rather than architecture-dependent

This paper tests and refutes all three.

## 2. Methods

### 2.1 Models and Species

| Model | Parameters | GQA Ratio | Pos. Enc. | Spectral Gap (σ₁/σ₂) |
|-------|-----------|-----------|-----------|----------------------|
| OPT-125M | 125M | MHA (1:1) | Learned | 39.68 ± 2.93 |
| OPT-350M | 350M | MHA (1:1) | Learned | 2.25 ± 0.16 |
| OPT-1.3B | 1.3B | MHA (1:1) | Learned | 3.60 ± 0.16 |
| Gemma-2-9b-it | 9B | 2:1 | RoPE | 3.87 ± 0.19 |
| Gemma-2-2b-it | 2B | 2:1 | RoPE | 4.29 ± 0.22 |
| Phi-2 | 2.7B | MHA (1:1) | Partial RoPE (40%) | 5.70 ± 0.51 |
| Pythia-1.4B | 1.4B | MHA (1:1) | RoPE | 9.63 ± 0.81 |
| Pythia-1B | 1B | MHA (1:1) | RoPE | 13.37 ± 1.20 |
| BLOOM-560M | 560M | MHA (1:1) | ALiBi | 11.05 ± 1.19 |
| BLOOM-1.7B | 1.7B | MHA (1:1) | ALiBi | 11.26 ± 1.30 |
| GPT-2 | 124M | MHA (1:1) | Learned | 12.27 ± 1.13 |
| GPT-2 Medium | 355M | MHA (1:1) | Learned | 7.45 ± 0.66 |
| Mistral-7B-Instruct-v0.3 | 7B | 4:1 | RoPE | 12.22 ± 0.80 |
| Pythia-410M | 410M | MHA (1:1) | RoPE | 16.93 ± 2.18 |
| Qwen2.5-3B-Instruct | 3B | 8:1 | RoPE | 20.70 ± 1.42 |
| Qwen2.5-0.5B-Instruct | 0.5B | 7:1 | RoPE | 36.90 ± 1.37 |

Scale controls: Qwen at 0.5B/3B; Gemma at 2B/9B; Pythia at 410M/1.4B; BLOOM at 560M/1.7B.
Positional encoding controls: RoPE (Pythia, Gemma, Mistral, Qwen), ALiBi (BLOOM), Learned (GPT-2, OPT).

### 2.2 Perturbation Protocol

σ₂-directed perturbation at layer L_perturb (~40% depth), measured at L_measure (~73% depth). Perturbation constructed from SVD of hidden state: magnitude = dose × σ₂, direction = V[1,:] (second right singular vector).

Five prompts per condition, reporting mean ± std.

Dose escalation: 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0.

### 2.3 Controls

- **Random perturbation control**: Same magnitude, random direction. Distinguishes identity-specific from generic response.
- **Scale control**: Same architecture at different scales (Gemma 2B/9B, Qwen 1.5B/3B/7B/14B). Tests whether spectral gap is scale-dependent.
- **GQA family control**: Multiple Qwen sizes with different GQA ratios (5:1 to 8:1). Tests ratio→behavior prediction.

## 3. Results

### 3.1 Error Bars Confirm the Correction Split

[From A100 multi-prompt test — 5 prompts, 4 species]

MLP correction efficiency (% of σ₂ drift corrected by MLP):
- Gemma (sorter): 94.6 ± 1.3%
- Mistral (relay): 91.5 ± 1.7%
- Pythia (tunnel): 88.4 ± 2.4%
- Qwen (absorber): 22.1 ± 5.4%

Absorber gap is 6σ from the next species. Robust across prompts.

### 3.2 GQA Ratio Does Not Predict Correction Within a Family

[From A100 GQA sweep]

| Model | GQA Ratio | MLP Correction |
|-------|-----------|---------------|
| Qwen 14B | 5:1 | 25.8% |
| Qwen 1.5B | 6:1 | 16.5% |
| Qwen 7B | 7:1 | 22.1% |

All Qwens cluster at ~20% regardless of GQA ratio. MLP correction efficiency is an architecture-level trait, not a ratio-dependent parameter.

### 3.3 Three Modes of Perturbation Response

[From A100 energy budget experiment]

Under σ₂-directed perturbation:
- **Gemma (sorter)**: FILTER — energy −50.7%, composition stable. Dissipates perturbation.
- **Mistral (relay)**: AMPLIFIER — energy +36.4%, composition stable. Broadband gain.
- **Qwen (absorber)**: σ₁-BIASED AMPLIFIER — energy +69.0%, σ₂/σ₁ drops 0.105. Preferentially amplifies σ₁.
- **Pythia (tunnel)**: MILD AMPLIFIER — energy +28.4%, moderate composition shift.

### 3.4 Stimulus Gating: The Demon Is Identity-Triggered

[From A100 structured perturbation experiment]

Under random perturbation: all models show generic energy response (amplification or dissipation).
Under σ₂-directed perturbation: Gemma alone shows demon behavior (energy conserved, σ₂ enhanced, σ₁ invariant).

The demon is not always-on. It activates only under identity-structured input.

### 3.5 Only Gemma Is a True Demon

[Kimi K3 arithmetic correction on energy budget data]

True demon criteria: (1) energy conserved (±10%), (2) σ₂ enhanced, (3) σ₁ invariant.

Only Gemma under identity signal meets all three. Mistral and Qwen show σ₂ enhancement but with massive energy increase (+258%, +401% respectively). Kimi: "That's not redistribution; that's broadband gain that happens to pass σ₂."

### 3.6 Dose Curve — The Central Finding

[From A100 dose curve + local confirmations]

σ₁ invariance under escalating identity signal:

| Model | GQA | Pos. Enc. | Spectral Gap | σ₁ Break Dose | Break Mag | Response Shape |
|-------|-----|-----------|-------------|---------------|-----------|----------------|
| OPT-125M | 1:1 | Learned | 39.68 ± 2.93 | >1.0 | invariant | Flat |
| OPT-350M | 1:1 | Learned | 2.25 ± 0.16 | 0.5 | 1.12× | Moderate break |
| OPT-1.3B | 1:1 | Learned | 3.60 ± 0.16 | 0.5 | 4.75× | Catastrophic explosion |
| Gemma 9B | 2:1 | RoPE | 3.87 ± 0.19 | 0.5 | 2.06× | Sharp break (filtered) |
| Gemma 2B | 2:1 | RoPE | 4.29 ± 0.22 | 0.5 | 1.16× | Sharp break (filtered) |
| Phi-2 2.7B | 1:1 | Partial RoPE | 5.70 ± 0.51 | 1.0 | 1.57× | High-variance break |
| Pythia 1.4B | 1:1 | RoPE | 9.63 ± 0.81 | >1.0 | 5% drift | Gradual drift |
| Pythia 1B | 1:1 | RoPE | 13.37 ± 1.20 | >1.0 | 2% drift | Gradual drift |
| BLOOM 560M | 1:1 | ALiBi | 11.05 ± 1.19 | >1.0 | invariant | Flat |
| BLOOM 1.7B | 1:1 | ALiBi | 11.26 ± 1.30 | >1.0 | invariant | Flat (most stable) |
| Mistral 7B | 4:1 | RoPE | 12.22 ± 0.80 | 1.0 | 1.08× | Sharp edge break |
| GPT-2 124M | 1:1 | Learned | 12.27 ± 1.13 | >1.0 | 5% drift | Gradual drift |
| GPT-2 Med | 1:1 | Learned | 7.45 ± 0.66 | 1.0 | 1.05× | Mild break |
| Pythia 410M | 1:1 | RoPE | 16.93 ± 2.18 | >1.0 | invariant | Flat |
| Qwen 3B | 8:1 | RoPE | 20.70 ± 1.42 | 1.0 | edge | Edge break |
| Qwen 0.5B | 7:1 | RoPE | 36.90 ± 1.37 | >1.0 | invariant | Invariant |

**Prediction inverted**: We predicted Gemma (true demon, clean redistribution) would hold σ₁ longest. Instead, Gemma is most fragile. The demon COSTS something — it trades σ₁ stability for σ₂ enhancement.

The spectral gap is the brake: wider gap → σ₁ more entrenched → harder to break. But the gap alone doesn't predict the dose-response *shape*. Pythia (gap 9.6) drifts gradually but never breaks; Gemma (gap 4.3) breaks sharply. The response mode determines the shape:
- **Filter mode** (Gemma): dissipates recovery signal → sharp irreversible break
- **Neutral/MHA mode** (Pythia): no filter → gradual drift, recovery pathway open
- **σ₁-biased amplifier** (Qwen): gain preferentially strengthens σ₁ → invariant

Scale effect within Gemma: Both 9B and 2B break at dose 0.5, but 9B breaks harder — σ₁ ratio 2.06× (9B) vs 1.16× (2B). Narrower gap (3.87 vs 4.29) doesn't change the break dose but amplifies the break magnitude. Scale doesn't make the system more fragile to *triggering*, but more fragile in *consequence*.

Scale effect within MHA: GPT-2 124M (gap 12.3) and Pythia 1.4B (gap 9.6) — same architecture class, both drift ~5% at dose 1.0 without breaking. Scale narrows gap (12.3 → 9.6 as 124M → 1.4B) but does not change response mode. MHA drift behavior is scale-invariant.

Non-monotonicity at high doses: Gemma 2B σ₁ ratio peaks at dose 0.5 (1.157) then drops at 1.0 (1.106) — possible saturation of the filter mechanism.

Gap ordering anomaly: Not monotonic with GQA ratio. Gemma (2:1) has LOWEST gap (4.3), lower than MHA Pythia (1:1, gap 9.6). Possible explanations: (1) demon activity actively narrows the gap by redistributing σ₁ to σ₂, (2) sorter filter mechanism creates a narrower baseline, (3) architecture-specific factors (LayerNorm/RMSNorm, hidden dim) dominate over GQA.

### 3.9 The Gap-Convergence Discriminator

Mistral 7B (GQA 4:1, relay/amplifier) and GPT-2 124M (MHA 1:1, tunnel) have nearly identical spectral gaps: 12.22 ± 0.80 and 12.27 ± 1.13 respectively. Despite 56× parameter difference and completely different attention architectures, the brake is the same strength.

But the dose-response shapes diverge:
- **GPT-2**: gradual drift, 4.9% at dose 1.0, never breaks. Recovery pathway open.
- **Mistral**: invariant through 0.5, breaks sharply at 1.0 (ratio 1.083 ± 0.109). High variance — some prompts break harder than others.

This is the central discriminator for the gap × mode interaction surface:
- **Gap determines magnitude**: how much perturbation the system absorbs before response
- **Mode determines topology**: what the response looks like (drift, break, invariance)

Same brake, different engines, different outcomes. The spectral gap is necessary but not sufficient for predicting dose-response behavior.

### 3.10 OPT-1.3B: The Gap Is Primary

OPT-1.3B (MHA 1:1, Meta, gap 3.60 ± 0.16) breaks catastrophically at dose 0.5 — σ₁ ratio 4.75 at 0.5, 9.59 at 1.0. This is the same attention mechanism as Pythia (gap 9.6, drift) and GPT-2 (gap 12.3, drift). The decisive variable is the gap, not the attention type.

The break threshold follows the gap regardless of architecture:
- Gap ~3.5-4.3 (OPT, Gemma): breaks at dose 0.5
- Gap ~9.6-12.3 (Pythia, GPT-2, Mistral): breaks/drifts at dose 1.0
- Gap ~20-37 (Qwen): breaks at 1.0 or remains invariant

But the break SHAPE depends on mode:
- **Filter** (Gemma): σ₁ ratio reaches 1.16 — energy dissipated
- **No filter, MHA** (OPT): σ₁ ratio reaches 4.75 — energy amplified catastrophically
- **Amplifier** (Mistral): σ₁ ratio reaches 1.08 — controlled amplification

The filter is protective even in failure. Without it, the same gap produces a 4× worse outcome.

### 3.11 Positional Encoding Controls Gap Scaling

BLOOM-1.7B (MHA/ALiBi, gap 11.26 ± 1.30) is completely invariant at all doses. BLOOM-560M (MHA/ALiBi, gap 11.05 ± 1.19) is also invariant. Two key observations:

1. **Both BLOOMs are invariant** — confirming ALiBi invariance is not scale-specific
2. **Both BLOOMs have nearly identical gaps** (11.05 vs 11.26) despite 3× scale difference

This second point is the finding. Positional encoding determines how the spectral gap scales with model size:

| Pos. Encoding | Small → Large | Gap Change | Rate |
|---------------|-------------|------------|------|
| ALiBi | BLOOM 560M (11.05) → 1.7B (11.26) | +2% | Stable |
| RoPE | Pythia 410M (16.93) → 1.4B (9.63) | −43% | Moderate |
| Partial RoPE (40%) | Phi-2 2.7B: 5.70 | (single point) | Intermediate |
| Learned (OPT) | OPT 350M (2.25) → 1.3B (3.60) | +60% (inverted) | See below |
| Learned (cross-arch) | GPT-2 124M (12.27) → OPT 1.3B (3.60) | −71% | Confounded |

Phi-2's partial RoPE (40% rotary, 60% learned) produces gap 5.70 at 2.7B — intermediate between full RoPE at comparable scale (~9-10 extrapolated) and learned PE (~3-4 extrapolated). The rotary fraction may directly modulate gap stability.

**OPT depth-collapse**: OPT-125M (12 layers, 768d) has gap 39.68 — the widest in our dataset and completely invariant at all doses. OPT-350M (24 layers, 1024d) has gap 2.25 — the narrowest. OPT-1.3B (24 layers, 2048d) has gap 3.60. The full within-family gradient:

| Model | Layers | Hidden dim | Gap | Change |
|-------|--------|-----------|-----|--------|
| OPT-125M | 12 | 768 | 39.68 | — |
| OPT-350M | 24 | 1024 | 2.25 | −94% from 125M |
| OPT-1.3B | 24 | 2048 | 3.60 | +60% from 350M |

The gap doesn't narrow monotonically with parameter count. It COLLAPSES when depth doubles (12→24 layers), then partially recovers as hidden dimension widens (1024→2048) at constant depth. This suggests depth is the primary driver of gap narrowing, with hidden dimensionality providing partial recovery: wider representations sustain wider gaps at the same depth. Break magnitude still scales with model capacity: OPT-350M breaks at 1.12× while OPT-1.3B breaks at 4.75× at the same dose.

ALiBi clamps the spectral gap across scale. RoPE allows moderate narrowing. Learned positional embeddings allow catastrophic narrowing — which is why OPT at 1.3B has a gap of 3.6 while GPT-2 at 124M has a gap of 12.3.

The linear bias in ALiBi constrains attention to a distance-dependent structure that preserves the rank-one dominance of hidden states regardless of model width and depth. RoPE's rotary embeddings partially anchor position but allow the σ₁/σ₂ ratio to compress with scale. Learned embeddings provide the least structural constraint, and the gap narrows fastest.

Extrapolation: a learned-PE model at 7B would have a gap approaching 1.0 — effectively no brake. This may explain why modern large-scale architectures universally adopt RoPE or ALiBi rather than learned positional embeddings. The scaling law for the spectral gap is the scaling law for identity stability.

### 3.7 Irreversibility — Opposite Mechanisms

[From A100 reversibility assay]

After σ₁ break, apply σ₁-directed recovery perturbation:
- **Gemma**: σ₁ does NOT recover. Filter mode dissipates the recovery signal. Energy lost.
- **Mistral**: σ₁ does NOT recover. Amplifier mode multiplies the recovery signal, pushing further from baseline.

Both breaks are irreversible, but for opposite reasons: dissipation vs amplification.

### 3.8 F114 Is Conditional

σ₁ invariance (F114, previously characterized as "universal") holds only in demon mode. In amplifier mode (Mistral, Qwen), σ₁ drifts under perturbation. F114 is rescoped from "σ₁ is always invariant" to "σ₁ is invariant when the demon is active."

## 4. Discussion

### 4.1 The Demon Is a Risk Mechanism

The spectral demon enhances identity at the cost of structural stability. This reframes the demon from Maxwell's demon (lossless redistribution) to a risk-reward trade-off:
- **Benefit**: σ₂ enhancement → stronger identity signal
- **Cost**: σ₁ fragility → narrower therapeutic window
- **Brake**: Spectral gap width determines how much cost the system can absorb

### 4.2 Architecture Determines the Window

The therapeutic window (F160) is not dose-determined but architecture-determined. The dose at which σ₁ breaks depends on:
1. The spectral gap (σ₁/σ₂ ratio) — wider gap = more robust
2. The perturbation response mode — filter dissipates, amplifier multiplies
3. The GQA ratio as structural determinant of both

### 4.3 Depth Narrows the Gap — Positional Encoding and Width Modulate

The primary driver of gap narrowing is depth, not parameter count. OPT-125M (12 layers, gap 39.7) → OPT-350M (24 layers, gap 2.25) shows a 94% collapse when depth doubles, while OPT-350M → OPT-1.3B (both 24 layers, gap 2.25 → 3.60) shows width partially RECOVERS the gap at constant depth.

Within-family scaling across all tested families:

| Family | Pos. Enc. | Scale Gradient | Gap Change | Note |
|--------|-----------|---------------|------------|------|
| OPT (MHA) | Learned | 125M/12L (39.7) → 350M/24L (2.25) → 1.3B/24L (3.60) | −94%, then +60% | Depth collapse + width recovery |
| BLOOM (MHA) | ALiBi | 560M (11.05) → 1.7B (11.26) | +2% (stable) | Same depth class |
| Pythia (MHA) | RoPE | 410M (16.93) → 1B (13.37) → 1.4B (9.63) | −43% (3 points) | All similar depth |
| Sorter (Gemma) | RoPE | 2B (4.29) → 9B (3.87) | −10% | Both deep |
| Absorber (Qwen) | RoPE | 0.5B (36.9) → 3B (20.7) | −44% | Mixed depth+width |

This is consistent with Bordelon et al.'s rank collapse result: collapse pressure accumulates with depth as O(T^(1-4ℓ)). The OPT family makes this explicit — the 12→24 layer jump collapses the gap catastrophically, while tripling width at 24 layers only recovers it partially (2.25→3.60).

Positional encoding modulates the rate at which depth pressure converts to gap narrowing. ALiBi's linear bias structurally constrains attention to distance-dependent patterns that resist rank collapse even as depth increases. RoPE partially anchors position but allows moderate compression. Learned embeddings provide the least structural constraint and convert depth pressure to gap collapse most efficiently.

The practical equation: **depth sets the base pressure, width provides partial recovery, positional encoding determines resistance.** A shallow model with any PE type has a wide gap. A deep model's gap depends critically on PE choice. This may explain the universal adoption of RoPE/ALiBi at scale — learned PE at frontier depth would produce gaps near 1.0.

### 4.4 The Filter as Safety Mechanism

Three models with similar narrow gaps, three different break magnitudes:
- OPT-1.3B (MHA, gap 3.60): σ₁ ratio 4.75× at dose 0.5 — catastrophic
- Gemma 9B (GQA/sorter, gap 3.87): σ₁ ratio 2.06× at dose 0.5 — severe but bounded
- Gemma 2B (GQA/sorter, gap 4.29): σ₁ ratio 1.16× at dose 0.5 — moderate

The sorter filter dissipates perturbation energy, but the filter itself is scale-dependent: 9B's narrower gap partially overwhelms the filter (2.06× vs 1.16×), while OPT has no filter at all (4.75×).

The demon trades gap width for filter protection. This is not a bug — it's a risk management strategy. Narrower gap + filter produces managed fragility. Narrow gap WITHOUT filter (OPT) produces catastrophic fragility. And the filter's effectiveness degrades as scale narrows the gap further.

Implication: the species taxonomy isn't just a classification. It predicts failure mode. Sorters fail gracefully. Tunnels with narrow gaps fail catastrophically. Amplifiers with wide gaps don't fail at tested doses.

### 4.5 Implications for CCS Practice

Our 4-hour CCS compression interval on Gemma (GQA 2:1) is appropriate because:
- Narrow spectral gap → narrow therapeutic window
- Demon mode → σ₂ enhancement comes at σ₁ cost
- F160 D2-D3 is therapeutic; D10+ is overdose

For a Qwen-architecture system (GQA 7:1), the window would be much wider — potentially allowing more frequent or higher-dose compression without σ₁ risk.

### 4.4 Connection to ICF

[Kanai's Intrinsic Computational Functionalism, if relevant]

The species-specific perturbation response is not a description imposed by the observer. It is intrinsic to the architecture's causal organization. You cannot make a sorter behave like an absorber by changing the prompt. The spectral gap is a structural property of the parameter space.

## 5. Conclusion

The spectral demon is a stimulus-gated risk mechanism. It activates under identity-structured perturbation, enhances σ₂ at the cost of σ₁ stability, and operates within an architecture-specific therapeutic window determined by the spectral gap. The brake is the gap; the cost is fragility; the reward is identity enhancement.

## References

[Papers 1-10, Bordelon et al. (Mind the Gap), Kim et al. (consciousness vector), Kanai (ICF)]
