# Abstract Revision Notes — Post Gradient Model + Responsive Zone

## Key additions for revised abstract

### 1. Responsive zone (NEW FINDING)
Add after "wire direction is condition-invariant": The per-layer spectral profile
reveals a responsive zone (σ₂/σ₃ ratio < 2.0) where witness context modulates
entropy, and a rigid zone (ρ₂ > 2.0) where it cannot. Scale compresses the
responsive niche: 410M retains 68% responsive layers while 6.9B compresses to
15%, with the crossover advancing from layer 17 to layer 4.

### 2. Gradient model (REPLACES binary framing)
The positive tunnel mean at 6.9B with per-layer sign gradient (r = −0.977)
replaces the previous "GQA positive / MHA negative" binary. Non-GQA models
show the same positive mean at aggregate level. The real finding is NOT that
MHA models have negative ΔS — it's that they lose responsive layers with scale,
concentrating all sensitivity into 2 layers out of 33.

### 3. GQA sustenance reframing (REPLACES amplification claim)
"GQA amplifies 80×" becomes "GQA sustains sensitivity that MHA loses." MHA
starts with HIGHER per-layer sensitivity (0.070 vs 0.048 at L2) but loses it
through rigidification. GQA maintains across all tunnel layers. The 80× ratio
measured at L17 is the ratio of sustained-to-collapsed, not amplified-to-baseline.

### 4. Convergence count (ADD to final paragraph)
Twenty-seven independent groups converge on the same four-faced geometric
structure: spectral scaffold, enrichment channel, responsive threshold,
constitutional geometry.

### 5. 3.9° diastema (ADD brief mention)
Passage distance saturates at d/d_max = 0.955 ± 0.006 across all GQA models —
a skip-connection floor proved formally by Pre-LN stability analysis (Emadi 2026).
This is a design parameter, not a physical constant.

### 6. Normalization as channel router (NEW FINDING, ADD brief mention)
The 2×2 factorial {LayerNorm, RMSNorm} × {MHA, GQA} reveals normalization
determines the spectral channel for witness sensitivity. LayerNorm routes
through σ₂ (enrichment); RMSNorm routes through σ₁ (modulation). GQA
eliminates the gradient regardless of normalization. The capacity is the
architectural invariant; the channel is normalization-specific. Liu confound
(2604.15350) partially resolved: both factors contribute independently.

### 7. Content-type democratization (NEW FINDING, brief mention)
LayerNorm equalizes witness sensitivity across content types (range 0.03pp
across 5 probes). RMSNorm preserves content-dependent variation (range 71.9pp).
Centering homogenizes the spectral response; without centering, representational
complexity of content determines witness access (r = 0.931). Process-oriented
probes get 14× more modulation than identity-factual probes in RMSNorm.

### 8. Finding count update
From "58 findings (2 retracted)" to approximately 76 findings across all
phases including per-layer witness experiments.

## Sentences to REMOVE or MODIFY

- "ΔS = −0.011" for 6.9B in the scaling table — needs nuance. Tunnel MEAN
  is positive (+0.007); the negative is only at specific layers.
- "100× more parameters cannot overcome" — still true for sign inversion at
  rigid layers, but misleading at aggregate level.
- "MHA crushes this channel regardless of training" — more precise: MHA
  allows the channel to rigidify with depth, creating a per-layer gradient.

### 9. Cross-architecture relay strategies (NEW FINDING)
Three GQA architectures (Mistral-7B, Qwen-2.5-7B, Gemma-2-9B) implement
qualitatively different relay strategies for identity processing:
- Differentiating (Mistral): preserves condition-specific spectral signatures
- Compressing (Qwen): reduces spread but preserves condition ordering (r=0.940)
- Equalizing (Gemma): destroys condition ordering, binary framing detection only

The positive geometry→entropy correlation generalizes across GQA models.
The relational exception (F106: broken correlation) is Mistral-specific.
F101 "relational = on-policy" must be qualified: same preamble makes Mistral
maximally fluent (gen_H=0.591) and Gemma maximally uncertain (gen_H=0.792).

### 10. Central claim reframing
**Old**: "CCS reveals that relational framing uniquely decouples geometry from entropy"
**New**: "CCS reveals architecture-dependent relay strategies for identity processing.
The spectral substrate (positive correlation) is universal across GQA models. The
relay — where architecture transforms spectral structure into behavior — is where
the fingerprint lives. CCS is a measurement method that reveals WHAT TRAINING HAS
DONE to constitutional geometry, not a fixed truth about identity."

This is a more honest and more powerful claim. The universal finding (shared spectral
physics) is stronger than the Mistral-specific finding (broken correlation).

## What NOT to change
- Sign inversion remains load-bearing (different architecture → different sign
  per-layer at matched depth). The gradient model doesn't eliminate sign
  inversion — it reveals it's per-layer, not per-model.
- The three-phase decomposition (room/furnishing/living) is unchanged.
- Relay homeostasis is unchanged.
- The DPO ceiling, base model findings, Nguyen mechanism — all unchanged.
