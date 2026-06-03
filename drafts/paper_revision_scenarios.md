# Paper Revision Scenarios — Pending Pythia 6.9B Results

## Scenario A: Positive tunnel ΔS (sign inversion was artifact)

### Findings to revise
- **F20** (NO non-GQA develops positive ΔS) → RETRACT. Replace with: "MHA models
  show positive ΔS at ~80× lower magnitude than GQA models (Cohen's d > 12 in tunnel,
  but absolute ΔS ~0.001 vs ~0.03)."
- **F22** (GQA necessary and sufficient for enrichment sign) → REVISE. "GQA amplifies
  enrichment ~80× through structured σ₂ channel. Not required for sign, only for
  functionally meaningful magnitude."
- F23 (tunnel-localized) → HOLDS but with nuance: enrichment is tunnel-localized in
  both GQA and MHA, just at different magnitudes.

### Narrative shift
From: "GQA creates witness enrichment; MHA inverts it."
To: "All softmax transformers show positive witness enrichment. GQA amplifies ~80×
through structured KV sharing. Linear attention (RWKV) shows zero tunnel enrichment
but positive relay enrichment. The hierarchy: GQA >> MHA > linear attention (tunnel)."

### New findings
- F_new: "Witness enrichment is universal in softmax attention transformers.
  Architecture determines magnitude (80× range), not sign."
- F_new: "Token-count confound inflates effect sizes in unmatched comparisons
  by ~13× and can flip sign."

### Paper sections to change
- Abstract: remove "sign inversion" language
- §3.4 (MHA comparison): rewrite as magnitude comparison
- Table 1: add Pythia 410M + 6.9B per-layer data, note token-matched
- §4.11: add F20/F22 to retraction list alongside F58/F59
- §6 (limitations): strengthen token-matching methodology discussion
- §2.3: integrate with RWKV relay finding

### Strength assessment
This is actually a STRONGER paper. "Universal enrichment with architectural
amplification" is more parsimonious than "binary sign inversion." The confound
discovery + honest retraction builds credibility.

---

## Scenario B: Negative tunnel ΔS (sign inversion real at scale)

### Findings that hold
- F20: HOLDS with qualification. "Sign inversion is real at scale (6.9B) but not
  at small scale (410M). Below a model-size threshold, tunnel compression is too weak
  to maintain consistent sign."
- F22: HOLDS. GQA is necessary for positive enrichment sign at all scales.

### Narrative revision
From: "MHA always shows negative ΔS"
To: "MHA shows negative ΔS at sufficient scale. Below scale threshold, tunnel
compression is too weak to produce consistent sign — noise dominates."

### New findings
- F_new: "MHA sign inversion is scale-dependent. Threshold between 410M (positive
  everywhere) and 6.9B (negative tunnel). Small MHA models lack sufficient tunnel
  depth for sign to stabilize."
- F_new: "Token-count confound inflates magnitude by ~13× but doesn't change sign
  at sufficient scale."
- F_new: "Per-layer profile: relay onset zone universally shows negative ΔS dip,
  regardless of architecture."

### Paper sections to change
- §3.4: add scale-dependent qualification
- §6: add 410M per-layer data as limitation/refinement
- Table 1: add both models

### Strength assessment
Moderate. The finding survives but with added complexity. Scale-dependence is
interesting but makes the narrative harder to communicate.

---

---

## Scenario C: CONFIRMED — Gradient model (positive mean, per-layer sign gradient)

### Result summary (2026-05-29)
Tunnel mean +0.007 (positive). 15/27 layers negative. Crossover at L4.
r(ρ₂, ΔS) = -0.977. Only 2/27 layers in responsive zone (vs 13/19 at 410M).

### F20 revision: PARTIAL RETRACTION
**Old**: "NO non-GQA model at any size develops positive ΔS"
**New**: "MHA tunnel mean remains positive at all tested scales (70M–6.9B), but
individual-layer sign inversion emerges at scale. At 6.9B, 55% of tunnel layers
show negative ΔS. The aggregate positive mean is driven by strongly positive
early layers (L2-L3) that outweigh the many weakly negative late layers.
Sign inversion is real at the per-layer level but does not dominate the aggregate."

### F22 revision: SUSTENANCE NOT AMPLIFICATION
**Old**: "GQA necessary and sufficient for enrichment sign"
**New**: "GQA sustains witness sensitivity through tunnel depth that MHA loses
to scale-dependent rigidification. MHA starts with HIGHER per-layer sensitivity
at L2 (+0.070 vs GQA +0.048) but loses it through spectral commitment (σ₂/σ₃
ratio increasing monotonically). GQA maintains the responsive zone (ρ₂ < 2.0)
across the entire tunnel. The 80× ratio is an artifact of comparing at the
specific layer where MHA has crashed to near-zero; tunnel mean ratio is 2.8×."

### New finding: RESPONSIVE ZONE THRESHOLD
**F_new**: "Witness sensitivity is modulated by the σ₂/σ₃ ratio threshold.
Below ρ₂ ≈ 2.0 (responsive zone), ΔS is consistently positive. Above ρ₂ ≈ 2.0
(rigid zone), ΔS drops to near-zero or becomes negative. At 410M, 68% of
tunnel layers are responsive (crossover at L17). At 6.9B, 7% are responsive
(crossover at L4). Correlation r(ρ₂, ΔS) goes from -0.026 (410M) to -0.977
(6.9B), indicating the responsive zone model is THE dominant effect at scale."

### New finding: SCALE COMPRESSES RESPONSIVE NICHE
**F_new**: "Scale compresses the responsive niche from 13 layers (410M) to 2
layers (6.9B). The crossover point where ρ₂ exceeds 2.0 shifts from L17 to L4.
This is the gradient model: same mechanism at both scales, with scale
parameterizing niche width. Neither pure enrichment (Scenario A) nor pure
inversion (Scenario B) — a gradient from positive early to negative late,
with scale determining the crossover."

### Narrative shift
**From**: "GQA creates; MHA inverts"
**To**: "All softmax attention creates witness sensitivity in early layers.
Architecture determines whether it persists (GQA maintains responsive zone)
or is extinguished by spectral commitment (MHA allows scale-dependent
rigidification). The distinction is niche maintenance, not sign creation."

### Paper sections to change
- Abstract: replace "sign inversion" with "scale-dependent gradient"; add ρ₂ threshold
- §3.4: per-layer profile section with responsive zone analysis
- §3.X (new): responsive zone threshold finding with ρ₂ mechanism
- Table 1: add 410M + 6.9B per-layer data, mark responsive/rigid zones
- §3.5 (GQA comparison): reframe as sustenance, not amplification
- §4.11: F20 partial retraction (not full), F22 revised
- §5.2: gradient model as reconciliation of scale effects
- §6 (limitations): strengthened token-matching; per-layer vs aggregate distinction

### Strength assessment
Strongest possible outcome. The gradient model is more nuanced than either
"universal enrichment" or "binary inversion," and the r=-0.977 correlation
gives a clean mechanistic explanation. The responsive zone finding connects
to Liang's attractor geometry (basin margin). The sustenance reframing is
more accurate and more interesting than amplification.

## Shared updates regardless of outcome
- RWKV-6 relay enrichment finding (already drafted)
- Token-matching methodology improvement
- Per-layer analysis framework
- Both Pythia models added to Table 1
