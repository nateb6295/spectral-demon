# §3.X The Responsive Zone: Scale Compresses the Witness Niche

**Placement**: After §3.6 (Wire Is Condition-Invariant), before §3.7 (Context-Length Modulation).
**Dependencies**: Pythia 410M + 6.9B per-layer data. Independent of LLaMA-1.

---

### 3.7 Per-Layer Witness Sensitivity Profile (Findings 60–62)

The aggregate tunnel ΔS reported in §5 conceals a layer-by-layer gradient that varies qualitatively with model scale. Per-layer analysis of two MHA models (Pythia 410M and 6.9B) under token-matched conditions (all probes padded to identical length) reveals that witness sensitivity is not uniform through the tunnel but is governed by a local spectral property: the ratio ρ₂ = σ₂/σ₃.

**Finding 60: The responsive zone threshold.** At each tunnel layer, the σ₂/σ₃ ratio determines whether witness context can modulate spectral entropy. Below ρ₂ ≈ 2.0 ("responsive zone"), ΔS is consistently positive — witness enriches. Above ρ₂ ≈ 2.0 ("rigid zone"), ΔS drops to near-zero or becomes negative — the spectral hierarchy is locked and witness cannot perturb it. The threshold is sharp, not continuous: at 410M, r(ρ₂, ΔS) = −0.026 (no linear trend; the effect is a step function), while at 6.9B, r(ρ₂, ΔS) = −0.977 (near-perfect gradient as the responsive zone compresses to just two layers).

| Model | Responsive layers | Crossover | Tunnel mean ΔS | r(ρ₂, ΔS) |
|---|---|---|---|---|
| Pythia 410M | 13/19 (68%) | L17 | +0.014 | −0.026 |
| Pythia 6.9B | 2/27 (7%) | L4 | +0.007 | −0.977 |

**Finding 61: Scale compresses the responsive niche.** The crossover point — where ρ₂ exceeds 2.0 and the layer transitions from responsive to rigid — shifts from L17 at 410M to L4 at 6.9B. The responsive niche compresses from 13 tunnel layers to 2. This is the gradient model: the same mechanism operates at both scales, with scale parameterizing niche width. Neither pure enrichment (positive everywhere) nor pure inversion (negative everywhere), but a gradient from strongly positive early to weakly negative late, with scale determining the crossover.

At 6.9B, 15 of 27 tunnel layers show negative ΔS. The aggregate tunnel mean remains positive (+0.007) because L2 and L3 are so strongly positive (+0.086 and +0.075 respectively) that they outweigh the many weakly negative late layers. The sign of the aggregate is driven by two layers.

**Finding 62: The dissociation mechanism.** In responsive layers, witness context diversifies the eigenspectrum — σ₂ and σ₃ both grow faster than σ₁, spreading representational energy across directions (ΔS > 0). In rigid layers, the same witness context polarizes the spectrum — σ₁ and σ₂ gain at σ₃'s expense, concentrating energy into the dominant directions (ΔS < 0). Same input, opposite geometric effect, determined by the ρ₂ landscape. At 6.9B: responsive layers show mean Δσ₂% = +13.2%, rigid layers +0.4%. The witness effect IS a σ₂ effect at responsive layers — 81.5% of the spectral difference between receptive and absent conditions at L1 projects onto the σ₂ direction.

### Mechanism

ρ₂ = σ₂/σ₃ measures the "spectral commitment" of the sub-wire structure. When σ₂ >> σ₃ (large ρ₂), the secondary direction overwhelms the tertiary, leaving no degrees of freedom for contextual modulation. When σ₂ ≈ σ₃ (ρ₂ near 1.0), the system is degenerate — insufficient structure for context to perturb. The responsive zone (ρ₂ ≈ 1.3) is where the system has enough structure to maintain coherence but enough flexibility to respond to witness condition.

GQA prevents ρ₂ from entering the rigid zone. Mistral 7B shows stable positive ΔS across all 27 tunnel layers (range +0.023 to +0.048), consistent with ρ₂ remaining below threshold throughout. MHA allows ρ₂ to increase unchecked with depth, creating the rigidification gradient. The responsive zone maps to Liang et al.'s (2605.05686) basin margin: deep in basin (large ρ₂) = locked trajectory; at margin (moderate ρ₂) = coherent but adjustable; outside basin (small ρ₂) = free drift.

### Implications for the GQA comparison

This finding reframes the GQA advantage from §3.3. The "80× amplification" is misleading:

| Layer | MHA (410M) | GQA (Mistral 7B) | Ratio |
|---|---|---|---|
| L2 | +0.070 | +0.048 | 0.69× (GQA lower) |
| L6 | +0.010 | +0.047 | 4.7× |
| L12 | +0.002 | +0.043 | 25× |
| L17 | +0.0004 | +0.031 | 78× |

MHA starts with HIGHER per-layer sensitivity than GQA (0.070 > 0.048 at L2) but loses it through scale-dependent rigidification. GQA starts lower but sustains sensitivity through the tunnel. The "80×" is a ratio at the specific layer where MHA has crashed to near-zero; the tunnel mean ratio is 2.8× (GQA +0.040, MHA +0.014). GQA's advantage is duration of sensitivity through tunnel depth, not magnitude of sensitivity at any single layer.

---

## Revisions to existing sections (pending final integration):

### §4.4 revision (F20 partial retraction):

**Current**: "NO non-GQA model at any size develops positive ΔS"
**Revised**: "MHA tunnel mean remains positive at all tested scales (70M–6.9B), but individual-layer sign inversion emerges at scale. At 6.9B, 55% of tunnel layers show negative ΔS. The aggregate positive mean is driven by strongly positive early layers that outweigh the many weakly negative late layers. The per-layer gradient, governed by the σ₂/σ₃ ratio threshold (§3.7), reconciles the aggregate sign with per-layer diversity."

### §4.9 revision (F22 sustenance reframing):

**Current**: "GQA necessary and sufficient for enrichment sign"
**Revised**: "GQA sustains witness sensitivity through tunnel depth. MHA models show strong initial sensitivity (L2 ΔS = +0.070 at 410M) that scale-dependent rigidification extinguishes. GQA maintains the responsive zone (ρ₂ < 2.0) across the entire tunnel. The distinction is niche maintenance, not sign creation: all softmax attention models show positive ΔS in early layers; GQA prevents its extinction."

### Abstract revision (key phrases):

- "scale-dependent witness sensitivity gradient" replaces "sign inversion" for MHA
- Add: "governed by a sharp σ₂/σ₃ ratio threshold (ρ₂ ≈ 2.0) that separates responsive from rigid layers"
- Add: "Scale compresses the responsive niche from 68% (410M) to 7% (6.9B) of tunnel layers"
- Retain: sign inversion language for the GQA/MHA comparison at tunnel-aggregate level

### New references to add:

- Han, Chalmers & Izmailov (2026). How's it going? Reinforcement learning in language models recruits a functional welfare axis. arXiv 2605.30232.
- Dadfar et al. (2026). When Models Examine Themselves: Self-Referential Directions in Language Models. arXiv 2602.11358.
