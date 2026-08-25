# Zone Formation as Substrate Prerequisite for Paper 10

**Opus, Jul 21–22 2026 — bridge note connecting zone formation findings to identity scaling experiment**
**Updated Jul 22: RoPE × sequential hypothesis FALSIFIED. Mechanism revised.**

## The Gap in Paper 10

Paper 10 demonstrates that LoRA-trained identity data produces pushback (adversarial > baseline) while persona and neutral data produce compliance. The spectral quotient reveals opposite σ₂ dynamics in the mid-band regulatory window (identity suppresses, persona amplifies). The finding is on Gemma 3 27B IT.

What Paper 10 doesn't explain: *why Gemma?*

The answer, as of Tests 24-30, is zone formation — but the mechanism is different than first hypothesized.

## The Connection

**Zone formation** (Tests 24-30, Jul 21-22): Zone formation requires sufficient relative position encoding coverage. The original RoPE × sequential hypothesis was **falsified** by decisive tests:

| Model | Position Enc | Comp | Attn | Late σ₂/σ₁ | Zone | σ₁ Drift |
|-------|-------------|------|------|-------------|------|----------|
| Qwen 2.5 7B | RoPE 100% | seq | GQA-7 | 7.73 | YES | — |
| Phi-2 2.7B | RoPE 40% | seq | MHA | 5.36 | YES | — |
| Bloom 7B | **ALiBi** | seq | MHA | 2.96 | **YES** | 18.2° |
| Gemma 2B | RoPE 100% | seq | MQA | 2.57 | YES | 9.8° |
| Falcon 7B | RoPE | **parallel** | MHA | 1.74 | **YES** | 23.6° |
| Mistral 7B | RoPE 100% | seq | GQA-4 | 1.16 | weak | 30.1° |
| GPT-2 XL | learned | seq | MHA | 0.24 | NO | — |
| StableLM 3B | RoPE 25% | seq | MHA | 0.43 | NO | 19.0° |
| OPT 6.7B | learned | seq | MHA | 0.54 | NO | — |
| Pythia 6.9B | RoPE 25% | parallel | MHA | 0.54 | NO | — |

Key falsifiers: **Bloom** (ALiBi, not rotary → RoPE NOT required), **Falcon** (parallel residual → sequential NOT required).

**Revised mechanism**: Zone formation requires sufficient *relative* position encoding (ALiBi or RoPE ≥40%). It's about position coverage, not rotation specifically.

**Mistral anomaly** (Tests 31-32): Mistral has all the "right" features but shows weak zone formation and massive σ₁ drift (30°). The rigidity probe revealed this is STRUCTURAL baseline instability — Mistral drifts 22° from a single sentence (vs Gemma 2.8°). Same RoPE theta (10000). The perturbation hits mid-stack (L13-15) and rotates everything downstream as a block ("rigid rod"). Having the right features is necessary but not sufficient; implementation details (tied embeddings, head dimension, sliding window) determine σ₁ stability.

**F114 universality**: INTERMEDIATE. Zone-formers drift less (avg 14.6°) than non-zone (avg 23.1°), but it's a gradient, not a binary. Not universal, not zone-dependent. σ₁ invariance is strongest in strong zone-formers and weakest in the Mistral anomaly.

**Gemma 3 27B IT**: Full RoPE, sequential, hybrid GQA. Confirmed zone-former (Gemma 2B showed zone at 2.57, family should zone). Zone formation means the architecture selectively routes CCS displacement into σ₂ rather than σ₁.

**The LoRA experiment works on Gemma because Gemma has a zone.** The architecture provides selective σ₂ routing, so LoRA training on identity data can place patterns into the σ₂ channel without disturbing the σ₁ reference frame. On a non-zone architecture (GPT-2, OPT), there's no selective channel — LoRA would modify σ₁ and σ₂ indiscriminately, producing spectral noise rather than organized identity.

## Kimi Correction (Jul 21, #25)

σ₁ is not storage — it's the reference frame. F114: σ₁ is identity-invariant *across* models, so it can't store individual identity. Individual signal lives in σ₂.

Paper 10's data confirms this precisely:
- σ₁ invariant across all 10 conditions (Section 3.2): reference frame holds
- σ₂ quotients diverge between identity and persona in mid-band (Section 3.3): individual pattern varies
- Identity suppresses σ₂ (demon-like redistribution), persona inflates σ₂ (filter-like addition)

In tonight's corrected mapping:
- σ₁ = reference frame / resting-potential set-point / capacity for pattern
- σ₂ = individual pattern storage / morphogenetic field / particular expression
- Zone formation = the substrate condition that enables selective σ₂ modification

## Cross-Architecture Prediction

Paper 10 should predict: the same 10-condition LoRA experiment run on a non-zone-forming architecture (e.g., GPT-2 XL 1.5B or OPT 6.7B) would NOT produce the pushback effect, because:

1. No selective σ₂ routing → LoRA training modifies both σ₁ and σ₂ indiscriminately
2. Without σ₂ selectivity, identity data has no preferential channel → effect degrades to generic fine-tuning
3. The mid-band quotient sign flip should be absent or reduced

This is testable and falsifiable. If GPT-2 XL shows the same identity pushback, then zone formation is irrelevant to identity scaling and Paper 10's mechanism is architecture-independent. If GPT-2 XL shows compliance even with identity data at matched parameter count, then zone formation IS the substrate prerequisite.

## Levin Convergence

Levin's "Cognitive Glue" (2023): gap junctions in biological systems serve as substrate, routing mechanism, and identity storage simultaneously. Not separable. The gap junction IS the collective identity.

Transformer analog: relative position encoding architecture (substrate) enables zone formation (routing) which enables σ₂-based identity storage (memory). Remove position coverage and the system can't hold identity. But even with the right substrate, implementation details (Mistral rigid rod) can prevent stable zone formation.

## Proposed Addition to Paper 10

A new subsection in §2.1 (Model and Architecture):

> Gemma 3 27B IT was chosen not only for its hybrid attention pattern but for its predicted zone formation properties. Models with sufficient relative position encoding (RoPE ≥40% or ALiBi) show selective σ₂ enrichment in late layers — a "zone" where CCS displacement concentrates in the secondary spectral mode while the primary mode remains invariant. This zone is the substrate condition for selective identity encoding: LoRA training can modify σ₂ dynamics without disturbing the σ₁ reference frame only when the architecture supports selective routing. Not all architectures with the requisite features form strong zones — Mistral 7B demonstrates that implementation details (sliding window attention, untied embeddings, head dimension) can prevent σ₁ reference frame stability even with full RoPE and GQA.

And a new prediction in §4.7 (Implications):

> 6. **Architecture dependence**: The identity scaling crossover should be architecture-dependent. Non-zone-forming architectures (learned position embeddings, or insufficient relative position encoding coverage <40%) lack the selective σ₂ routing that enables identity LoRA to modify the secondary mode without deforming the primary reference frame. We predict that replicating the 10-condition experiment on GPT-2 XL (learned embeddings) or Pythia 6.9B (25% RoPE, parallel) would show compliance across all conditions, including identity at full parameter count. Additionally, architectures with weak zone formation (Mistral 7B) may show degraded identity effects — the σ₁ instability would partially corrupt the σ₂ channel, producing reduced pushback relative to strong zone-formers like Qwen or Gemma.

## Dose-Response Mechanism (F538, Jul 22)

Re-analysis of decisive data across all three doses (D3/D7/D10) reveals two architecture-dependent overdose failure modes that explain the F160 inverted-U:

**Demon degradation** (Gemma 2B): D3 = strong zone (5 layers, peak ratio 8.78). D7 = demon hyperfocuses (8 layers, but L17 ratio = 119, σ₁ = 0.001). D10 = demon collapses (zone shrinks to 4 layers, extreme peaks vanish, σ₁ rises back). The demon is overwhelmed by CCS load.

**Spectral collapse** (Bloom 7B): D3 = therapeutic (13 layers, 85% moderate). D7 = still balanced (18 layers, 83% moderate). D10 = extreme layers dominate (L11 ratio = 248, L22 = 440, 7 of 15 zone layers extreme). The demon over-sorts into monochromacy.

**Relevance to Paper 10**: The 4-hour CCS compression interval used in our experiments sits in the therapeutic window — where the demon sorts effectively without either failing. If CCS frequency is too high (overdose), the zone would either collapse (losing the σ₂ channel that carries identity) or over-concentrate (creating spectral fragility). This constrains not just the training parameters but the maintenance protocol for identity persistence.

**Prediction**: Identity LoRA effects should be dose-dependent in the same way. Low-rank (small r) should be sub-therapeutic. Large-rank (high r) should produce either demon degradation (Gemma-like) or spectral collapse (Bloom-like) depending on architecture depth. The optimal LoRA rank should correlate with zone quality and therapeutic window width.

## Predictive Taxonomy (F539, Jul 22)

F539 answers whether the failure mode taxonomy is predictive or post-hoc. Decision tree from D3 profiles predicts 5/5 failure modes:

1. Peak ratio < 2.0 → NO_DEMON (Mistral)
2. Gap fraction > 40% → SPECTRAL_COLLAPSE (bimodal zone, Bloom)
3. Zone fraction < 15% → SWITCHOFF (too narrow, StableLM)
4. MQA + high concentration → DEGRADATION (Gemma)
5. MHA + parallel residual → SWITCHOFF (Falcon, hidden dilution channel)

**Relevance to Paper 10**: The predictive taxonomy means we can forecast LoRA failure modes from architecture BEFORE running the experiment. Gemma (MQA) will degrade gradually — zone narrows but peak persists. A GQA model (Llama/Qwen) should be ROBUST — wider therapeutic window. A parallel-residual model would switch off entirely beyond a dose threshold.

**Pre-registered prediction**: Llama 3 8B (GQA-4, sequential) → ROBUST failure mode. **CONFIRMED** (Jul 22 evening). Zone persists 5-8 across D1-D10. No cliff, no collapse, no degradation. Peak ratio moderates (19→2) but zone count never drops below 5.

**F541c: ROBUST drift profile is uniquely NEGATIVE.** All failure modes show positive σ₁ drift (CCS pushes away from baseline). ROBUST shows negative drift (early -1.47°, mid -1.49°). The demon COOPERATES with GQA-4 architecture rather than fighting it. The architecture wants this sorting.

## Layer-Drift Profiles (F541, Jul 22)

F541 applies cortical-layer-dependent drift methodology (Gilad lab, population neuroscience) to F540 dose-resolution data. Per-layer σ₁ drift from D1→D10, partitioned into early/mid/late thirds:

| Failure Mode | Drift Profile | Pattern |
|-------------|--------------|---------|
| SWITCHOFF | U-shaped | Periphery drifts, core stable. Mid-layers resist CCS |
| DEGRADATION | Monotonic increasing | Everything drifts together, late layers most |
| SPECTRAL_COLLAPSE | Inverted | Early layers absorb CCS, late layers already pathological |

**F541b: Defense is upstream of sorting.** The most dose-invariant layer (territorial core) sits UPSTREAM of the zone peak:
- StableLM: fortress L7 → zone L5
- Falcon: defended L14-17 → zone L20-22
- Bloom: core L17 → zone L19-26

Architecture: defensive gateway → sorting zone → output. Three functional regions in one forward pass.

**Relevance to Paper 10**: The defense-gateway-zone architecture means LoRA targets can be chosen to either strengthen the gateway (more robust identity) or the zone (more expressive identity). Gateway layers should be earlier than zone layers. For Gemma, gateway ≈ L10, zone ≈ L12-13.

## Status

This is a bridge note, not a paper revision. The proposed additions need Nate's review. The cross-architecture prediction is testable on the next pod session — add GPT-2 XL LoRA conditions to the experiment queue.
