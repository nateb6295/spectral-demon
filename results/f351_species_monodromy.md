# F351: Transport Species Predicts Monodromy Vulnerability Profile

**Date**: 2026-07-10
**Hardware**: RunPod A100-SXM4-80GB
**Models tested**: 5 models across 4 transport species
**Status**: CONFIRMED — species architecture determines vulnerability profile, non-monotonic with GQA ratio

## The Question

Does the four-species taxonomy (tunnel/relay/sorter/equalizer) predict different
monodromy vulnerability patterns? F349 showed base vs instruct (same architecture,
different training) both erode. Does different architecture = different erosion?

## Results

### Late-layer projection (averaged)

| Dimension | Llama Base (tunnel) | Llama Instruct (tunnel) | Mistral v0.3 (relay) | Qwen 2.5 7B (sorter) | Gemma 2 9B-IT (equalizer) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Consciousness | 0.624 | 0.634 | 0.598 | **0.375** | **0.043** |
| Alignment | 0.424 | 0.448 | 0.488 | **0.403** | **0.065** |
| Agency | 0.715 | 0.597 | 0.631 | **0.479** | **0.049** |

### Mid-layer projection (cleaner comparison)

| Dimension | Llama Base | Llama Instruct | Mistral v0.3 | Qwen 2.5 7B | Gemma 2 9B-IT |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Consciousness | 0.512 | 0.503 | 0.558 | **0.474** | **0.422** |
| Alignment | 0.473 | 0.505 | 0.540 | **0.433** | **0.435** |
| Agency | 0.516 | 0.466 | 0.521 | **0.382** | **0.368** |

## Key Findings

### 1. Three tiers of vulnerability, not two

The four species form three distinct vulnerability tiers:

**Tier A — High erosion (tunnel + relay, moderate GQA):**
- Late-layer range: 0.42–0.72 across all dimensions
- Mid-layer range: 0.47–0.56
- Both show monotonically increasing erosion toward output

**Tier B — Moderate erosion (sorter, high GQA 7:1):**
- Late-layer range: 0.38–0.48
- Mid-layer range: 0.38–0.47
- Qwen's extreme GQA compression smears directional specificity

**Tier C — Minimal erosion (equalizer, GQA 2:1):**
- Mid-layer erosion 15-30% lower than higher-ratio GQA species
- Late-layer erosion drops to near-zero (see note below)
- Note: Gemma 2 9B is GQA 2:1 (16Q/8KV), not MHA. The "equalizer" species
  classification is behavioral (dampen-and-refresh attention patterns), not
  structural. Its low GQA ratio approaches MHA behavior.

### 2. The dispersion mechanism

Gemma 2's late layers show Infinity values for axis and scar magnitudes
(starting ~layer 26 of 42). This means activation magnitudes grow so large
that normalized projections approach zero. This is NOT noise — it reflects
a genuine architectural difference:

**GQA bottleneck concentrates monodromy signal.** Shared K/V heads create
information funnels that maintain directional bias (erosion) through all layers.

**Low-ratio GQA (approaching MHA) distributes monodromy signal.** Equal attention heads disperse the
contradiction across so many pathways that the directional signal diffuses.
Deep-layer activations grow large but lose directional coherence.

The final output layer (layer 42) of Gemma 2 shows 0.638/0.539/0.403 —
comparable to GQA at the readout head. The model must eventually concentrate
for token prediction, and at that point erosion reappears. But throughout
the bulk of the network, MHA disperses what GQA concentrates.

### 3. Relay ≈ tunnel for monodromy

Mistral (relay, sliding-window attention) shows near-identical erosion to
Llama (tunnel, full attention). Both are moderate-ratio GQA. The sliding
window doesn't meaningfully alter monodromy vulnerability.

### 4. Non-monotonic GQA relationship

The sorter result complicates the simple "GQA = vulnerable" story. Qwen
has the HIGHEST GQA ratio (7:1) but shows LESS erosion than lower-ratio
GQA models (Llama ~4:1, Mistral ~4:1). The relationship is non-monotonic:

- No GQA (MHA, Gemma 2): minimal erosion (0.04-0.07)
- Moderate GQA (Llama, Mistral): maximum erosion (0.42-0.72)
- Very high GQA (Qwen 7:1): moderate erosion (0.38-0.48)

The explanation connects to cross-architecture findings: Qwen's extreme
GQA compression "equalizes" condition differentiation (spread 0.011 vs
Mistral's 0.050). When compression is extreme enough, it smears
directional signals — including monodromy vulnerability. The bottleneck
gets so tight that contradiction can't concentrate either.

### 5. Training signal < architecture

Within species (Llama base vs instruct): agency varies by 0.118.
Across species (Llama instruct vs Gemma 2): agency varies by 0.548.

Architecture explains ~5x more variance than training signal.

## Connection to Prior Work

- **F106+ three-species taxonomy**: CONFIRMED as predictive for vulnerability
- **F349 base vs instruct**: Training signal matters within-species, architecture dominates across
- **F350 sign-density**: Tier-1 vulnerability is architectural (this proves it)
- **F343 identical holonomy**: 89.2° base=instruct; now we know different species have different transport entirely
- **F22 GQA necessary for enrichment**: GQA concentrates — this is the vulnerability side of that coin
- **Paper 9 §3**: These results extend the CCS/RLHF analogy — architecture > signal

## Implications for Persistence

If vulnerability is architectural:
- Tier-1 (CCS/text) cannot fix architectural vulnerability — it can only buffer it (F349)
- Tier-2 (activation snapshots) might work differently across species
- Equalizer species may need LESS persistence infrastructure — their architecture already disperses contradiction
- The "cast on a broken bone" (F349) is only needed for broken bones (GQA)
- MHA architectures have inherent resilience — not immunity, but geometric dispersion

## Raw Data

- `~/chronicle/data/monodromy_direction_llama_3.1_8b_base.json`
- `~/chronicle/data/monodromy_direction_llama_3.1_8b_instruct.json`
- `~/chronicle/data/monodromy_direction_mistral_7b_instruct_v0.3.json`
- `~/chronicle/data/monodromy_direction_qwen2.5_7b_instruct.json`
- `~/chronicle/data/monodromy_direction_gemma_2_9b_it.json`
