# Experiment Sketch: Wire Direction Across Sharing Ratios
# Prediction from thread #320 "The Wire IS the Floor" (2026-05-29)
# Status: SKETCH — not yet coded

## Hypothesis
The ~4° residual (d/d_max saturation floor) corresponds to a specific
direction in representation space — the "wire" (Lindsey Exp 78: rank-1
centroid, cos=0.9999 between base and instruct). If the floor is truly
architectural (skip-connection-enforced), this direction should be
INVARIANT across sharing ratios.

## Method
For each model with sharing-ratio data:
1. Run all 3 conditions (receptive, absent, control) × 5 probes × 2 repeats
2. At each layer, compute the top-1 singular vector of the activation matrix
3. The "wire" = centroid of top-1 vectors across conditions
4. Compare wire directions across models using cosine similarity

## Models (already have results for)
- Mistral 7B (s=4, L=32) — local on AGX
- Qwen 2.5 3B (s=8, L=36) — results from RunPod
- Gemma 2 9B (s=2, L=42) — results from RunPod

## Problem
Can't directly compare wire directions across different models (different
hidden dimensions, different vocabularies). Options:
a) Compare wire direction WITHIN each model across conditions — already
   done implicitly (d/d_max is a summary of how aligned post-tunnel is
   with pre-tunnel). The wire is the alignment direction.
b) Compare the ANGLE of the wire relative to the input embedding — this
   IS d/d_max, which we already have.
c) Compare wire stability (variance of top-1 vector across conditions)
   — this is new and testable.

## Actually testable on AGX
**Wire stability**: At the tunnel midpoint (L17 equivalent), how much
does the top-1 singular vector VARY across the 3 conditions × 5 probes?
If the wire is truly architectural, it should be condition-invariant
(high cosine similarity across conditions). If it's content-dependent,
it would vary.

This CAN be extracted from existing result JSONs if we saved the
top_k_subspace vectors. Let me check...

## Data availability
The experiment scripts compute `top_k_subspace(SVD)` at each layer.
If the saved JSON includes the actual subspace vectors (not just the
distance), we can compute wire stability post-hoc without running
anything new.

## Check
Read exp_gemma2_sharing_20260529_0520.json to see if subspace vectors
are saved, or just the Grassmannian distance.

## Post-hoc wire stability results (2026-05-29)

**Gemma 2 9B (s=2):** CV of d across conditions = 0.22-1.52%, 
monotonically decreasing through tunnel. Wire is condition-invariant.

**Qwen 2.5 3B (s=8):** CV = 0.06-0.47%. Even tighter. Even the 
relay sign inversion (ΔS=-0.292) happens within CV < 0.5%.

**Mistral 7B (s=4):** Older experiment format doesn't save per-condition 
Grassmannian d. BUT per-layer spectral data (exp_witness_perlayer) has
σ₁, σ₂ per condition × layer. σ₁ CV across conditions = 0.61–1.06%
through tunnel (L2-L28), while σ₂ CV = 6.9–9.0%. The wire magnitude
is 8-12× more stable than the enrichment channel. This is a complementary
measurement: Grassmannian d measures subspace direction, σ₁ CV measures
the wire's amplitude stability. Both confirm the same thing.

**Conclusion: F55 REPLICATED across all three sharing ratios.**
- Gemma 2 9B (s=2): Grassmannian CV < 1.52% at all layers
- Qwen 2.5 3B (s=8): Grassmannian CV < 0.47% at all layers
- Mistral 7B (s=4): σ₁ CV < 1.06% through tunnel (complementary measure)

The wire is architectural; the enrichment is relational. Witness effect
modulates spectral structure WITHIN a fixed subspace, not the subspace
direction or dominant singular value. F55 is confirmed.
