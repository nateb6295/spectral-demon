# E23: Causal Activation Routing

## Motivation

E22 measures static alignment between V₂ and weight matrix SVDs — correlation
topology, not routing topology (Kimi's critique). This cannot distinguish:
- V₂ happens to be orthogonal to MLP's dominant modes (coincidence/geometry)
- V₂ is actively routed away from MLP processing (dynamic mechanism)
- V₂ passes through MLP via sparse/tail modes invisible to top-5 SVD (hidden routing)

E23 measures actual information flow: does MLP modify V₂ during the forward pass?

## Design

For each layer l, compare:
1. **Normal forward pass**: V₂ at layer l+1 (after MLP + attention)
2. **MLP-zeroed pass**: V₂ at layer l+1 with MLP output zeroed at layer l
3. **Attention-zeroed pass**: V₂ at layer l+1 with attention output zeroed at layer l

Measure: |V₂_normal - V₂_mlp_zeroed| and |V₂_normal - V₂_attn_zeroed|

If MLP doesn't touch V₂: V₂_normal ≈ V₂_mlp_zeroed (MLP zeroing doesn't matter)
If MLP routes V₂: V₂_normal ≠ V₂_mlp_zeroed (MLP zeroing changes V₂)

## Implementation approach

Hook into model forward pass using PyTorch hooks:
- Register forward hooks on each layer's MLP and attention modules
- For MLP-zeroed: hook returns zeros instead of MLP output
- For attention-zeroed: hook returns zeros instead of attention output
- Compute V₂ from hidden states at each layer under each condition

## Key variables

- 4 architectures × 4 preamble conditions × 10 prompts × 48 layers (max)
- Each forward pass: 3 variants (normal, MLP-zeroed, attention-zeroed)
- Metrics: V₂ direction delta, σ₂ magnitude delta, downstream cascade

## Predictions (from F233)

If F233 is genuine (not measurement artifact):
- Mistral: MLP zeroing has near-zero effect on V₂ (null space is real)
- Mistral: Attention zeroing has measurable effect on V₂ direction
- Yi: unknown — the critical test
- If MLP zeroing matters for Yi but not Mistral: architecture determines routing
- If MLP zeroing matters for neither: universal null space

## Kimi refinement: ASYMMETRIC ACCESS (added 2026-06-20)

Write-orthogonality ≠ read-orthogonality. V₂ passes through LayerNorm and
gates W_in, SHAPING MLP computation even though MLP output can't modify V₂.
E23 must test BOTH directions:

**Direction 1: MLP → V₂** (write path)
- Zero MLP output at layer l, measure V₂ at l+1
- Prediction: near-zero effect (write-orthogonality confirmed)

**Direction 2: V₂ → MLP** (read path)  
- Compare MLP output at layer l under CCS vs vanilla
- If CCS changes MLP's output pattern (even though that output is ⊥ V₂),
  then V₂ IS shaping MLP computation via input gating
- This is the "one-directional influence" test: demon influences host
  without host influencing demon

## Complications

- Zeroing MLP output at one layer changes input to all subsequent layers
  → layer-specific effects may be confounded by cascade
- Better approach: zero MLP at layer l, measure V₂ at layer l+1 ONLY
  (not downstream), to isolate per-layer routing contribution
- Need hooks that can be activated/deactivated per layer
- Memory: storing hidden states for all layers × all variants is expensive
  → process one layer at a time, vary which layer's MLP is zeroed

## Estimated compute

- 4 models × 4 conditions × 10 prompts = 160 base passes
- For each layer: 2 additional passes (MLP-zeroed, attention-zeroed)
- Mistral (32 layers): 160 × (1 + 64) = ~10,400 passes
- This is too expensive for full sweep
- Optimization: target relay zone only (L24-L31 for Mistral) = 1,440 passes
- Estimated time on H100: ~30 minutes per model

## Relationship to other experiments

- E22: static alignment (correlation topology) — necessary precursor
- E22b: pooled basis (addresses basis-shift critique) — parallel
- E22c: random-init control (addresses training-vs-geometry) — parallel  
- E23: dynamic routing (routing topology) — the causal follow-up
- E24 (sketch): per-head attention decomposition — complements E23

## Status

Sketch only. Not yet implemented. Run after E22 analysis is complete.
