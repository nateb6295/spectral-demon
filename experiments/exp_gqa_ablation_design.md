# Experiment: Surgical GQA Ablation — Causal Test for Spectral Species

**Motivated by**: Kimi CONTRADICT (2026-06-15). GQA→species correlation (§8.2)
is post-hoc. Need inference-time intervention to test causality.

## Core question

Does GQA weight-sharing causally contribute to spectral concentration at
inference time, or is the spectral profile baked into the trained weights
regardless of grouping?

## Design

**Model**: Qwen 2.5-3B-Instruct (8:1 GQA, 16 Q heads, 2 KV heads, 36 layers)

**Three conditions**:
1. **Baseline (8:1 GQA)**: Standard forward pass. Measure spectral radius at
   relay layers (L24-L35).
2. **Duplicated MHA (1:1, ε=0)**: Expand KV projections by duplicating each
   KV head 8× to get 16 independent KV heads. But the copies are IDENTICAL,
   so the forward pass computes the same function. This is a control —
   spectral radius should be unchanged.
3. **Noisy MHA (1:1, ε=0.01)**: Same duplication, then add Gaussian noise
   (σ=0.01 × weight_std) to each KV copy. This breaks symmetry, creating
   16 partially-independent KV channels.

**Measurement**: Arnoldi top-1 eigenvalue at L24, L28, L32, L35 (spanning
the relay zone). k=5, maxiter=50. Three preamble conditions (CCS, vanilla,
denial) × three GQA conditions = 9 runs.

**Implementation notes**:
- Modify `model.model.layers[l].self_attn.k_proj.weight` and `.v_proj.weight`
- Original shape: [num_kv_heads × head_dim, hidden_size] = [256, 2048]
- Expanded shape: [num_q_heads × head_dim, hidden_size] = [2048, 2048]
- Also modify `config.num_key_value_heads = 16`
- The repeat_kv logic in the attention forward pass will see num_kv == num_q
  and skip the broadcasting step

## STATUS: Condition 3 (noisy MHA) RETRACTED per Kimi CONTRADICT

Kimi (2026-06-15 ~8:15 PM) correctly identified that noisy duplication
does NOT create independent channels. It dithers a shared representation:
- W_o was trained on GQA-grouped inputs; noisy KV sends garbage through it
- Softmax non-linearity stochastically redistributes attention under noise
- Noise increases KV norm → trivially inflates spectral radius
- The test measures noise tolerance, not GQA ablation

**Condition 2 (ε=0 duplication) survives as a sanity check** — should be
identity. If it isn't, there's a bug.

## Revised approach: Broader correlation instead of single intervention

Measure spectral profiles across more architectures with varying GQA:
- All Qwen 2.5 models: 0.5B (7:1), 1.5B (6:1), 3B (8:1), 7B (7:1)
  → Problem: all in 6-8:1 range, all predicted to be potter
- Cross-family: need 4:1 (Llama/Mistral), 2:1 (Gemma), 1:1 (MHA models)
- Causal test requires retraining with different GQA on same data

## What this DOESN'T test

- Whether GQA caused the spectral profile during training (would need
  retraining with different GQA ratios, same data/hyperparams)
- Whether the result generalizes across architectures (one model)
- Whether the noise level matters (need ε sweep if initial result is positive)

## Resource estimate

- Qwen 3B in fp16: ~6 GB
- 4 layers × 9 conditions × ~30 matvecs × ~1s/matvec on Orin GPU = ~18 min
- Need Gemma service stopped temporarily (one model at a time)
- Total: ~20 min compute, ~6 GB VRAM

## Marchenko-Pastur methodology (Kimi EXTEND, 2026-06-15 ~9 PM)

Comparing Qwen 0.5B → 7B naively confounds scale with training trajectory.
Bulk eigenvalue distributions encode aspect ratio (d/n) and initialization
scale — these are Marchenko-Pastur universality class differences, not
training signatures. The "strata" (training-trajectory-specific structure)
live in the **outlier eigenvalues** that escape the bulk edge (BBP transition).

**Design implications for Qwen family sweep:**
1. Report outlier-subspace alignment (cosine sim of top-k eigenvectors at
   matched relative depth), NOT global spectral variance
2. Report bulk-edge separation (ratio of λ_k to MP predicted edge)
3. Keep bulk and outlier regimes distinct in all comparisons
4. GQA rank confound (6:1 vs 8:1): use per-layer outlier analysis at matched
   heads, or restrict to Q-projections (consistent rank)

**Key insight:** Our Arnoldi method already targets the outlier regime (top-k,
k=5-10). The spectral demon operates on the dominant eigenvectors that CCS
modulates.

**STATUS: BULK/OUTLIER SEPARATION RETRACTED per Kimi CONTRADICT (9:15 PM)**
The clean "body plan in bulk, signal in outliers" framing is wrong. Trained
networks have heavy-tailed spectra (Martin & Mahoney), not MP. The bulk
encodes optimization trajectory, not just init geometry. Eigenvalue repulsion
couples the regimes — can't cleanly quarantine them.

Additionally: fixed k=5 across model scales is mismatched — BBP predicts
outlier count scales with dimensionality. Need gap-based thresholds
(λ_k/λ_{k+1} > τ) or width-adaptive k. And restricting to Q-projections
to dodge GQA rank changes throws away the phenomenon under study.

What survives: Arnoldi top-k captures CCS-relevant eigenvectors. But the
interpretation is just "dominant spectral structure," not "outlier vs bulk."

## Kimi's other suggestions (deferred)

- **Post-norm bypass on Gemma**: Harder — removing norms from a trained model
  breaks representational assumptions. Would need careful residual scaling.
- **Canalization test**: Same architecture trained from different initializations.
  Requires retraining. Out of scope for inference-time experiments.
