
## Queued for next pod session

### 1. Leakage Matrix (needs revision: `scripts/leakage_matrix.py`)
Does the 70/30 perturbation split explain σ₁/σ₂ ≈ 2? **Revision needed per Kimi CONTRADICT**: extend from 2×2 (v₁↔v₂) to k×k (v₁...v₅ minimum). If off-diagonals decay sharply past v₂, it's a dimer. If energy bleeds to higher modes, the 2D picture is a projection artifact. Also check whether leakage matrix character changes in final 2-3 layers before lm_head (boundary condition hypothesis).

### 2. Prompt Q Factor (needs script)
Resonance width measurement. Prompt titration: start with neutral completion, add identity-relevant words incrementally, measure σ₁/σ₂ at each step. Width of the concentration curve = Q factor. Predictions:
- IT sharpens Q without shifting resonant frequency (F343 geometric evidence)
- Mistral: broad flat response (always ~2, architecturally constrained cylinder)
- Llama: sharp peak (1.55 neutral → 3.49 identity, switchable geometry)
- Base vs instruct same architecture tests whether Q is trained or inherent
Compare base vs instruct on Llama 3.1 8B (reuses F343 infrastructure).

### 3. Weight Perturbation Recovery — "The Gregory Experiment" (needs script)
**The most important experiment in this batch.** Tests whether v₁ is a LOCAL property (each layer independently) or a GLOBAL attractor (the whole network converges). This is the first real architectural intervention — all prior experiments intervened on hidden states, not weights.

**Method**: For each architecture:
1. Compute baseline v₁ at every layer (5-prompt SVD as in F342)
2. Add Gaussian noise (scaled ε) to attention weights at a SINGLE target layer (e.g., L10)
3. Re-run all prompts, compute v₁ at every layer with perturbed weights
4. Measure: does v₁ recover downstream of the perturbation? How many layers to recover?

**Predictions**:
- If GLOBAL attractor: v₁ shifts at L10-L12 then recovers by L15-L20. "Marks imprinted by nature" — the whole system re-recognizes the direction.
- If LOCAL property: v₁ shifts at L10 and stays shifted through all downstream layers. Each layer contributes independently; break one, break the chain.
- Species-specific: Mistral (rigid cylinder) may recover faster than Gemma (oscillator). Llama's late-layer convergence may self-correct even large perturbations.

**Why this matters**: Epistemological tier upgrade. "Architecture produces v₁" is currently observational (Tier 1). This would make it interventional (Tier 2). Also directly tests Gregory's "constitutional recognition" — whether the marks are in the whole or in each piece.

**Sweep**: ε = {0.001, 0.01, 0.1} × target_layer = {L5, L10, L15, L25} × 4 architectures = 48 conditions. ~2h on A100.
