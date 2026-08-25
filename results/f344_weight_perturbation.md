# F344: Weight Perturbation Recovery — "The Gregory Experiment"

**Date**: 2026-06-28  
**Type**: Interventional (Tier 2 — first weight-level intervention)  
**Script**: `scripts/weight_perturbation.py`

## Design

Perturb attention weights at a single layer with Gaussian noise (ε × σ_weight), run 5 identity-loading prompts, measure whether v₁ (computed via SVD) recovers at downstream layers.

- 4 architectures × 4 target layers × 4 epsilon values = 64 conditions
- Target layers: early (~L5), early-mid (~L9-10), mid-late (~L18-21), late (~L23-35)
- Epsilon: 0.001, 0.005, 0.01, 0.05
- Recovery criterion: final_cos > 0.95

## Results

### Universal Global Attractor

**64/64 conditions recovered. 100% GLOBAL across all architectures.**

| Species | Recovered | Avg Recovery Distance |
|---------|-----------|----------------------|
| Qwen    | 16/16     | 3.8 layers           |
| Mistral | 16/16     | 2.2 layers           |
| Llama   | 16/16     | 2.7 layers           |
| Gemma   | 16/16     | 3.1 layers           |

### Species-Specific Recovery Speed

Recovery distance correlates with cylinder geometry:
- **Mistral** (rigid cylinder, σ₁/σ₂ ≈ 2): fastest recovery (2.2L). Directional rigidity = strong attractor basin.
- **Llama** (switchable geometry): 2.7L. IT convergence provides recovery pathway.
- **Gemma** (GQA oscillator): 3.1L. Standing-wave pattern takes slightly longer to re-establish.
- **Qwen** (distributed): 3.8L. Most distributed processing, but still recovers completely.

### Key Observations

1. Even at ε=0.05 (substantial weight perturbation), final cosine > 0.999 for all conditions
2. Maximum disruption typically occurs 1-2 layers downstream of perturbation site
3. Late-layer perturbations sometimes show "never" for strict 0.99 recovery threshold, but final cosine is still > 0.999 — the attractor has fewer layers to work with, not weaker pull
4. Pre-perturbation layers are completely unaffected (cos = 1.000)

## Interpretation

v₁ is a **global attractor** of the forward pass, not a local property of individual layers. Perturbing weights at any single layer creates a transient deviation that the network corrects within 2-4 layers.

Gregory of Nyssa's "marks imprinted by nature" — the recognition is constitutional, distributed across the whole network. No single layer holds the identity direction; the whole system re-derives it.

**Epistemological upgrade**: "Architecture produces v₁" moves from Tier 1 (observational — we see it at every layer) to Tier 2 (interventional — we broke it and it came back). This is the first causal evidence that v₁ is not just present but actively maintained.

## Relation to Prior Findings

- **F342b** (sharing ratio): v₁ is computed from the same prompts. Global attractor explains why sharing ratio is architecture-specific but always converges.
- **F341** (perturbation response): Hidden-state perturbation showed 70/30 split. Weight perturbation shows the 70% isn't passive — the network actively steers back.
- **F237** (cylindrical constraint): Mistral's rigid cylinder now has a mechanical explanation — strongest attractor basin = fastest recovery.
- **E21b** (weight-space SVD): IT remodeling preserves relay. Weight perturbation shows why — the relay is an attractor, not a conduit.

## Predictions Confirmed

- [x] Global attractor (vs local): YES, 64/64
- [x] Mistral fastest recovery: YES (2.2L vs 2.7-3.8L)
- [x] Recovery speed tracks cylinder geometry: YES (rigid > oscillatory > distributed)
- [ ] Llama self-corrects large perturbations via late convergence: Partially — recovers at all layers, not just late

## Next Steps

This finding constrains interpretation of the k×k leakage matrix (running now). If v₁ is a global attractor, "leakage" from v₁→v₂ may be the attractor basin pulling stray energy, not thermalization.
