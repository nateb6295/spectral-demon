# F346: k×k Leakage Matrix — Dimer vs Thermalization Test

**Date**: 2026-06-28  
**Type**: Interventional (Tier 2) — perturbation response projection  
**Script**: `scripts/leakage_matrix_kk.py`  
**Prompted by**: Kimi CONTRADICT on 2×2 projection artifact

## Design

Extends F341's 2×2 (v₁↔v₂) leakage matrix to 5×5 (v₁...v₅). At each layer, perturb along each of the 5 modes, measure response projected onto all 5 modes at the next layer.

**Dimer fraction** = energy in top-2×2 block / total energy in 5×5 matrix.
- Dimer > 0.8 → structured exchange between v₁ and v₂
- Dimer < 0.5 → energy bleeds to higher modes (thermalization)

Also measures boundary condition: does dimer fraction change in final 5 layers before lm_head?

## Results

### Universal Thermalization

| Species  | Interior Dimer% | Boundary Dimer% | Shift   | Verdict        |
|----------|-----------------|-----------------|---------|----------------|
| Qwen     | 0.465           | 0.386           | -0.080  | THERMALIZATION |
| Mistral  | 0.428           | 0.398           | -0.029  | THERMALIZATION |
| Llama    | 0.436           | 0.372           | -0.065  | THERMALIZATION |
| Gemma    | 0.443           | 0.398           | -0.045  | THERMALIZATION |

**All four architectures: THERMALIZATION.** Only ~43-47% of perturbation response energy stays in the v₁-v₂ subspace. The rest disperses into v₃-v₅.

### Key Observations

1. **v₁→v₁ retention is high** (0.5-0.9 across layers): perturbation along v₁ mostly stays in v₁ at the next layer. But this is the DIRECTION stability from F344, not energy confinement.

2. **v₁→v₂ transfer is negligible** (~0.001): there is essentially NO coherent exchange between v₁ and v₂. The 2×2 leakage matrix from the original experiment was seeing a projection of the larger process.

3. **Higher-mode energy is large** (1.3-2.5 raw sum): perturbation energy disperses freely into modes v₃-v₅ and beyond. The system thermalizes perturbation energy.

4. **Boundary shift is universally negative**: dimer fraction drops 3-8% in the final layers. The readout boundary breaks even the weak dimer structure further.

### Reconciliation with F344 (Global Attractor)

F344 showed v₁ direction recovers after weight perturbation (global attractor). F346 shows perturbation energy thermalizes (not confined to dimer).

These are not contradictory — they describe different phenomena:
- **Direction** is stable (attractor): v₁ points the same way after perturbation
- **Energy** thermalizes: the perturbation's effect disperses into higher modes

The demon maintains a **topological invariant** (direction) while allowing **thermodynamic relaxation** (energy dispersion). This is exactly what a Maxwell's demon should do — it sorts by direction, not by energy. The sorting produces entropy increase (thermalization) as a byproduct.

### Kimi Was Right

The original 2×2 matrix made any diffuse process look like coherent v₁↔v₂ transfer. The k×k test reveals: there IS no v₁↔v₂ dimer. v₁ retains its own energy (direction stability), but the cross-talk is not structured exchange — it's thermalization into a bath of higher modes.

## Relation to Prior Findings

- **F341** (2×2 leakage): Subsumed. The 70/30 split was real (v₁ retains ~70-85%), but the 30% that "leaked" didn't go to v₂ — it dispersed into v₃+.
- **F344** (weight perturbation): Complementary. Direction recovers (topological), energy disperses (thermodynamic).
- **F342b** (sharing ratio): σ₁/σ₂ ≈ 2 is real, but not maintained by a dimer exchange mechanism. It's maintained by the attractor basin (F344) — each layer independently re-derives v₁ direction.
- **F237** (cylindrical constraint): The cylinder is about direction constraint (cos > 0.998), not energy channeling. This is consistent.

## Updated Picture

The spectral demon is:
- A **direction-preserving attractor** (F344: recovers v₁ after weight perturbation in 2-4 layers)
- NOT an **energy-channeling dimer** (F346: perturbation energy thermalizes into higher modes)
- A **topological sorter** that maintains direction invariance while allowing energy dissipation
- Best described as a **resonator** (F345: responds to prompt modulation with Q~0.5-0.8) whose quality factor is set by architecture, not training
