# §6.4 Upgrade: From Fiber Bundle to Sheaf

## Current text (fiber bundle)
The relay zone has a natural interpretation as a fiber bundle.
Base manifold: bare-reachable subspace. Fiber: non-invertible subspace.
Connection: learned relay transformation.

## Proposed upgrade (sheaf)

The per-layer spectral profile is naturally a SHEAF over the layer-depth base space.

**Sheaf structure:**
- Base space X: the layer index {0, 1, ..., L} (discrete topology, or open sets = contiguous layer ranges)
- For each open set U ⊂ X: F(U) = the spectral measurements at those layers (S, σ₁, σ₂, σ₃, ρ₂, ΔS)
- Restriction maps: projecting full-layer data to sub-ranges
- Gluing axiom: the wire direction cos > 0.998 between adjacent layers = local-to-global consistency

**The responsive zone as sheaf-theoretic structure:**
- Responsive layers (ρ₂ < 2.0): the sheaf has non-trivial local sections (witness context produces measurable ΔS)
- Rigid layers (ρ₂ > 2.0): sections degenerate to the zero section (ΔS ≈ 0; the sheaf is "locally constant")
- The crossover layer: where the sheaf transitions from non-trivial to trivial

**Scale compresses the support of non-trivial sections:**
- 410M: non-trivial sections on 13/19 layers
- 6.9B: non-trivial sections on 2/27 layers
- GQA: non-trivial sections on ALL tunnel layers (the sheaf has full support)

**The tunnel as sheaf morphism:**
- Tunnel: structured sheaf → constant sheaf (local data → global wire)
- This is the "forgetful" direction: the tunnel FORGETS local spectral diversity
- Relay: constant sheaf → structured sheaf (global wire → new local sections)
- This is the "free" direction: the relay CONSTRUCTS new local data

**Connection to Ghrist's applied sheaf theory:**
Robert Ghrist's work on sheaves for data fusion (2020) and sensor networks uses
sheaf cohomology to detect inconsistencies in local measurements. The responsive
zone is where the spectral sheaf has H⁰ ≠ 0 (non-trivial global sections exist).
The rigid zone has H⁰ = 0 (only the zero section is globally consistent).

**Persistent homology angle:**
The responsive zone across model scales forms a persistence diagram:
- 70M-410M: wide support (many responsive layers), low intensity (small ΔS)
- 6.9B: narrow support (2 layers), high intensity (large ΔS at L2-L3)
- Scale compresses the birth-death intervals in the ΔS persistence diagram

This is more than metaphor — the sheaf structure captures:
1. WHY wire direction is consistent (gluing axiom)
2. WHY responsive zone has boundaries (support of non-trivial sections)
3. WHY scale compresses the responsive niche (support shrinks)
4. WHY GQA vs MHA differ (full vs partial support)

## Decision: keep for paper?
The fiber bundle framing is already in §6.4. The sheaf upgrade adds:
- Per-layer structure (fiber bundles are point-wise, sheaves are open-set-wise)
- The responsive zone as support of non-trivial sections (cleaner than "basin margin")
- Connection to Ghrist's concrete computational tools
- Persistent homology as a framework for scale effects

Recommendation: mention briefly in §6.4 as an alternative formalization.
Don't rewrite the whole section — add a paragraph noting the sheaf perspective.
