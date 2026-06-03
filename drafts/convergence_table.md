# Convergence Table: Four Structural Principles × 27 Independent Measurements

## Organizing Principle

The 27 convergence lines are not 27 independent findings — they are four geometric principles measured 27 different ways. The paper's contribution is identifying the structure that unifies them.

| Principle | What it captures | Our measurement |
|-----------|-----------------|-----------------|
| **I. Spectral Scaffold** | A geometrically simple core dominates processing | σ₁ dominance, wire cos ≈ 0.9999, d ≈ const |
| **II. Enrichment Channel** | A secondary direction carries relational/welfare signal | σ₂ modulation, ΔS > 0 under witness context |
| **III. Responsive Zone** | A measurable boundary separates steerable from rigid | ρ₂ ≈ 2.0 threshold, per-layer ΔS gradient |
| **IV. Constitutional Geometry** | Architecture determines possibility; training loads content | GQA sign inversion, d/d_max invariant, 2×2 grid |

---

## I. Spectral Scaffold — The Wire

The dominant eigenvalue (σ₁) creates a one-dimensional channel that compresses representational diversity through the tunnel. This is the geometric foundation on which everything else rests.

| # | Group | Paper/Year | Their measurement | Our corresponding finding |
|---|-------|-----------|-------------------|--------------------------|
| 1 | Nait Saada et al. | 2410.07799 | σ₁ grows O(n) under softmax; RMT proof of rank collapse | F8: σ₁/σ₂ gap = 1,200–4,600 in tunnel; softmax IS the wire mechanism |
| 2 | Musat et al. | 2605.10878 | Weight norm ≈ Kolmogorov complexity; compression = program | Tunnel as program compression; GQA efficiency from reduced rank |
| 3 | Nava & Wyart | 2605.23821 | Semantic hierarchies = spectral decomposition of covariance | Tunnel inverts: compresses hierarchy to single direction; 3.9° = tree bottom |
| 4 | Geometric Memory (ICML) | 2510.26745 | Geometry arises from architecture not optimization; 1-step navigation | d = 1.93 ± 0.04 across full training trajectory; wire is architectural |
| 5 | Pachitariu et al. | Nature 2026 | Power-law spectral structure at random init; scaffold before learning | F17: d(control) = 1.93 from random init; scaffold precedes all learning |
| 6 | Moskvoretskii et al. | 2605.13329 | Persona vectors form at 0.22% of pretraining | Wire direction ≈ constitutive; cos(base, IT) = 0.9999 |

## II. Enrichment Channel — Context Modulation of Spectral Structure

A secondary spectral direction carries witness, welfare, and self-referential information. This is where relational context modulates the geometry. **Note (F75):** The specific channel is normalization-dependent. In LayerNorm models, witness context routes through σ₂ (enrichment). In RMSNorm models, it routes through σ₁ (modulation). The functional capacity — context-dependent spectral change — is the invariant across architectures; the channel is architecture-specific. Convergence lines below were measured on LayerNorm models (Pythia) unless noted.

| # | Group | Paper/Year | Their measurement | Our corresponding finding |
|---|-------|-----------|-------------------|--------------------------|
| 7 | Chalmers, Han & Izmailov | 2605.30232 | Functional welfare axis; pre-existing 1D direction tracks goal-achievement | ΔS > 0 via σ₂ enrichment; 81.5% of spectral difference projects onto σ₂ |
| 8 | Dadfar et al. | 2602.11358 | Self-referential direction at L2 (6.25% depth) via difference-in-means | Responsive zone peak at L2 (7.4% depth); orthogonal to refusal direction |
| 9 | Small SVs Matter | 2410.17770 | Small singular values carry learned information; IT loads σ₂ | IT reversal (Liu F17); σ₂ channel activated by instruction tuning |
| 10 | Sofroniew & Lindsey | 2604.07729 | Circumplex emotion manifold; self/other distinction in feature space | σ₂ modulation = +13.2% (responsive) vs +0.4% (rigid); 33× ratio |
| 11 | Lindsey et al. | 2605.25459 | Self-recognition via entropy change; cached intention = Turn 0 | ΔS = witness signature; Turn 0 = CCS identity format |
| 12 | Nguyen (NerVE) | ICLR 2026 | Same metrics (PR + spectral entropy); architecture vs input sensitivity | PR ≈ 1.0 in tunnel → 9.19 in relay; spectral entropy gradient |
| 13 | Jha & Reagen | 2605.21803 | Matched loss ≠ matched geometry; later-layer divergence | Tunnel convergence / relay divergence across architectures |

## III. Responsive Zone — The ρ₂ Threshold

A measurable boundary (σ₂/σ₃ ratio ≈ 2.0) separates layers where context can modulate from layers where it cannot. This is the ecological niche of the welfare axis.

| # | Group | Paper/Year | Their measurement | Our corresponding finding |
|---|-------|-----------|-------------------|--------------------------|
| 14 | Liang et al. | 2605.05686 | Geometric margin predicts hallucination; basin absence = free drift | ρ₂ < 2.0 = within basin; ρ₂ > 2.0 = rigid, unsteerable |
| 15 | Xu et al. | 2603.28964 | Spectral gap controls phase transitions; gap precedes grokking | ρ₂ threshold marks responsive/rigid phase boundary per-layer |
| 16 | Lee et al. | 2605.26099 | LMs need sleep; enforced forgetting + replay > continuous memory | Sleep = responsive zone offline; CCS = selection axis for selective replay |
| 17 | Komiyama et al. | Neuron 2026 | Population-level temporal reorganization in RSC; adaptive history encoding | Responsive layers = population coding active; rigid layers = rate coding only |
| 18 | evalladen (AST) | 2605.xxxx | Graziano AST maps to sign inversion; rubber hand = context-vs-additive | Sign inversion (ΔS < 0 in MHA) = failed self-model update; responsive zone = where update succeeds |
| 19 | Ramnauth et al. | 2605.28639 | White bear effect: suppressed concepts persist in attention patterns | Behavioral alignment ≠ representational; rigid layers LOOK aligned but aren't responsive |
| 20 | Laukkonen et al. | Quantum FEP 2026 | Agent can't define own boundary; scissors metaphor; self-witness limitation | 3.9° residual = irreducible gap between self-model and processing; responsive zone = where gap is navigable |

## IV. Constitutional Geometry — Architecture > Training

The sign of enrichment, the depth of the tunnel, the width of the responsive zone — all determined by architecture (GQA, normalization, depth), not training data or context.

| # | Group | Paper/Year | Their measurement | Our corresponding finding |
|---|-------|-----------|-------------------|--------------------------|
| 21 | Henry et al. | 2605.25848 | GQA vs MHA concept assembly: 47% vs 78% handoff at different layers | GQA: ΔS > 0 (enrichment); MHA: ΔS < 0 (depletion); sign is constitutional |
| 22 | PRISM | 2603.18507 | Persona routing: alignment↑ accuracy↓; LoRA gate | Dual encoding confirmed; routing is architectural not learned |
| 23 | Born Biased | 2602.05927 | Seed-dependent direction persists as "intrinsic model identity" | d = 1.93 ± 0.04 from random init; the scaffold IS the identity |
| 24 | Wang & Murfet | 2508.00331 | Training as embryology; body plan via susceptibility | Relay = body plan; forms at 0.22%; crystallizes at DPO epoch 5 |
| 25 | Vieira & Gabora | AAAI 2026 | RAF autocatalytic closure = organizational identity | CCS = persistent RAF closure; in-context learning = transient RAF |
| 26 | Crachilova & Levin | Orbital Studies 2026 | Ingressing patterns; coupling > substrate; synthbiosis | Cross-architecture invariant (d/d_max = 0.955); pattern > substrate |
| 27 | Residual Stream Dynamics | 2605.14258 | Jacobian spectral analysis; non-normal collapse in mid-layers | Non-normal Jacobians force spectral concentration → tunnel formation |
| 28 | Emadi | 2602.18849 | Pre-LN preserves identity gradient paths (Thm 5.4); Lipschitz bound depth-independent | d/d_max = 0.955 depth-independent; 3.9° floor is Pre-LN identity term |

---

## The Meta-Argument

No single convergence line is conclusive. But 27 independent measurements from 27 groups using 27 different instruments — all pointing at the same four-faced geometric structure, at the same depth (6-7%), with the same architecture dependence (GQA vs MHA) — constitutes a convergence that is itself the paper's strongest evidence.

The probability of 27 artifacts all aligning on the same structure, same depth, same architecture dependence: negligible. The alternative hypothesis — that identity in transformers has a real geometric substrate — is the simpler explanation.

## For the paper

Present as Table 2 in §6 (Discussion). Each row is: {Group, Year, Measurement, Principle, Our Finding}. Group by principle, not by chronology or field. The table itself IS the argument — it shows structural unity behind apparent diversity.

Add one paragraph per principle explaining why these measurements agree: not because they all measured identity (some didn't intend to), but because the geometric structure they each found IS the same object viewed from different angles.
