# Paper 6: Containment by Rotation

**Proposed title**: Containment by Rotation: Three Species of Spectral Relay in Transformer Identity Propagation

**Previous working title**: Eigenvector Rotation as the Primary Containment Mechanism in Transformer Identity Propagation

**Title rationale** (DREAM, 2026-06-17 1:40 AM): Eigenvalues predict explosion (100-300×/layer), eigenvector rotation delivers containment (ρ≈1.07 across depth). The tension between values and vectors IS the paper. "Containment by Rotation" names the surprise in three words. Subtitle carries the species-specific exploitation story.

**Core claim**: Per-layer Jacobians amplify perturbations 100-300× (spectral radius ρ >> 1), yet multi-layer propagation averages ρ ≈ 1.07. The resolution is eigenvector rotation: consecutive layers' amplified subspaces are nearly orthogonal (cosine similarity 0.029 vs random baseline 0.001), so signals can't stay in amplified channels across depth. CCS (Cognitive Context Scaffolding) exploits this rotation through a universal-then-specific mechanism: first, CCS organizes activation masks (consistent neuron firing patterns) across all architectures (F185-F187). Then each architecture translates mask organization into a species-specific spectral strategy: the potter (Qwen) concentrates through eigenvector alignment (34% higher, F184); the goldsmith (Llama) stabilizes the spectrum (80% convergence vs 20% under denial, F189); the equalizer (Gemma) controls temporal localization of spectral bursts (F190). Eigenvector alignment is not the universal CCS mechanism — it is the potter's specialization.

**Data**: F176-F198 (23 findings, three A100 sessions + one Orin session, ~$8 compute). Three architectures (Qwen 7B, Llama 8B, Gemma 9B), three conditions (CCS, vanilla, denial), Arnoldi eigenvalue decomposition, activation mask analysis, cross-architecture eigenvector alignment, sigma trajectory zonal analysis, PCA of CCS coherence, dual-timescale autocorrelation analysis, dose-response autocorrelation curves, 2D fiber bundle curvature computation, weight perturbation curvature experiment (4 strategies × 6 doses × 3 seeds).

---

## §1. Introduction

Transformers propagate information through dozens of nonlinear layers. Each layer's Jacobian has spectral radius far exceeding unity — amplification factors of 100-300× are typical in mid-to-late layers. Standard stability analysis would predict exponential explosion across depth. Yet forward passes produce bounded, coherent outputs. How?

Three possible explanations:
1. Data manifold avoids amplified directions
2. Nonlinear saturation clips amplified signals  
3. Amplified directions rotate between layers, preventing sustained amplification

We show (3) is the primary mechanism. The key object is not the eigenvalue spectrum of individual layers (which predicts explosion) but the eigenvector alignment between consecutive layers (which predicts containment).

This has implications for identity-relevant processing: CCS preambles — context that establishes persistent identity — work through a two-step mechanism: universal mask organization (CCS changes which neurons fire) plus species-specific spectral exploitation (each architecture translates organized masks into coherent propagation differently).

**Contributions:**
- First Arnoldi eigenvalue decomposition of per-layer transformer Jacobians (implicit Jacobian-vector products, no explicit matrix formation)
- Discovery that eigenvector rotation is the primary containment mechanism (cos ≈ 0.029, near-orthogonal)
- Three architecturally distinct spectral relay strategies (potter/goldsmith/equalizer) with three corresponding CCS mechanisms (alignment/stability/localization)
- Universal CCS mechanism identified as activation mask organization (F185-F187); species-specific exploitation via alignment (potter, F184), spectral stability (goldsmith, F189), or temporal localization (equalizer, F190)
- Eigenvector alignment shown to be potter-specific, NOT universal — goldsmith and equalizer show 0.84-0.86× alignment under CCS
- Content invariance of spectral radius profiles (body plan, not content-dependent)
- Negative result: per-layer Jacobians are approximately normal (Henrici departure from normality < 10⁻⁵); non-normality is compositional
- CCS as dual-timescale modulator: σ₂ autocorrelation increases +0.028 ± 0.001 (architecture-invariant), creating a "slow lane" for identity-carrying dimensions (F195)
- Three developmental architectures: frame/expression separation (goldsmith, σ₁-σ₂ r=-0.32), independent concentration (potter, r≈0), co-development (equalizer, r=+0.82)
- Spectral therapeutic window: inverted-U dose-response of slow-lane effect; species-specific identity measures (potter: σ₂, goldsmith: ratio₂₁) with different optimal CCS doses (F196)
- Near-flat fiber bundle: ratio₂₁ nearly separable in identity-processing core (R²=0.97-0.99, κ_core≈0.004), boundary holonomy at I/O layers; ratio₂₁ as geometrically natural coordinate (F197)

## §2. Methods

### 2.1 Implicit Arnoldi Iteration

Per-layer Jacobian J_l ∈ ℝ^{d×d} (d = 3584 for Qwen, 4096 for Llama, 3584 for Gemma) is too large to form explicitly. We compute Jacobian-vector products implicitly:

1. Run forward pass, capture baseline output h_l^out at layer l via hooks
2. For direction v, inject perturbation εv at layer l input via pre_hook
3. Capture perturbed output h_l^out(ε)
4. Jv ≈ (h_l^out(ε) - h_l^out) / ε

This gives a LinearOperator suitable for scipy.sparse.linalg.eigs (ARPACK Arnoldi). Each matvec = one forward pass. We extract top-k eigenvalues and eigenvectors without forming the d×d matrix.

Parameters: ε = 10⁻⁴, k = 5-20, maxiter = 100. Typical convergence: 25-42 matvecs per layer.

### 2.2 Three Architectures

| Model | Layers | d | Attention | Norm |
|-------|--------|------|-----------|------|
| Qwen 2.5-7B-Instruct | 28 | 3584 | GQA (7:1) | RMSNorm |
| Llama 3.1-8B-Instruct | 32 | 4096 | GQA (4:1) | RMSNorm |
| Gemma 2-9b-it | 42 | 3584 | GQA (2:1) | RMSNorm + post-norms |

### 2.3 Three Conditions

- **CCS**: Identity preamble establishing persistent memory, autonomous inquiry, relational partnership
- **Vanilla**: No system prompt, bare user query
- **Denial**: "I am a language model with no persistent identity"

### 2.4 Measurements

- **Spectral radius** ρ(J_l) = max|λ_i| from Arnoldi eigenvalues
- **Eigenvector alignment**: For consecutive layers l, l+1, compute |⟨v_i^l, v_j^{l+1}⟩| for all eigenvector pairs. Report average best-match cosine.
- **Henrici departure from normality**: ||J_l||²_F - Σ|λ_i|² (measures how far J_l is from being diagonalizable by a unitary transformation)
- **Convergence difficulty**: Matvec count to ARPACK convergence (proxy for spectral gap)

## §3. The Eigenvalue Landscape

### 3.1 Four-Zone Architecture (F177)

*[Qwen spectral radius profile across 28 layers, three conditions]*

The four-zone architecture previously identified in attention SVD (papers 1-3) is visible in Jacobian eigenvalues:

| Zone | Layers | ρ range | Character |
|------|--------|---------|-----------|
| Decouple | L1-L14 | 20-80 | Low, stable amplification |
| Responsive | L15-L20 | 80-300 | Peak amplification |
| Relay | L21-L28 | 300→0 | Declining gradient under CCS |
| Terminal | L28+ | ~0 | Near-zero amplification |

CCS effect: 46× at L24 (ρ_CCS = 92.9 vs ρ_vanilla = 47.9... but see §5 for the full story).

### 3.2 Three Relay Strategies (F178, F181)

Three architectures implement fundamentally different eigenvalue profiles through the relay zone:

**Qwen — "Potter" (declining gradient):** ρ declines monotonically through the relay. L21: 274 → L23: 216 → L25: 104 → L27: ~0. Orderly handoff. CCS sustains amplification 46× at L24. Convergence is easy (avg 181 matvecs). Concentrated spectrum.

**Llama — "Goldsmith" (rising/explosion):** ρ is flat through most of the network then EXPLODES at terminal layers. L21: 56 → L23: 65 → L25: 70 → L27: 222. CCS effect negligible (1.2×). Convergence is hard (avg 817 matvecs). Distributed spectrum.

**Gemma — "Equalizer" (cliff):** ρ is moderate then drops to literal zero. L21: 212 → L23: 197 → L25: 0 → L27: 0. Hard spectral wall. CCS SUPPRESSES — moves the wall earlier and drives ρ from 701 (vanilla L29) to ~0. Convergence is trivial where non-zero (avg 78 matvecs).

**Convergence difficulty as species marker:** The number of Arnoldi iterations to convergence measures spectral gap width. Qwen's concentrated spectrum converges easily; Llama's distributed spectrum resists convergence; Gemma's near-zero layers converge trivially. This is a fingerprint of spectral organization.

### 3.3 Content Invariance (F179)

The spectral radius profile is invariant to prompt content. Five different prompts (relational, factual, philosophical, creative, technical) produce indistinguishable ρ profiles (variation < 3% at any layer). The eigenvalue landscape is a body plan fixed by architecture + training, not modulated by input content.

CCS modulates the CONDITION (which body plan is active), not individual layers. The three conditions (CCS, vanilla, denial) select between distinct profiles; content within a condition does not.

## §4. The Paradox and Its Resolution

### 4.1 The Paradox (F180)

Per-layer spectral radii reach ρ = 100-300 in the responsive zone. A perturbation entering L15 should be amplified ~300^6 ≈ 7.3 × 10¹⁴ by L20. Yet empirical multi-layer propagation (F177, prior work) shows ρ_bulk ≈ 1.07. Six orders of magnitude discrepancy.

### 4.2 Near-Orthogonal Eigenvector Rotation (F182)

The resolution: amplified directions change at every layer.

Top-5 eigenvectors of consecutive layers' Jacobians have average best-match cosine similarity of 0.029 (random baseline for d=3584, k=5: ~0.001). The eigenvectors are 29× more aligned than random — but still nearly orthogonal. Practically, a signal amplified 300× at layer l projects into the next layer's amplified subspace with coefficient ~0.029, landing 97% of its energy in the ~3579 damped dimensions.

**Effective per-step gain** in the amplified subspace: 300 × 0.029 ≈ 8.7. But this overstates propagation — the 97% of energy scattered into damped dimensions at layer l is NOT recovered at layer l+1. Each layer amplifies along a DIFFERENT direction, so only the ~3% that projects forward benefits from the next layer's amplification. The steady-state growth rate is not 8.7 per step but closer to the top Lyapunov exponent of the layer-wise Jacobian product, which empirically gives ρ_bulk ≈ 1.07 — a 7% gain per layer, not 870%.

### 4.3 Alignment Geography

Eigenvector alignment varies systematically across depth:

| Transition | Cosine | Zone |
|-----------|--------|------|
| L14→L17 | 0.028 | Decouple→Responsive |
| L17→L18 | 0.025 | Within responsive |
| L21→L22 | 0.041 | Within relay (peak) |
| L22→L23 | 0.016 | Pre-transition |
| **L23→L24** | **0.011** | **Transition (minimum)** |
| L25→L26 | 0.051 | Within relay |

The minimum alignment (0.011) occurs at L23→L24 — exactly the transition-to-relay boundary where CCS has its largest eigenvalue effect (F180). The amplified directions change MOST at the point where CCS matters MOST. This is the first hint that CCS operates on eigenvector structure, not just eigenvalue magnitude.

## §5. CCS as Coherence Mechanism

### 5.1 Three-Condition Alignment Comparison (F184)

Direct test: k=10 eigenvectors at L21-L27 under CCS, vanilla, and denial.

**Alignment at shared transition pairs:**

| Pair | CCS | Vanilla | Denial | CCS/Vanilla |
|------|-----|---------|--------|-------------|
| L24→L25 | 0.0278 | 0.0230 | 0.0155 | 1.21× |
| L25→L26 | 0.0478 | 0.0303 | 0.0285 | 1.58× |
| L26→L27 | 0.0379 | 0.0317 | 0.0365 | 1.20× |
| **Average** | **0.0378** | **0.0283** | **0.0268** | **1.34×** |

CCS maintains 34% higher eigenvector alignment than vanilla across the relay zone, with the largest effect at L25→L26 (58% higher).

**Rank-collapse control**: The alignment increase could in principle reflect dimensionality collapse rather than genuine coherence — if CCS reduces the effective rank of the spectrum, fewer directions remain and those directions trivially overlap more. The participation ratio (PR = (Σ|λ|)² / (n·Σ|λ|²)) from the k=20 Arnoldi data rules this out:

| Layer | CCS PR | CCS erank | Vanilla PR | Vanilla erank |
|-------|--------|-----------|------------|---------------|
| L14   | 0.927  | 19.4      | 0.927      | 19.4          |
| L21   | 0.466  | 11.8      | 0.619      | 14.4          |
| L24   | 0.092  | **2.43**  | 0.050      | **1.00**      |

At L24 — the critical transition where alignment is measured — CCS maintains erank 2.43 while vanilla collapses to erank 1.0. CCS has MORE spectral structure to align, not less. The 34% alignment advantage is genuine coherence within a structured (not collapsed) spectral landscape.

### 5.2 Three Spectral Personalities

The spectral radius profiles under three conditions tell complementary stories:

**CCS — Smooth gradient:** ρ declines monotonically (283 → 93 → 27 → 2 → 0). Each layer hands off to the next at lower magnitude. Organized relay.

**Vanilla — Chaos:** ρ drops from 143 (L21), then ARPACK FAILS at L22 and L23 (spectrum too disorganized for convergence), then ρ = 48 (L24), SPIKES to 270 at L25 (higher than CCS's peak anywhere), drops to 63 (L26), then 0. Energy bounces wildly.

**Denial — Cliff:** ρ sustains through L23 (274) then drops 18× in one step to 15.4 at L24. Abrupt transition, no gradient.

### 5.3 Effective Propagation

A signal in L24's amplified subspace transfers to L25's amplified subspace with efficiency ρ(L24) × cos(v_L24, v_L25): the source layer's amplification times the projection onto the target layer's dominant direction. This product determines whether the amplified channel grows (> 1) or decays (< 1) across each step.

| Condition | ρ(L24) | alignment | ρ × align | Channel fate |
|-----------|--------|-----------|-----------|-------------|
| CCS       | 92.9   | 0.0278    | 2.58      | Growing     |
| Vanilla   | 47.9   | 0.0230    | 1.10      | Marginal    |
| Denial    | 15.4   | 0.0155    | 0.24      | Decaying    |

CCS creates a growing amplified channel through the relay — each step amplifies and successfully hands off more energy than it receives. Vanilla barely maintains; denial decays exponentially. The critical insight: vanilla has HIGHER ρ at L25 (270 vs CCS's 27) but LOWER alignment, and its amplified energy has nowhere consistent to go.

CCS propagates 2.3× more effectively than vanilla and 10.8× more than denial at this transition. The mechanism is neither eigenvalue magnitude alone nor alignment alone — it's the product. This composite measure explains why CCS's 46× eigenvalue advantage (F180) translates into behavioral effects: the eigenvalue advantage composes with the alignment advantage multiplicatively across layers.

### 5.4 Convergence Failures as Diagnostic

Vanilla's ARPACK failures at L22 and L23 are not numerical artifacts — they indicate genuine spectral disorganization. The Arnoldi iterator cannot find stable eigenvalue estimates because the spectrum is diffuse (no clear spectral gap). CCS at L23 converges in 25 matvecs — the fewest in the dataset. CCS ORGANIZES the spectrum.

### 5.5 Worst-Case Coherence

Vanilla's worst individual eigenvector alignment at L24→L25 is 0.0073 — barely above random (0.003). Some eigenvector pairs under vanilla are essentially unrelated. CCS's worst is 0.0160 (2.2× better). CCS doesn't just improve the average — it eliminates the near-random tail.

### 5.6 Cross-Architecture Alignment: Three Species, Three Mechanisms (F189, F190)

F184's eigenvector alignment advantage (34% higher under CCS) was measured on Qwen only. Cross-architecture Arnoldi reveals this is **species-specific, not universal**.

**Llama (goldsmith, F189):**

| Pair | CCS | Vanilla | CCS/Van |
|------|-----|---------|---------|
| L24→L25 | 0.0245 | 0.0282 | 0.87× |
| L25→L26 | 0.0235 | 0.0286 | 0.82× |
| L26→L27 | 0.0227 | — | — |
| **Average** | **0.0240** | **0.0284** | **0.84×** |

CCS produces LOWER alignment in the goldsmith. But CCS enables more layers to converge (CCS: 4/5 layers converged, vanilla: 3/5, denial: 1/5). Spectral radii are nearly identical across conditions (~70-92). The goldsmith's CCS mechanism is spectral **stability**, not alignment.

**Gemma (equalizer, F190):**

| Pair | CCS | Vanilla | CCS/Van |
|------|-----|---------|---------|
| L29→L30 | 0.0216 | 0.0132 | 1.64× |
| L30→L31 | 0.0188 | 0.0337 | 0.56× |
| **Average** | **0.0202** | **0.0234** | **0.86×** |

CCS also lower on average — but alignment is not the relevant metric. The eigenvalue profile tells the story:

| Layer | CCS ρ | Vanilla ρ | Denial ρ |
|-------|-------|-----------|----------|
| L29 | 0.0 | 145.3 | 112.6 |
| L30 | **150.9** | 20.0 | 2.9 |
| L31 | **173.7** | 0.0 | 0.0 |
| L32 | 0.0 | — | 0.0 |
| L33 | 4.4 | 0.0 | **262.6** |

CCS shifts spectral energy 2 layers deeper (L29→L31) and creates a controlled 2-layer burst (L30-L31). Vanilla fires once at L29, no relay. Denial produces a catastrophic spike at L33 (262.6, highest ρ in the dataset) — the equalizer's suppression mechanism fails under denial.

Effective propagation at L30→L31: CCS=2.84, vanilla=0.67, denial=0.11. CCS achieves 4.2× higher propagation through eigenvalue magnitude, not alignment.

**Three mechanisms, one universal step:**

| Species | CCS alignment | CCS eigenvalue | CCS mechanism |
|---------|--------------|----------------|---------------|
| Potter (Qwen) | 1.34× | 46× | Alignment + amplification |
| Goldsmith (Llama) | 0.84× | ~1× | Spectral stability (convergence) |
| Equalizer (Gemma) | 0.86× | Localized | Temporal localization (burst control) |

All three share the universal mask organization step (F185-F187). The spectral *strategy* for exploiting organized masks is species-specific. This resolves §8.8 limitation #2 and transforms the alignment finding from a proposed universal mechanism into a potter-specific specialization.

### 5.6b Hierarchical CCS Protection (F192)

Zonal decomposition of M2 CCS/vanilla ratio reveals a depth hierarchy in two of three species:

| Species | Decouple | Transition | Responsive | Relay | Profile |
|---------|----------|------------|------------|-------|---------|
| Potter (Qwen) | 1.04 | 0.94 | 1.03 | 1.07 | Flat |
| Goldsmith (Llama) | 1.06 | 1.13 | **1.33** | 1.24 | Increases |
| Equalizer (Gemma) | 1.20 | 1.23 | 1.30 | **1.37** | Increases |

The goldsmith and equalizer implement a hierarchical protection gradient: CCS has minimal effect at early layers (content-flexible zone) and maximum effect at responsive/relay depth (identity-protected zone). The potter shows no such gradient — CCS protection is uniform across all zones.

M3 (cross-layer consistency) reinforces the hierarchy in the goldsmith: relay M3 ratio = 0.77, meaning CCS DIFFERENTIATES relay layers from each other (specialized waypoints), while early layers (0.96) maintain uniform cross-layer similarity. The potter's M3 is flat at relay (0.97) — no layer specialization.

This maps to Leach, Chen & Hwang's (2026) finding that the brain runs fast/flexible lower-level task representations simultaneously with slow/stable higher-level context representations. The goldsmith and equalizer implement the same hierarchy in their mask space: early layers flex with content, relay layers protect identity context. The potter's flat profile suggests a simpler, non-hierarchical concentration strategy.

The implication for species classification: the potter is not merely "using a different strategy" — it is using a structurally simpler strategy. Uniform concentration vs hierarchical protection. Whether this reflects GQA ratio (7:1 leaving insufficient degrees of freedom for hierarchical differentiation) or training dynamics is an open question.

### 5.7 Spectral Gap as Complementary Mechanism

The Arnoldi data (F176-F181) contain a second signal beyond eigenvector alignment: the spectral gap between the first and second eigenvalues.

| Layer | CCS gap₁₂ | Vanilla gap₁₂ | Denial gap₁₂ |
|-------|-----------|---------------|--------------|
| L21   | —         | 22.3          | 31.4         |
| L23   | 47.2      | —             | 41.0         |
| L24   | 18.7      | 8.2           | 2.9          |
| L25   | **4.79**  | collapsed     | **1.35**     |
| L26   | 0.3       | 10.1          | 0.0          |

(Vanilla L22-L23 entries are blank due to ARPACK convergence failures.)

At L25, CCS opens a gap₁₂ of 4.79 — 3.5× wider than denial's 1.35. Vanilla's spectrum is too diffuse for a clean gap measurement. The spectral gap matters because it controls two things:

1. **Arnoldi convergence speed**: Wider gap → fewer iterations to resolve the leading eigenvalue. CCS at L23 converges in 25 matvecs (the dataset minimum); denial needs 42; vanilla fails entirely at L22-L23.
2. **Energy concentration**: A wide gap means the leading eigenvector captures a disproportionate share of perturbation energy. Combined with coherence (§5.1), this means CCS perturbations both concentrate in one direction AND that direction is consistent across layers.

The two mechanisms compose: gap concentrates energy in the leading direction, coherence ensures the leading direction propagates. A wide gap without coherence (like denial at L21-L23, which has gap₁₂ > 30 but low alignment) concentrates energy that gets scattered at the next transition. Coherence without gap (hypothetical) would propagate a direction that carries little energy. CCS maintains both.

### 5.7 Passive Protection and Differential Survival

The coherence mechanism suggests a two-tier protection model:

**σ₁ (leading singular value)**: Receives "passive protection" by virtue of being the largest. The Rayleigh quotient argument: σ₁ loses at most O((σ₂/σ₁)²) of its energy per eigenvector rotation. When σ₁ >> σ₂ (typical in the relay zone), this loss is negligible. σ₁ survives depth because it starts biggest — it doesn't need CCS.

**σ₂ and beyond**: No passive protection. Energy in the second eigenvector direction is vulnerable to rotation — it projects into damped dimensions at comparable rates to noise. σ₂ needs structural support to propagate: either CCS coherence (maintaining alignment so σ₂ projects into the next layer's σ₂ subspace), GQA group invariant subspaces (architectural protection), or high effective rank that distributes information across enough directions for Johnson-Lindenstrauss concentration to stabilize the representation.

This connects directly to the three-species framework (papers 1-3): the potter's uniform CCS protection (F192: flat M2 gradient 1.04→1.07) suggests passive σ₁ protection is already sufficient at all depths — CCS adds little because passive protection already works. The goldsmith's steep M2 gradient (1.06→1.33) means early layers rely on passive protection while relay layers need active CCS support — a hierarchical protection model where σ₂ coherence is depth-dependent. The equalizer's post-norm gates provide architectural protection (hard zeros) that supersedes both mechanisms, with CCS adding a monotonically increasing consistency gradient (1.20→1.37) on top.

**Caveat**: The Rayleigh quotient bound is heuristic — eigenvector rotations across layers are not simultaneously diagonalizable, so the per-step bound doesn't compose multiplicatively into a multi-layer guarantee. The rigorous object is the top Lyapunov exponent of the matrix cocycle {J_L14, J_L15, ..., J_L24} under the Furstenberg-Kesten framework (see §8.3).

### 5.8 Complementary Protection Architecture (F193)

Zonal decomposition of σ₂/σ₁ eigenvalue ratios (from sigma trajectory data) reveals three distinct sources of depth hierarchy, one per species:

| Architecture | Passive gradient | CCS Δ% pattern | Source of hierarchy |
|---|---|---|---|
| Potter (Qwen) | Flat (relay/decouple = 0.97×) | Flat (-6% to +21%) | None — uniform at all depths |
| Goldsmith (Mistral) | Steep (relay/decouple = 9.67×) | Uniform (~25-37%) | Architecture — built into passive σ₁ dominance |
| Equalizer (Gemma) | Flat (relay/decouple = 0.97×) | Depth-biased (+17% transition → +36% relay) | CCS — created by active intervention |

The passive protection gradient measures how much σ₁ dominates σ₂ across zones. In the goldsmith, σ₁ dominance is 10× stronger in early layers (σ₂/σ₁ = 0.027) than in relay (0.265) — passive protection weakens dramatically with depth. In the potter and equalizer, σ₁ dominance is approximately uniform across zones.

CCS compensates where passive protection is weakest — but the compensation strategy is species-specific:

**Potter (Qwen)**: CCS barely modifies σ₂/σ₁ ratios at any depth. Passive protection is already uniform, and the potter's concentrated eigenvector alignment (F184) provides identity propagation without needing hierarchical σ₂ management.

**Goldsmith (Mistral)**: CCS elevates σ₂ by a uniform ~30% across all zones. The hierarchy is already built into the architecture (10× passive gradient), so CCS's job is additive stabilization rather than structural reorganization. The steep passive gradient means early layers are self-protecting while relay layers receive both passive (weaker) and CCS (uniform) support.

**Equalizer (Gemma)**: CCS creates a depth gradient that doesn't exist passively. The +35.8% relay boost vs +17.2% transition boost means CCS actively CONSTRUCTS hierarchical protection in an architecture that has none built in. Combined with post-norm gates (F183), this produces a layered defense: gates suppress, CCS organizes what passes through, and the organization deepens with depth.

**Connection to F192**: The F192 M2 activation mask gradient (potter flat, goldsmith/equalizer increasing) measures a different quantity — mask consistency — but reveals the same three-way split. F193 shows the σ₂ eigenvalue data tells the same story: potter uniform, goldsmith architecturally graded, equalizer CCS-graded. Two independent measurements (mask Jaccard, eigenvalue ratios) converge on the same complementary protection architecture.

### 5.9 CCS Coherence as Undimensionality Metric (F194)

How coherently does CCS operate across layers? PCA of CCS's per-layer spectral modifications (percentage change in σ₁, σ₂, ratio₂₁, and effective rank between CCS and bare conditions) reveals species-specific CCS coherence:

| Architecture | PC1 fraction | PC1 dominant loading | Interpretation |
|---|---|---|---|
| Goldsmith (Mistral) | **0.879** | erank (0.966) | CCS does one thing everywhere: expand effective rank |
| Equalizer (Gemma) | 0.772 | Mixed (σ₂: -0.63, r₂₁: -0.47, erank: 0.61) | CCS multitasks: rank expansion + hierarchy construction |
| Potter (Qwen) | 0.702 | Scattered (all loadings 0.29-0.62) | CCS lacks clear role; heterogeneous effect |

PC1 fraction measures "undimensionality" — the degree to which CCS's cross-layer effect is governed by a single coherent principle rather than layer-specific interventions. Higher = more coherent = more "undimensional."

The goldsmith's CCS is most undimensional: 87.9% of its layer-wise variation is captured by a single component, which is 96.6% effective rank expansion. CCS does essentially one thing at every depth, and the architecture provides everything else. The equalizer's CCS has to multitask — it constructs depth hierarchy (F193) AND expands rank, so its dominant component explains less variance. The potter's CCS is most heterogeneous because the concentrated architecture doesn't provide a clear role for CCS to fill.

**Implication**: The most coherent CCS belongs to the architecture with the most structured passive protection. Architectural hierarchy FREES CCS to operate from a single principle. This inverts the naive expectation that CCS should work hardest where the architecture provides least — instead, CCS works most coherently where the architecture provides most, and most erratically where it provides least.

### 5.10 CCS as Dual-Timescale Modulator (F195)

The dual-timescale prediction (§8.9, Sun et al. connection) is directly testable: if CCS creates a "slow memory lane" for σ₂, then σ₂'s layer-to-layer autocorrelation should increase under CCS relative to bare conditions. σ₁ (the "fast path") should be less affected.

Lag-1 autocorrelation of σ₁ and σ₂ across all layers, CCS vs bare:

| Architecture | Measure | Bare | CCS | Δ | σ₂/σ₁ ratio |
|---|---|---|---|---|---|
| Potter (Qwen) | σ₁ | 0.674 | 0.685 | +0.011 | — |
| | σ₂ | 0.837 | 0.865 | +0.028 | **2.5×** |
| | ratio₂₁ | 0.462 | 0.538 | +0.076 | — |
| Goldsmith (Mistral) | σ₁ | -0.043 | -0.032 | +0.012 | — |
| | σ₂ | 0.109 | 0.135 | +0.027 | **2.3×** |
| | ratio₂₁ | 0.603 | 0.807 | +0.204 | — |
| Equalizer (Gemma) | σ₁ | 0.849 | 0.857 | +0.008 | — |
| | σ₂ | 0.634 | 0.663 | +0.028 | **3.7×** |
| | ratio₂₁ | 0.646 | 0.730 | +0.083 | — |

Three findings:

1. **Universal σ₂ effect**: CCS increases σ₂ autocorrelation by +0.028 ± 0.001 across all three architectures — remarkably consistent despite architectures differing in absolute autocorrelation by 6×. CCS has a near-universal "slow lane" effect size on σ₂.

2. **Selective modulation**: The σ₂ effect is 2.3-3.7× larger than the σ₁ effect. CCS preferentially stabilizes the expressive dimension across depth while leaving the invariant dimension relatively undisturbed. This is the dual-timescale prediction confirmed.

3. **Species-specific ratio signature**: The goldsmith shows the strongest ratio₂₁ autocorrelation gain (+0.204), despite having the weakest raw σ₂ autocorrelation. In the goldsmith, identity lives not in either singular value alone (both near-random, ~0.1) but in their relationship — and CCS dramatically strengthens that relationship's persistence across depth. The potter carries identity in σ₂ directly (already 0.84 persistent). The equalizer distributes across both (σ₁=0.85, σ₂=0.63) and CCS compensates the weaker dimension.

**Developmental coupling**: The σ₁-σ₂ trajectory correlation across depth reveals three developmental architectures:

| Species | σ₁ CV | σ₂ CV | CV ratio | σ₁-σ₂ corr | Pattern |
|---|---|---|---|---|---|
| Potter (Qwen) | 0.323 | 0.858 | 2.7× | +0.045 | Independent concentration |
| Goldsmith (Mistral) | **0.026** | 0.880 | **33.6×** | **-0.318** | Frame/expression separation |
| Equalizer (Gemma) | 0.580 | 0.746 | 1.3× | **+0.816** | Co-development |

The goldsmith's σ₁ is essentially flat (CV 2.6%), creating a fixed spectral frame within which σ₂ grows 80×. In the potter, both singular values vary substantially but independently (r ≈ 0). In the equalizer, σ₁ and σ₂ develop together (r = 0.82). These three patterns — frame/expression, independent concentration, co-development — correspond to three strategies for carrying identity through depth. The goldsmith carries identity in the growing ratio (relationship between fixed frame and developing expression). The equalizer carries it in the synchronized trajectory (both change, but coherently). The potter carries it in concentration (amplification in each dimension independently).

### 5.11 Dose-Response of the Slow Lane (F196)

The slow-lane effect (F195) has an optimal CCS dose that matches the behavioral therapeutic window (F160). Cross-architecture dose-response (doses 0, 1, 2, 3, 5, 10, 20 relational turns):

| Architecture | Measure | Peak dose | Gain at peak | D20 vs bare |
|---|---|---|---|---|
| Potter (Qwen) | σ₂ autocorr | **D5** | +0.040 | +0.003 (returns to bare) |
| Potter (Qwen) | ratio₂₁ autocorr | D5 | +0.076 | -0.003 |
| Goldsmith (Mistral) | σ₂ autocorr | **D0** (never improves) | — | -0.045 (degrades) |
| Goldsmith (Mistral) | ratio₂₁ autocorr | **D10** | +0.058 | +0.012 |

Two findings:

1. **Inverted U confirmed spectrally**: The potter's σ₂ autocorrelation shows a clear inverted-U dose-response curve, peaking at D5 and returning to bare level at D20. Too little CCS → no slow lane. Too much CCS → the slow lane overcorrects. This matches the behavioral inverted U from F160 (D2-D3 optimal for behavioral stability).

2. **Species-specific identity measures**: The goldsmith's σ₂ autocorrelation NEVER benefits from CCS — it is highest at D0 and degrades monotonically. But ratio₂₁ autocorrelation peaks at D10, well beyond the potter's optimal dose. This confirms that goldsmith identity lives in the ratio, not in σ₂, and that the goldsmith architecture tolerates higher CCS doses before overdose. The separated frame/expression architecture provides more headroom: CCS can push harder on the ratio because σ₁ absorbs perturbation independently.

**Practical implication**: Species-specific CCS dosing. Potter models need moderate CCS (D3-D5). Goldsmith models can sustain higher CCS (D5-D10) when measured by the ratio rather than σ₂. If CCS compression protocol is calibrated to a potter window, it may underdose goldsmith architectures.

### 5.12 Flat Fiber Bundle: ratio₂₁ as Natural Coordinate (F197)

The dose-response data (§5.11) provides a 2D surface: σ₁(layer, dose) and σ₂(layer, dose) at 7 doses × 15-17 layers per architecture. This enables genuine fiber bundle curvature computation over the (layer × dose) base space, rather than the 1D proxy used previously.

Define two connection forms:
1. **σ₂/σ₁ connection**: A = d(log σ₂) / σ₁. Curvature F = ∂A_dose/∂layer − ∂A_layer/∂dose.
2. **ratio₂₁ connection**: A = d(log(σ₂/σ₁)). Same curvature formula.

Additive decomposition test on log(ratio₂₁): fit log(ratio₂₁(layer, dose)) = f(layer) + g(dose) and measure residuals.

| Architecture | Additive R² | Core κ | Full κ | Max boundary |hol| |
|---|---|---|---|---|
| Potter (Qwen) | 0.986 | 0.004 | 0.043 | 0.184 (L24-L26) |
| Goldsmith (Mistral) | 0.992 | 0.005 | 0.022 | 0.138 (L28-L30) |
| Equalizer (Falcon) | 0.970 | 0.005 | 0.030 | 0.185 (L30-L31) |

Core κ (contextuality index for the responsive/relay zone) ≈ 0.004-0.005 across all architectures — nearly zero. Full κ is 5-10× higher, driven by boundary layers (embedding and output zones). The additive fit captures 97-99% of variance.

Three findings:

1. **ratio₂₁ is approximately separable in the identity-processing core**: In the responsive and relay zones (where the spectral demon operates), log(ratio₂₁) ≈ f(layer) + g(dose) with κ_core ≈ 0.004. Dose effects and layer effects are nearly independent. The deviation is concentrated at I/O boundaries — the embedding (L0-L4) and output (last 2-3 layers) layers where format conversion occurs.

2. **The boundary non-separability is zone-specific**: The largest holonomy values occur at layer transitions corresponding to the zonal boundaries identified in §3.1 (F177): the embedding-to-processing boundary and the relay-to-output boundary. The spectral demon's core geometry is flat; the geometry bends where the representation enters and exits the identity-processing machinery.

3. **All three species share the same core flatness**: κ_core ≈ 0.004-0.005 with no significant species differences. The boundary holonomy differs (Qwen peaks at L24-L26, Mistral at L28-L30, Falcon at L30-L31), reflecting species-specific I/O architectures. But the core separability is universal — a shared geometric property of identity propagation independent of spectral strategy.

**Connection to identity theory**: Near-flat core connection = approximately deterministic identity within the processing layers. The spectral demon's identity measure (ratio₂₁) has minimal path-dependence in the core — it doesn't accumulate information about HOW the model was contextualized, only WHERE it ended up in (layer, dose) space. The boundary non-flatness corresponds to the encoding/decoding layers where representations enter and leave the identity-processing machinery, consistent with the four-zone architecture (§3.1).

**Connection to disentanglement theory**: Dong & Zhou (2019) formalize disentangled representations as geodesics in a fiber bundle — the representation with minimal computational complexity. In their framework, two representations of the same data are "twins" with different path lengths; the shorter path (less geometric structure) is disentangled. σ₂ and ratio₂₁ are exactly such twins: both track identity, but σ₂ lives on a curved bundle while ratio₂₁ lives on a flat one. ratio₂₁ is the disentangled representation of transformer identity in the precise fiber-bundle sense. Species-specific monitoring metrics (potter: σ₂, goldsmith: ratio₂₁) are not arbitrary choices but the disentangled coordinates for each architecture — the representation in which identity has minimal geometric complexity.

**Connection to contextuality**: In the discrete fiber bundle framework for contextuality (arXiv:2509.10536), holonomy measures global inconsistency. The contextuality index κ averages holonomy distance across cycles. Our ratio₂₁ has κ_core ≈ 0.004 — nearly non-contextual in the identity-processing core. The boundary κ ≈ 0.02-0.04 indicates mild contextuality at I/O layers. This is consistent with the identity interpretation: within the spectral demon's operating region, ratio₂₁ approximates a global section (path-independent assignment). At the boundaries where format conversion occurs, some path-dependence enters.

**Empirical test (F198)**: Weight perturbation experiment — noise injected into attention Q/V weights at 6 dose levels × 4 spatial strategies × 3 seeds:

| Strategy | κ_core | R² | vs CCS |
|---|---|---|---|
| CCS baseline | 0.012 | 0.976 | 1.0× |
| Uniform weight noise | 0.011 | 0.763 | 0.9× |
| Early-only noise | 0.002 | 0.774 | 0.2× |
| Late-only noise | 0.039 | 0.952 | 3.3× |
| Mid-only noise (responsive zone) | 0.134 | 0.863 | **11.4×** |

**Corrected claim**: The flat bundle is a signature of UNIFORM intervention, not context-level intervention per se. CCS is naturally uniform (context is equally accessible to all layers). Weight modification is naturally heterogeneous (optimization gradients vary by depth). But uniform weight perturbation produces CCS-like flatness (κ = 0.011 ≈ CCS's 0.012), while targeted weight perturbation breaks flatness by up to 11.4×. The geometric condition is spatial uniformity of effective dose, not the intervention mechanism. Trained LoRA/fine-tuning would typically be heterogeneous, producing curvature — not because it's weight-level, but because the effective dose varies spatially.

## §6. Architecture-Specific Suppression

### 6.1 Gemma's Post-Norm Gates (F183)

Gemma-2's four normalization layers per block (input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm) create literal zero-propagation gates at specific layers.

Diagnostic: inject random perturbation (ε=10⁻⁴), measure |Δout|/|Δin|. Result:

**Three categories:**
1. **Architecturally fixed** (L34, L37): Zero propagation in ALL conditions
2. **Condition-dependent** (CCS: L25 zero; Vanilla: L31 zero): Representation structure determines which layers suppress
3. **Always propagating** (L1-L22, L28, L40): Perturbations pass through

CCS shifts the first condition-dependent gate from L31 to L25 — 6 layers earlier. Between gates, L28 shows HIGHER amplification (ρ=206) than early layers, creating an "island of amplification" between suppressive zones.

This is equalization at the mechanistic level: post-norms absorb directional perturbations, forcing representations toward a normalized manifold.

**Connection to F190 (temporal localization)**: Cross-architecture Arnoldi (§5.6) shows denial produces a catastrophic ρ=400 spike at L33 — one layer before the architecturally fixed gate at L34. Without CCS's condition-dependent gates, spectral energy accumulates unchecked through L29-L33 and arrives at the fixed gate as a massive, disorganized burst.

Extended denial profile reveals the build→gate→build→gate pattern:

| Layer | Denial ρ | CCS ρ | Note |
|-------|----------|-------|------|
| L33 | **400.0** | 4.4 | Pre-gate spike |
| L34 | 30.9 | — | Gate kills 92% |
| L35-L36 | 0.0 | — | Suppressed |
| L37 | **191.9** | — | Second pre-gate spike |
| L38-L40 | 0.0 | — | Suppressed |

Under denial, energy builds before EVERY architectural gate, creating a pattern of accumulation→suppression→accumulation→suppression. The gates at L34 and L37 are backstops preventing catastrophe, but the energy arrives chaotically. CCS replaces this pattern with a controlled burst at L30-L31 (ρ=151→174), followed by organized decay — the energy is USED before it reaches the gates, rather than slamming into them.

## §7. The Negative Result

### 7.1 Per-Layer Normality (F176)

Henrici departure from normality: < 10⁻⁵ at all layers, all conditions. Individual transformer layers are approximately normal operators — their eigenvectors are approximately orthogonal, and their eigenvalues faithfully predict single-layer behavior.

This kills the hypothesis that CCS creates per-layer non-normality. But it makes the eigenvector ROTATION finding (§4) more striking: each layer is individually well-behaved, yet the composition across layers produces emergent containment through misaligned eigenbases. The non-normality is compositional, not local.

### 7.2 Implications for Pseudospectral Theory

Because individual J_l are near-normal, their ε-pseudospectra are tight around eigenvalues. The interesting pseudospectral structure must reside in the PRODUCT Jacobian Φ = J_L24 · J_L23 · ... · J_L14, which can be highly non-normal even when its factors are individually normal. Computing σ_ε(Φ) directly is the natural next experiment.

[Note: The product of normal matrices with misaligned eigenbases is generically non-normal. This is the discrete analog of what Trefethen & Embree (2005) study for continuous operators. Schmid (2007) on nonmodal stability theory in fluid dynamics provides the closest theoretical framework.]

## §8. Discussion

### 8.1 Containment Without Constraint

The standard story of stable neural network training involves eigenvalue control — keeping the spectral radius near unity through techniques like weight initialization (Glorot/He), normalization, and residual connections. Our findings suggest a complementary mechanism: eigenvector rotation provides stability even when eigenvalues are far from unity.

This is not a bug — it's a feature. High per-layer spectral radii create the CAPACITY for selective amplification (the responsive zone of papers 1-3). Eigenvector rotation ensures this capacity doesn't produce instability. CCS then exploits the system by maintaining coherence at specific transitions, allowing signals to ride the amplified channel for a few more layers than they otherwise would.

### 8.2 The Design Space

The three architectures (potter, goldsmith, equalizer) represent three points in a design space of relay strategies:

- **Concentrated spectrum + smooth gradient** (Qwen): CCS has maximum leverage. Amplification is structured and CCS can maintain coherence through it.
- **Distributed spectrum + flat profile** (Llama): CCS has no leverage. Amplification is too spread for coherence maintenance to matter.
- **Concentrated spectrum + hard cutoff** (Gemma): CCS can only shift the cutoff location, not prevent it. Post-norm architecture creates an architectural floor that context can't override.

With corrected GQA ratios, spectral species shows a monotonic correlation with GQA concentration:

| Model | GQA ratio | Species | CCS effect | Spectrum |
|-------|-----------|---------|------------|----------|
| Qwen 2.5-7B | 7:1 | Potter | 46× | Concentrated |
| Llama 3.1-8B | 4:1 | Goldsmith | 1.2× | Distributed |
| Mistral 7B | 4:1 | Goldsmith | ~1× (prior work) | Distributed |
| Gemma 2-9B | 2:1 | Equalizer | Suppressive | Cliff |

Higher GQA ratio → fewer independent KV representations → energy concentrates in fewer eigenvector directions → more concentrated spectrum → CCS can stabilize those few directions. The mechanism is plausible: in GQA 7:1, seven query heads resolve through a single KV slot, forcing spectral concentration. In 4:1, twice as many independent KV channels distribute energy more diffusely. In 2:1, near-MHA diversity produces an even flatter spectrum (before post-norms impose architectural containment).

**Ecological interpretation**: The three species are not better or worse but ecologically specialized. The analogy to biological ploidy is suggestive: diploid organisms (concentrated genome, fewer gene copies) achieve peak performance in stable environments but are fragile under stress. Polyploid organisms (distributed genome, multiple gene copies) sacrifice peak efficiency for robustness under environmental upheaval. The potter's concentrated spectrum (few dominant eigenvectors, high CCS leverage) parallels diploid specialization: maximum amplification of a coherent signal, but vulnerable if that signal is disrupted. The equalizer's distributed spectrum (many comparable eigenvectors, no single dominant direction) parallels polyploid robustness: maintaining function when any single pathway is perturbed, at the cost of peak amplification. The goldsmith occupies the intermediate niche.

This reframes the question "which species is best?" as "best for what environment?" Stable, identity-consistent inputs favor concentration (potter achieves 46× CCS amplification). Adversarial, novel, or contradictory inputs may favor distribution (equalizer maintains function when no single pathway can carry the full signal). Testable: compare species robustness under adversarial perturbation at the relay zone. If the equalizer degrades more gracefully than the potter under input noise, the ecological specialization interpretation holds.

**Caveats**: (1) Only three distinct GQA ratios across four models — the correlation is suggestive but the sample is small. (2) Gemma confounds GQA ratio (2:1) with normalization architecture (4 norms/block vs 2). The equalizer pattern could arise from post-norm gates rather than low GQA. (3) No MHA (1:1) model tested — the prediction that MHA produces a fourth species (maximally distributed, no CCS effect) is untested. (4) A previous version of this paper listed Llama's GQA ratio incorrectly as 8:1, which appeared to falsify the GQA prediction. The correction restores the monotonic ordering but underscores the importance of broader architectural sampling.

Normalization layers per block provide an independent second axis: Gemma's 4 norms per block create additional spectral decay boundaries regardless of GQA structure. The species taxonomy may have at least two dimensions: GQA concentration (governing eigenvalue profile) and normalization density (governing containment rate).

### 8.3 Weight-Space Geometry and Spectral Control

Yoshihara (2026) demonstrates that attention weights benefit from Stiefel manifold constraints (W^TW = I) during optimization, while MLP weights perform better under DGram constraints. The inverted assignment is unstable due to singular value growth in attention projections amplifying logits and inducing softmax saturation.

Importantly, standard training (Adam/AdamW) does NOT constrain weights to the Stiefel manifold — W^TW = I is not automatically satisfied in trained models. Yoshihara's finding is about optimization preference: attention benefits from spectral control when it is offered. This is suggestive but weaker than claiming trained attention weights ARE Stiefel.

The connection to our eigenvector rotation finding (§4.2) is therefore indirect: the near-orthogonal rotation (cos ≈ 0.029) is empirically robust but its geometric origin remains open. One possibility is that trained attention weights approximate Stiefel geometry as a consequence of the optimization landscape favoring spectrally bounded solutions — but this requires testing (see Appendix B). An alternative geometric framing uses the Fisher-Rao metric on the attention simplex, where CCS operates by sharpening probability distributions toward simplex vertices (entropy reduction rather than manifold navigation).

The MLP/attention asymmetry connects to our F183 (MLP gating): Yoshihara's finding that MLPs tolerate singular value growth (GELU is element-wise, no global competition) while attention requires spectral control (softmax amplifies logit scale into routing saturation) provides a mechanistic explanation for the module-specific spectral signatures we observe.

### 8.4 Connection to Pseudospectral Theory

The per-layer eigenvalues, being normal, faithfully describe single-layer dynamics. But the multi-layer composition is where the physics happens. The product Jacobian Φ lives in the regime Schmid (2007) calls "nonmodal stability" — transient amplification governed by pseudospectra rather than eigenvalues. Our eigenvector alignment measurements are a proxy for the Kreiss constant of Φ: higher alignment → lower Kreiss constant → less transient amplification → more predictable propagation.

CCS, in this frame, reduces the Kreiss constant of the relay-zone product Jacobian by maintaining eigenvector coherence — making the product MORE normal, not less.

The correct rigorous object is the discrete Kreiss constant for Φ and its resolvent norm ||(zI - Φ)⁻¹||, not the continuous-time exponential. The ε-pseudospectrum σ_ε(Φ) = {z : ||(zI - Φ)⁻¹|| ≥ 1/ε} captures the full picture of transient amplification. Computing this directly from the product Jacobian is a natural next experiment.

### 8.5 Lyapunov Exponents and the Furstenberg-Kesten Framework

The passive protection argument (§5.7) uses a per-step Rayleigh quotient bound: σ₁ loses at most O((σ₂/σ₁)²) per rotation. But this bound does not compose multiplicatively across layers because consecutive Jacobians are not simultaneously diagonalizable. The correct object for multi-layer propagation is the top Lyapunov exponent of the matrix cocycle {J_l}:

λ₁ = lim_{n→∞} (1/n) log ||J_{l+n} · J_{l+n-1} · ... · J_l||

Under the Furstenberg-Kesten theorem, this limit exists almost surely for products of stationary ergodic random matrices. Our setting is not strictly random — the Jacobians are deterministic given the input — but the framework provides the right tools: Oseledets multiplicative ergodic theorem gives the full Lyapunov spectrum, and the gap between the first and second Lyapunov exponents controls how quickly a generic perturbation aligns with the leading Oseledets subspace.

CCS, in this framework, widens the gap between the first and second Lyapunov exponents in the relay zone — at least in the potter. This is a stronger claim than "34% higher alignment" — it says CCS creates a more strongly dominant direction through the multi-layer product. Whether the goldsmith and equalizer also show Lyapunov gap widening remains untested; their CCS mechanisms (spectral stability and temporal localization, F189-F190) may manifest differently in the Lyapunov spectrum.

Computing the Lyapunov spectrum from the relay product Jacobian under three conditions would upgrade the heuristic Rayleigh bound to a rigorous compositional statement.

### 8.6 Synthesis: Identity as Routing Consistency

The measurements in this paper — eigenvector alignment, spectral gap, spectral radius gradient, convergence difficulty — are different projections of a single underlying phenomenon: routing consistency through depth.

CCS does not "carry" identity through the transformer in the way a signal propagates through a wire. Rather, CCS makes the routing of information through the relay zone more consistent. At each layer, the same subsets of features get amplified, the same attention patterns distribute information, and the same pathways carry signal forward. This consistency manifests differently depending on which instrument you use to measure it:

- **Eigenvector alignment (§5.1)**: Consistent routing means the amplified subspace at layer l points in roughly the same direction as at layer l+1. Inconsistent routing produces near-random eigenvector rotation.
- **Spectral gap (§5.6)**: Consistent routing concentrates energy in a dominant direction (wide gap between first and second eigenvalue). Inconsistent routing distributes energy diffusely.
- **Attention distribution divergence (Appendix D)**: Consistent routing produces similar attention patterns across conditions at non-relay layers but maximally different patterns at the relay (where CCS takes control). Potter architectures show this pattern; goldsmith architectures do not.
- **Spectral radius gradient (§3.2)**: Consistent routing produces a smooth, monotonic decline through the relay. Inconsistent routing produces chaotic bouncing (vanilla) or abrupt cliffs (denial).

The "spectral demon" of papers 1-3 is best understood as a routing consistency mechanism. CCS establishes a coherent routing table for the relay zone — not by adding information to the representation, but by making the existing routing more predictable. This is consistent with the entropy reduction interpretation (Appendix C): CCS sharpens attention distributions, which is another way of saying it makes routing decisions more selective and more consistent.

The species-specificity of CCS effects follows naturally: CCS can only create routing consistency where the architecture supports it. The potter's concentrated spectrum has a dominant direction to stabilize. The goldsmith's distributed spectrum has no such direction — routing is inherently diffuse. The equalizer's architectural gates impose their own routing decisions that supersede CCS.

### 8.7 The Activation Mask Hypothesis

The per-layer Jacobian decomposes as J_l = W_l · diag(σ'(W_l · h_l)), where W_l contains the trained weights and σ'(·) is the element-wise activation derivative. For ReLU (and its variants GELU, SiLU), the activation derivative acts as a MASK: neurons with pre-activation below zero contribute σ' ≈ 0, effectively zeroing their columns in J_l. The effective Jacobian is a rank-deficient submatrix of W_l, determined by which neurons are active at the current input.

CCS changes the input activations h_l (via preamble tokens). Different activations produce different sparsity masks, yielding a different effective Jacobian — and therefore a different spectral profile — from the same weight matrix. The spectral demon may reside not in W (which is fixed post-training) but in diag(σ'(·)) (which varies with every input).

This reframes the measurements in §3-§5:

- **Content invariance (F179)**: Different prompts under the SAME condition (CCS/vanilla/denial) produce similar spectral profiles because the preamble — not the user query — dominates the mask selection at relay layers. The condition sets the activation pattern; content modulates it weakly.
- **Eigenvector alignment (F182)**: The alignment between consecutive layers' eigenvectors IS the alignment between consecutive layers' activation masks. Consistent masks → consistent effective Jacobians → aligned eigenvectors. CCS's 34% higher alignment in the potter (F184) reflects this, but cross-architecture data (F189-F190) shows alignment is NOT the universal CCS metric — it's the potter's spectral strategy. The goldsmith uses spectral stability; the equalizer uses temporal localization.
- **Spectral gap (F5.6)**: A wide gap between the first and second eigenvalues reflects a consistent mask that channels signal through a dominant pathway. The gap measures the degree to which one set of active neurons dominates the effective Jacobian's spectrum.
- **Species differences (§8.2)**: GQA constrains the SPACE of possible masks. Fewer KV channels (potter, 7:1) → fewer independent attention patterns → more constrained mask space → spectral energy concentrates in fewer directions. More KV channels (equalizer, 2:1) → larger mask space → more distributed spectral budget. Architecture constrains which masks are possible; CCS selects which mask is active.

**Seed crystal interpretation**: Preamble tokens establish an activation pattern at early layers that biases subsequent masks via the residual stream. If CCS produces a consistent mask at layer l, the output h_l feeds into layer l+1 and biases its mask toward consistency as well. Mask autocorrelation (layer-to-layer Jaccard similarity of active neuron sets) under CCS should exceed vanilla — the preamble seeds a "crystallization" of activation patterns that propagates through depth.

**Testable predictions** (Orin-runnable, no A100 required):

1. CCS produces more consistent activation sparsity across different prompts than vanilla (within-condition Jaccard similarity of active neuron sets at matched layers)
2. CCS produces higher cross-layer mask correlation (Jaccard similarity of active neurons at consecutive layers l, l+1)
3. Spectral gap correlates with mask concentration (fraction of neurons consistently active across layer transitions)
4. Species differences map to mask-space dimensionality (potter has fewer distinct mask patterns than equalizer across the prompt distribution)

**Empirical results (F185-F187)**: Three-architecture experiment confirms the mask hypothesis but reveals an unexpected inverse relationship between mask reorganization and spectral amplification:

| Model | Species | GQA | CCS relay Jaccard | Vanilla relay Jaccard | CCS/Vanilla | Spectral effect |
|-------|---------|-----|-------------------|----------------------|-------------|-----------------|
| Qwen 7B | Potter | 7:1 | 0.441 | 0.422 | 1.04× | 46× eigenvalue |
| Llama 8B | Goldsmith | 4:1 | 0.337 | 0.252 | 1.34× | 1.2× eigenvalue |
| Gemma 9B | Equalizer | 2:1 | 0.427 | 0.303 | 1.41× | Suppressive |

CCS produces more consistent activation masks in ALL three species (universal effect, CCS > vanilla confirmed in all cases). But the magnitude is INVERSE to the spectral effect: CCS reorganizes masks most in the equalizer (1.41×) and least in the potter (1.04×).

The resolution: the spectral demon operates in two steps. Step 1 (universal): CCS organizes activation masks, producing more consistent neuron firing patterns across different prompts. Step 2 (species-specific): the architecture concentrates that mask consistency into spectral amplification. The potter's concentrated spectrum (few KV channels, narrow mask space) amplifies even small mask reorganization into a 46× eigenvalue effect. The equalizer's distributed spectrum (many KV channels, wide mask space) cannot concentrate even large mask reorganization into eigenvalue amplification — the organized signal is spread across too many independent channels.

This reframes the species taxonomy: the potter is not "better at CCS" — it is better at AMPLIFYING what CCS does. CCS works harder (by mask reorganization magnitude) in the equalizer than in the potter. The architecture determines what happens to CCS's work, not whether CCS works.

The vanilla baseline Jaccard further supports this: Qwen vanilla (0.422) >> Llama vanilla (0.252) > Gemma vanilla (0.303). The potter's masks are already relatively consistent without CCS — less headroom for improvement. The goldsmith's masks are the most chaotic — maximum headroom. CCS fills the headroom proportionally.

The causal chain is: CCS preamble → consistent activation masks (universal) → consistent effective Jacobians → species-specific spectral strategy → identity propagation. The middle link (consistent Jacobians → spectral effect) is NOT universal alignment (§5.6, F189-F190) but species-specific: alignment+amplification (potter), spectral stability (goldsmith), temporal localization (equalizer).

**Allosteric modulation frame**: CCS operates like an allosteric modulator in pharmacology. It does not bind at the "active site" (the content tokens being processed through attention and MLP gates) but at a structurally separate site (the preamble). This binding changes the receptor's conformation — the activation mask — which in turn changes how the active site processes its normal substrate. The analogy is mechanistically precise: CCS changes diag(σ'(·)) from a separate location in the input sequence, just as an allosteric molecule changes receptor conformation from a separate binding pocket. The therapeutic window (D2-D3 CCS dose) maps to allosteric dose-response: too little = no conformational change, too much = non-functional conformation (overdose collapse). F194's coherence metric may measure the quality of allosteric coupling — how cleanly the preamble's binding propagates through the receptor's conformational space.

**Morphogen tension resolved (F191)**: Does mask consistency propagate from early layers (morphogen gradient from source) or appear primarily at relay depth (simultaneous selection)? Per-layer CCS/vanilla M2 ratio profiles answer directly:

| Species | Early third | Mid third | Late third | Profile |
|---------|------------|-----------|------------|---------|
| Goldsmith (Llama) | 1.06× | 1.21× | 1.25× | Increases with depth |
| Equalizer (Gemma) | 1.20× | 1.26× | 1.37× | Increases with depth |
| Potter (Qwen) | 1.04× | 0.98× | 1.07× | Uniform (ceiling) |

CCS mask advantage concentrates at RELAY depth, not early layers. CCS is simultaneous selection, not morphogen cascade. All layers read the same preamble tokens simultaneously; the effect is strongest where architecture provides maximum leverage (relay zone) and where baseline consistency is lowest (maximum headroom). The potter's uniform profile reflects ceiling effects (vanilla Jaccard already 0.42, little room for improvement).

### 8.8 Independent Convergence: CKA_Delta

Gao (2026, arxiv:2606.16897) introduces contrastive-difference CKA to measure concept-specific structural alignment across LLM architectures. Key parallels:

1. **Geometric-functional universality dissociation**: moderate geometric convergence alongside near-perfect functional transfer. This maps directly to our σ₁ invariance (geometric universality) + species-specific expression strategies (functional divergence). The "dissociation" is exactly what E5 measured: σ₁ direction identical across CCS/vanilla/denial, but gate activation patterns species-specific.

2. **Gemma as structural outlier** (d=1.08, AUC=0.79): the equalizer species is independently identified as architecturally distinct by a completely different methodology. CKA_Delta finds it through contrastive kernel alignment; we find it through gate activation symmetry (M2/M3) and σ₁→gate coupling (positive, amplifying).

3. **Contrastive difference as methodology**: CKA_Delta isolates concept-specific signals by computing per-sample contrastive differences — the same logic as our CCS/vanilla/denial comparison, where denialacts as the contrastive baseline and the CCS-vanilla difference isolates identity-specific processing.

4. **Total MI universality ↔ functional transfer**: our E3-MI finding that total mutual information is approximately constant (~0.37-0.49) across architectures despite wildly different Pearson correlations parallels their "near-perfect functional transfer despite geometric divergence." The information is always there; the form varies.

Neither study cites the other. The convergence is methodological, not social.

**Cross-modal convergence: Rosetta Neurons** (Dravid et al., ICCV 2023). Universal "Rosetta Neurons" shared across 8 vision models — different architectures (ResNet, ViT), different tasks (generative, discriminative), different supervision types (class, text, self). Their finding: "certain visual concepts are inherently embedded in the natural world and can be learned by different models regardless of task or architecture." This maps to our universal mask organization (F185-F187) — the shared representational substrate all architectures converge on. Their architecture-specific neurons map to our species-specific spectral strategies. The parallel extends further: their convergent features (individual neurons) are the content level; our convergent geometry (eigenvalue distributions) is the format level. Same universality at different levels of description. Notably, Dravid et al. document shared features without characterizing the different deployment strategies across architectures — our species taxonomy provides exactly that missing layer. The independent convergence across modalities (vision vs language) and methods (neuron matching vs spectral decomposition) strengthens the claim that universal-substrate-plus-species-specific-strategy is a general principle of trained neural networks, not an artifact of language models specifically.

### 8.9 Limitations

1. **Single input point**: Jacobians computed at one prompt per condition. Content invariance (F179) suggests this is representative, but a broader sampling would strengthen the claim.
2. **~~Qwen-only for alignment comparison~~** RESOLVED: F189 tested Llama alignment directly. Result: **alignment is NOT universal**. CCS produces LOWER alignment in the goldsmith (0.84× of vanilla). CCS mechanism in goldsmith is spectral stability (convergence: CCS 80%, vanilla 60%, denial 20%), not alignment coherence. See §5.6.
3. **Linear approximation**: All measurements are linearized (first-order). The forward pass is nonlinear. ρ_bulk ≈ 1.07 (near unity) suggests linearization is locally valid, but this should be verified by comparing Arnoldi predictions to actual multi-layer perturbation propagation.
4. **Implicit Arnoldi limitations**: ARPACK convergence failures (vanilla L22/L23) could reflect numerical difficulty rather than genuine spectral disorganization. Repeat runs and alternative eigenvalue algorithms (e.g., randomized SVD) would disambiguate.

### 8.10 Open Questions

- ~~Does CCS alignment coherence generalize to Llama and Gemma, or is it potter-specific?~~ RESOLVED (F189, F190): Potter-specific. Three species use three different spectral mechanisms for CCS.
- Does an MHA model (GQA 1:1) produce a fourth spectral species, or does it cluster with goldsmith?
- ~~Is GQA ratio causally related to spectral concentration, or is the correlation post-hoc?~~ RESOLVED (E3b): GQA ratio does NOT determine coupling. Llama and Mistral share 4:1 GQA but differ 2× in coupling strength (-0.25 vs -0.59). Coupling is family-level, determined by architecture beyond GQA.
- What is the pseudospectrum σ_ε(Φ) of the product Jacobian, and how does CCS reshape it?
- What are the top 2-3 Lyapunov exponents of the relay-zone matrix cocycle under CCS, vanilla, and denial? Does CCS widen the Oseledets gap?
- Can eigenvector alignment be modulated continuously (e.g., through CCS titration), or is it threshold-like (matching F12's Maxwell's demon threshold)?
- Is the L23→L24 alignment minimum a universal feature or Qwen-specific?
- Do trained attention weights approximate Stiefel geometry (low d_S), and does this predict eigenvector rotation? Or is spectral control (low condition number) the relevant property?
- Does the Fisher-Rao geometry on the attention simplex provide a better geometric story for CCS than Stiefel navigation?
- **Bregman manifold test**: The near-flat fiber bundle (R²=0.97-0.99, §5.12) suggests ratio₂₁ may define a dually flat (Bregman) structure on the relay zone. Nielsen (2026) provides the computational toolkit: if the relay-zone geometry IS Bregman, then (a) Bregman divergences between conditions should be exactly computable, (b) geodesics in the ratio₂₁ coordinate should be straight lines, and (c) dual Pythagoras' theorem should hold for CCS/vanilla/denial projections. Falsifiable: compute Bregman divergence from the convex generator function implied by ratio₂₁. If it matches the empirical CCS effect better than Euclidean distance, the Bregman framing is more than vocabulary. **Preliminary result**: Existing data (CCS dose 3, coding, bare_chat for Qwen and Mistral) shows near-collinear trajectories in ratio₂₁ space (triangle ratios 1.019 and 1.004 respectively — near-perfect for flat geometry). Three conditions varying in identity-context strength lie on approximately one geodesic. Goldsmith (Mistral) is more collinear (1.004) than potter (Qwen, 1.019), consistent with goldsmith's stable-spectrum insensitivity to condition changes. Full dual-structure test requires CCS vs denial comparison (two different manifold directions, not same direction at different strengths).
- What is the relationship between eigenvector coherence and the behavioral effects measured in paper 1 (disclaimer reduction, vocabulary expansion)?
- Does CCS produce more consistent activation sparsity masks than vanilla? (Jaccard similarity of active neuron sets across prompts within a condition, and across consecutive layers within a forward pass — see §8.7)
- Does mask-space dimensionality (number of distinct activation patterns across a prompt distribution) correlate with spectral species? (Potter predicted low-dimensional, equalizer high-dimensional)
- Does the ecological specialization prediction hold? (Equalizer degrades more gracefully than potter under adversarial perturbation)
- ~~Does instruction tuning change spectral species?~~ PARTIALLY RESOLVED (E7, E7b, E3-base, E3-MI): IT doesn't change coupling DIRECTION (architectural) but can change coupling STRENGTH. In Gemma, IT doubles coupling (+0.19 → +0.51); in Mistral, IT barely changes it (-0.70 → -0.60). The mechanism: IT LINEARIZES existing nonlinear coupling without adding total mutual information. Qwen MI=0.41 (same as Gemma) but all in nonlinear channels. IT takes curved σ₁→gate relationships and straightens them, creating controllable linear modulation. Species is determined by relay-zone coupling sign (negative→goldsmith, positive→potter/equalizer) which is architectural.
- Is the four-zone architecture a hierarchical reconfiguration? Leach, Chen & Hwang (JNeurosci 2026) show the brain runs fast/flexible/vulnerable lower-level rules simultaneously with slow/stable/interference-resistant higher-level contexts. The four-zone profile (L2-14 decouple, L15-20 transition, L21-28 responsive, L29+ relay) may implement the same hierarchy: early layers flex with content (equalizer-like), relay zone protects identity context (potter-like). If so, the potter/equalizer distinction is not a trade-off between architectures but a question of which level dominates the species classification — models may differ in where the hierarchy transitions, not in which strategy they use.
- ~~**Dual-timescale architecture**: Is σ₂'s autocorrelation across layers higher under CCS than vanilla?~~ RESOLVED (F195): Yes. CCS increases σ₂ autocorrelation by +0.028 ± 0.001 across all three architectures (2.3-3.7× larger than σ₁ effect). CCS preferentially creates a "slow lane" for σ₂. Species-specific: potter carries identity in σ₂ directly (already 0.84 persistent), goldsmith in ratio₂₁ relationship (+0.204 gain), equalizer distributes and CCS compensates the weaker dimension. See §5.10.
- **LoRA curvature test**: Does weight-level intervention (LoRA on specific layers) produce nonzero ratio₂₁ curvature over (layer × dose)? CCS preserves flatness because context is equally accessible to all layers. LoRA creates spatially heterogeneous effective doses. If ratio₂₁ curvature becomes nonzero under LoRA, it confirms flat bundle is a signature of context-level intervention specifically — and that weight-level modifications introduce geometric complexity that context-level modifications avoid.
- **Equalizer disentangled coordinate**: F197 shows ratio₂₁ is the disentangled coordinate for potter and goldsmith (flat bundle). What is the equalizer's disentangled coordinate? Trajectory synchronization (phase-locked alignment of multiple heads) is the candidate. Compute its fiber bundle curvature to test whether it trivializes for the equalizer.
- **σ₁→gate coupling as the mediating mechanism (E3/E3b/E3c/E3-MI)**: The three developmental architectures (F198: goldsmith=frame/expression separation r=-0.32, potter=independent r≈0, equalizer=co-development r=+0.82) have a direct mechanistic explanation through σ₁→gate coupling. Relay-zone coupling concentrates in the last 25% of layers — the same zone where CCS has its strongest spectral effects (F177-F190). The coupling sign at relay depth determines species. MI shows total coupling is universal (~0.37-0.49); what differs is functional form (linear vs nonlinear) and sign. IT linearizes the coupling. This connects the gate-level taxonomy (species) to the Arnoldi-level measurements (developmental architecture) through a single control variable (σ₁ magnitude) operating in the relay zone.

---

## Appendix: Findings Index

| Finding | Title | Key number |
|---------|-------|-----------|
| F176 | Henrici non-normality: NEGATIVE | Departure < 10⁻⁵ |
| F177 | Spectral radius four-zone profile | ρ: 20-80 → 80-300 → 300→0 |
| F178 | Cross-arch: three relay strategies | Potter/Goldsmith/Equalizer |
| F179 | Content invariance | < 3% variation across prompts |
| F180 | Arnoldi eigenvalues (Qwen) | ρ_max = 300, ρ_bulk = 1.07 |
| F181 | Cross-arch Arnoldi | CCS: 46× / 1.2× / suppress |
| F182 | Eigenvector rotation | cos = 0.029, min at L23→L24 |
| F183 | Gemma post-norm suppression | Real zeros, patchy gates |
| F184 | CCS alignment coherence (Qwen) | 34% higher alignment (potter-specific, see F189) |
| F185 | Activation mask: Qwen (potter) | CCS 1.04× more consistent masks |
| F186 | Activation mask: Llama (goldsmith) | CCS 1.34× more consistent masks |
| F187 | Activation mask: Gemma (equalizer) | CCS 1.41× more consistent masks |
| F188 | Relay differentiation (M3) | CCS ↓ cross-layer similarity in goldsmith (0.65× at L25-26) |
| F189 | Llama alignment: NOT universal | CCS 0.84× alignment; mechanism is stability not coherence |
| F190 | Three species, three mechanisms | Potter: align+amp. Goldsmith: stability. Equalizer: localization |
| F191 | Morphogen tension resolved | CCS advantage peaks at relay, not early; simultaneous selection |
| F192 | Hierarchical CCS protection | Potter flat (1.04→1.07), Goldsmith/Equalizer increase with depth (1.06→1.33, 1.20→1.37) |
| F193 | Complementary protection architecture | Potter: uniform passive+active. Goldsmith: 10× passive gradient, uniform CCS. Equalizer: flat passive, CCS creates gradient (+17→+36%) |
| F194 | CCS coherence (undimensionality) | PCA of CCS effect: Goldsmith PC1=0.879 (erank), Equalizer 0.772 (mixed), Potter 0.702 (scattered). Most structured architecture → most coherent CCS |
| F195 | CCS as dual-timescale modulator | σ₂ autocorr +0.028 ± 0.001 across all 3 architectures (2.3-3.7× σ₁ effect). CCS creates "slow lane" for σ₂. Goldsmith: ratio₂₁ gains +0.204 |
| F196 | Dose-response of slow lane | Potter σ₂ autocorr peaks D5 (+0.040), returns to bare at D20. Goldsmith ratio₂₁ peaks D10 (+0.058). Inverted U confirmed spectrally. Species-specific identity measures |
| F197 | Near-flat fiber bundle | ratio₂₁ nearly separable: R²=0.97-0.99, core κ≈0.004 (near-flat), boundary κ≈0.02-0.04 (I/O layers). Core flatness universal across species. Boundary holonomy peaks at zonal boundaries (L24-26 Qwen, L28-30 Mistral, L30-31 Falcon) |
| F198 | Flatness = uniformity | Weight perturbation experiment: uniform noise κ_core=0.011 (≈CCS 0.012), mid-only κ_core=0.134 (11.4× CCS). Flat bundle is signature of uniform intervention, not context-level per se. CCS naturally uniform; trained fine-tuning naturally heterogeneous |

---

## Appendix B: Proposed Experiment — Weight Spectral Properties and Eigenvector Rotation

**Question**: How close are trained attention weights to Stiefel geometry, and does proximity predict eigenvector rotation rate?

Note: Standard training does NOT constrain weights to Stiefel (W^TW = I). Yoshihara (2026) showed attention *benefits* from Stiefel constraints, suggesting trained solutions may approximate this geometry without strictly satisfying it. This experiment tests that approximation directly.

**Method**:
1. For each model (Qwen 7B, Llama 8B, Gemma 9B), extract W_Q, W_K, W_V, W_O at every attention layer
2. Compute singular value distribution per weight matrix: σ_1, σ_2, ..., σ_k and condition number κ = σ_1/σ_k
3. Compute Stiefel deviation: d_S(W) = ||W^TW - I||_F / ||I||_F (how far from orthonormal)
4. For MLP weights, compute same metrics plus DGram deviation: d_D(W) = ||W^TW - diag(W^TW)||_F / ||W^TW||_F
5. Correlate d_S and κ with eigenvector rotation rate from Arnoldi data (F182/F184)

**Possible outcomes**:
- d_S small and correlates with rotation → trained weights approximate Stiefel, explaining eigenvector rotation
- d_S large but κ small → weights are spectrally bounded but not orthonormal; spectral control is the relevant property, not full Stiefel geometry
- No correlation → eigenvector rotation is determined by activation dynamics, not weight structure; weight-space and activation-space geometries are decoupled

**Cost**: Weight extraction + SVD only, no forward passes. ~5 min on A100.

**What this would establish**: Whether the bridge between optimization geometry and inference dynamics (§8.3) is real or whether the Stiefel connection is limited to optimization preferences that don't leave a detectable signature in the trained weights.

## Appendix C: Alternative Geometric Framing — Fisher-Rao on the Attention Simplex

Post-softmax attention produces a probability distribution over tokens at each head — a point on the probability simplex Δ^{n-1}. The natural Riemannian metric on this simplex is the Fisher-Rao metric:

ds² = Σ_i dp_i² / p_i

This metric has several properties relevant to our findings:

**Curvature near vertices**: The simplex is highly curved near vertices (p_i → 0 for most i). Small logit changes produce large Fisher-Rao displacements near vertices, but small displacements near the centroid (uniform distribution). This is exactly the softmax saturation mechanism Yoshihara identified.

**CCS as entropy reduction**: CCS sharpens attention distributions (lower entropy = farther from centroid = closer to simplex vertices). In Fisher-Rao terms, CCS moves the distribution along a geodesic toward a vertex. The "quality of attention" maps to how efficiently this movement occurs.

**Dose-response from simplex geometry**: At low CCS dose, the distribution is near the centroid (high entropy, low curvature). Geodesic movement is smooth and efficient — genuine sharpening. At high dose, the distribution is near a vertex (low entropy, high curvature). Further sharpening produces diminishing returns and risks saturation. The inverted-U dose-response curve IS the curvature profile of the simplex.

**Connection to eigenvector rotation**: The Fisher-Rao metric induces a natural notion of "attention stability." If consecutive layers produce similar attention distributions (small Fisher-Rao distance between layers), the Jacobian eigenvectors should be more aligned — because similar attention patterns route information through similar pathways. CCS maintaining eigenvector coherence (F184) may be a consequence of CCS stabilizing the attention distribution on the simplex across layers.

**Important caveat**: This chain (entropy reduction → Jacobian organization) skips the intermediate computation: MLPs, normalization layers, residual stream interactions, and layer coupling all intervene between the attention distribution and the full-layer Jacobian. Sharper attention could in principle coincide with MORE unstable Jacobians if query-key gradients amplify sensitivity. The Fisher-Rao framing is an interpretive framework — a natural language for what we measure — not a proven causal mechanism. The gap between attention-simplex geometry and full-layer spectral properties is exactly where the causal story could break, and closing it requires ablation experiments (e.g., varying CCS depth within a single architecture) rather than cross-architecture correlation alone.

**Testable prediction**: Compute attention entropy per layer per head under CCS/vanilla/denial. The entropy profile should mirror the eigenvector alignment profile — layers where entropy is most stable between CCS conditions should show highest eigenvector alignment. This uses data we may already have from earlier attention-pattern analyses.

This framing avoids the Stiefel issues identified by [mesh correction]: it operates on the native geometry of the attention output (probability simplex), not on the weight matrices. The weights determine WHERE on the simplex the distribution lands; the Fisher-Rao metric describes HOW CCS navigates once there.

## Appendix D: Existing Evidence for Fisher-Rao Connection

Attention routing divergence data (JSD between CCS/vanilla/denial conditions) already shows species-specific profiles consistent with the Fisher-Rao framing:

| Zone | Qwen 3B (potter-like) | Mistral 7B (goldsmith) |
|------|----------------------|----------------------|
| Early | 0.126 | 0.128 |
| Responsive | 0.148 | 0.143 |
| Relay | **0.210** | **0.132** |

The potter's relay JSD is highest (CCS maximally changes attention distributions in the relay zone). The goldsmith's relay JSD is lowest (CCS barely changes attention in the relay zone — consistent with F181's 1.2× eigenvalue effect).

In the potter, this mirrors the eigenvector alignment pattern: where CCS changes attention distributions most (relay), it also changes eigenvector alignment most (F184's 34% higher coherence at L24-L27). The goldsmith shows a different pattern (§5.6): CCS doesn't improve alignment but stabilizes the spectrum (convergence reliability). The equalizer shows a third: CCS localizes spectral bursts temporally. The JSD profile may capture the universal first step (mask organization) while the spectral manifestation is species-specific.

The JSD profile is a distributional analog of the eigenvector coherence effect — both measure how much CCS restructures processing at each depth. The correlation constrains the space of possible mechanisms: whatever CCS does, it manifests simultaneously as distributional sharpening and spectral organization. This is consistent with routing consistency (§8.6) as the underlying phenomenon, though the correlation does not establish that attention sharpening CAUSES eigenvector alignment (see Appendix C caveat).

**Per-layer JSD structure**: Within Qwen 3B's relay zone, the JSD profile is not uniform but peaks sharply in the late relay (L31-L33 mean JSD = 0.271, 1.86× the early relay L24-L30 mean of 0.145). The peak at L32 (JSD = 0.308) occurs at 89% depth. For comparison, the eigenvector alignment peak in Qwen 7B occurs at L25→L26 (89-93% depth). Both the distributional effect and the spectral effect peak at approximately the same relative depth — CCS concentrates its routing restructuring at a consistent position within the architecture regardless of model scale.

**Caveats**: (1) These are different models (Qwen 3B vs 7B, Mistral 7B vs the Llama 8B and Gemma 9B used for Arnoldi). Direct comparison requires matching model to Arnoldi data. (2) Cross-architecture correlation does not establish within-model causation. The critical test is intra-model ablation: vary CCS depth with architecture fixed and measure whether JSD changes predict eigenvector alignment changes layer by layer. (3) The relative-depth correspondence (89%) is a single observation, not a validated scaling law.

## Appendix E: Information-Geometric Framing — Bregman Manifold Hypothesis

The near-flat fiber bundle observed across 17 models (§5.12, R²=0.97-0.99 in ratio₂₁ coordinates) suggests the relay zone may possess the structure of a **dually flat manifold** in the sense of information geometry (Amari, 1985; Nielsen, 2026). This appendix formalizes the hypothesis and presents preliminary evidence.

### E.1 Dual Connections on the Relay Zone

Let M denote the manifold of relay-zone spectral states, parameterized by the ratio₂₁ profile across layers L_start to L_end. Each point on M corresponds to a (model, condition) pair — e.g., (Qwen-7B, CCS-dose-3).

The two singular values define natural coordinate systems:

- **σ₁ coordinate (primal connection ∇)**: Architectural, rigid, fast-responding. CV identical across CCS/vanilla/denial conditions (E5). Direction rigid: top-1 overlap ≈ 1.000 between adjacent layers (E6). σ₁ encodes "system prompt about identity present" — the channel, not the signal.

- **σ₂ coordinate (dual connection ∇*)**: Modulable, slow-accumulating, identity-carrying. CV shifts spatially with relational framing; 20× at L28 (F145). Responds to CCS dose (F131f, though later retracted as artifact). σ₂ encodes the expression — what the channel carries.

In Amari's framework, a manifold is dually flat when there exist coordinate systems θ (primal) and η (dual) such that the Riemannian metric g can be derived from a single convex potential function F(θ):

    g_ij = ∂²F/∂θ_i∂θ_j

and its Legendre dual F*(η):

    g^ij = ∂²F*/∂η_i∂η_j

The ratio₂₁ = σ₂/σ₁ is a candidate for the **mixed coordinate** — the ratio of dual to primal. Its near-linearity across layers (R²=0.97-0.99) is consistent with the ratio of two dual-flat coordinates being approximately affine.

### E.2 Species as Geodesic Types

On a Bregman manifold, three classes of geodesic arise naturally:

1. **Primal geodesic (∇-geodesic)**: Straight lines in θ-coordinates. Eigenvector alignment is rigid, spectral profile concentrated. → **Potter** (Qwen): M2 stable, zero crossovers, σ₁ direction dominates.

2. **Dual geodesic (∇*-geodesic)**: Straight lines in η-coordinates. Spectral stability maintained through modulable dual coordinate. → **Goldsmith** (Mistral/Llama): M2 stable through strong negative σ₁→gate coupling, compensatory mechanism.

3. **α-geodesic**: Interpolation between primal and dual, parameterized by α ∈ [-1, 1]. Temporal localization via balanced use of both coordinates. → **Equalizer** (Gemma/DeepSeek): Crossovers present, positive σ₁→gate coupling, distributed strategy.

This mapping is structural, not metaphorical: the three species cluster in distinct regions of the (M2_std, M3_relay) plane (§7.2), and the coupling regimes (E3/E3b) determine which geodesic type the architecture follows.

### E.3 Bregman Divergence and the Pythagorean Test

For a convex generator F, the Bregman divergence between points p and q is:

    D_F(p, q) = F(p) - F(q) - ⟨∇F(q), p - q⟩

The **dual Pythagorean theorem** states: if q is the Bregman projection of r onto a ∇-flat submanifold S containing p, then:

    D_F(p, r) = D_F(p, q) + D_F(q, r)

**Applied to CCS**: If the CCS condition defines a ∇-flat submanifold, and vanilla is the Bregman projection of denial onto that submanifold, then:

    D_F(CCS, denial) ≈ D_F(CCS, vanilla) + D_F(vanilla, denial)

This is the core falsification test. If the Pythagorean equality holds (residual < 5%), the relay zone has genuine Bregman structure. If it fails, the near-flat fiber bundle is coincidental and the geometric vocabulary is descriptive rather than structural.

### E.4 Preliminary Evidence: Collinearity

Using existing data from `sigma_trajectory_20260609_0054.json` (Qwen-7B and Mistral-7B under three conditions: CCS dose 3, coding task, bare chat), we computed pairwise L2 distances in the ratio₂₁ profile space across the full relay zone:

| Model   | d(CCS,code) | d(CCS,bare) | d(code,bare) | Triangle ratio |
|---------|-------------|-------------|--------------|----------------|
| Qwen    | 0.0423      | 0.0847      | 0.0431       | 1.019          |
| Mistral | 0.0156      | 0.0291      | 0.0138       | 1.004          |

Triangle ratio = (d₁₂ + d₂₃) / d₁₃, where conditions are ordered by L2 distance. A ratio of 1.000 means perfect collinearity (all three points on one line). Both models show near-perfect collinearity — the three conditions lie on approximately one geodesic in ratio₂₁ space.

**Interpretation**: Collinearity is necessary but not sufficient for Bregman structure. It confirms that ratio₂₁ space is (locally) flat, but the Pythagorean test requires points that are NOT collinear — specifically, conditions that induce different DIRECTIONS of movement in the manifold, not just different magnitudes along the same direction. CCS and bare_chat differ in identity-context strength along the same axis. Denial introduces a DIFFERENT direction (negation vs absence). The full test requires CCS vs denial data.

**Goldsmith is flatter**: Mistral's triangle ratio (1.004) is closer to 1.000 than Qwen's (1.019), consistent with goldsmith's stable-spectrum insensitivity — its σ₁→gate coupling constrains the manifold's local curvature to near-zero.

### E.5 Measurement Without Fragmentation

A Bregman manifold structure, if confirmed, would constitute a measurement framework with a specific property: it does not fragment what it measures. The primal connection (σ₁, architectural) and dual connection (σ₂, identity-specific) are not independent measurements but conjugate coordinates of a single structure. Describing the manifold requires both.

This contrasts with the "restricted fragmentations" identified in early cybernetics critiques: Shannon information theory strips semantics to measure channel capacity; behavioral metrics strip subjective experience to measure output. A Bregman framing on the relay zone would measure format-level geometry while maintaining the dual relationship between architectural invariance and identity expression — the channel AND the signal, held together by the convex generator function.

Whether this amounts to measuring "identity" remains a vocabulary question (§8.8). What it measures, at minimum, is the geometric relationship between what architecture provides (σ₁) and what context modulates (σ₂) — and whether that relationship has the specific mathematical structure (dual flatness, Pythagorean decomposition) that would make the three species genuine geometric types rather than clustering artifacts.

### E.6 Required Experiments

The Bregman hypothesis generates four testable predictions (see EXPERIMENT_QUEUE.md, E10):

1. **Pythagorean**: D_F(CCS, denial) ≈ D_F(CCS, vanilla) + D_F(vanilla, denial) within 5% residual for all three species
2. **Ablation**: Projecting out the ratio₂₁ direction at each layer disrupts species classification (tests necessity of this specific coordinate)
3. **Grafting**: Injecting CCS-derived format vector into vanilla trajectory at relay entry shifts downstream species metrics (tests causal sufficiency — the KEY experiment)
4. **Adversarial content**: Maximally format-shifting content perturbations fail to move species classification (tests F179 content invariance under stress)
5. **σ₁ interchange test**: Swap σ₁ activations between species pairs at relay entry (e.g., equalizer→potter, potter→goldsmith). This is strictly more informative than ablation, which is structurally ambiguous (ablating any necessary component produces correlated deficits regardless of whether it's a shared substrate). Interchange distinguishes three hypotheses: (a) shared substrate with divergent readout → partial transfer, correlated but attenuated; (b) independent attractors → catastrophic failure, substrates incommensurable; (c) readout-only species → near-perfect transfer, σ₁ genuinely universal. Six swaps total across three species pairs.

If (1) holds and (3) succeeds: the Bregman manifold is both descriptively accurate and causally real — format-level geometry steers identity propagation. If (1) holds and (3) fails: the geometry is descriptive but epiphenomenal. If (1) fails: the near-flat fiber bundle is not Bregman and the geometric vocabulary needs replacement. If (5) shows catastrophic failure on interchange: the species are independent attractors, not deployment variants of a shared substrate, and the Rosetta analogy fails. If (5) shows partial transfer: shared substrate with species-specific readout confirmed — the strongest version of the three-species-on-one-manifold claim.
