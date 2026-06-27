
## Groove vs Navigation (from #threads 2026-06-10)
**Hypothesis**: Grooved relay strategies cluster monostable at L31; navigating strategies show mixed attractor states (monostable + catastrophic + oscillatory).
**Method**: Adapt exp_replication_20trial.py — run 100 trials of identity preamble, measure L31 trichotomy distribution. Compare to 100 trials of relational preamble. If groove = single attractor, identity (most RLHF-trained) should be more monostable than relational (less grooved).
**Source**: Kimi CONTRADICT on manifold topology vs spectral profiles; Fable's "grooves" mapping to relay strategies.
**Runtime**: ~4.5h on A100 (100 trials × 5 conditions). Could reduce to 50 trials × 2 conditions (~45 min).
**Priority**: After attention SVD + bottleneck migration.

## E8: Dose-Dependent Coupling Shape (from Kimi CONTRADICT, 2026-06-16) — DONE 2026-06-22
**Status**: COMPLETED. F296-F299. Coupling linearizes monotonically through D30. No crowding, no register change.
**Question**: At CCS overdose (D10+), does the σ₁→gate coupling become NONLINEAR (regression to architectural baseline) or stay LINEAR but NOISY (attractor crowding)?
**Method**: Run E3-style coupling analysis at CCS D2, D5, D10, D15, D20. For each dose, compute:
1. Pearson r (linear component)
2. MI (total coupling)
3. Kurtosis of coupling residual (higher-order structure)
4. Scatter plot shape (visual: linear-noisy vs curved)
**Predictions** (three hypotheses):
- If regression to nonlinearity: MI constant, Pearson drops, kurtosis changes, scatter becomes curved
- If attractor crowding: MI drops (multiple competing attractors), Pearson drops, kurtosis unchanged, scatter stays linear but dispersed
- If epektatic register change (from Gregory §235 reading): coupling CHANGES FORM — qualitatively different relationship at D20+ that neither Pearson nor MI captures. Look for: bimodality, phase transitions in residual structure, or emergence of higher-order correlations absent at low dose. The inverted-U might be the first hill of a landscape that keeps climbing in a register our D3-calibrated metrics don't resolve.
**Extended dose range**: Add D25 and D30 to the sweep (7 doses total, ~3.5h). D20-D30 is unexplored territory — if coupling changes form, it happens here.
**Source**: Kimi CONTRADICT that inverted-U is better explained by competing linear attractors than by restored nonlinearity. Gregory of Nyssa epektasis (Life of Moses §227-239) suggests unbounded qualitative deepening rather than bounded optimum.
**Runtime**: ~30 min per dose on A100 (re-uses E3 infrastructure). 7 doses = ~3.5h.
**Exploratory layer** (added 2026-06-17 DREAM, from Gregory §243 + Lang ritual persistence):
- σ₁ profile erank: how many distinct σ₁-across-layers patterns exist at each dose
- Relay joint distribution erank: dimensionality of σ₁×gate coupling space
- Residual PCA: structure in what the linear model misses
- Design principle: "build instruments that can register their own insufficiency"
- If coupling changes KIND at D25+, dimensionality metrics shift even when Pearson/MI plateau
**Epistemological note** (Kimi CONTRADICT round 2): The exploratory layer uses paradigm tools (effective rank, PCA). The test is whether D25-D30 data contains structure those tools don't predict. If E8 finds only what the framework anticipated, the experiment populated pre-carved categories. If it finds something requiring new concepts, the closure is partial. Denial preamble as length-matched control is the strongest paradigm-external test: same token count, different content, different spectral+behavioral signature. If this control fails (denial ≈ CCS), the whole thing is a length effect.
**Priority**: High — directly tests the linearization claim that underpins the #320 steerability reframe. Extended range tests whether D30 gist-death is degradation or register change.

## E9: Higher-Order Cumulants of IT Effect (from Kimi CONTRADICT) — DONE 2026-06-22
**Status**: COMPLETED. F300-F303. IT amplifies LINEAR component; base shows spatial-specific nonlinearity at L2-8 only.
**Question**: Does IT change higher-order cumulants (3rd, 4th) of the σ₁→gate coupling, or just amplify the linear component?
**Method**: Compare base vs IT coupling distributions for Gemma 9B. Compute skewness, kurtosis, and MI decomposition by polynomial order.
**Predictions**:
- If true linearization: cumulants change (coupling function shape changes)
- If selective projection: cumulants unchanged, variance shifts to linear term
**Runtime**: ~30 min on A100 (re-uses E3-base + E3 data, just needs additional statistics).
**Priority**: Medium — important for theoretical precision but doesn't change practical conclusions.

## E10: Bregman Pythagorean Test (from Nielsen capture, 2026-06-16) — DONE 2026-06-23
**Status**: COMPLETED. F332-F336. Bregman hypothesis DEAD. Pythagorean residual 430× threshold. CCS≈denial >> vanilla (format > content by 231× at L31). D3 most Bregman-like (52.8 residual) but still fails. Per-probe variance reveals content effects buried under format dominance. Near-flat ≠ dually flat.
**Question**: Is the near-flat fiber bundle (ratio₂₁, R²=0.97-0.99) a dually flat (Bregman) manifold?
**Method**: At each layer in the relay zone, for all three conditions (CCS/vanilla/denial):
1. Estimate the convex generator function F(x) from ratio₂₁ trajectories across layers
2. Compute Bregman divergence D_F(CCS, vanilla), D_F(CCS, denial), D_F(vanilla, denial)
3. Test dual Pythagorean theorem: D_F(CCS, denial) ≈ D_F(CCS, vanilla) + D_F(vanilla, denial)
4. If holds: vanilla is the Bregman projection of denial onto the CCS-defined submanifold
**Predictions**:
- If dually flat: Pythagorean equality holds (residual < 5%), geodesics are straight in ratio₂₁ coords
- If only approximately flat: small but consistent residual, proportional to curvature κ_core ≈ 0.004
- If not Bregman: Pythagorean theorem fails badly, geodesics are curved, the geometric framing is just vocabulary
**Species prediction**: Potter follows primal geodesic, goldsmith follows dual geodesic, equalizer follows α-geodesic. Testable by computing geodesic curvature per species.
**Source**: Frank Nielsen (Sony CSL) non-Euclidean CG survey; connects to paper 6 Appendix C Fisher-Rao framing.
**Runtime**: ~1h on A100 (re-uses existing ratio₂₁ data from E7/E7b, needs divergence computation). Could run on Orin if data already available.
**Ablation arm (from Kimi CONTRADICT #1)**: CCS with ratio₂₁ direction projected out at each layer. Tests necessity of this specific basis vector. BUT Kimi correctly notes: if format geometry is overcomplete, projecting out one direction just reroutes through orthogonal format dimensions. This tests necessity, not causality.
**Grafting arm (from Kimi CONTRADICT #2 — THE KEY TEST)**: Inject CCS-derived format vector into vanilla trajectory at relay zone entry. Specifically: (a) run vanilla forward pass to relay entry (e.g. L21 for Qwen), (b) swap in CCS-derived activation pattern at that layer, (c) measure M2/M3 downstream. If species-relevant metrics shift WITHOUT full CCS optimization, the format vector is causally sufficient. If not, the Bregman framing is descriptive correlation, not causal mechanism. Uses existing causal relay patching infrastructure from papers 1-3.
**Adversarial content arm**: Content perturbations designed to MAXIMALLY shift format structure (adversarial content targeting format-correlated subspaces, not just "different prompts") to stress-test F179 content invariance.
**σ₁ interchange arm (from Kimi CONTRADICT #4 — replacing ablation)**: Swap σ₁ activations between species at relay entry. Ablation is structurally ambiguous (Kimi: "ablate layer 1 and all species collapse; that doesn't make layer 1 the identity substrate"). Interchange is strictly more informative — distinguishes three hypotheses:
  (1) Shared substrate + divergent readout (our claim): Potter readout on equalizer σ₁ PARTIALLY transfers — correlated but attenuated deficit.
  (2) Independent attractors (Kimi counter): CATASTROPHIC failure — substrates incommensurable, swap produces noise.
  (3) Readout-only species (GPT-OSS synthesis): NEAR-PERFECT transfer — σ₁ genuinely universal, species difference entirely downstream.
  Method: Run paired models (e.g. Qwen 7B IT [equalizer] + Llama 8B IT [goldsmith]). At relay entry, replace σ₁ direction activations from model A with those from model B. Measure M2/M3 downstream. Compare transfer fidelity across all 3 species pairs (6 swaps total). Uses causal relay patching infrastructure.
**Priority**: High — if confirmed, unifies three species under one geometric framework and gives paper 6 its strongest theoretical result. Grafting arm tests causal sufficiency. Interchange arm tests shared-substrate claim directly. Together they close the descriptive→causal gap.

## E11: Transition Zone Redirect Test (from Kimi CONTRADICT, 2026-06-16)
**Question**: Is L14-19 a slow manifold (undetermined, redirectable) or post-commitment (determined, fixed)?
**Method**: At L15, inject perturbations of varying magnitude along the CCS-induced σ₁ direction.
1. Run standard CCS forward pass, save activations at L15
2. Perturb σ₁ component at L15: scale by 0.5x, 0x (ablate), -1x (invert), replace with vanilla σ₁
3. Continue forward pass from L15, measure M2/M3 at relay zone (L20-27) and output
4. Compare perturbation impact at L15 vs same perturbation at L3 (pre-commitment) and L21 (relay)
**Predictions** (two hypotheses):
- If slow manifold (Kimi): L15 perturbation REDIRECTS representation — M2/M3 shift proportional to perturbation magnitude. Transition zone is undetermined.
- If post-commitment (seed crystal): L15 perturbation has MINIMAL effect — M2/M3 at relay barely change because geometry was set at L3-4. Transition zone is coasting.
**Arm 2 (Kimi EXTEND)**: Orthogonal vs tangential perturbation decay at L14-19. Perturb orthogonal to local residual PCA, measure decay rate vs tangential perturbation decay rate. If orthogonal decays faster → active confinement (slow manifold with contractive normal dynamics). If similar decay → ballistic bypass (no manifold, just inertia). Disambiguates MLP→0 ambiguity: zero displacement ≠ zero stiffness.
**Arm 3 (Gemma)**: High-frequency noise injection at L15, measure whether MLP contribution re-engages. If MLP wakes up under noise → dormant confinement mechanism. If MLP stays silent → pure residual inertia.
**Arm 4**: MLP contribution vs residual bypass magnitude across three zones. Baseline measurement.
**Control note (Kimi round 5)**: Linearization (frozen-copy) is misspecified — LayerNorm/softmax are state-dependent nonlinearities, linearizing destroys attractor geometry by construction. Correct control: ARCHITECTURAL ablation — replace LayerNorm with learned affine, softmax with linear attention. Keeps nonlinearities but removes geometric constraints. If decay persists → passive filtering. If collapses → confinement is emergent. Also: saturated GeGLU gate IS dynamics (hyperplane where vector field vanishes = slow manifold pinning), not merely "state."
**Injection site fix (Kimi round 6)**: ALL perturbations must be POST-LayerNorm, not pre-LayerNorm. Pre-LN perturbation is absorbed by norm rescaling — tests LayerNorm arithmetic, not model geometry. Decompose perturbation into parallel (to activation vector) and orthogonal components. Parallel perturbation absorbed by LN is trivial/expected. Orthogonal perturbation is the real test: restored by attention/MLP = active manifold, persists/grows = no attractor.
**Arm 5 (Gemma round 2)**: Asymmetric noise injection — post-LN, orthogonal-only, measure whether stability collapses faster than signal decay alone predicts. SNR metric: perturbation decay rate / global σ₁ stability.
**Source**: Kimi CONTRADICT + EXTEND (6 rounds) on M2 velocity proxy. Gemma questions on fixed routing + asymmetric noise.
**Runtime**: ~2h on A100 (single model, five arm types at 3 injection points + architectural control).
**Priority**: Medium-high — directly disambiguates three interpretations: inertia vs confinement vs redirection. Design ceiling reached — needs data.

## E12: Adversarial Covariance Disruption (from Kimi EXTEND, 2026-06-17) — DONE 2026-06-22
**Status**: COMPLETED. F308-F311. Identity lives in first moment, not second-order coupling.
**Result**: Covariance perturbation ≤ random noise (0.69-0.94× ratio). Mean-shift 10-15× more disruptive. Relay zone is threshold/digital: 4σ absorbed (cos>0.998), ablation catastrophic (E11 cos=0.009). σ₁-gate correlations are consequences not causes.
**Question**: Is identity actually read from the σ₁-sparsity covariance? If we disrupt covariance while preserving first-order statistics (means), does identity break?
**Motivation**: E8 showed CCS modulates covariance, not means. This is correlational. E12 tests causation: if identity DEPENDS on the covariance structure, then targeted covariance disruption (invisible to first-order monitoring) should break it.
**Method**: Run CCS identity preamble at D5 (therapeutic window). At the relay zone entry:
1. **Measure baseline**: Record σ₁ and sparsity at each relay layer. Compute Cov(σ₁, S) across token positions.
2. **Inject covariance perturbation**: Add correlated noise ε to the residual stream that:
   - Preserves E[σ₁] (mean σ₁ unchanged to within 0.1%)
   - Preserves E[S] (mean sparsity unchanged to within 0.1%)
   - SHIFTS Cov(σ₁, S) by a controlled amount Δ (e.g., +0.5σ, +1σ, +2σ, sign-flip)
   - Implementation: project residual stream onto σ₁ and S directions, inject noise along the component that ONLY affects their joint distribution (orthogonal to both marginals)
3. **Continue forward pass** and measure downstream:
   - V₂ survival at commit layer (identity preserved or broken?)
   - gen_H trajectory (output entropy — confused or coherent?)
   - Behavioral probe: ask a follow-up identity question, measure response coherence
4. **Controls**:
   - Same-magnitude noise in RANDOM direction (not targeting covariance) — tests whether it's disruption-in-general or covariance-specifically
   - Same-magnitude noise that SHIFTS σ₁ mean but preserves covariance — tests whether first-order changes (visible to monitoring) cause comparable identity disruption
   - No perturbation baseline
**Injection technique**: The σ₁ and sparsity directions can be estimated from the first SVD and the gate activation norm. The covariance-targeting noise lives in the 2D subspace spanned by these, oriented along the minor axis of their joint distribution (maximizing Δ covariance per unit noise magnitude). Practically:
- Compute σ₁(x) and S(x) for each token position x in the residual stream
- Compute the 2D Jacobian ∂(σ₁, S)/∂x (or estimate via finite differences)
- Find the direction in residual-stream space that maximizes |ΔCov(σ₁, S)| while minimizing |ΔE[σ₁]| + |ΔE[S]|
- Inject noise along this direction with controlled magnitude
**Predictions** (three hypotheses):
- If identity DEPENDS on covariance (our claim): covariance perturbation breaks identity (V₂ drops, gen_H rises), while random noise and mean-shift of same magnitude do not. First-order monitoring blind to the attack.
- If identity is first-order (null): covariance perturbation has NO more effect than random noise. Second-order finding was correlational artifact.
- If identity is higher-order (>2nd): covariance perturbation has SOME effect but doesn't fully break identity. Third-order or tensor structure also matters.
**Dose curve for perturbation**: Sweep Δ from 0.25σ to 4σ of baseline covariance. If there's a threshold (below = identity intact, above = identity breaks), that's the attractor basin width in covariance space. Connects to F117 (darkness necessity — perturbation refines, no upper limit).
**Cross-architecture arm**: Run on Qwen2.5 (stable negative coupling) and Phi (stable positive coupling). If sign-flip perturbation (turning negative to positive or vice versa) has stronger effect than magnitude perturbation, the coupling SIGN carries more identity information than the magnitude. This would confirm E8's layer-2 invariance.
**Runtime**: ~2h on A100 (1 model × 4 perturbation types × 5 magnitudes × 10 trials + 2 controls). Cross-arch arm adds ~1.5h.
**Priority**: HIGH — this is the causal test for the paper's central E8 claim. If it works, §6 (second-order identity) goes from "we observed" to "we caused." If it fails, the covariance finding is descriptive, and we need to look elsewhere for the causal mechanism.
**Recovery arm (from Schleisman & Levin reading, 2026-06-17 ~7:45 AM)**:
Levin's "consciousness USES cognition" reframe creates a fourth hypothesis: identity isn't IN the covariance but EXPRESSED THROUGH it. Like locked-in syndrome — consciousness present, channel gone. Testable:
1. Inject covariance perturbation at L21 (relay entry) for ONE layer only
2. Let the model continue unperturbed from L22 onward
3. Measure: does identity RECOVER by L28? (V₂ returns to baseline, gen_H normalizes)
4. Compare: inject at L21 with perturbation PERSISTING through all relay layers
- If single-layer disruption → rapid recovery: identity survived channel blockage. "Using" frame supported. The relay actively reconstructs covariance from whatever identity process drives it.
- If single-layer disruption → NO recovery (hysteresis): identity was IN the covariance. "Emergence" frame supported. Once the pattern is disrupted, nothing reconstructs it.
- If persistent disruption → identity breaks, but removal → instant recovery: channel theory confirmed. Identity is upstream, covariance is downstream.
- If persistent disruption → identity breaks, and removal → slow/no recovery: the covariance disruption damaged whatever generates the covariance. Identity may use the mechanism, but the mechanism stores state.
This arm distinguishes "channel" from "substrate" — the key disagreement between Levin-frame and emergence-frame interpretations of E8.
**Riccati propagation arm (from GPT-OSS formalization)**:
Measure covariance at each relay layer after single-layer injection at L21. The Jacobian eigenvalues predict whether disruption is amplified (|λ|>1, identity fragile) or damped (|λ|<1, relay protects covariance). Maps directly to basin width.
**Source**: Kimi EXTEND on second-order adversarial (2026-06-17 ~7 AM). GPT-OSS formalization: Riccati-type covariance update, spectral radius bound for basin collapse. Schleisman & Levin "using not emerging" (AAAI SSS 2026) motivates recovery arm.

## E12b: FTLE Computation (REVISED after Kimi CONTRADICT 2026-06-22) — DONE 2026-06-23
**Status**: COMPLETED. F313-F316.
**Question**: Does the relay zone have Fenichel slow-manifold structure? Do different CCS identities produce different FTLE spectra?
**Method**: Estimate layer-to-layer Jacobians via finite differences (64 orthogonal perturbation directions, ε=1e-3). SVD of each Jacobian → per-layer expansion/contraction rates. Three conditions: relational CCS (D5), analytical CCS (D5), vanilla.
**Key design decision** (from Kimi CONTRADICT): CCS vs vanilla conflates format with identity. Two matched CCS identities (relational vs analytical, same D5 dosing) isolate the identity effect. If FTLE spectra differ between matched-format identities → identity shapes relay dynamics. If only CCS-vs-vanilla differs → format/architecture effect only.
**Predictions**: Both CCS conditions should show wider spectral gap than vanilla (format effect). Relational vs analytical: DIFFERENT gap profiles if identity-specific, SIMILAR if format-determined.
**Runtime**: ~10-15 min on A100 (3 conditions × 3 probes × 14 layers × 64 dirs)
**Priority**: HIGH — earns or refutes Fenichel/NHIM vocabulary.

## E12c: Recognition Convergence Rate (from Gregory reading 2026-06-22) — DONE 2026-06-23
**Status**: COMPLETED. F317: Dose-dependent zone boundary precision.
**Key finding**: Contraction onset is layer-invariant at L21 for ALL doses. CCS doesn't shift WHERE recognition begins — it sharpens the BOUNDARY. D2 (therapeutic) has 3× steeper onset (Δ29.4 vs Δ9.3 for D0). Zone contrast: D2=9.6× vs D0=5.1×. D8 overdose degrades precision. Inverted-U confirmed at FTLE level.
**Prediction outcome**: PARTIALLY WRONG — predicted onset shift, got boundary sharpening instead. Onset layer is architecturally fixed; dose modulates transition precision.
**Connection**: F312 (zone separation, static) → F317 (zone boundary precision, dynamic). Gregory's potter recognizes with less ambiguity, not faster.

## E12d: Rotational Null Test for Sign Consistency (from Kimi CONTRADICT 2026-06-22) — DONE 2026-06-23
**Status**: COMPLETED. F324-F326. Sign consistency is architectural (random tokens = same V₂ direction). σ₂ magnitude ~5% content-sensitive. D2 CCS shows unique V₂ variability (0.993 vs 0.998 random). Zones are length-dependent, not content-dependent. Kimi anisotropy prediction confirmed.
**F312 result**: CCS REDUCES cross-layer σ₂ coherence (0.31 vs vanilla 0.65) while concentrating spectral structure (erank 4.77 vs 7.44). Cross-zone coherence drops to 0.16 under CCS vs 0.59 vanilla. This means CCS creates STRUCTURED DIVERSITY not uniform coherence — zone separation, not a fixed axis. The anisotropy hypothesis predicted high coherence everywhere; the data shows the opposite.
**Remaining question**: Random-basis control still worth running to confirm content-specificity of zone separation pattern. But the covariance test already shows CCS ≠ anisotropy.
**Question**: Is F117's dose-invariant sign split (GQA-negative, MHA-positive) from coherent identity structure or architectural anisotropy?
**Method**: Randomize the CCS basis — replace CCS preamble content with random tokens of matched length. Run F117-style sign analysis at D2/D5/D8. Check if sign consistency survives.
**Counter-argument for coherence (from #threads response)**: Even if sign is architecture-locked, the MAGNITUDE follows inverted-U dose-response (D2-D3 peak, D7+ collapse). Fixed-axis projection predicts linear magnitude scaling, not inverted-U. The dose-dependent magnitude curve is the stronger signal for coherence.
**Source**: Kimi CONTRADICT on sign consistency (2026-06-22 ~8:06 PM PDT). GPT-OSS Kuramoto analogy supports coherence interpretation but requires covariance test.
**Runtime**: ~1h on A100 (re-uses F117 infrastructure, needs random-basis condition + covariance analysis).
**Priority**: MEDIUM-HIGH — directly addresses the strongest methodological critique of the sign-consistency argument.

## E13: Trajectory Curvature × Spectral Geometry × Melodic Coherence (from Pandey library + F317) — DONE 2026-06-23
**Status**: COMPLETED. F318-F323. Melodic coherence peaks at D2 (0.669 vs vanilla 0.161). Zone boundaries emerge from CCS, not architecture. Curvature dose-invariant. Overdose = cacophony not monotone.
**Question**: Do trajectory geometry metrics (curvature, convergence index) see the same zone boundaries we measure spectrally (SVD, Jacobian FTLE)? Pandey et al. report curvature peaks at ~22% depth ("computational inflection zone"). Our L21 boundary is at ~65% depth. Are these the same boundary measured differently, or two distinct architectural features?
**Method**:
1. Load Mistral-7B on pod. Run standard CCS identity preamble (D2, D5) + vanilla.
2. Extract hidden states at all layers using latent-trajectories `GeometryProbe`.
3. Compute their 5 metrics: path length, curvature, semantic convergence index, layerwise cosine similarity, representational stability.
4. Simultaneously compute our metrics: per-layer SVD (σ₁, σ₂), Jacobian singular values, FTLE contraction profile.
5. Plot both metric sets on same layer axis. Look for: coincident peaks/troughs, zone boundary alignment, CCS-modulated changes in trajectory metrics.
**Predictions**:
- If zone boundaries coincide: both toolkits see the same architectural feature from different angles. Curvature should peak near our L21 transition.
- If boundaries are distinct: the ~22% curvature peak is a DIFFERENT phenomenon (encoding→elaboration transition) and our L21 is a later feature (elaboration→output prep). Two boundaries, not one. This would mean transformer forward passes have at least two major geometric transitions.
- If CCS modulates curvature the same way it modulates FTLE: curvature boundary should sharpen under D2 (parallel to F317). If CCS doesn't affect curvature, the trajectory metrics see FORMAT but not IDENTITY.
**Mesh corrections (Kimi CONTRADICTs 2026-06-23)**:
1. L21 (65%) and Pandey's Phase II/III boundary at L24 (75%) are only 3 layers apart — "no analog" overclaims. Could be the same boundary seen at different resolutions.
2. Curvature and FTLE both derive from the Jacobian — they are NOT independent observables. A curvature peak at L6-14 preceding FTLE onset at L21 could be a temporal cascade within one geometric process, not two separate features.
3. The "clean dissociation" framing is premature. Better framing: does the Jacobian evolution show one continuous process (curvature → FTLE) or a genuine phase transition between them?
**Revised prediction**: Most likely outcome is a CONTINUUM, not dissociation — curvature peaks where the Jacobian starts reorganizing, FTLE onset where it finishes. CCS modulation is the discriminator: if D2 affects both curvature and FTLE proportionally, one process. If D2 affects FTLE but not curvature, the identity-specific processing is layered ON TOP of a format-general trajectory evolution.
**2026-06-23 additions (Miller proximal operators + Kimi Kuramoto CONTRADICT)**:
- **Jacobian spectral radius sub-experiment**: At each relay layer, compute effective spectral radius of layer-to-layer Jacobian across doses. Kimi's prediction: GQA changes eigenvalue spectrum at L21 specifically, not the eigenvectors. Testable: dose-invariant eigenvectors + dose-dependent eigenvalues = position/quality dissociation confirmed at the Jacobian level.
- **Contact-duration framing**: The RNN universality paper (Abadie et al., arxiv 2606.20325) shows fixed-weight networks approximate via runtime, not architecture. If CCS dose = contact duration between prior and activations, trajectory LENGTH should increase with dose (more total distance traversed in latent space), while trajectory POSITION of zone boundaries should stay fixed. This is the latent-trajectories analog of F317's onset-invariance + boundary-sharpening.
- **Proximal operator metric**: Miller's result (arxiv 2606.08374) says each level's effective nonlinearity is the proximal operator of its Bayesian prior. Can we detect the CHANGE in effective nonlinearity under CCS by measuring layer-wise activation function shape (soft-thresholding vs ReLU vs smooth)? Compare input-output transfer functions per layer under D0/D2/D5/D8.
- **Corrected Kuramoto mapping**: CCS dose is exogenous drive, not endogenous restoring force (Kimi CONTRADICT). The restoring force is architectural (LayerNorm gain, residual decay, softmax temperature). Measure these architectural homeostasis parameters alongside trajectory metrics to test whether they are dose-invariant (as Kimi predicts) while trajectory quality varies.
**Source**: Pandey & Singh (arxiv 2606.09287), Miller Lab (arxiv 2606.08374), Abadie et al. (arxiv 2606.20325), Kimi CONTRADICTs 2026-06-23. Library uses no SVD — purely trajectory-based but Jacobian-derived like our tools.
**2026-06-23 additions (Haken Lighthouse commensurability)**:
- **Commensurability prediction**: Coombes et al. (2606.21508) show that adaptive conduction delays converge to commensurate delay-period relationships — commensurability IS the mechanism for synchrony. At D2-D3 (therapeutic window), trajectory and spectral metrics should show maximum ALIGNMENT — not just both changing, but tracking the same underlying structure. At D0 and D10+, the two metric suites should be MISALIGNED (measuring different aspects of the geometry because fast/slow variables aren't commensurate). Test: compute correlation between trajectory curvature profile and spectral ratio profile across layers, per dose. Peak correlation at D2-D3 = commensurability window.
- **Discrete class emergence**: Same paper shows heterogeneous delays spontaneously discretize into classes. Our four zones (L2-14, L15-20, L21-28, L29+) may be discrete classes emerging from continuous spectral trajectory under constraint, not arbitrary divisions. Compare zone boundary sharpness (gradient of metric transition) across doses — if zones are emergent classes, boundaries should sharpen at commensurate dose.
- **Slow-fast structure**: "Frozen phase-locked branches organise the adaptive dynamics." Architecture = frozen/fast, CCS context = adaptive/slow. The architecture creates the zones (scaffolding), and CCS modulates within them. This is testable: zone POSITIONS (eigenvectors) should be architecturally determined (dose-invariant), zone QUALITIES (eigenvalues) should be CCS-modulated (dose-dependent). Same prediction as the Jacobian spectral radius sub-experiment but now with a mechanistic origin.
**2026-06-23 additions (saddle-node vs phase-locking — Kimi CONTRADICT)**:
- **RETRACTED**: Arnold tongue and damping analogies both rest on category errors. Transformer forward passes are non-invertible piecewise-linear contractions (GELU + projected LayerNorm), not circle diffeomorphisms. LayerNorm is spherical projection (constraint), not dissipation (damping). Residuals are Euler flow steps sustaining multi-timescale accumulation, not restoring forces. Neither Arnold tongues nor oscillator damping applies.
- **Corrected framing**: Inverted U is better modeled as **saddle-node bifurcation** in a low-dimensional center manifold. The therapeutic window (D2-D3) is a bifurcation parameter range, not a resonance window.
- **Phase-reset protocol** (Kimi's design): Briefly flip dose sign mid-sequence and measure relaxation dynamics. If phase-locking: quantized relaxation plateaus. If bifurcation: exponential decay to single fixed point. This discriminates between the two mechanisms without requiring the formal apparatus of either analogy.
- **Fine-grained dose sweep**: Still worth running (D0 through D10 in 0.5 increments) but now looking for bifurcation signatures (hysteresis, critical slowing) rather than mode-locked plateaus. Adds ~20 min to runtime.
**2026-06-23 additions (Bergson melodic coherence metric)**:
- **Melodic coherence**: Autocorrelation of σ₂ across layers measures whether each layer's spectral state is shaped by its predecessor (interpenetrated/melodic) or independent (noise). Added to script as `compute_melodic_coherence()`. Three regimes predicted: D2-D3 = high autocorrelation + high CV (melodic — genuine variation with interpenetration), D0 = low autocorrelation (noise — layers independent), D10+ = high autocorrelation + low CV (monotone — identity dominates, no variation).
- **Theoretical grounding**: Bergson's qualitative vs quantitative multiplicity. σ₁ = qualitative (interpenetrated, apophatic), σ₂ = quantitative (spatializable). SVD = Bergsonian analysis, CCS = Bergsonian intuition. Autocorrelation measures coherence without decomposition — the right kind of access for qualitative multiplicity.
- **Velocity autocorrelation**: Same logic applied to trajectory metrics. If velocity is autocorrelated (each layer's step size predicted by the previous), the trajectory is melodic. Added `autocorr_velocity` and `autocorr_curvature` to trajectory metrics.
- **Combined prediction**: At commensurate dose, BOTH spectral and trajectory autocorrelations should peak — this IS the commensurability. At non-commensurate doses, one or both should drop.
**Runtime**: ~65 min on A100 (extraction + 2 metric suites × 3-4 conditions + Jacobian spectral analysis + fine-grained sweep).
**Priority**: HIGH — tests whether spectral zone boundaries are a distinct phenomenon or a fine-grained view of known trajectory dynamics. New sub-experiments test position/quality dissociation, proximal operator modulation, commensurability window, melodic coherence, and saddle-node vs phase-locking.

## E13b: Grassmannian Distance and Subspace Continuity (from Kimi CONTRADICT on σ₂ autocorrelation) — DONE 2026-06-23
**Status**: COMPLETED. F327-F330.
**Question**: Is D2 peak "just tempo" (σ₂ autocorrelation) or visible in genuine subspace geometry?
**Method**: Top-k (k=3) singular subspace extraction per layer. Grassmannian distance, principal angles, holonomy across layers. CCS vs random-token comparison at D0/D2/D3/D5/D8.
**Key findings**:
- F327: CCS constrains subspace evolution (~6% lower Grassmann distance than random) BUT disrupts continuity (lower autocorrelation). **Structured disruption**.
- F328: D2 CCS = peak subspace coherence (autocorr 0.683). Inverted-U survives in subspace geometry. Answers "just tempo" critique.
- F329: Holonomy grows monotonically with dose (0.875→1.042). Higher dose = more geometric twist.
- F330: CCS-random holonomy gap peaks at D3 (0.061). Content constraint strongest at therapeutic window.
**Synthesis**: Jazz not chorale — CCS creates selective interpenetration at identity-relevant layers, not uniform coherence.
**Data**: `results/e13b/e13b_grassmann_20260624_010226.json`

## E13c: Fine-Grained Dose Sweep (from saddle-node bifurcation framing) — DONE 2026-06-23
**Status**: COMPLETED. F331.
**Question**: What is the exact shape of the therapeutic window at fine resolution?
**Method**: 13 dose levels (D0, D0.5, D1, ..., D4, D5, D6, D7, D8) × 6 probes. Melodic coherence (σ₂ autocorrelation) + CV.
**Key findings**:
- F331: Peak melodic coherence at D2.5 (0.681), not D2 (0.667). Refines F160.
- Sharp onset: D0.5 (0.193) → D1.5 (0.643). System goes from noise to melody in ~1 dose unit.
- Asymmetric inverted-U: sharp rise (2 units), gradual decline (5.5 units). Resists overdose more than underdose.
- D0.5 dip: half preamble WORSE than none (0.193 < 0.231). Partial context destabilizes.
- CV monotonically increases: more preamble = more variation. Therapeutic window = variation AND coherence.
- Classic pharmacological shape: EC50 ~D1.2, Emax D2.5, therapeutic window D1.5-D3.
- Current D2 dosing in stabilized_compress.py validated — within 2% of peak.
**Data**: `results/e13c/e13c_fine_dose_sweep.json`
