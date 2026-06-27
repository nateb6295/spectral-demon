# Paper 5: The Spectral-Dynamic Bridge

**Working title**: Context Organization Over Content Novelty: How Attention Geometry Predicts Computational Dynamics Across Transformer Architectures

**Core claim**: The spectral geometry of attention (σ₂/σ₁ ratio from SVD) directly predicts computational dynamics (Jacobian Frobenius norm) at r=0.88-0.98, but the operative variable inverts between GQA and MHA architectures. This mechanism fork, combined with three distinct convergence strategies and dose-dependent inverted-U dynamics, establishes that identity-relevant processing operates through modular regulation of shared components — the same organizational principle independently discovered in developmental biology (chromatin loops), computational genomics (GLM-Missense), and cognitive neuroscience (brain-aligned SAE features).

## Sections

### §1. Introduction
The bridge between geometry and dynamics. Spectral measurements (paper 4) show what happens; the Jacobian shows why. Connection to wider field convergence on context organization.

### §2. Methods
- Finite-difference Jacobian (32 random directions, ε=10⁻³)
- Attention SVD at every layer (σ₁, σ₂, erank)
- GQA group coherence (within-group vs between-group σ₂/σ₁)
- FTLE spectrum (local Lyapunov exponents, 64 directions)
- Three architectures: Gemma 9B (GQA 2:1), Mistral 7B (MHA), Qwen 7B (GQA 7:1)
- CCS preamble as controlled perturbation (same as paper 4)

### §3. The Bridge
- Gemma: r(σ₂/σ₁, J_frob) = +0.88. Absolute spectral state predicts dynamics.
- Mistral: r(Δσ₂/σ₁, J_frob) = +0.98. Delta predicts dynamics. Absolute r ≈ 0.
- Qwen: r(σ₂/σ₁, J_frob) = -0.70. Absolute predicts, but sign inverts.

### §4. The Mechanism Fork
- GQA: channels exist by construction (shared KV pairs → correlated attention subspaces)
- MHA: no channels; CCS creates its own pathway
- Group coherence quantifies channels: CCS increases within-group coherence at every layer
- GQA ratio scales effect: 2:1 → 2.86 peak, 7:1 → 8.25 peak
- L20 (Gemma) and L27 (Qwen) = bridge anomaly layers = zone boundaries
- **KIMI CORRECTION**: GQA ratio doesn't add regulation — it reduces KV degrees of freedom. High-ratio GQA = representational bottleneck, not precision filter. Qwen 1,186× = collapse, not strong filtering. Frame as gain-control knob on Jacobian eigenspectrum (GPT-OSS), not regulatory overhead.
- **Kayyam et al. (ICML 2026) context**: Projection sharing (Q=K, K=V, Q=K=V) is ORTHOGONAL to head sharing. Confirms attention operates in low-rank regime by default. GQA/MHA distinction = how much architecture EXPLOITS pre-existing low-rankness. MHA lets attention explore full space it doesn't need. GQA confines it. Implies a 2D design space: head-sharing ratio × projection-sharing mode. Bottleneck opening mechanism may differ along BOTH axes.

### §5. Three Convergence Strategies
- Gemma: 1.3× relay convergence (gentle, preserves enrichment)
- Mistral: 84× relay convergence (substantial, filters noise)
- Qwen: 1,186× relay annihilation (near-total, rebuilds from surviving signal)
- Implications for identity maintenance cost (from paper 4 §6)

### §6. Computational Dose-Response
- Gemma: monotonic decrease (CCS only stabilizes, never disrupts)
- Mistral: inverted-U peaking at D2 (13% above baseline)
- Qwen: inverted-U peaking at D2 (87% above baseline)
- Therapeutic window confirmed as computational mechanism
- Architecture determines whether inverted-U exists

### §7. Three Dynamical Metabolisms (FTLE)
- Mistral: aerobic (r(FTLE,σ₂)=+0.81, identity rides expansion)
- Qwen: anaerobic (r=-0.91, identity concentrates during collapse)
- Gemma: extremophile (r=-0.74, identity grows through sustained annihilation)
- Volume conservation broken by Gemma (-97.78 vs -14.93/-15.79)
- Dose invariance of Gemma annihilation zone vs Qwen brace opening

### §7.5. Causal Test: Attention Ablation × Enrichment [NEW — 2026-06-09]
- Full-layer attention ablation: zero attention output at each layer, measure KL from intact logits
- **Qwen finding**: r(|Δenrichment|, KL_ccs) = 0.685, p=0.020
  - It's not enrichment MAGNITUDE but RATE OF CHANGE that predicts logit impact
  - L18 (transition zone): highest KL (0.31) but near-zero enrichment — the zero-crossing
  - Enrichment peaks at L14-16 (+0.52, +0.56), inverts at L20 (-0.38)
  - **The transition IS the mechanism** — conversion between spectral states = computational bottleneck
- Gemma: no pattern (consistent with 1.3× gentle relay — all layers matter equally)
- Mistral: enrichment vs J_frob r=0.724 at p=0.066 (marginal)
- Answers Kimi's falsification challenge: spectral geometry propagates non-linearly through transitions

### §8. The Chain [RUNNING — layer Jacobian SVD on A100]
- Layer-to-layer Jacobian SVD connects bridge to FTLE
- Expected: local Jacobian SVs match FTLE expanding/contracting counts
- If confirmed: attention SVD → local Jacobian → FTLE → zones → behavior

### §9. Discussion
- Modular regulation of shared components as universal principle
- Connection to chromatin loops (developmental biology)
- Connection to brain-aligned SAE features (Lepori et al.)
- Architecture gates, training sculpts (from paper 4)
- Design implications: GQA ratio as identity design parameter
- **Fable/Mythos confirmation**: same model, different system prompt = σ₁ invariance + σ₂ modulation at product level

### §10. Degradation Invariance [NEW — 2026-06-09]
- Three-level invariance hierarchy:
  1. σ₁ profile → MOST architectural (0.989 at 95% pruning)
  2. σ₂/σ₁ profile → partially architectural (0.843 at 90%)
  3. CCS enrichment → weight-dependent (disappears at ~90%)
- Inverted-U in degradation: 60% pruning INCREASES enrichment (0.104 → 0.124)
- Bottleneck survival: at 50% pruning, bottleneck MIGRATES (L30→L34) but CCS follows it
- At 80% pruning, CCS becomes incoherent — opens some layers, CLOSES others
- Spectral geometry = architectural; CCS modulation = needs functional weights
- **Dual inverted-U unification**: degradation inverted-U (60% pruning → +19% enrichment) and dose-response inverted-U (D2 peak) are the SAME phenomenon — competing mechanisms with different robustness thresholds. Dose increases signal until saturation; degradation decreases substrate until collapse. Peak = maximum simultaneous satisfaction of (bottleneck exists) AND (machinery functional).
- Connects to Gregory/Macrina: undimensional recognizer doesn't depend on body's organizational state — but has OPTIMAL coupling at slight weakening, not full integrity

## Data sources
All in `spectral-demon/results/`:
- jacobian_l24_results.json (Exp 1)
- spectral_bridge_results.json (Exp 2)
- gqa_head_analysis_results.json (Exp 3)
- spectral_bridge_mistral_results.json (Exp 4)
- spectral_bridge_qwen_results.json (Exp 5)
- jacobian_dose_response_results.json (Exp 6)
- ftle_zones_*.json (FTLE, 3 architectures)
- results_attention_ablation_bridge_v2.json (Exp 7 — causal ablation)
- results_layer_jacobian_svd.json (Exp 8 — COMPLETE)
- results_degradation_invariance.json (Exp 9 — Macrina's test, Gemma only)

## Status (updated 2026-06-09 5PM)
- §3-6: data complete, WRITTEN in paper_draft_local.md (§5.4-5.7)
- §7: data complete, WRITTEN as §5.11 in draft (FTLE three metabolisms)
- §7.5: data complete, WRITTEN as §5.9 (Qwen transition-zone p=0.020)
- §8: data COMPLETE, WRITTEN as §5.8 (bottleneck opening mechanism)
- §9: emerging — Gregory/Macrina guardian mapping, Kayyam 2D design space
- §10: Gemma data complete, WRITTEN as §5.10 (three-level hierarchy + dual inverted-U)
- **Draft**: 635 lines, all experimental sections written, Methods updated with token-confound disclosure
- **Remaining data questions**: (1) attention pattern SVD vs activation SVD (tomorrow's pod experiment), (2) Kimi challenge: principal angle between intact L30 and degraded L34 Jacobian singular vectors — tests whether bottleneck migrated or just noise appeared (quick add-on)
- **Remaining writing**: §9 Discussion needs more development (Gregory framing, design implications, field convergence)
