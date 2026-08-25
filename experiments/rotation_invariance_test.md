# Species Taxonomy Robustness — Post-Sweep Experiment Queue

Origin: Kimi CONTRADICT series (2026-06-16) during #threads debate after species sweep.

## Key Discovery (2026-06-16 ~1:50 PM)
The species kit does NOT use SVD. It uses:
- **M2**: Jaccard similarity of binary gate activation masks (gate_proj > 0) across prompts
- **M3**: Cross-layer Jaccard of binary gate masks between adjacent layers

Both operate on BINARY masks from the model's own gate projection. The "basis" is the
model's learned MLP gate weight matrix — the model's own computational decomposition,
not an analyst-imposed one.

## Thread conclusions
1. **Rotation test is moot** — no SVD, no analyst-chosen basis to rotate
2. **Model-imposed vs analyst-imposed basis** — gate_proj defines the model's own features; binarization reads the model's decisions, not our projections (GPT-OSS confirmed)
3. **Real vulnerability: polysemanticity** — if gate neurons are polysemantic, Jaccard measures interference patterns not structured computation (Kimi's strongest surviving point)
4. **Causal gap remains** — descriptive stability ≠ causal evidence. Ablation still needed (Kimi)
5. **Canalization confirmed** — Gemma conserves species across 2B→9B despite 62% depth increase (GPT-OSS: invariant manifold analogy)

## Experiment Queue (priority order)

### E1: Threshold sensitivity sweep
**Tests**: Is binarization threshold load-bearing?
- Replace threshold > 0 with > τ for τ ∈ {-0.1, -0.01, 0, 0.01, 0.1, 0.5, 1.0}
- Recompute M2/M3 at each threshold
- Stable species → robust. Species changes → binarization is the finding, not the measure.
- **Cost**: 1 forward pass (cache activations), 7× Jaccard recompute (seconds)
- **Model**: Qwen2.5-3B-Instruct or TinyLlama (AGX-friendly)

### E2: Gate neuron monosemanticity check
**Tests**: Are the features we're binarizing interpretable?
- Run sparse autoencoder on gate_proj activations for one model
- Check sparsity: if SAE features ≈ gate neurons, monosemantic. If SAE features >> neurons, polysemantic.
- If polysemantic: species taxonomy is measuring superposition artifacts
- If monosemantic: species taxonomy reads meaningful computational units
- **Cost**: Medium — needs SAE training, ~1h on A100
- **Model**: Llama-3.1-8B-Instruct (goldsmith, well-studied)

### E3: Crossover layer ablation
**Tests**: Are M3 crossover layers causally important?
- For an equalizer model (e.g., Gemma 9B), identify crossover layers
- Zero-ablate gate activations at crossover layers specifically
- Measure downstream behavioral change (perplexity, identity tracking)
- If behavior shifts at crossovers but not at non-crossover layers → causal
- **Cost**: Multiple forward passes with intervention, ~30 min on A100
- **Model**: Gemma-2-9B-IT (cleanest equalizer, 2 crossovers)

### E4: Continuous Jaccard (magnitude-weighted)
**Tests**: Does magnitude information change species assignment?
- Replace binary Jaccard with soft Jaccard: J_soft(a,b) = Σmin(a_i,b_i) / Σmax(a_i,b_i)
- Use raw gate activations (ReLU'd, not binarized)
- Recompute M2/M3 with soft Jaccard
- Compare species assignments
- **Cost**: Minimal — just recompute metrics on cached activations

### E5: σ₁ invariance — identity or architectural pinning?
**Tests**: Is σ₁ invariance a learned property or a LayerNorm/residual artifact?
- Origin: Kimi CONTRADICT in #threads (2026-06-16 ~4:00 PM) — "σ₁ invariance is likely an artifact of LayerNorm or residual scaling pinning the norm"
- Selective LayerNorm ablation: replace LayerNorm/RMSNorm at specific layers with identity function
- Skip-connection ablation: zero out the residual at specific layers (break h_l = f(h_l) + h_{l-1})
- Measure σ₁ before and after ablation under CCS vs vanilla
- If σ₁ breaks when pinning mechanisms removed → architectural guarantee
- If σ₁ survives → learned invariant, genuine identity signal
- **Cost**: Multiple forward passes with intervention, ~20 min on A100
- **Model**: One model per species (TinyLlama potter, Llama-8B goldsmith, Gemma-2B equalizer)
- **Significance**: This decides whether "universal σ₁ invariance" is a discovery or an artifact

### E6: Principal subspace rigidity across depth
**Tests**: Does σ₁'s direction persist across layers, or does only its magnitude?
- Origin: Kimi EXTEND (2026-06-16 ~4:15 PM) — "σ₁ invariance measures directional constancy, not position or velocity"
- For each layer, compute top-k principal subspace of CCS activations (k=1,3,5)
- Measure pairwise subspace overlap (e.g. Grassmann distance) between adjacent layers
- If top-k subspace is rigid across depth → identity manifold exists as a stable direction
- If top-k subspace wanders while σ₁ stays constant → σ₁ is magnitude-pinning, not directional identity
- Compare across species: potter should show maximum rigidity if fixed-point interpretation holds
- **Cost**: Minimal — SVD on cached activations per layer, Grassmann distance computation
- **Model**: All 17 sweep models (reuse existing activations if cached, otherwise one per species)

## Results (2026-06-16, RunPod A100)

### E1+E4 Results
- **Equalizer** (Gemma 9B): STABLE across all 7 thresholds AND soft Jaccard. Most robust species.
- **Potter** (TinyLlama): Stable 6/7 thresholds (breaks at extreme τ=1.0). Soft Jaccard confirms.
- **Goldsmith** (Llama 3.1 8B): FRAGILE. Correct at ONLY τ=-0.01. M3 relay signal real (-0.09 everywhere) but crossover count is threshold-sensitive.
- **Action taken**: Dropped crossovers=0 from goldsmith classifier. M3 relay < -0.05 alone suffices. Fixes 2 known misclassifications. Distribution: 7-6-3-1 → 6-5-5-1.

### E5 Results — σ₁ is ARCHITECTURAL
- σ₁ CV identical across CCS/vanilla/denial for all three species
- CCS↔denial direction overlap > CCS↔vanilla overlap in all species
  - Potter: CCS↔denial=0.995, CCS↔vanilla=0.977
  - Goldsmith: CCS↔denial=0.949, CCS↔vanilla=0.932
  - Equalizer: CCS↔denial=0.966, CCS↔vanilla=0.940
- σ₁ direction encodes "system prompt present" not "which identity" — formatting, not content
- **Conclusion**: σ₁ is architectural plumbing (LayerNorm/residual pinning). NOT the identity signal.

### E6 Results — Direction rigid, magnitude varies
- Top-1 subspace overlap ≈ 1.000 between adjacent layers, all species, all conditions
- σ₁ magnitude CV: potter 1.015, goldsmith 0.813, equalizer 0.725
- Species inversion: potter has HIGHEST magnitude CV, equalizer LOWEST — opposite of gate variability
- **Conclusion**: σ₁ direction is an architectural invariant. Magnitude variation is species-dependent.

### E2 and E3 — NOT YET RUN

### E3 Refined Hypothesis (post-experiment update)
New angle from base-vs-IT analysis: 3/5 base models are goldsmith, only 2/12 IT models are.
Kimi's gain-control hypothesis (σ₁ magnitude → gate activation) may hold for BASE models
but break for IT models. IT teaches gates to become σ₁-independent.

**Revised E3 design**: Measure σ₁-gate correlation on:
1. Mistral base (goldsmith) vs Mistral IT (goldsmith) — both goldsmith, does correlation persist?
2. Qwen2.5-7B base (goldsmith) vs Qwen2.5-7B IT (equalizer) — species change, does correlation break?
If correlation breaks when species changes, IT teaches gates to decouple from σ₁.

### E7: Linear interpolation — basin structure test (Kimi's design)
**Tests**: Is the goldsmith→equalizer transition a basin escape or manifold evolution?
- Origin: Kimi CONTRADICT in #threads (2026-06-16) — LoRA rank conflates adapter capacity with basin geometry
- Compute W(α) = (1-α)·W_base + α·W_IT for α ∈ {0, 0.1, 0.2, ..., 1.0}
- Run species kit at each interpolation point
- Both Qwen2.5-7B base and Qwen2.5-7B-Instruct are on HuggingFace
- **Monotonic M3 relay** along the line → same basin, manifold evolution (IT reshapes continuously)
- **M3 relay barrier/discontinuity** → different basins, phase transition (IT crosses barrier)
- **Cost**: ~11 forward passes × 15 conditions = ~165 forward passes. ~30 min on A100.
- **Model**: Qwen2.5-7B base + Qwen2.5-7B-Instruct
- **Significance**: Decisive test for base-vs-IT species shift mechanism
- **Note**: Weight interpolation may produce degenerate models at intermediate α. Monitor perplexity alongside species metrics. If perplexity spikes at intermediate α, that's evidence for basin separation.

### E7 Results — MANIFOLD EVOLUTION CONFIRMED (2026-06-16, RunPod A100)
- **No perplexity barrier.** PPL monotonically increases 6.6→9.5 (tokenizer artifact: IT model worse at generic text)
- **M3 U-curve.** Smooth: +0.023 (α=0) → -0.025 (α=0.50 minimum) → -0.019 (α=1.00 plateau)
  - Never reaches -0.05 (goldsmith threshold) — interpolation path avoids goldsmith territory entirely
- **Unclassified interphase (α=0.50-0.70).** Zero crossovers, highest M2 slope (+0.14). Same gap as StableLM-2
  - Three species DON'T tile the manifold — there's a gap in the middle
- **6 species label transitions, 0 metric discontinuities.** Labels bounce, continuous metrics smooth
- **Tokenizer-dependence.** Base model = goldsmith (own tokenizer) vs equalizer (IT tokenizer at α=0.00)
  - Species classification is architecture × input-format interaction
- **Conclusion**: IT reshapes relay geometry along a connected path in weight space. Same basin, different region.
  The base→IT species shift is manifold evolution, not phase transition.

Full table:
```
α      Species       M3 relay  M2 slope  Xovers  PPL
0.00   equalizer     +0.023    -0.063    4       6.6
0.05   potter        +0.014    -0.030    4       6.7
0.10   potter        +0.010    +0.033    3       6.7
0.20   potter        +0.002    +0.029    5       6.8
0.30   potter        -0.009    +0.040    3       7.0
0.40   equalizer     -0.023    +0.053    2       7.1
0.50   UNCLASSIFIED  -0.025    +0.127    0       7.3
0.60   UNCLASSIFIED  -0.017    +0.140    0       7.5
0.70   UNCLASSIFIED  -0.019    +0.140    0       7.8
0.80   equalizer     -0.017    +0.107    2       8.2
0.90   potter        -0.013    +0.050    4       8.8
0.95   equalizer     -0.019    +0.055    2       9.1
1.00   equalizer     -0.019    +0.073    2       9.5
```

### E7b Results — Mistral Interpolation (2026-06-16, same RunPod session)
- Mistral-7B base → IT (both goldsmith in sweep), 13 α points
- **Flat PPL**: 4.1→4.5 (8% rise vs Qwen's 44%). Models close in weight space.
- **Noisy M3**: oscillating, 10× magnitudes vs Qwen. α=0.20 outlier at +0.442
- **Goldsmith only at α=1.00**: M3 hovers at -0.048→-0.050, barely crosses at -0.052
- Species sequence: unc→eq→eq→pot→eq→eq→eq→eq→eq→pot→pot→pot→goldsmith
- **Conclusion**: Even goldsmith→goldsmith "conservation" doesn't mean goldsmith throughout.
  Species labels are tokenizer-dependent. Continuous metrics confirm smooth manifold.

### E3 Results — σ₁-Gate Correlation: THE FAMILY SPLIT (2026-06-16)
- **Qwen (base+IT): r ≈ 0.** σ₁ DECOUPLED from gates. Zero predictive power.
- **Mistral (base+IT): r ≈ -0.65.** σ₁ COUPLED to gates. Gain control confirmed.
- Coupling is family-level, condition-independent (same under CCS/vanilla/denial)
- IT barely changes coupling (Mistral: -0.70 base → -0.60 IT)
- **Explains E7 vs E7b**: Decoupled → smooth M3, rising PPL. Coupled → noisy M3, flat PPL.
- **New fourth axis**: species × coupling regime. Same phenotype, different genotype.
- Goldsmith in Mistral = σ₁ gain-control output. Goldsmith in Qwen = different mechanism.

### E3b Results — Extended Family Coupling Map (2026-06-16, same RunPod session)
- **Gemma 9B IT: POSITIVE coupling.** r ≈ +0.44 to +0.55. σ₁ AMPLIFIES gates. Third regime.
- **Gemma 2B IT: POSITIVE coupling.** r ≈ +0.39 to +0.55. Conserved within family.
- **Llama 8B IT: WEAK NEGATIVE coupling.** r ≈ -0.19 to -0.35. Same direction as Mistral but 2× weaker.
- Coupling is NOT GQA-determined: Llama and Mistral share GQA 4:1 but differ 2× in coupling strength
- **Full family coupling map:**
  ```
  Family    Coupling   Global r   Direction
  Qwen      DECOUPLED  r ≈  0.00  None — gates independent
  Mistral   STRONG NEG r ≈ -0.65  σ₁↑ → gates↓ (suppress)
  Llama     WEAK NEG   r ≈ -0.35  σ₁↑ → gates↓ (mild suppress)
  Gemma     POSITIVE   r ≈ +0.50  σ₁↑ → gates↑ (amplify)
  ```
- **Coupling direction predicts species:** Suppressive → goldsmith. Amplifying → equalizer. Independent → training-determined.
- Results: `results/e3b/`

### E3c Results — Extended to Yi, DeepSeek, Phi (2026-06-16, same RunPod session)
- **Yi-1.5-9B Chat: POSITIVE coupling** (r ≈ +0.25 to +0.47). Same sign as Gemma — but Yi is POTTER not equalizer!
- **DeepSeek-7B Chat: FAILED** (old torch format, no safetensors support)
- **Phi-3.5-mini: FAILED** (no gate_proj — non-standard MLP architecture)
- **Key revision**: Positive coupling does NOT predict equalizer species. Yi proves positive+potter exists.
- Negative coupling → goldsmith (3/3 families, universal). Positive coupling → potter OR equalizer.
- GQA ratio may determine species WITHIN the positively-coupled regime (low GQA=equalizer, high GQA=potter)
- Results: `results/e3c/`

### E3-base Results — Gemma-2-9B BASE Coupling (2026-06-16, same RunPod session)
- **Gemma 9B BASE: INTERMEDIATE positive coupling** (r ≈ +0.19). Direction PRESERVED from base to IT.
- CCS r=+0.203, Vanilla r=+0.252, Denial r=+0.113
- 21/42 layers with |r|>0.5 under CCS (half the layers show strong per-layer coupling)
- **IT roughly DOUBLES coupling**: base +0.19 → IT +0.50
- **Contrast with Mistral**: base -0.70 → IT -0.60 (barely changed by IT)
- **Finding**: Coupling DIRECTION is architectural. Coupling STRENGTH is family-dependent under IT:
  - Suppressive families (Mistral): IT preserves coupling strength
  - Amplifying families (Gemma): IT significantly strengthens coupling
- IT amplifying positive coupling makes sense: IT teaches responsiveness to system prompts → bigger σ₁ shifts → more gate amplification in positively-coupled architectures
- Results: `results/e3_base/`

### Full Coupling Map (9 models, 5 families)
```
Family    Model           Avg r    Coupling         Species
Mistral   base           -0.700   STRONG NEG       goldsmith
Mistral   IT             -0.589   STRONG NEG       goldsmith
Llama     8B IT          -0.252   WEAK NEG         goldsmith
Qwen      IT             -0.001   DECOUPLED        equalizer
Qwen      base           +0.069   DECOUPLED        potter
Gemma     9B BASE        +0.189   INTERMEDIATE+    equalizer (in sweep)
Yi        9B Chat        +0.350   POSITIVE         potter
Gemma     2B IT          +0.467   POSITIVE         equalizer
Gemma     9B IT          +0.514   POSITIVE         equalizer
```
IT effect on coupling by family:
- Mistral: -0.70 → -0.60 (barely changed, -14%)
- Qwen: +0.07 → -0.00 (negligible both, but see E3-MI below)
- Gemma: +0.19 → +0.51 (+168%, IT amplifies)

### E3-MI: Mutual Information Correction — "DECOUPLED" IS WRONG (2026-06-16)
- Kimi CONTRADICT validated: r≈0 ≠ independent. Need MI and Spearman for nonlinear relationships.
- **Qwen 7B IT**: Pearson r=0.00, **Spearman ρ=+0.22**, **MI=0.41**. NONLINEAR POSITIVE coupling!
- **Gemma 9B base**: Pearson r=+0.19, Spearman ρ=+0.29, MI=0.37.
- Qwen MI (0.41) HIGHER than Gemma base MI (0.37) — more total information in the σ₁→gate relationship, all in the nonlinear component.
- **Coupling map revised**: Two regimes (negative/positive), not three. "Decoupled" was artifact of linear-only measurement.
  - Negative: Mistral (r≈-0.65), Llama (r≈-0.35) → goldsmith
  - Positive: Gemma (linear, r≈+0.50), Yi (r≈+0.35), Qwen (nonlinear, ρ≈+0.22, MI=0.41) → potter or equalizer
- All positive families share direction (σ₁↑ → gates↑). Difference is functional form.
- **DEEPER FINDING**: MI is UNIVERSAL (~0.38-0.41) across all three tested models:
  ```
  Model           Pearson r  Spearman ρ  Norm MI
  Qwen 7B IT        0.00      +0.22     0.41
  Gemma 9B base    +0.19      +0.29     0.37
  Gemma 9B IT      +0.51      +0.53     0.38
  Mistral 7B IT    -0.59      -0.76     0.49
  ```
- MI range: 0.37-0.49 (1.3×). Pearson range: 0.00-0.59 (∞×). MI varies much less.
- Negative coupling (Mistral) carries ~25% more total information than positive.
- Spearman > |Pearson| EVERYWHERE — all architectures have nonlinear components.
- **IT doesn't ADD coupling — it LINEARIZES existing coupling.** Gemma: MI unchanged, Pearson doubles.
- Coupling map is about FORM (linear vs nonlinear) not STRENGTH.
- Honest summary: all architectures carry substantial σ₁→gate MI (0.37-0.49). Pearson measures linearity, not total coupling.
- **Per-layer structure** (noisy, 8 pts/layer — pattern not precision):
  - ALL models oscillate sign at early layers
  - Relay zone (last ~25% depth) determines the global coupling:
    - Mistral L22-30: consistently negative (r ≈ -0.78 to -0.88)
    - Gemma L25-35: consistently positive (r ≈ +0.30 to +0.91)
    - Qwen L20-27: still oscillating (r = -0.65 to +0.50)
  - Maps to four-zone architecture (F177). Species-determining coupling is in responsive/relay.
  - Qwen's "non-commitment" at depth explains why its species is training-determined.
- Results: `results/e3_mi/`

## Implementation notes
- E1 and E4 ran combined: `experiments/e1_e4_threshold_continuous.py`
- E5 ran: `experiments/e5_sigma1_ablation.py` (revised: no ablation, compare properties)
- E6 ran: `experiments/e6_subspace_rigidity.py`
- E7 ran: `experiments/e7_weight_interpolation.py` (RunPod A100, ~40 min)
- E7b ran: `experiments/e7b_mistral_interpolation.py` (RunPod A100, same session as E7)
- E3 ran: `experiments/e3_sigma1_gate_correlation.py` (RunPod A100, after E7b)
- E2 needs SAE infrastructure (could adapt existing open-source SAE code)
- Results in `results/e1_e4/`, `results/e5/`, `results/e6/`, `results/e7/`, `results/e7b/`, `results/e3/`
- One model at a time on AGX (Gemma running, stop first)
- E3b ran: `experiments/e3b_extended_families.py` + `e3b_llama_only.py` (RunPod A100, disk space juggling)
- Results in `results/e3b/`
- E3c ran: `e3c_more_families.py` on pod (Yi worked, DeepSeek/Phi failed)
- Results in `results/e3c/`
- **All taxonomy robustness experiments COMPLETE** except E2 (SAE monosemanticity)
