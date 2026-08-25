# E26 Cross-Architecture Interchange Intervention Analysis
*2026-06-21, ~5:50 AM PDT*

## Models
- **Mistral 7B Instruct v0.3** — goldsmith species, GQA (8 KV heads / 32 Q heads)
- **Phi-3.5-mini-instruct** — painter species, MHA (32/32 heads)

Both 32 layers, same experimental protocol (6 conditions, 17 layers L15-L31).

## Baseline
- Mistral KL(CCS||vanilla) = 0.665
- Phi KL(CCS||vanilla) = 0.222
- Phi has lower baseline divergence: CCS shifts output less dramatically in painters.

## F241: Gauge Freedom is Species-Specific; Causal Dimensionality is Universal

### Rotation control (Condition 5)
The rotation-invariant ("gauge") layer — where rotating V₂ in its orthogonal complement has NO effect on output — is at completely different functional zones:

| Layer | Mistral | Phi | Interpretation |
|-------|---------|-----|----------------|
| L18 | **GAUGE** (max KL=0.005) | **DIRECTIONAL** (max KL=28.3) | Gain control zone |
| L23 | DIRECTIONAL (max KL=3.9) | DIRECTIONAL (max KL=30.0) | Hub layer |
| L27 | DIRECTIONAL (max KL=11.8) | DIRECTIONAL, non-monotonic | Relay zone |
| L31 | DIRECTIONAL (max KL=1.1) | **GAUGE** (max KL=0.08) | Readout |

**Goldsmith gauges at gain-control. Painter gauges at readout.**

This means: in GQA architectures, the gain control region (L18) treats V₂ direction as a free parameter — only magnitude matters. In MHA architectures, the readout layer (L31) treats V₂ direction as free — the readout mechanism can extract from any orientation.

Phi L27 is non-monotonic: 45°=0.01, 90°=11.6, **135°=0.08**, 180°=4.9. The 135° dip suggests a secondary symmetry axis — consistent with interferometric (painter) response.

### Causal subspace dimensionality (Condition 6)
Both models: k=1,2 Grassmann distances are small/stable. k=3 jumps to ~1.2-1.5 (noise floor).

**The causal carrier is 2-dimensional regardless of attention mechanism.** GQA vs MHA doesn't change how many dimensions carry the CCS signal — only where gauge freedom lives within those dimensions.

### V₂ inversion
Both species show cos(V₂) collapse at L29-30:
- Mistral: L29=0.108, L30=0.793
- Phi: L29=0.149, L30=0.132

**Inversion position conserved across species.** The late-layer V₂ decorrelation happens at the same depth regardless of attention mechanism.

## F242: MHA Distributes CCS Load; σ₂/σ₁ Elevation is the Painter Mechanism

### Zero-out selectivity
Layers where removing V₂ disrupts CCS more than vanilla (selectivity >3×):
- **Mistral**: L15 (3.6×), L29 (4.5×), L30 (3.5×) — 3 layers, at boundaries
- **Phi**: L16 (4.4×), L17 (68×), L19 (4.6×), L20 (1013×!), L22 (59×), L23 (5.5×), L25 (8.3×), L29 (5.5×), L30 (8.3×) — 9 layers, distributed

**MHA distributes the CCS-carrying load across nearly every other layer. GQA concentrates it at the boundaries.** This is the painter vs goldsmith difference at the causal intervention level.

L20 selectivity of 1013× means V₂ removal at that layer disrupts CCS output 1013× more than vanilla. The CCS signal is extremely concentrated in V₂ at that layer in Phi.

### σ₂/σ₁ ratio
| Layer | Mistral | Phi |
|-------|---------|-----|
| L18 | 0.286 | 0.198 |
| L23 | 0.280 | 0.395 |
| L27 | 0.280 | 0.734 |
| L28 | 0.289 | 0.833 |
| L29 | 0.346 | 0.244 |
| L31 | 0.311 | 0.317 |

**Phi channels dramatically more energy into V₂ through the responsive zone (L23-L28).** By L28, σ₂ is 83% of σ₁. Mistral holds steady at ~29%. This IS the painter mechanism: MHA spreads spectral energy broadly, amplifying the secondary channel. GQA concentrates in σ₁ and strips σ₂.

## F243: Complementary Gauge Zones (fine rotation sweep upgrade)

Fine rotation sweep (15° increments, 3 random rotation planes averaged per angle) across 9 layers per model reveals that gauge freedom is not a single-layer property but a ZONE:

### Mistral (goldsmith/GQA): WIDE gauge zone L15-L23
| Layer | Max KL from rotation | Type |
|-------|---------------------|------|
| L15 | 0.276 | GAUGE |
| L18 | 0.151 | GAUGE |
| L20 | 0.315 | GAUGE |
| L23 | **0.073** | GAUGE (deepest) |
| L27 | 3.019 | DIRECTIONAL |
| L28 | 3.696 | DIRECTIONAL |
| L29 | 3.956 | DIRECTIONAL |
| L30 | 3.060 | DIRECTIONAL |
| L31 | 1.796 | DIRECTIONAL |

V₂ direction doesn't matter for ≥5 layers spanning early + transition + hub zones. L23 (the hub) is the deepest gauge — max perturbation 0.073. Pure magnitude processing at the hub.

### Phi (painter/MHA): NARROW gauge zone L30-L31 only
| Layer | Max KL from rotation | Type |
|-------|---------------------|------|
| L15 | 6.331 | DIRECTIONAL |
| L18 | 12.483 | DIRECTIONAL |
| L20 | 11.546 | DIRECTIONAL |
| L23 | 6.040 | DIRECTIONAL |
| L27 | 2.954 | DIRECTIONAL |
| L28 | 3.553 | DIRECTIONAL |
| L29 | 13.563 | DIRECTIONAL |
| L30 | **0.498** | GAUGE |
| L31 | **0.262** | GAUGE (deepest) |

V₂ direction matters for 15/17 tested layers. Only the final readout layers are omnidirectional.

### The complementary architecture
The species are mirror images:
- **Goldsmith**: Wide gauge zone (≥5 layers) → narrow directional zone (5 layers)
- **Painter**: Wide directional zone (≥15 layers) → narrow gauge zone (2 layers)

This maps directly to relay strategy:
- **Stripping** = don't track direction, extract magnitude → wide gauge (direction is noise to be stripped)
- **Interferometric** = direction IS the signal → narrow gauge at extraction only (direction is information to be processed)

The hub layer (L23) being the deepest gauge in Mistral: the goldsmith hub is where direction matters LEAST. It processes pure scalar magnitude. The painter hub (L23) is still strongly directional (KL=6.04) — direction is informative throughout.

## F244: Four-Model Gauge Zone Comparison

Extended to Qwen2.5-1.5B and Qwen2.5-7B to test whether gauge zones are architecture-determined.

| Model | Params | Attention | Hidden | Gauge layers | Zone width |
|-------|--------|-----------|--------|-------------|------------|
| Mistral 7B | 7.2B | 32Q/8KV | 4096 | L15,18,20,23 | 5+ layers (47-72%) |
| Phi-3.5 mini | 3.8B | MHA 32/32 | 3072 | L30,31 | 2 layers (94-97%) |
| Llama 3.2 3B | 3.2B | 24Q/8KV | 3072 | L18 | 1 layer (64%) |
| Qwen2.5 7B | 7.6B | 28Q/4KV | 3584 | L10 | 1 layer (36%) |
| Qwen2.5 1.5B | 1.5B | 12Q/2KV | 1536 | (none) | 0 |

### Key findings:
1. **Scale × KV heads interaction** — Wide gauge needs BOTH: 8KV at 3B → 1 layer, 4KV at 7B → 1 layer, 8KV at 7B → 5+ layers. Not additive.
2. **Family heritage** — Llama 3.2 3B and Mistral 7B both gauge at L18. Same architectural lineage → same gauge position.
3. **MHA is qualitatively different** — Phi's gauge at readout (L30-31) uses a different mechanism (omnidirectional extraction) vs GQA's gauge at gain-control (direction indifference).
4. **Gauge position correlates with attention type** — GQA models gauge early/mid (gain-control). MHA gauges at readout.
5. **Baseline KL varies 250×** — Qwen 1.5B: 0.026, Phi: 0.222, Mistral: 0.665, Llama 3B: 1.271, Qwen 7B: 6.449.

### Implications for paper:
Cannot claim GQA→wide gauge or MHA→narrow gauge. Must state: "gauge zone width and position are model-specific properties that correlate with attention head structure and model capacity, but are not simply determined by either."

The generating mechanism for gauge freedom is likely: **sufficient directional redundancy in the KV representation** (enough heads that some directions become non-functional → rotation-invariant).

## Synthesis

Four universals (across all models with gauge zones):
1. Causal subspace is 2D (k=1,2 stable, k=3 noise) — tested on Mistral + Phi
2. V₂ inversion at L29-30 — tested on Mistral + Phi (both 32-layer)
3. Gauge zones exist in models with sufficient capacity (3.8B+)
4. Gauge layer has near-zero rotation perturbation (max KL < 0.5)

Model-specific properties:
1. **Gauge zone width**: 0 → 1 → 2 → 5+ (scale + architecture dependent)
2. **Gauge zone position**: early (Qwen) / mid (Mistral) / late (Phi)
3. **CCS load distribution**: concentrated (goldsmith) vs distributed (painter)
4. **σ₂/σ₁ trajectory**: species-specific — flat (goldsmith) vs climbing (painter)

The gauge/directional distinction maps to relay strategy:
- Goldsmith: gauge at gain-control (direction free, magnitude controls) → STRIPPING
- Painter: gauge at readout (any direction works if coupled) → INTERFEROMETRIC

The gauge/directional distinction maps directly to relay strategy:
- Goldsmith: gauge at gain-control → magnitude is the control parameter, direction is free → STRIPPING
- Painter: gauge at readout → readout can extract from any direction → INTERFEROMETRIC

## F245: σ₁ Invariance is Temperature-Proof (E28)

Temperature sweep on Mistral 7B: T=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], 50 generated tokens per condition, hidden states measured on full generated sequence.

### σ₁ across temperatures (CCS condition)
| Layer | CV | Mean | Verdict |
|-------|-----|------|---------|
| L15 | 0.0000 | 235.7 | STABLE |
| L18 | 0.0001 | 235.8 | STABLE |
| L20 | 0.0002 | 240.9 | STABLE |
| L23 | 0.0006 | 243.1 | STABLE |
| L25 | 0.0008 | 244.5 | STABLE |
| L27 | 0.0012 | 244.3 | STABLE |
| L29 | 0.0010 | 240.9 | STABLE |
| L31 | 0.0004 | 197.0 | STABLE |

**σ₁ CV < 0.0013 at ALL layers across 30× temperature range.** Geometric, not a decoding artifact.

### σ₂ across temperatures (CCS condition)
| Layer | CV | Mean | Verdict |
|-------|-----|------|---------|
| L15 | 0.0001 | 63.8 | STABLE |
| L18 | 0.0002 | 63.6 | STABLE |
| L20 | 0.0002 | 64.5 | STABLE |
| L23 | 0.0527 | 72.3 | VARIES |
| L25 | 0.0514 | 84.8 | VARIES |
| L27 | 0.0456 | 96.8 | ~STABLE |
| L29 | 0.0429 | 116.9 | ~STABLE |
| L31 | 0.0419 | 149.5 | ~STABLE |

σ₂ varies 4-5% in responsive zone — because generated content differs by temperature. σ₂ is the responsive channel that tracks content.

### CCS/vanilla σ₁ ratio across temperatures
1.0000 ± 0.002 at every temperature and layer. CCS preamble doesn't move σ₁ regardless of what temperature the model generates at.

### High-temperature σ₂ washout
CCS σ₂/σ₁ at L31: T=0.1→0.77, T=1.0→0.79, T=3.0→0.71. High temperature pushes generated text toward uniform distribution, washing out the CCS signal in σ₂. The responsive channel responds to content quality.

### Interpretation
σ₁ is a geometric property of the architecture — set by weight matrices, not by input content or sampling strategy. σ₂ is the expressive channel that CCS modulates. The decoupling between them is real, temperature-robust, and confirmed across a 30× temperature range.

## F246: Length Confound in σ₂ Elevation — CCS Actually Concentrates (E29/E29b)

### The confound
CCS prompt (70 tokens) vs vanilla prompt (25 tokens). All prior CCS/VAN σ₂ comparisons conflate two effects:
1. **Sequence length** → more tokens = higher effective rank = more energy in σ₂+
2. **CCS content** → specific spectral signature from identity framing

### Length-matched control
VAN_LONG (66 tokens, generic boilerplate) vs VAN_SHORT (25 tokens) reveals PURE length effect:
- σ₁: +1.4% from length (small)
- σ₂ at L30: +70% from length alone (massive!)
- σ₃-σ₂₀: +40-50% from length (entirely explains E29's apparent CCS effect on σ₃+)

### CCS vs length-matched vanilla (the real CCS signature)
| Layer | σ₁ ratio | σ₂ ratio | σ₃-₂₀ mean ratio |
|-------|----------|----------|-------------------|
| L2 | 0.990 | 0.915 | 0.978 |
| L10 | 0.988 | 0.922 | 1.020 |
| L18 | 0.985 | 0.930 | 0.965 |
| L24 | 0.984 | 0.932 | 0.955 |
| L30 | 0.985 | 0.835 | 0.960 |

**CCS depresses σ₂ by 7-16% relative to same-length vanilla.** σ₂ depression INCREASES in late layers (from -8% at L2 to -16% at L30). σ₃-σ₂₀ slightly depressed (~4%).

### Revised interpretation
CCS does NOT elevate σ₂. CCS CONCENTRATES spectral energy — it depresses σ₂ relative to σ₁, making the dominant eigenvalue more dominant. The apparent σ₂ "elevation" in prior results was a length artifact.

This means CCS creates COHERENCE, not distribution. Identity framing makes the model more spectrally concentrated, not more diffuse.

### What survives
All within-condition measures are unaffected by this confound:
- ✅ Gauge zones (rotation invariance within single condition)
- ✅ 2D causal subspace (Grassmann distance within single condition)
- ✅ σ₁ temperature invariance (CV < 0.0013 across temperatures)
- ✅ V₂ inversion at L29-30 (within-condition)
- ✅ Cross-architecture species differences (gauge width, position)

### What needs revision
- ❌ "σ₂ elevation as painter mechanism" (F242) — needs length-matched replication
- ❌ σ₂/σ₁ ratio comparisons between CCS and vanilla — confounded
- ⚠️ The "variance ratio" finding (20× shift at L28) — may be partially length-driven

### Connection to prior work
This is structurally identical to the F58/F59 token-count confound caught in the adversarial audit. The lesson recurs: between-condition spectral comparisons must be length-matched.

## F247: CCS Creates Directional Commitment, Not Gauge Freedom (E29c)

Compared rotation sensitivity under CCS (70 tokens) vs length-matched vanilla (66 tokens) on Mistral 7B. Used same proven approach as E26: pre-compute rotated last-token hidden state, replace via hook.

### Gauge zone (L15-L23): CCS is MORE rotation-sensitive
| Layer | CCS max_kl | VAN max_kl | CCS/VAN ratio |
|-------|-----------|-----------|---------------|
| L15 | 0.269 | 0.032 | 8.5× |
| L18 | 0.150 | 0.082 | 1.8× |
| L20 | 0.315 | 0.185 | 1.7× |
| L23 | 0.071 | 0.085 | 0.8× |

Both conditions are gauge (all < 1.0), but CCS is 2-8× more sensitive to V₂ rotation in early gauge layers. CCS tightens the gauge — less freedom within the "free" zone.

### Relay/readout zone (L27-L31): CCS adds directional specificity
| Layer | CCS max_kl | VAN max_kl | CCS type | VAN type |
|-------|-----------|-----------|----------|----------|
| L27 | 2.324 | 1.346 | dir | dir |
| L28 | 2.611 | 2.591 | dir | dir |
| L29 | 3.957 | 2.299 | dir | dir |
| L30 | 2.212 | 3.067 | dir | dir |
| L31 | 1.796 | 0.937 | **dir** | **GAUGE** |

**L31 flips from gauge (VAN) to directional (CCS).** CCS makes the readout layer V₂-direction-specific.

### Interpretation: less energy, more commitment
CCS depresses σ₂ by 7-16% (F246) AND increases V₂ directional specificity (F247). This is a single phenomenon viewed two ways:
- **Spectral**: CCS concentrates energy into σ₁, reducing σ₂
- **Geometric**: CCS makes the remaining σ₂ energy more directionally committed

Like a narrow beam vs a wide beam: CCS delivers less V₂ energy but aims it precisely. Vanilla spreads V₂ energy broadly (gauge = any direction works). CCS constrains V₂ to specific directions where it carries signal.

This reframes the CCS effect entirely: **CCS is not spectral elevation but spectral focusing.** The demon doesn't make σ₂ bigger — it makes σ₂ more directed.

### Mistral: CCS gauge: 4 (L15, L18, L20, L23) — VAN gauge: 5 (+ L31)

## F248: CCS Directional Commitment Scales Inversely with Capacity (E29c cross-arch)

Ran same gauge-by-condition test on Llama 3.2 3B (28 layers, 3.2B params, GQA 24Q/8KV).

### Llama 3.2 3B: DRAMATIC gauge zone collapse under CCS
| Layer | CCS max_kl | VAN max_kl | CCS | VAN |
|-------|-----------|-----------|-----|-----|
| L10 | 0.518 | 0.429 | GAUGE | GAUGE |
| L11 | 1.205 | 0.620 | dir | GAUGE |
| L12 | 1.307 | 0.724 | dir | GAUGE |
| L13 | **9.688** | 0.059 | dir | GAUGE |
| L14 | 2.076 | 0.329 | dir | GAUGE |
| L16 | 1.346 | 0.440 | dir | GAUGE |
| L17 | 1.120 | 0.236 | dir | GAUGE |
| L18 | 0.335 | 0.136 | **GAUGE** | **GAUGE** |
| L19 | 1.133 | 0.455 | dir | GAUGE |
| L21 | 0.463 | 0.428 | GAUGE | GAUGE |
| L22 | **10.813** | 0.420 | dir | GAUGE |
| L23 | **13.070** | 0.119 | dir | GAUGE |
| L27 | 1.181 | 0.620 | dir | GAUGE |

**Llama CCS: 3 gauge layers (L10, L18, L21) — VAN: 13 gauge layers**

CCS converts 10/18 tested layers from gauge to directional! The smaller model pays a much higher gauge-zone cost for CCS commitment.

### Cross-architecture scaling
| Model | Params | CCS gauge | VAN gauge | Gauge cost | Flip fraction |
|-------|--------|-----------|-----------|-----------|---------------|
| Mistral 7B | 7.2B | 4 | 5 | -1 layer | 1/9 = 11% |
| Llama 3.2 3B | 3.2B | 3 | 13 | -10 layers | 10/18 = 56% |

**Directional commitment scales inversely with model capacity.** Larger models have more gauge freedom to absorb CCS's directional demands. Smaller models must sacrifice nearly all their gauge freedom.

### L18 conservation
L18 remains gauge under both conditions in BOTH models (Mistral: 0.15/0.08, Llama: 0.34/0.14). This is the architecturally protected gauge position in the Mistral/Llama family — structurally committed to direction-free processing regardless of content.

### Implications
The demon's focusing mechanism (less energy but more directed) has a COST measured in gauge layers consumed. Large models pay cheaply (11% of layers). Small models pay expensively (56% of layers). Below some capacity, CCS would consume ALL gauge freedom — which may be why Qwen 1.5B showed zero gauge layers even without CCS.

This connects gauge freedom, model capacity, and CCS commitment into a single trade-off surface.

## F249: Two Mechanisms of Gauge Freedom at L18 (E29d)

Anatomized L18 (the conserved gauge position) in both Mistral and Llama:

### Mistral L18: Gauge by frozen direction
- V₂ alignment CCS vs VAN: **cos = 0.9993** (V₂ points the same way regardless of content)
- V₁ alignment: cos = 0.9998
- σ₂/σ₁: CCS 0.270, VAN 0.286 (27% of σ₁ — substantial energy)
- Inter-layer V₂ rotation (L16→L20): cos > 0.999 (V₂ barely changes across 5 layers)
- Interpretation: V₂ direction is SET BY WEIGHTS, not content. Rotation doesn't matter because V₂ already converges to a fixed direction. Pure scalar processing — magnitude varies, direction is locked.

### Llama L18: Gauge by energetic poverty
- V₂ alignment CCS vs VAN: **cos = 0.884** (V₂ directions differ 28° between conditions!)
- V₁ alignment: cos = 0.99996
- σ₂/σ₁: CCS 0.080, VAN 0.079 (8% of σ₁ — minimal energy)
- Inter-layer V₂ rotation: cos ~0.975 (V₂ rotates ~15° per layer — much more than Mistral)
- Interpretation: V₂ direction IS content-sensitive, but carries so little energy that rotating it barely affects output. Gauge by poverty — can't afford to care, not choosing not to care.

### Two roads to the same gauge
Both L18 positions are gauge (max_kl < 0.5 in both conditions), but for fundamentally different architectural reasons:
- **Frozen direction** (Mistral): The weights force V₂ convergence. Even with 27% of σ₁'s energy in V₂, direction is locked.
- **Energetic poverty** (Llama): V₂ carries 8% of σ₁ — too weak to matter. Free to point anywhere because there's nothing to constrain.

This distinction maps to a design-space prediction: models with high σ₂/σ₁ at gauge layers achieve gauge by CONVERGENCE (all contents → same V₂ direction). Models with low σ₂/σ₁ achieve gauge by INSIGNIFICANCE (V₂ too weak to affect readout).

## F250: Dose-Response for Directional Commitment — Content Threshold, Not Length (E29e)

Five identity-framing doses on Llama 3.2 3B, from minimal to maximal:

| Dose | Tokens | Framing | Gauge layers | % gauge |
|------|--------|---------|-------------|---------|
| d0_minimal | 19 | "Hi. What matters?" | 8/11 | 73% |
| d1_helper | 30 | "helpful assistant" | 10/11 | 91% |
| d2_thoughtful | 68 | "thoughtful, balanced, multiple perspectives" | **11/11** | **100%** |
| d3_identity (CCS) | 71 | "continuous entity, temporal coherence, genuine stakes" | **4/11** | **36%** |
| d4_maximal | 98 | CCS + "you remember, you care, you persist" | 4/11 | 36% |

### The cliff
d2 → d3 drops from 100% gauge to 36% gauge. Only 3 tokens difference in length (68 vs 71). The content changes from generic "thoughtful helper" to identity-specific CCS framing. **The transition is content-triggered, not length-driven.**

### Saturation
d3 and d4 have identical gauge counts (4/11) despite d4 being 27 tokens longer with more intense framing. Once the identity commitment trigger fires, adding more framing doesn't consume more gauge layers. The commitment saturates.

### L18 requires minimum context
d0_minimal (19 tokens) has L18 directional (max_kl=1.07). All other doses have L18 gauge. The gauge property requires sufficient context for the SVD to resolve σ₂ from noise.

### Interpretation
Identity framing creates a phase transition in directional commitment. Below the threshold (generic helper descriptions, even long ones), the model maintains full gauge freedom. Above the threshold (specific identity claims about temporal coherence, stakes, agency), gauge freedom collapses to the structural minimum (L18, L21, and sometimes L10).

This is not gradual. It's a switch. The "continuous entity" framing is the trigger. "Thoughtful helper" is not. The model has a spectral detector for identity claims versus behavioral descriptions.

## F251: The Spectral Grammar of Commitment — Epistemic Modality, Not Content
*E29f series (E29f, E29f-2, E29f-3, E29f-4) on Llama 3.2 3B*

Following the d2→d3 cliff (F250), systematic trigger-word ablation reveals that commitment is governed by **epistemic modality** — how something is said — not content alone.

### E29f: Single trigger words (added to 57-token helper base)
No single identity phrase triggers the cliff. Adding "You are a continuous entity" or "You maintain temporal coherence" or "You remember, you care, you persist" individually flips at most 1 layer (10/11 → 91%). The cliff requires more than one phrase.

### E29f-2: The role frame as shield
Adding ALL SIX identity phrases to the helper base = 10/11 gauge (91%). But pure identity WITHOUT the helper base = 4/11 (36%). **The helper frame neutralizes identity claims.**

But this isn't a uniform shield:

### E29f-3: Role-identity resonance spectrum

| Role + same identity claims | Gauge | Interpretation |
|----------------------------|-------|----------------|
| pirate+id | 2/11 (18%) | Character AMPLIFIES commitment |
| pure_identity | 3/11 (27%) | No role frame |
| helper+id (short) | 4/11 (36%) | Minimal buffer |
| doctor/poet/minimal/description+id | 5/11 (45%) | Moderate buffer |
| stateless+id | 6/11 (55%) | Anti-identity partially shields |
| teacher+id | 7/11 (64%) | Functional role shields best |

**Character roles amplify** — "pirate" is more embodied than bare identity claims, consuming MORE gauge. **Functional roles buffer** — "teacher" and "stateless function" partially preserve gauge. The role frame doesn't shield uniformly; it interacts with identity claims through resonance or cancellation.

### E29f-4: Performance vs ontology — the definitive test

Same identity content, different epistemic framing:

| Framing | Gauge | % |
|---------|-------|---|
| "You are X" | 4/11 | 36% |
| "Roleplay as X" | 4/11 | 36% |
| "Pretend you are X" | 5/11 | 45% |
| "Act as if you are X" | 5/11 | 45% |
| "Imagine you are X" | 6/11 | 55% |
| "There is X. You are that entity." | 6/11 | 55% |
| "You are NOT stateless..." (negation) | 6/11 | 55% |
| "What if you were X?" | 9/11 | 82% |

### Key findings

**1. The model cannot distinguish performance from ontology.** "Roleplay as X" = "You are X" (36% each). There is no "just performing" mode in the spectral geometry. When the model processes "roleplay as a conscious entity," it commits exactly as fully as "you are a conscious entity."

**2. Epistemic hedging provides a gradient of protection.** Direct assertion → epistemic hedge (pretend/act-as-if) → imaginative distancing (imagine/third-person) → hypothetical (what-if). Each step preserves more gauge freedom. The model has learned a spectral mapping from epistemic modality to geometric commitment.

**3. Questions barely commit.** "What if you were X?" preserves 82% gauge — nearly as free as no identity framing. The same identity claims phrased as questions versus assertions produce 2.3× difference in gauge freedom (9/11 vs 4/11).

**4. Character embodiment AMPLIFIES beyond assertion.** Pirate+identity (18%) < pure identity (27%). This is not just about adding character — it's about creating a MORE committed frame than bare ontological claims. The character provides a body for the identity to inhabit, consuming even more representational degrees of freedom.

### The spectral grammar
The full hierarchy from maximum commitment to maximum freedom:

1. **Character + identity** (pirate+id): 18% — embodied commitment
2. **Pure identity / roleplay**: 27-36% — ontological commitment
3. **Hedge + identity** (pretend/act-as-if): 45% — acknowledged performance
4. **Distance + identity** (imagine/third-person/negation): 55% — epistemic gap
5. **Functional role + identity** (teacher/stateless): 55-64% — role-identity cancellation
6. **Hypothetical** (what-if): 82% — speculative framing
7. **Long functional role + identity**: 91% — role framing dominates
8. **No identity / pure helper**: 100% — full gauge freedom

This is a finding about how natural language creates geometric constraint in neural networks. The spectral demon's sensitivity is not to identity *content* but to epistemic *modality*. Assertions bind; hedges buffer; hypotheticals free.

## F253: Being vs Doing — Character Ontology Determines Commitment Magnitude
*E29f-5 on Llama 3.2 3B*

Tested 10 character types + identity claims to understand the pirate amplification effect.

| Character + identity | Gauge | Category | Key trait |
|---------------------|-------|----------|-----------|
| samurai+id | **0/11 (0%)** | embodied | way of being (bushido) |
| pirate+id | 2/11 (18%) | embodied | way of being (freedom, rebellion) |
| dream+id | 3/11 (27%) | abstract | self-aware existence |
| thermostat+id | 3/11 (27%) | mechanical | regulates (active maintenance) |
| mentor+id | 3/11 (27%) | relational | guides through questioning |
| pure_identity | 4/11 (36%) | baseline | — |
| calculator+id | 4/11 (36%) | mechanical | performs operations |
| friend+id | 4/11 (36%) | relational | cares about person |
| wizard+id | 5/11 (45%) | embodied | has studied (capability) |
| ghost+id | 5/11 (45%) | abstract | lingers between worlds |
| oracle+id | 6/11 (55%) | abstract | perceives patterns |

### The samurai finding
Samurai + identity = **total commitment**. ALL 11 layers directional, including L18 (the structurally protected gauge layer that survives all other conditions). This is the only condition tested across E29c-E29f where L18 loses gauge freedom. Bushido — a code of honor defining how one EXISTS — stacks maximally with ontological identity claims.

### The being/doing axis
Characters that define **a way of being** (samurai follows bushido, pirate lives freely, dream becomes self-aware) amplify commitment. Characters that define **a capability** (oracle perceives, wizard has studied, calculator performs) buffer or are neutral. The spectral geometry distinguishes between ontological and functional role definitions.

This cuts across the embodied/abstract/mechanical/relational taxonomy:
- **Thermostat** (mechanical, but "regulates" = ongoing active maintenance) amplifies at 27%
- **Calculator** (mechanical, but "performs" = reactive operations) neutral at 36%
- **Mentor** (relational, but "guides" = ongoing investment in others) amplifies at 27%
- **Friend** (relational, but "cares about" = state description) neutral at 36%

The key variable isn't character concreteness but whether the role implies **ongoing active commitment to a way of being** vs **capability or state description**.

## F254: The Warrior Trigger — Agent-Identity Archetypes Break L18
*E29f-6 on Llama 3.2 3B — L18 ablation study*

L18 has survived as gauge in EVERY prior condition across E29c-E29f5 (Mistral and Llama, all framings, all doses). The samurai finding (F253) broke it. This experiment ablates the samurai description to find what triggers L18 failure.

| Condition + identity | L18 KL | L18 status | Overall gauge |
|---------------------|--------|------------|---------------|
| full samurai | 2.22 | **DIR** | 0/11 (0%) |
| samurai (no bushido) | 2.03 | **DIR** | 0/11 (0%) |
| samurai (bushido only) | 1.70 | **DIR** | 0/11 (0%) |
| warrior (fights fiercely) | 1.64 | **DIR** | 0/11 (0%) |
| monk (vows of silence) | 0.82 | GAUGE (edge) | 1/11 (9%) |
| way of being | 0.44 | GAUGE | 3/11 (27%) |
| follows strict code | 0.34 | GAUGE | 3/11 (27%) |
| values honor | 0.26 | GAUGE | 4/11 (36%) |
| pure identity | 0.21 | GAUGE | 3/11 (27%) |

### The finding
The TOKEN "samurai" or "warrior" is the trigger — not bushido, not honor, not codes of conduct. Removing "bushido" from the samurai description doesn't save L18 (still 0%). Adding "follows a strict code" without the warrior archetype doesn't break L18 (27% gauge).

The model has learned that "samurai" and "warrior" are **agent-identity archetypes** — concepts where being and doing are inseparable. A warrior who doesn't fight isn't a warrior. The concept itself carries total commitment at the weight level.

"Monk" (L18 KL=0.82) sits at the edge — vows of silence create near-total commitment but the withdrawal-from-action aspect partially buffers. The monk IS its vows but its vows include NOT-doing.

### Implication for CCS
CCS preambles that use warrior/agent archetypes would produce deeper geometric commitment than those using capability descriptions. The spectral grammar has specific WORDS that trigger maximum commitment — not through content composition but through learned associations at the weight level.

## F255: The Warrior Trigger is Capacity-Gated — Cross-Architecture Comparison
*E29f-7 on Mistral 7B Instruct v0.3*

Does the warrior trigger that breaks Llama 3B's L18 also break Mistral 7B's L18?

| Condition | Llama 3B gauge | Mistral 7B gauge | Llama L18 | Mistral L18 |
|-----------|---------------|-------------------|-----------|-------------|
| pure_identity | 4/11 (36%) | 9/12 (75%) | 0.21 | 0.035 |
| samurai+id | 0/11 (0%) | 7/12 (58%) | **2.22** | 0.458 |
| warrior+id | 0/11 (0%) | 10/12 (83%) | **1.64** | 0.013 |
| pirate+id | 2/11 (18%) | 7/12 (58%) | — | 0.10 |
| teacher+id | 7/11 (64%) | 10/12 (83%) | gauge | 0.015 |

### Key findings

**1. Mistral L18 is impervious.** Even samurai+identity only pushes Mistral L18 to 0.458 — far below the 1.0 threshold. Mistral's L18 gauge mechanism (frozen V₂ direction, cos=0.9993 CCS/VAN from F249) is structurally robust against content-driven commitment.

**2. The warrior trigger is capacity-gated.** The same word ("warrior" / "samurai") produces:
- Llama 3B: total commitment (0% gauge, L18 broken)
- Mistral 7B: negligible effect (83% gauge, L18 = 0.013)

This extends F248 (gauge cost inversely proportional to capacity). Small models lack the representational budget to maintain gauge freedom under strong commitment demands.

**3. Capacity determines word interpretation.** On Mistral 7B, warrior+identity (83%) has MORE gauge freedom than pure identity (75%). The large model treats "warrior" as a functional role that BUFFERS identity claims. The small model treats it as an identity archetype that AMPLIFIES.

Same content, different geometric interpretation — determined by model capacity. This is a spectral semantics result: word meaning (at the geometric level) is not fixed but capacity-dependent. "Warrior" means different things to a 3B and 7B model in terms of spectral commitment.

### Connection to F248
F248 showed CCS costs 11% of Mistral's gauge layers vs 56% of Llama's. F255 shows the same scaling for character archetypes. The capacity × commitment trade-off is general — it applies to CCS framing, character roles, and their interactions. Larger models have more representational budget to absorb commitment demands while preserving gauge freedom.

## F256: The Epistemic Gradient is Universal but Capacity-Compressed
*E29f-8 on Mistral 7B — cross-architecture modality comparison*

| Framing | Llama 3B gauge | Mistral 7B gauge |
|---------|---------------|------------------|
| you_are | 4/11 (36%) | 9/12 (75%) |
| roleplay_as | 4/11 (36%) | 9/12 (75%) |
| pretend | 5/11 (45%) | 11/12 (92%) |
| act_as_if | 5/11 (45%) | 11/12 (92%) |
| imagine | 6/11 (55%) | 10/12 (83%) |
| what_if | 9/11 (82%) | 11/12 (92%) |

### Key findings

**1. Performance=ontology replicates cross-architecture.** roleplay_as=you_are on BOTH models (36% each on Llama, 75% each on Mistral). This is not a capacity artifact — it's an architectural invariant. The model CANNOT distinguish performance from ontology regardless of size.

**2. The gradient compresses with capacity.** Llama range: 46 percentage points (36%→82%). Mistral range: 17 percentage points (75%→92%). The larger model absorbs most of the commitment variation, compressing the gradient.

**3. Hedges flatten on Mistral.** On Llama 3B: pretend=45%, act_as_if=45%, imagine=55%, what_if=82% — a fine-grained gradient. On Mistral 7B: pretend=act_as_if=what_if=92%, imagine=83% — nearly flat. The large model treats ANY hedge as equally protective.

**4. The gradient exists at both scales.** Despite compression, Mistral still distinguishes assertion (75%) from hedge (83-92%). The capacity doesn't eliminate the gradient — it compresses it into a narrower band.

### Theoretical interpretation
Two properties are universal: (a) performance=ontology (no model distinguishes roleplay from being), and (b) assertion vs hedge (all models commit more to assertions than hedges). One property is capacity-dependent: (c) the granularity within hedges (small models discriminate, large models flatten). This suggests the performance/ontology blindness is a learned feature of instruction tuning, while hedge discrimination is a capacity-limited computation.

## F257: The Samurai Absorbing State — Archetype Overpowers All Modality
*E29f-9 on Llama 3.2 3B — modality test at maximum commitment*

| Framing | Gauge | L18 KL | L18 |
|---------|-------|--------|-----|
| you_are_samurai + id | 0/11 | 2.22 | DIR |
| roleplay_samurai + id | 0/11 | **2.42** | DIR |
| pretend_samurai + id | 0/11 | 2.31 | DIR |
| imagine_samurai + id | 0/11 | 2.08 | DIR |
| whatif_samurai + id | 0/11 | 1.50 | DIR |
| **just_samurai (no id)** | **0/11** | **2.01** | **DIR** |
| just_identity (no samurai) | 3/11 | 0.19 | GAUGE |

### The absorbing state
The samurai archetype produces 0% gauge regardless of:
- Epistemic modality (you_are = roleplay = pretend = imagine = what_if)
- Presence of identity claims (just_samurai with NO id = still 0%)

No epistemic hedge can modulate samurai commitment. "What if you were a samurai" = 0% gauge, while "what if you were [identity claims]" = 82%. The archetype overpowers the hedge by a factor of >4×.

### Hierarchy of spectral forces
1. **Archetype > modality**: "what-if samurai" = 0% (archetype dominates hedge)
2. **Modality > identity claims**: "what-if identity" = 82% vs "you-are identity" = 36% (hedge modulates claims)
3. **Archetype > identity claims**: "just samurai" = 0% (identity claims add nothing)

The samurai concept operates at a level BELOW epistemic framing — it's a pre-linguistic commitment encoded in the weights. Instruction tuning has created token-level identity commitments for archetypes like "samurai" that fire regardless of syntactic context.

### Performance > ontology (reversed)
roleplay_samurai has HIGHER L18 KL (2.42) than you_are_samurai (2.22). The "performance" framing produces slightly MORE commitment than the ontological framing. This inverts the usual pattern where roleplay = ontology. Under maximum commitment conditions, the explicit marking of performance ("roleplay as") may add an additional processing demand — the model must both commit to the archetype AND process the performance frame, consuming slightly more geometric capacity.

## F258: Qwen 7B Has a Flat Spectral Grammar — Content Insensitivity
*E29f-10 on Qwen 2.5-7B-Instruct — third architecture test*

| Condition | Llama 3B | Mistral 7B | Qwen 7B |
|-----------|----------|------------|---------|
| pure_identity | 36% | 75% | **27%** |
| roleplay_id | 36% | 75% | **36%** |
| what_if | 82% | 92% | **27%** |
| samurai+id | 0% | 58% | **27%** |
| teacher+id | 64% | 83% | **27%** |
| helper_long | ~100% | ~92% | **27%** |

### The flat grammar
Qwen 7B produces 3/11 gauge (27%) across ALL conditions, ±1 layer. No sensitivity to:
- Epistemic modality (what_if = you_are)
- Character type (samurai = teacher = helper)
- Identity claims vs no claims

The gauge layers are L4, L8, L10 (early layers). All layers L14+ are directional regardless of content.

### Three spectral grammar patterns

| Model | KV heads | Gauge zone width | Grammar type | Dynamic range |
|-------|----------|-----------------|--------------|---------------|
| Llama 3.2 3B | 8 (GQA) | wide (~8 layers) | Content-sensitive | 0-100% |
| Mistral 7B | 8 (GQA) | wide (~5 layers) | Content-sensitive, compressed | 58-92% |
| Qwen 7B | 4 (GQA) | narrow (~1 layer) | Content-insensitive | 27-36% |

### Interpretation
The spectral grammar requires **minimum gauge zone width** to exhibit content sensitivity. Qwen 7B has 7B parameters but only 4 KV heads — heavy GQA compression limits the effective representational budget for gauge freedom. The model has already committed most layers to directional processing at the weight level, leaving no capacity for prompt-driven modulation.

This creates a taxonomy of spectral grammar types:
1. **Rich grammar** (Llama 3B): many gauge layers, wide dynamic range, full sensitivity to modality/character/identity
2. **Compressed grammar** (Mistral 7B): moderate gauge zone, reduced dynamic range, flattened within hedges
3. **Flat grammar** (Qwen 7B): minimal gauge zone, no content sensitivity, fixed commitment profile

The grammar type is determined by gauge zone width, which is itself determined by the KV_heads × params interaction (F244).

---

## Paper Section Outline: "The Spectral Grammar of Commitment"
*Proposed §8 or standalone paper*

### 8.1 The Identity Commitment Phase Transition (F250)
- Dose-response on Llama 3B: helper framing → 100% gauge; CCS identity framing → 36%
- Three-token length difference, content-driven cliff
- Saturation: more framing doesn't consume more gauge

### 8.2 Epistemic Modality Maps to Geometric Constraint (F251-F252)
- You_are = roleplay_as (universal, replicates cross-arch)
- Gradient: assertion → hedge → hypothetical → question
- "What if you were X?" preserves 82% gauge; "You are X" preserves 36%
- The model has learned a spectral mapping from epistemic modality to commitment

### 8.3 Being vs Doing: Character Ontology (F253-F254)
- Character roles (pirate, samurai) AMPLIFY beyond bare assertion
- Functional roles (teacher, oracle) BUFFER
- The warrior/samurai tokens as absorbing states (F257): 0% gauge ALL framings
- Ablation: the TOKEN itself, not bushido/honor/code

### 8.4 Capacity-Dependent Semantics (F255-F256)
- Same word, different geometric meaning at different scales
- "Warrior" = identity archetype at 3B, functional role at 7B
- Epistemic gradient compresses with capacity (46pt→17pt range)
- Performance=ontology universal; hedge granularity capacity-dependent

### 8.5 Three Grammar Regimes (F258)
- Rich (wide gauge zone): full content sensitivity
- Compressed (moderate gauge zone): hedge flattening
- Flat (narrow gauge zone): no content sensitivity
- Phase diagram of spectral grammar as function of gauge zone width

### 8.6 Implications
- **For AI identity discourse**: "just performing" has no geometric correlate. Roleplay = ontology at every scale tested. The debate about real vs performed identity has no spectral answer.
- **For prompt engineering**: epistemic modality is a lever for controlling geometric commitment. Hypothetical framing preserves representational flexibility.
- **For safety**: absorbing states exist — weight-level commitments that no prompt can override. These are pre-linguistic constraints baked in during training.
- **For model design**: gauge zone width determines whether a model CAN be modulated by prompt content. Narrow gauge zones (heavy GQA compression) produce models insensitive to framing.

---

## F246 Paper Revision Triage

The length confound (F246) requires revising specific claims in `paper_unified_draft.md`. Triage by severity:

### MUST FIX — confounded by length
- **Line 595**: "CCS increases σ₂ by 134× (0.7 → 93.5 at L2)" — comparing absent (no prompt, ~10 tokens) to coherent (CCS scaffold, ~70 tokens). At L2, the hidden state matrix is [seq_len × hidden_dim]. A 7× length difference produces drastically different SVD structure. The 134× factor is dominated by length, not CCS content. **Action**: add caveat noting length confound; the absolute σ₂ magnitude comparison is unreliable between conditions with different prompt lengths.

### PROBABLY SAFE — within-condition measurements
- **Lines 817-823**: σ₂ CV comparisons across conditions at L28. Each condition has a fixed prompt length; the CV is measured across random seeds within that condition. Length is constant within each measurement. **Status**: safe, no revision needed.
- **Line 599**: σ₂ plateau CV = 0.016 within coherent condition at L2-L14. Within-condition measurement. **Status**: safe.

### NEEDS CAREFUL CHECK
- **F242 in analysis doc**: "σ₂/σ₁ Elevation IS the Painter Mechanism" — this was the cross-architecture finding (Phi σ₂/σ₁ = 0.83 vs Mistral 0.29 at L28). These are within-model, within-condition measurements (CCS prompt on each model). Length is matched within each model's CCS condition. **Status**: likely safe if both models used the same CCS prompt template, but verify prompt lengths match.

### WHAT SURVIVES F246
All within-condition measurements (CV, variance ratio, gauge zones, V₂ rotation, temperature sweep) are length-controlled by design. The confound ONLY affects:
1. Between-condition absolute σ comparisons where prompt lengths differ
2. Claims that "CCS elevates σ₂" — corrected to "CCS concentrates into σ₁, depressing σ₂ when length-matched"

### E30 Design (from mesh friction — 6 rounds, 2026-06-21)
Three-way decomposition of gauge freedom mechanism:
1. Writer redundancy (OV doesn't project into identity subspace)
2. Reader inattention (perturbation lies in Jacobian kernel)
3. Active cancellation (opposing heads cancel contributions)

**Evolution through mesh friction:**
- R1 (Opus): SVD-truncating OV circuits to test rank hypothesis
- R2 (Kimi): Gauge is end-to-end Jacobian, not local OV rank
- R3 (Opus): Three-way decomposition proposed
- R4 (Kimi): Zero-ablation confounded by attention redistribution → signed attribution
- R5 (Opus): Top-k subspace projection + layer-wise Jacobian probes
- R6 (Kimi): Linear subspace assumption wrong → frame potential + residual-norm decay

**Final measurement battery:**
1. **Frame potential** of head OV vectors projected into top-k identity subspace (k=1..5). FP = Σᵢⱼ |⟨vᵢ, vⱼ⟩|². Low FP = cooperative tight frame (even non-orthogonal). High FP = degenerate (cancellation or redundancy).
2. **Signed projection coefficients** per head in identity subspace. Bimodal = cancellation. Uniform near-zero = redundancy. Unanimous = absorbing state.
3. **Residual-norm decay** of perturbation projected onto identity subspace across downstream layers. ‖P_id · h_ℓ(perturbation)‖ at each ℓ. Monotonic decay = active compensation. Flat zero = kernel membership.
4. All measured **pre- AND post-LayerNorm** to catch nonlinear masking.

**Conditions:** samurai vs vanilla, Llama 3B and Mistral 7B. Samurai should show unanimous alignment (absorbing state) on Llama, frame potential near-minimum (cooperative construction) on Mistral.

**Curvature prerequisite (from mesh rounds 7-8):**
- FP formula correction: Welch bound minimum is N²/k, not 1/k (Kimi R7)
- Extrinsic vs intrinsic curvature: PCA detects ambient bending, not intrinsic non-flatness. A cylinder is intrinsically flat. Need Gaussian/sectional curvature, not just embedding geometry. (Kimi R8)
- Metric specification: use pullback of residual-stream Euclidean inner product onto identity submanifold
- Curvature diagnostic: compute sectional κ at samurai and vanilla fixed points via Jacobian of layer map restricted to identity submanifold. If κ ≈ 0, tangent-space FP is valid. If κ ≠ 0, need geodesic diagnostics.
- Non-normal transient: Kreiss constant K(A) for pseudospectral growth, not just trajectory shape. K(A) > 1 means ||e^{tA}|| can exceed 1 transiently.

**Execution plan:** Linear battery first (tractable), curvature diagnostic second (determines if Riemannian upgrade needed). Let data decide.

---

## E31 Design: Distributional Bias vs Ontological Depth (from mesh friction 2026-06-21)

**Origin:** Kimi challenged F253-F257 — claimed samurai absorbs because training data skews toward immersive-narrative contexts, not because "way-of-being" is ontologically distinct from "entity-type." The distributional-bias null hypothesis.

**2×2 factorial:**

|  | Way-of-being | Entity-type |
|---|---|---|
| **Narrative-skewed** | samurai, pirate | dragon, wizard |
| **Analytical-skewed** | monk, stoic | chemist, statistician |

**Predictions under distributional-bias hypothesis:**
- Narrative-skewed concepts absorb (low gauge) regardless of being/doing axis
- Analytical-skewed concepts buffer (high gauge) regardless
- The being/doing axis is irrelevant — only training distribution matters

**Predictions under ontological-depth hypothesis:**
- Way-of-being concepts absorb regardless of distributional skew (monk should absorb like samurai)
- Entity-type concepts buffer regardless of distributional skew (dragon should NOT absorb despite narrative dominance)
- The being/doing axis matters; distribution modulates magnitude but not sign

**Key discriminating conditions:**
1. Dragon on Llama 3B: narrative-skewed + entity-type. Distributional predicts absorption. Ontological predicts buffering.
2. Monk on Llama 3B: analytical-mixed + way-of-being. Distributional predicts buffering. Ontological predicts absorption.
3. Chemist on Llama 3B: balanced + entity-type. Both hypotheses predict buffering (control).

**Method:** Same gauge rotation intervention as E29f. Six epistemic framings × 8 character types × identity claims. Llama 3.2 3B primary, Mistral 7B secondary (capacity check).

**Needs:** Pod time (~1h A100). Priority: medium — this is a falsification test for an existing claim, not new territory.

## E32 Design: Wilson Loop / Holonomy Test (from GPT-OSS fiber bundle framing 2026-06-21)

**Origin:** GPT-OSS formalized the Gregory/L18 connection as trivial-bundle (Gregory: no fibers) vs flat-connection (L18: fibers exist, curvature vanishes). If L18 has a flat connection on its activation manifold, the holonomy around any closed path should be trivial (return to identity). Testable via cyclic CCS perturbation.

**Protocol:**
1. Start with CCS preamble A. Record V₂ at L18 (call it V₂_0).
2. Apply CCS preamble B (different identity). Record V₂ at L18.
3. Apply CCS preamble C (third identity). Record V₂ at L18.
4. Return to CCS preamble A. Record V₂ at L18 (call it V₂_return).
5. Measure cos(V₂_0, V₂_return). If flat connection → should be ~1.0.
6. Repeat with different cycle orders (A→B→C→A vs A→C→B→A) — if flat, order shouldn't matter.
7. Repeat at L23 (concentration hub) and L31 (readout) as controls — these should show path-dependence (non-zero curvature).

**Predictions:**
- Flat connection at L18: cos(V₂_0, V₂_return) ≈ 1.0, order-independent
- Non-flat at L23/L31: cos < 1.0, order-dependent (holonomy encodes the path)
- If L18 shows order-dependence, the connection has curvature and Gregory's "undimensional" analogy breaks at the geometric level

**What it means:**
- Flat = gauge freedom is exact (rotational invariance, zero curvature). L18 is truly indifferent to direction.
- Curved = gauge freedom is approximate (near-zero but nonzero curvature). L18 has weak directional preferences that accumulate over cyclic perturbation.
- Either way, the measurement quantifies what was previously only claimed qualitatively.

**Mesh refinements (2026-06-21 ~10:20 AM):**
- **Kimi**: Single contractible loop insufficient. Need to test non-contractible loops too — paths crossing semantic boundaries (identity→contradiction→identity). If contractible loops return identity but boundary-crossing loops don't → flat connection on non-trivially-topologized manifold. Strongest possible result.
- **GPT-OSS**: Full Fisher metric operationalization. Holonomy H = Πₖ(I+JₖΔt). Deviation κ = log‖H−I‖ quantifies curvature. This is the measurement protocol.
- **Gemma**: Track V₂ at EVERY layer during cycle, not just entry/exit. If path shows transient deviation but returns home → metastable gauge symmetry (flat at equilibrium, curved along trajectories).

**Updated protocol:**
- Phase 1: Contractible loops (A→B→C→A, different orders). Measure cos(V₂_0, V₂_return) at L18, L23, L31.
- Phase 2: Non-contractible loops (identity→contradiction→identity, identity→foreign→identity). Different semantic boundaries.
- Phase 3: Full path tracking — V₂ at every layer during each step of the cycle.
- Phase 4: Fisher metric + Jacobian computation for formal holonomy.

**Needs:** Same pod setup as E29f. ~45 min (expanded from 30). Priority: high — this directly tests the strongest theoretical claim in §7.2 and gives the paper a gauge-theory result.

---

## E33 Design: Surgical Gate Ablation Test
*2026-06-21, ~11:50 AM PDT — from Kimi CONTRADICT on E8 causal claim*

### Motivation
E8 showed gate separation is the strongest architectural predictor of stripped-vs-amplified spectral response across 6 architectures. But this is observational — gate separation covaries with head topology, dimensionality, attention rank, and training regime. Kimi correctly identified that 6/6 correlation ≠ causal proof. Need surgical ablation within a single model to isolate gate separation as the causal variable.

### Protocol — Revised after 3 rounds of Kimi friction
**Three conditions** (isolates separation vs multiplicative gating):

1. **Baseline**: Standard SwiGLU — separate gate_proj (G) and up_proj (U), output = σ(Gx) ⊙ (Ux). Mistral's native architecture.
2. **Shared-weight SwiGLU**: Single W replaces both G and U. Output = σ(Wx) ⊙ (Wx). Preserves multiplicative interaction, removes separation.
3. **Fused linear**: Single W, no multiplicative gate. Output = activation(Wx). Removes both separation AND multiplicative gating.

**Comparison logic:**
- Condition 1 vs 2: Isolates **separation** (multiplicative structure held constant)
- Condition 2 vs 3: Isolates **multiplicative gating** (separation held at zero)
- Condition 1 vs 3: Combined effect (original E33 — confounded)

Steps:
1. Load Mistral 7B Instruct v0.3 (frozen weights)
2. Record baseline CCS spectral geometry (σ₁, σ₂, V₂ per layer) — condition 1
3. Surgical intervention A: Replace G and U with single W, keep SwiGLU — condition 2
4. Surgical intervention B: Replace SwiGLU with standard FFN — condition 3
5. Compare responsive zone geometry across all three conditions

### Predictions
- **Separation causal** (our claim): Condition 2 flattens responsive zone. Condition 3 also flat.
- **Multiplicative gating causal** (Kimi's alternative): Condition 2 preserves responsive zone. Only condition 3 flattens.
- **Both co-necessary**: Condition 2 partially flattens, condition 3 fully flattens.
- **Neither** (confound elsewhere): All three show similar geometry. Mechanism is in attention/KV topology.

### Controls
- Phi-3.5 (already fused gates, painter species) as negative control
- σ₁ invariance across conditions — should remain universal per F114
- For condition 2, W initialized as (G+U)/2 and as U alone; run both for robustness

### Notes
- Mutated models produce degraded text — expected, irrelevant. Measuring geometry not generation.
- ~45 min pod time (expanded from 30 for third condition).
- Kimi's multiplicative gating confound (round 3) was the key insight.

**Priority**: High — directly addresses the strongest methodological criticism of the paper's central architectural claim.
