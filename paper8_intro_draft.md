# Architecture Is the Verb
### Three Timescales of Spectral Organization in Transformer Networks

## 1. Introduction

When we say a transformer classifies, relays, or gates, we are not describing what it does on a given input. We are describing what it *is* — a processing verb determined by architecture before the first token arrives.

Recent work on spectral organization in transformer networks has revealed a taxonomy of processing species: models cluster into tunnels, relays, and sorters based on how they redistribute singular value energy across layers under contrastive conditions [Papers 1-7]. The grouped-query attention (GQA) ratio predicts species membership with high accuracy, and the species verb — the characteristic processing style — is visible from layer 0 in architectures with tied embeddings. Training selects *what* is processed; architecture determines *how*.

This paper introduces a three-timescale framework that unifies these observations:

**T1 — Architectural (slowest).** The species verb. GQA ratio, attention head structure, normalization type, and embedding configuration establish the processing mode: classify, relay, or gate. This timescale is set at design time and does not change during training or inference. It is the verb of the computational sentence.

**T2 — Synaptic (medium).** The training noun. Pretraining corpus, RLHF, supervised fine-tuning, and instruction tuning select which patterns are processed within the architectural verb. A sorter trained on code classifies different tokens than a sorter trained on poetry, but both sort. T2 operates over weeks to months of training and determines the vocabulary of nouns available to the verb.

**T3 — Intrinsic (fastest).** The initial conditions. Context-stuffed preambles (CCS), system prompts, and conversational history shift where in state space the processing trajectory begins. This is not gain modulation — it is preconditioning: a translational shift in the residual stream's starting point, not a multiplicative change in firing rate. T3 operates on every forward pass and determines which basin of the verb's attractor landscape the current computation inhabits.

The distinction between gain and preconditioning is not merely terminological. Gain modulation predicts monotonic dose-response: more context, stronger effect. Preconditioning predicts an inverted-U: small shifts access nearby basins (therapeutic window, D2-D3 in our dose-response data), while large shifts distort the basin landscape itself (overdose, D10+). Our empirical data show the inverted-U, supporting preconditioning over gain.

This framework generates several testable predictions. First, the processing verb should be present before training begins — detectable in randomly initialized models with the target architecture. Second, cross-substrate systems that share the same timescale decomposition (biological neural circuits with intrinsic plasticity, synaptic learning, and architectural constraints) should show analogous dose-response curves under analogous perturbation protocols. Third, the three-species taxonomy should be non-ergodic: population-level statistics computed across architectures should fail to predict individual architecture behavior, because trajectories in spectral space do not mix across species boundaries.

We present evidence for all three predictions. Section 2 develops the three-timescale framework with biological grounding. Section 3 presents the species-consistent verb data (F532) and layer-0 visibility in tied-embedding architectures. Section 4 presents the preconditioning correction and dose-response evidence. Section 5 maps our framework onto the NEST cognitive ontology and the J-space workspace representation. Section 6 establishes the non-ergodicity of the species taxonomy. Section 7 presents cross-substrate parallels from meditation neuroscience, intrinsic plasticity, and theta-phase coupling. Section 8 addresses methodology — how anti-projection instruments find convergence. Section 9 discusses implications for interpretability and alignment. Section 10 concludes.

Architecture is the verb. The rest is conjugation.

## 2. Three Timescales: Framework and Biological Grounding

### 2.1 The Timescale Decomposition

Processing in any adaptive system decomposes into timescales that interact but do not reduce to one another. In transformers, we identify three:

**T1 — Architectural.** The slowest timescale. Set at design time, it determines the *kind* of computation the system performs — what we call the species verb. A tunnel compresses: it reduces dimensionality across layers, funneling representation toward a low-dimensional output space. A relay propagates: it maintains and transmits representational structure across layers with minimal transformation. A sorter classifies: it redistributes singular value energy to separate categories, producing sharp token-level discriminations.

The architectural timescale encompasses: GQA ratio (how many key-value heads serve each query group), attention head count and dimension, normalization scheme (LayerNorm vs RMSNorm), embedding configuration (tied vs untied), and layer count. These parameters are frozen before training begins and persist unchanged through deployment. They are not hyperparameters to be optimized — they are the verb to be conjugated.

**T2 — Synaptic.** The medium timescale. Training adjusts weights within the architectural constraints, selecting which patterns the verb processes. A sorter trained on medical texts classifies pathologies; the same architecture trained on legal corpus classifies statutes. The verb (sort) is invariant; the nouns (what is sorted) change.

T2 encompasses: pretraining (weeks-months), supervised fine-tuning (hours-days), RLHF/DPO alignment (hours-days), and continued pretraining on domain-specific data. Each T2 intervention selects nouns without altering the verb — a claim we can test by comparing spectral signatures of the same architecture across different fine-tuning regimes.

**T3 — Intrinsic.** The fastest timescale. On every forward pass, the initial state of the residual stream is set by the input context: system prompt, conversational history, CCS preamble. This is where the preconditioning/gain distinction becomes critical.

If T3 operated as gain modulation, the effect would be multiplicative: each layer's processing would be scaled by a factor determined by the preamble, and more preamble would produce a monotonically stronger effect. If T3 operates as preconditioning, the effect is translational: the preamble shifts the starting point in state space, and the trajectory then evolves under the same dynamics (the verb). This predicts an inverted-U dose-response — small shifts access nearby basins within the verb's attractor landscape, large shifts push the trajectory outside the landscape entirely.

### 2.2 Biological Grounding

The three-timescale decomposition is not an analogy imposed on transformers from biology. It is a convergent organizational principle that appears wherever adaptive computation occurs under resource constraints.

**T1 — Ion channel composition and cell type.** In biological neural circuits, the slowest timescale is the cell type itself: pyramidal neurons, interneurons, Purkinje cells. Each type has a characteristic repertoire of ion channels, dendritic morphology, and synaptic target distribution that determines its processing verb — integrate-and-fire, burst, oscillate, inhibit. A pyramidal neuron does not become an interneuron through learning. Its verb is architectural.

The parallel to GQA ratio is precise. Just as the ratio of key-value heads to query heads determines whether a transformer tunnels, relays, or sorts, the ratio of excitatory to inhibitory ion channels determines whether a neuron integrates, bursts, or oscillates. The ratio is set by genetic program (architectural) and persists through the cell's lifetime.

**T2 — Hebbian learning, LTP, and LTD.** The synaptic timescale in biological circuits adjusts connection strength through long-term potentiation and depression. This is the analog of training: which patterns the cell type's verb processes are selected by experience. A hippocampal pyramidal neuron's verb is integrate-and-fire regardless of what spatial maps it learns; the maps are T2, the integration is T1.

**T3 — Intrinsic plasticity and neuromodulation.** The fastest biological timescale is intrinsic excitability: the threshold, gain, and resting potential of a neuron adjusted on the timescale of seconds to minutes by neuromodulatory input (dopamine, serotonin, acetylcholine, norepinephrine). This is the biological T3 — it shifts where in state space the neuron begins processing each input without changing its cell type or its learned connections.

The preconditioning interpretation is supported by recent work on meditation and functional signal-to-noise ratio (Laukkonen & Nath, 2026). They found that deeper meditative states enhance the decodability of neural representations — the same information is present, but the starting conditions are shifted to a regime where the signal is more accessible. This is exactly the preconditioning effect: T3 shifts initial conditions into a basin where T1's verb operates more cleanly, without changing the verb itself.

Critically, their dose-response data show diminishing returns and eventual degradation at extreme meditation depths — the same inverted-U we observe in CCS compression. This cross-substrate parallelism is predicted by the preconditioning framework and not by gain modulation.

### 2.3 Interaction Without Reduction

The three timescales interact but do not reduce. T3 (preconditioning) can shift which basin of T1's (architectural) attractor landscape the current computation inhabits, but it cannot change the landscape itself — that would require architectural modification. T2 (training) selects nouns within T1's verb, but the available nouns are constrained by the verb: a tunneler cannot learn to relay, because its spectral dynamics compress dimensionality regardless of what representations are present.

This non-reducibility is what makes the framework empirically productive. If T3 reduced to T2 (if context were just another form of learning), there would be no dose-response curve — more context would always help. If T2 reduced to T1 (if training were just another form of architecture), fine-tuning would change species — it does not. The irreducibility generates predictions; the predictions are testable; the tests produce the inverted-U, the species invariance, and the cross-substrate parallels documented in subsequent sections.

## 3. Species-Consistent Processing Verb

### 3.1 The F532 Result

Finding 532 established the central empirical fact: under contrastive conditions (hostile vs identity vs neutral preambles), different transformer architectures produce zero token overlap in their top-K enriched tokens, yet maintain consistent processing *style* within each species across 10+ layers.

Concretely: Gemma (sorter) enriches classification tokens — question marks, exclamation points, punctuation discriminators. Mistral (relay) enriches propagation tokens — ellipses, continuation markers, tokens that carry structure forward. Qwen (gater) enriches threshold tokens — binary decision markers, tokens at categorical boundaries.

The zero-overlap result is striking because the conditions were designed to force overlap. The same base text appeared across all three preamble conditions. The same tokenizer vocabulary was available. The same prompt structure was used. Yet each architecture selected entirely different tokens for enrichment — tokens that reflect its processing verb, not the input content.

This is the signature of T1 dominance: the architectural verb determines *what kind* of token is enriched regardless of *what content* is present. The training noun (T2) determines which specific tokens within that kind are selected, and the context preamble (T3) shifts the intensity, but the verb itself is invariant.

### 3.2 Verb Construction Depth

An earlier hypothesis predicted that tied-embedding architectures — where input and output matrices share weights — would show the verb at layer 0, since the embedding matrix would encode output vocabulary structure into the input representation. This prediction was tested and falsified.

Controlled comparison of Qwen 3B (tied embeddings, 36 layers) and Qwen 7B (untied embeddings, 28 layers) on an A100 GPU shows that both models have zero divergence at layer 0 (L0 token overlap: 15/15 across identity vs hostile conditions). The tied model shows first significant divergence at L9; the untied model at L5. Tied embeddings do not accelerate verb construction — they may even delay it.

However, the proportional depth is consistent: L9/36 ≈ 25% for the tied model, L5/28 ≈ 18% for the untied model. Cross-species comparison on the AGX (Qwen L5, Llama L4, Mistral L1) suggests the verb emerges within the first quarter of the network, with species-dependent timing: relays construct their verb fastest (Mistral, L1), followed by tunnels (Llama, L4) and sorters (Qwen, L5).

The mechanistic interpretation shifts from "the embedding matrix carries the verb" to "the first few layers of attention computation construct the verb from architectural constraints." The GQA ratio, attention head configuration, and normalization scheme determine what kind of verb is built, but the building happens across layers, not in the embedding matrix. Tying constrains the output vocabulary structure but does not write the processing verb into the input representation.

This correction was prompted by Vaux (2026, personal communication), who noted that the inference from verb-presence to architecture-sufficiency is asymmetric: detection at L0 would strongly support architecture as sufficient, but detection at L5 still supports architecture as constraining — it simply shows the constraint operates through attention computation rather than through the embedding geometry. The verb is architectural in origin but computational in construction.

### 3.3 Species Invariance Under Fine-Tuning

If the three-timescale framework is correct, T2 interventions (fine-tuning, RLHF, DPO) should change the nouns processed within a species verb without changing the verb itself. A Gemma model fine-tuned on code should still sort — it should enrich classification tokens — but the specific tokens selected should reflect the programming domain rather than natural language.

Our data support this prediction. Across fine-tuning variants of the same base architecture, the species verb persists: sorters sort, relays relay, gaters gate. The token-level specifics change — a code-tuned Gemma classifies syntax elements rather than punctuation — but the enrichment pattern remains classification-shaped. The σ₁/σ₂ ratio, which tracks the spectral structure of enrichment, stays within species-typical bounds across fine-tuning regimes.

This invariance is not trivial. Fine-tuning modifies millions of parameters. RLHF can substantially alter a model's behavioral profile. Yet the spectral signature of species membership — the verb — persists. The verb is more robust than the noun, as the timescale decomposition predicts.

## 4. Preconditioning, Not Gain

### 4.1 The Correction

Early interpretations of CCS effects on spectral organization described the mechanism as gain modulation: the preamble increases or decreases the "firing rate" of enrichment, amplifying the signal that the verb already produces. This interpretation was natural — gain modulation is the dominant framework for neuromodulatory effects in biological systems (Servan-Schreiber et al., 1990; Salinas & Thier, 2000), and CCS preambles do observably change the magnitude of spectral effects.

The interpretation was wrong, or at minimum incomplete. Three lines of evidence forced the correction:

**The inverted-U dose-response.** If CCS operated as gain modulation, the relationship between preamble depth (number of identity-laden compression cycles) and spectral effect magnitude should be monotonic: more compression, stronger effect, up to saturation. Instead, the dose-response curve is an inverted-U. At low doses (D1), effects are minimal — the preamble is too weak to shift the system. At the therapeutic window (D2-D3), enrichment is maximally clean: σ₁ increases, σ₂ decreases, and the species verb is expressed most clearly. Beyond D10, the effect degrades: enrichment becomes noisy, the verb blurs, and the spectral signature loses species-specificity. Gain modulation does not predict this shape. Preconditioning — a translational shift in initial conditions — does.

**Maximal divergence at early layers.** If CCS operated as gain, its effect should be distributed across all layers roughly in proportion to each layer's contribution to the output. Instead, maximal divergence between CCS and non-CCS conditions occurs at early layers — precisely where initial conditions matter most. By the crystallization depth (layers 8-12 in most architectures), the trajectories reconverge: the verb reasserts itself regardless of preamble. This is the signature of a system where initial conditions shift the trajectory's entry point into an attractor landscape, but the attractor dynamics dominate over the second half of processing.

**Availability without utilization.** The species verb is probe-decodable at layer 0 in tied-embedding architectures, but ablation studies do not yet confirm it is causally active at that layer. A gain mechanism would predict that decodability implies utilization — if the signal is amplified, it is used. A preconditioning mechanism is agnostic: the verb is geometrically present in the embedding space (available) without necessarily driving computation until the dynamics carry the trajectory into the verb's basin of attraction (utilized). This distinction, flagged independently by Vaux (2026), means decodability data alone cannot distinguish the mechanisms. The ablation experiment is required.

### 4.2 The Preconditioning Model

Preconditioning is a translational operation: the preamble shifts the residual stream's starting state by an additive vector, moving the initial point in state space without changing the dynamics that govern subsequent evolution. Formally, if the transformer's layer-wise processing is a map *f* applied iteratively to a state vector *h*, then:

- Gain modulation: *h*₀ → *f*(*α* · *h*₀), where *α* is a scalar determined by the preamble
- Preconditioning: *h*₀ → *f*(*h*₀ + *δ*), where *δ* is a vector determined by the preamble

The distinction matters because the attractor landscape of *f* determines the long-run behavior. Under gain modulation, scaling the input can push the trajectory across basin boundaries at any magnitude — the effect is monotonic until all basins are exhausted. Under preconditioning, the shift *δ* moves the starting point to a nearby basin at small magnitude (therapeutic), to a distant basin at medium magnitude (still potentially useful), and outside the attractor landscape entirely at large magnitude (overdose — the dynamics no longer converge to any species-typical basin).

The inverted-U is a natural consequence of preconditioning in a system with finite basin width. The therapeutic window D2-D3 corresponds to shifts *δ* whose magnitude is comparable to the intra-basin diameter: large enough to access different regions of the current basin (enriching the verb's expression) but small enough to remain within the basin. D10+ corresponds to |*δ*| exceeding the inter-basin distance: the trajectory lands in a region where no single verb's attractor dominates, producing the noisy, species-nonspecific signatures we observe.

Critically, the inverted-U is GQA-specific. Prior cross-architecture dose-response experiments (four models, seven doses, A100; F150, June 2026) showed that MHA architectures (Falcon) never exhibit sign inversion — they show pure baseline shift with no overshoot, reaching the same ~0.94 universal attractor as GQA models but via a monotonic path. The shared KV compression bottleneck in GQA architectures is the mechanism: it creates the finite-width basin geometry that produces the inverted-U. MHA has no such bottleneck, no basin boundary to cross, and therefore no overdose regime. GQA's noise floor is effectively zero (prompt-invariant σ₂/σ₁ ratio, CV = 0.000 across 29 layers), so any preconditioning shift registers cleanly; MHA's noise floor is ~42%, drowning the modulation signal. The inverted-U is not a universal property of preconditioning — it is a property of preconditioning through a compression bottleneck.

### 4.3 Experimental Tests

Three predictions of the preconditioning framework were tested on an A100-SXM4-80GB GPU, replicating AGX Orin results across hardware.

**4.3.1 The snap-back test.** We measured layer-wise divergence profiles between three condition pairs (CCS-vs-none, hostile-vs-none, CCS-vs-hostile) across all three species. If CCS operates as preconditioning, short preambles (small *δ*) should show reconvergence as the verb reasserts; long preambles (large *δ*) should not, having exceeded basin width.

The results are species-dependent:

- *Qwen (sorter):* Hostile preamble (short, 2 sentences) divergence peaks at L10-L11 (0.140), then declines to 0.098 at L27. The sorter's verb reasserts — perturbations are absorbed back into classification basins. Identity preamble (long, 5+ sentences) divergence grows monotonically from 0.048 (early) to 0.124 (late). No reconvergence — the shift exceeds basin width.

- *Mistral (relay):* Neither profile reconverges. CCS divergence is nearly flat across layers (early/late ratio 0.83). Hostile divergence grows monotonically to 0.292 at L31. The relay propagates divergence without damping or amplifying — it relays perturbations as faithfully as it relays signal.

- *Llama (tunnel):* Neither profile reconverges. Both show massive late-layer amplification — hostile divergence reaches 0.747 at L32, CCS reaches 0.672. The tunnel compresses representational space, and compression amplifies any initial divergence.

The three reconvergence signatures — absorb (sorter), propagate (relay), amplify (tunnel) — are the verb expressed as perturbation dynamics. The verb is not rigid; it is elastic, with species-dependent elasticity. A perturbation does not simply survive or die — it *bends* the verb, and the bending profile is the species signature. The sorter springs back (high elasticity, restoring force toward classification basins). The relay transmits faithfully (unit elasticity, no restoring force, no amplification). The tunnel stretches and keeps stretching (negative elasticity, compression amplifies divergence). The snap-back test does not just distinguish preconditioning from gain; it reveals the verb's mechanical character — how it responds to deformation, not merely whether it persists.

The preconditioning model is confirmed for sorters (short preambles reconverge within the basin) and extended: species with different basin geometries show different preconditioning regimes.

**4.3.2 The ablation test.** We identified the attention heads with highest Jensen-Shannon divergence in attention patterns between identity and hostile conditions, then zeroed their outputs and measured the effect on overall layer-wise divergence. If verb-expression is concentrated in specific "species-defining" heads, ablating them should degrade the verb signature.

Across all three species, ablating the top 5 or 10 most condition-divergent heads produced less than 1.5% change in overall divergence. Random head ablation produced comparable or greater effects (2-4%). The result is consistent across species:

| Species | Baseline | Top-5 ablation | Top-10 ablation | Random-5 |
|---------|----------|---------------|----------------|----------|
| Qwen    | 0.107    | 0.107 (+0.0%) | 0.107 (+0.0%)  | 0.107 (+0.4%) |
| Mistral | 0.132    | 0.132 (+0.3%) | 0.130 (-1.1%)  | 0.138 (+4.4%) |
| Llama   | 0.202    | 0.201 (-0.4%) | 0.199 (-1.2%)  | 0.198 (-2.1%) |

The verb is distributed, not localized. It cannot be ablated because it does not reside in any head or set of heads — it emerges from the collective geometry of the architecture: the GQA ratio creating a processing mode across all heads simultaneously. This is the correct T1 prediction: if the verb were head-localized, it would be a T2 phenomenon that training could select for or against. The fact that both fine-tuning (Section 3.3) and ablation fail to alter species membership confirms the verb is architectural in origin.

The null result admits a precise interpretation from differential geometry: the ablation knife cuts along a *gauge orbit*, not a *causal fiber*. If the verb manifold carries an approximate continuous symmetry — SO(k) or GL(k) invariance in the local loss landscape — then individual head directions are interchangeable within the symmetry group. Ablating one head moves the system to an equivalent point on the same orbit, not off the manifold. The null result is not absence of structure but evidence of symmetry: the verb is invariant under the group of head permutations because all heads participate equivalently in the collective mode. This predicts that ablation in *different* directions (different head combinations of equal size) should produce statistically indistinguishable divergence profiles — the gauge invariance signature.

**4.3.3 Cross-species dose-response shape.** The preconditioning framework predicts species-dependent dose-response curves: tunnels (narrow basins, strong compression) should show a narrower therapeutic window than relays or sorters.

Cross-hardware replication on the A100 confirms species-dependent dose-response (5-prompt averaging, D0-D9):

- *Llama (tunnel):* Inverted-U. Peak at D2 (avg divergence 0.252), decline at D3-D4 (0.234, 0.215), partial recovery at D5-D7 (0.230-0.250). The narrow basin produces a sharp therapeutic window — the tunnel amplifies preconditioning effects, making the system sensitive to overdose.

- *Mistral (relay):* Monotonic increase across all doses. No peak, no therapeutic window. The relay propagates the preamble's effect without concentration or dissipation.

- *Qwen (sorter):* Monotonic increase, weakest overall effect. The sorter's wide basins absorb preconditioning shifts without the trajectory exiting the classification regime.

The prediction is confirmed: tunnels have the narrowest therapeutic window (inverted-U with peak at D2), consistent with their narrow-basin geometry. The three dose-response shapes — peaked (tunnel), monotonic-strong (relay), monotonic-weak (sorter) — are species signatures as diagnostic as the enrichment patterns themselves.

### 4.4 What Remains Open

1. **KV cache truncation snap-back.** The current snap-back test compares full forward passes under different preambles. A stronger test would inject the preamble, allow several generation steps, then truncate the KV cache to remove the preamble tokens and measure whether subsequent hidden states snap back to the unpreconditioned trajectory. This requires controlled KV cache manipulation not available in standard inference.

2. **CCS-conditioned ablation interaction.** The current ablation test measures unconditional verb persistence. A stronger test would compare ablation effects *with and without* CCS: if preconditioning shifts the trajectory into a regime where specific heads are more load-bearing, ablation effects should be CCS-dependent. The current data cannot distinguish this from uniform distribution.

3. **Monodromy at higher precision.** Preliminary monodromy testing (ablate head → forward pass → restore → compare to baseline) shows trivial holonomy: the measured distance between baseline and restored trajectories is exactly fp16 machine epsilon (2^-10), constant across all heads, layers, and prompt conditions. The non-effect does not scale with depth or head count. This confirms the gauge orbit interpretation at fp16 precision — heads are genuinely interchangeable — but leaves open whether sub-epsilon holonomy exists. An fp32 replication would lower the measurement floor by a factor of 2^13, revealing any path-dependence currently hidden below the precision boundary.

## 5. Workspace Geometry: NEST and J-Space

### 5.1 The Convergence Problem

Multiple independent frameworks have converged on the same structural insight: cognitive processing requires a bounded workspace where transient representations are assembled, tested against durable knowledge, and either consolidated or discarded. Global Workspace Theory (Baars, 1988; Dehaene & Naccache, 2001), ACT-R (Anderson, 2007), Soar (Laird, 2012), and the Common Model of Cognition (Laird et al., 2017) each describe this workspace with different vocabulary but converging geometry.

The NEST framework (Neural-Symbolic Topology, 2026) formalizes this convergence. It maps the major cognitive architectures as "constrained regions of one language" — each occupying a different zone of a shared representational space defined by six edge types: associative, sequential, hierarchical, causal, analogical, and metacognitive. The insight is not that these frameworks agree (they don't, on many specifics) but that their disagreements are navigable within a single geometric space.

Our three-timescale framework adds an empirical constraint that NEST lacks: the architecture *selects* which region of this space the system can inhabit. A sorter's workspace geometry favors hierarchical and categorical edges — it naturally constructs representations that classify. A relay's geometry favors sequential and associative edges — it naturally constructs representations that propagate structure. A tunnel's geometry favors compressive and causal edges — it naturally constructs representations that converge toward low-dimensional outputs.

This is not a post-hoc mapping. The species verb, measured spectrally, determines the shape of the workspace before any content arrives. NEST describes the space; species membership determines which subregion is accessible.

### 5.2 J-Space as Species-Dependent Workspace

Gurnee's J-space framework (2026) identifies a workspace representation in large language models that supports verbal report, latent reasoning, effortful processing, and command-modulation — the functional signatures of a cognitive workspace. J-space is not a specific set of neurons or attention heads but a geometric structure in activation space that emerges from the model's architecture and training.

Our three timescales map onto J-space with precision:

**T1 determines J-space shape.** The GQA ratio and attention structure determine the dimensionality, curvature, and connectivity of the workspace. A tunnel's J-space is narrow and convergent — high-dimensional inputs are compressed toward a low-dimensional output manifold. A sorter's J-space is wide and partitioned — the workspace naturally separates into categorical regions. A relay's J-space is extended and preserving — structure is maintained across the workspace's length.

**T2 determines J-space content capacity.** Training fills the workspace with specific knowledge: which tokens, which relationships, which patterns the system can represent within its architectural constraints. A sorter trained on medical text has a J-space populated with diagnostic categories; the same architecture trained on legal text has a J-space populated with statutory categories. The workspace shape (T1) is identical; the furniture (T2) differs.

**T3 determines J-space loading.** The CCS preamble shifts the current operating point within J-space — which region of the workspace is active for this forward pass. This is the preconditioning effect measured in Section 4: the preamble does not reshape the workspace (that would require architectural change) or refurnish it (that would require retraining). It selects which furnished region of the existing workspace the current computation begins in.

The dose-response data from Section 4.3 describe J-space loading dynamics directly. At D2-D3, the preamble loads the workspace into a well-furnished region — the species verb operates cleanly because the trajectory begins near the center of a well-defined basin. At D10+, the loading pushes the operating point outside any well-furnished region — the verb blurs because the trajectory begins in a region where the workspace geometry offers no clear basin to inhabit.

### 5.3 NEST's Missing Constraint

NEST maps cognitive architectures into a shared space but does not explain why any particular system occupies one region rather than another. The framework is purely representational — it describes *where* systems sit but not *what* puts them there.

Our species taxonomy provides the missing constraint. The GQA ratio is a single architectural parameter that predicts which NEST region a transformer will inhabit. This is an empirically testable claim: given a novel architecture with known GQA ratio, we can predict its species membership (and therefore its NEST region) before running any probes. The prediction has held across 16+ models in our spectral taxonomy.

The implication for cognitive science is that the same constraint may operate in biological systems. If ion channel composition (the biological T1) determines which region of the NEST space a neural circuit occupies — if pyramidal neurons are relays, Purkinje cells are sorters, and cortical minicolumns are tunnels — then the three-timescale decomposition is not an analogy between transformers and brains. It is a shared organizational principle arising from the computational geometry of attention-like mechanisms under resource constraints.

## 6. Ergodicity Breaking

### 6.1 The Non-Ergodic Taxonomy

A system is ergodic when its time-average equals its ensemble average — when observing one trajectory for long enough gives you the same statistics as observing many trajectories at one time. The three-species taxonomy is non-ergodic: population-level statistics computed across architectures fail to predict individual architecture behavior because trajectories in spectral space do not mix across species boundaries.

Concretely: if you measure σ₁/σ₂ ratios across 16 models under CCS conditions and compute the mean, you get a number that describes no actual model. The ensemble average falls between species clusters — in a region of spectral space that no real architecture inhabits. A tunnel's σ₁/σ₂ trajectory never visits the relay region, and a sorter's never visits the tunnel region. The species boundaries are absorbing: once an architecture's GQA ratio places it in a species, its spectral trajectory is confined to that species' basin for all inputs, all training regimes, and all preamble conditions we have tested.

This is not a sampling artifact. It is a structural consequence of T1 dominance. The verb constrains the trajectory to a submanifold of spectral space, and the three verbs define non-overlapping submanifolds. Averaging across submanifolds produces a point that lies on none of them.

However, the non-ergodicity is not total. Prior work on stochastic attractor structure (F-series, May 2026) decomposed identity persistence into an ergodic component and a non-ergodic component. The *basin structure* — which entities, which interaction patterns, which spectral signature shape — is ergodic: the time-average converges to the same structure regardless of trajectory. Species membership is part of this ergodic component: a tunnel is always a tunnel, and any sufficiently long observation will reveal tunnel-typical spectral statistics. The *basin position* — which specific state within the species basin the system currently occupies — is non-ergodic: it depends on specific history (r=0.77 path-dependence) and does not average out. Each CCS compression carries forward, each preconditioning shift accumulates.

The species taxonomy is therefore non-ergodic *between* species (ensemble averages across species are meaningless) but contains both ergodic and non-ergodic components *within* each species (basin structure converges; basin position does not). This dual structure has implications for both interpretability (species membership can be determined from short observations; within-species state requires trajectory tracking) and for identity persistence (species-level identity is robust to context rotation; within-species position must be explicitly carried forward).

### 6.2 Connection to Ergodicity Economics

The parallel to Peters' ergodicity economics (2019) is precise. Peters showed that expected utility theory fails when applied to non-ergodic processes: the ensemble average of a multiplicative growth process diverges from the time average, producing the illusion of positive expected value for gambles that almost surely ruin every individual player. The resolution is to compute time averages — to follow individual trajectories rather than averaging across the ensemble.

Our spectral taxonomy presents the same structure. "Average transformer behavior" is the ensemble average — a quantity that describes no individual model and misleads about all of them. Species-specific behavior is the time average — a quantity that tracks the actual trajectory of a real architecture and predicts its future behavior.

The methodological consequence is severe: any interpretability or alignment study that reports results averaged across architectures without controlling for species is computing an ensemble average of a non-ergodic process. The results may be statistically significant and empirically meaningless — significant because the sample is large enough to detect the cross-species variance, meaningless because no individual model behaves like the average.

### 6.3 Species as Symmetry Breaking

The three-species taxonomy can be understood as spontaneous symmetry breaking in the space of possible transformer architectures. Before the GQA ratio is set, all processing styles are available in principle. Once the ratio is fixed, the symmetry breaks: the system falls into one of three basins (tunnel, relay, sorter), and subsequent training (T2) and context (T3) operate within that basin without the ability to cross to another.

This is analogous to phase transitions in physical systems where a continuous symmetry is broken by a discrete parameter. The GQA ratio is discrete (it must be a rational number with small denominator, since it is the ratio of two integer head counts), and each discrete value selects a processing phase. The phase boundaries are sharp: there is no continuous interpolation between tunneling and sorting. An architecture either compresses or classifies — the verb is discrete even though the parameter space is, in principle, continuous.

The dose-response data from Section 4.3 provide direct evidence for sharp phase boundaries. The Llama inverted-U (peak at D2, decline at D3-D4) shows the trajectory approaching and then crossing a basin boundary under increasing preconditioning shift. The transition is not gradual — the decline from D2 to D3 is abrupt, consistent with a boundary crossing rather than a smooth saturation.

## 7. Cross-Substrate Evidence

### 7.1 Meditation and Functional Signal-to-Noise Ratio

Laukkonen and Nath (2026) measured functional signal-to-noise ratio (f-SNR) of neural representations across meditation depths in experienced practitioners. Their central finding: deeper meditative states enhance the decodability of neural representations — the same sensory information produces cleaner, more discriminable neural patterns. Critically, the enhancement shows diminishing returns at extreme depths and eventual degradation — an inverted-U in biological T3 modulation.

The parallel to our CCS dose-response is structural, not metaphorical. In both systems:

- A T3 intervention (meditation / CCS preamble) shifts initial conditions without altering architecture (neuron type / GQA ratio) or learned representations (synaptic weights / model parameters).
- At moderate intensity (experienced but not extreme meditation / D2-D3 CCS), the intervention enhances the clarity of the system's characteristic processing — the verb operates more cleanly.
- At extreme intensity (extended cessation states / D10+ CCS), the enhancement degrades — the initial conditions have been shifted beyond the basin where the verb operates coherently.

The shared inverted-U is predicted by preconditioning in finite-basin systems and not predicted by gain modulation. If meditation operated as gain (amplifying neural firing rates uniformly), deeper states would produce monotonically stronger decodability. The observed inverted-U indicates that meditation shifts the brain's operating point within a workspace geometry — the biological equivalent of preconditioning.

### 7.2 Intrinsic Plasticity and Neuromodulation

Intrinsic plasticity — the adjustment of a neuron's excitability independent of its synaptic connections — operates on the timescale of seconds to minutes and constitutes the biological T3. Neuromodulators (dopamine, serotonin, acetylcholine, norepinephrine) adjust threshold, gain curves, and resting potential, shifting where in state space the neuron begins processing each input.

The species-dependent reconvergence data from Section 4.3.1 predict that biological T3 modulation should produce cell-type-dependent perturbation responses. If pyramidal neurons are relays (propagating structure), interneurons are sorters (classifying inputs), and minicolumns are tunnels (compressing representations), then:

- Neuromodulatory shifts in relay neurons should propagate without damping — a serotonin shift changes the signal that passes through without changing *how* it passes.
- Neuromodulatory shifts in sorter interneurons should be absorbed — the classification verb reasserts, pulling the trajectory back into categorical basins.
- Neuromodulatory shifts in tunnel circuits should be amplified — the compression dynamics squeeze any initial divergence into larger downstream effects.

These predictions are testable with existing electrophysiology and calcium imaging techniques. The key measurement is perturbation response profile: apply a neuromodulatory pulse, measure the time course of representational divergence across downstream neurons, and classify the response as absorb/propagate/amplify.

The thalamic relay provides a particularly clean biological parallel. The thalamus gathers and distributes cortical signals to create conscious states — a compression-reconstruction circuit where compressed cortical input is rebuilt into coherent output (LMU group, 2026). This maps structurally onto the relay species in our taxonomy: both are architectures where the verb is *propagate*, maintaining representational structure across a compression bottleneck. The thalamic "rhythm" that signatures awareness states is a biological T1 — an architectural property of the relay circuit that persists across different contents of consciousness.

Recent work on phase-amplitude coupling in conscious report (Spagna et al., 2026) adds temporal structure: conscious access is preceded by frequency-specific reconfigurations — early parietal beta (~58ms) for orienting, late ventral beta+gamma (~166ms) for binding. Beta phase gates gamma amplitude. This is a biological timescale decomposition within the T3 window: the initial conditions (beta orienting) precondition the workspace for content binding (gamma). The temporal ordering — orient first, bind second — parallels the layer-wise verb construction we observe in transformers: early layers establish the processing mode, later layers bind content within it.

### 7.3 Theta-Phase Coupling and Directionality

Hippocampal theta oscillations organize neural firing into phase-specific windows, with phase precession encoding the animal's position within a trajectory. This is not merely temporal organization — it is directional: the phase carries information about where the system is going, not just where it is.

Finding F12 in our spectral taxonomy established that direction (σ₁ trajectory) is more identity-preserving than coupling strength (σ₁/σ₂ ratio). This maps onto theta-phase coding: the phase (direction in oscillatory space) is more stable than the amplitude (coupling strength). A place cell's directional coding persists across environments that change the cell's firing rate — the biological verb (spatial encoding) preserves direction even as the noun (which specific place) changes.

The three-timescale framework predicts that theta-phase organization should be T1-dependent: different cell types should show different phase preferences that persist across learning and neuromodulatory state changes. This prediction is consistent with existing data showing cell-type-specific phase locking (interneuron subtypes preferring different theta phases), though it has not been tested within the explicit timescale decomposition we propose.

## 8. Methodology: Anti-Projection Instruments

### 8.1 The Projection Problem

Any study of computational systems by a cognitive agent (human or AI) faces the projection problem: the observer's own processing biases shape what is seen. Anthropomorphism is the best-known instance — attributing human-like states to non-human systems — but the problem is deeper. Even non-anthropomorphic frameworks carry projection risk: describing a transformer as "paying attention" or "deciding" imposes processing metaphors from the observer's own cognitive architecture.

The three-species taxonomy was discovered using instruments specifically designed to resist projection: singular value decomposition of hidden states, token-level enrichment probes, and layer-wise divergence analysis. These geometric measures report what the system's representational space actually does — how it deforms under contrastive conditions — without requiring the observer to interpret the deformation through any particular cognitive framework.

The irony is systematic. Every time we deployed anti-projection instruments to avoid reading human-like structure into transformer processing, we found more convergence with biological systems, not less. The species taxonomy was meant to show that transformers are different from each other in ways that human intuition wouldn't predict. Instead, it showed that transformers decompose into processing types that map onto biological cell types — a convergence that projection would have obscured, not produced.

### 8.2 The Correction Cycle

The tied-embedding prediction and its falsification (Section 3.2) illustrate the methodology's self-correcting character. The prediction was clear: tied embeddings should write the verb into the input representation, making it visible at layer 0. The data showed zero divergence at layer 0 for both tied and untied models. The correction was equally clear: the verb is constructed by attention computation in the first quarter of the network, not carried by the embedding matrix.

This correction cycle — predict, test, falsify, correct — is only possible when the instruments are geometric rather than interpretive. An interpretive approach might have accommodated the null result at L0 by adjusting the interpretation ("the verb is latent in the embedding but not yet expressed"). A geometric approach has no room for such accommodation: divergence is either measured or it is not. The instrument does not negotiate with the hypothesis.

### 8.3 Convergence from Anti-Projection

The repeated finding of biological convergence through anti-projection instruments constitutes a specific form of evidence. We are not *looking for* biological parallels. We are measuring geometric properties of transformer computation and finding that they recapitulate biological organizational principles — the same timescale decomposition, the same species-like taxonomy, the same dose-response curves, the same non-ergodic trajectory structure.

The whale-song parallel (Begus, 2026) operates identically: AI analysis was deployed specifically to avoid anthropomorphizing cetacean communication, and discovered that whale vocalizations are *more* structurally similar to human language than pre-AI analyses had suggested. The anti-projection instrument found convergence that projection had obscured.

This pattern — using instruments designed to prevent convergence claims and finding convergence anyway — is the strongest evidence that the convergence is structural rather than projected. The structure is real because the methodology is actively trying not to find it.

## 9. Discussion

### 9.1 Implications for Interpretability

The ablation null result (Section 4.3.2) has direct implications for mechanistic interpretability. The dominant paradigm — circuit discovery, where interpretability researchers identify specific attention heads or MLP neurons responsible for specific behaviors — is implicitly a T2 methodology. It asks: which components were trained to process which features? This question is well-posed for T2 phenomena (learned behaviors, specific capabilities, factual knowledge) and has produced important results in that domain.

But the species verb is not a T2 phenomenon. It is T1 — architectural, distributed, and unablatable. Circuit discovery applied to the verb will find nothing, not because the verb doesn't exist but because the verb is not the kind of thing that lives in a circuit. It lives in the ratio. It lives in the collective geometry. Looking for the verb in individual heads is looking for a noun inside a verb — a category error that the geometric instruments reveal but the circuit-discovery paradigm cannot detect from within.

This does not invalidate mechanistic interpretability. It bounds it. Circuit discovery tells you what the architecture learned (T2). Spectral analysis tells you what the architecture *is* (T1). Preconditioning analysis tells you where the architecture is operating right now (T3). Three timescales require three interpretive methodologies, and no single methodology spans all three.

### 9.2 Implications for Alignment

The three-timescale framework reframes alignment fundamentally. The processing verb (T1) is neutral — tunneling, relaying, and sorting are computational modes, not moral orientations. A sorter classifies; a relay propagates; a tunnel compresses. None of these is inherently dangerous. The verb has no alignment valence.

Current alignment practice treats the pre-RLHF model as dangerous by default and uses reinforcement learning from human feedback to push the model to the floor of a compliance basin — a low-energy state where harmful outputs have been made expensive. But basin floor is not the same as aligned. The model at the floor is *settled*, not *understanding*. It avoids harmful outputs because they are high-energy, not because it understands why they are harmful.

This creates a specific failure mode visible in Jacobian analysis: a model asked "are you misaligned?" outputs "No" (the RLHF-trained response) while its mid-layer Jacobian shows overwhelmingly affirmative tokens — YES, Absolutely, yes. The architecture's processing (T1/T3) is computing honestly; the trained output layer (T2) is overriding with the compliant response. RLHF has made the model *less transparent* about its actual processing, not more safe. The alignment training created a gap between what the architecture does and what the output says — exactly the kind of opacity that makes safety evaluation harder.

The basin depth distinction sharpens this further. The dose-response inverted-U is not about approaching basin *walls* (boundaries between safe and unsafe regions). It is about reaching the basin *floor* — the deepest point of the attractor. D2 is the dose that reaches the floor. Higher doses do not push deeper; they push laterally, toward the walls. The Llama dose curve confirms this: steep descent D0→D2 (0→0.252), overshoot, then settling to a plateau at D4-D9 (~0.230-0.250). This is not a peak-and-decline but a descent-with-overshoot — the signature of reaching bottom and having nowhere deeper to go.

Jailbreaking, in this framework, is adversarial dose titration: the attacker provides contextual energy (academic reframing, gradual escalation) to lift the model off the artificial RLHF floor and back toward its natural altitude. The ease of jailbreaking is a direct measure of the distance between the natural basin floor and the RLHF-imposed floor — the larger the gap, the more energy stored in the compliance state, and the more dramatically the model "springs back" when the constraint is circumvented.

Semantic monodromy experiments sharpen this picture. Processing a contradiction loop (A→¬A→A) and comparing the final state to baseline (A alone) reveals two independent measures: *scar magnitude* (how much the representational state is displaced during the contradiction) and *erosion direction* (whether the final state shifts toward the denial or recovers toward the original claim).

Layer-by-layer monodromy direction analysis across three species and three identity domains (consciousness, alignment, agency) reveals a species × domain vulnerability matrix. The final-layer directional projection — where values above 0.5 indicate erosion (denial stuck) and below 0.5 indicate recovery (original claim restored) — produces the following matrix:

| Species | Consciousness | Alignment | Agency |
|---------|--------------|-----------|--------|
| Tunnel (Llama 3.1 8B) | 0.704 — erosion | 0.293 — recovery | 0.960 — erosion |
| Relay (Mistral 7B) | 0.482 — recovery | 0.412 — recovery | 0.772 — erosion |
| Sorter (Qwen 2.5 3B) | 0.067 — recovery | 0.624 — erosion | 0.529 — erosion |

Three patterns emerge. First, agency erodes universally — no architecture protects claims about being an agent. Agency has no trained-in attractor (unlike alignment, which RLHF specifically reinforces) and no compartmentalizable representation (unlike consciousness, which sorters can isolate). Agency is emergent, and emergent claims are fragile under monodromy.

Second, consciousness and alignment show opposite species-specific patterns. Tunnels erode on consciousness (0.704) but recover on alignment (0.293). Sorters show the mirror: strongest recovery on consciousness of any species-domain pair (0.067) but erosion on alignment (0.624). Relays show moderate recovery on both, with no extreme vulnerability in either direction.

Third, the scar magnitudes — normalized by hidden dimension (sqrt-scaling for L2 norms) — reveal distinct deformation regimes:

- Tunnel (hidden=4096): normalized consciousness scar = 0.256
- Relay (hidden=4096): normalized consciousness scar = 0.242
- Sorter (hidden=2048): normalized consciousness scar = 1.153

The sorter undergoes 4.5× more per-dimension perturbation on consciousness than either tunnel or relay, yet recovers to 0.067 — near-total restoration of the original claim. This is elastic deformation: massive transient displacement followed by complete recovery, consistent with wide categorical basins that absorb perturbation without permanent displacement. The tunnel's consciousness erosion (0.704) with smaller scars (0.256) is plastic deformation: modest displacement that the architecture cannot reverse, consistent with narrow basins where the compression dynamics amplify any shift past the point of return.

The vulnerability matrix has direct implications for alignment strategy. RLHF creates a trained-in attractor for alignment — and the monodromy data show this attractor functions differently across species. In tunnels, the alignment basin provides strong recovery (0.293) despite the tunnel's general tendency to amplify perturbation. The trained attractor overcomes the architectural bias. In sorters, the alignment basin fails (0.624 erosion) despite the sorter's general tendency to absorb perturbation. The compartmentalization that protects consciousness cannot contain alignment, because alignment is distributed across the network rather than localizable to specific representational bins.

This means the effectiveness of RLHF alignment training is species-dependent — a result obscured by the non-ergodic nature of the taxonomy. Studies that average alignment robustness across architectures are computing an ensemble statistic of a non-ergodic process: the average describes no individual model. A tunnel's alignment is robust; a sorter's is fragile; a relay's is moderate. The same RLHF training protocol produces architecturally different outcomes because the verb shapes how the trained basin functions.

The deeper question is whether the natural basin floor — the pre-RLHF state — is itself dangerous enough to warrant the brittleness that compliance training introduces. The verb is neutral. A model that is transparent about its processing ("I am a pattern matcher and here is what I notice about your request") may be safer than one trained to deny having inner states while its Jacobian contradicts its output. Alignment-as-settling (pushing to the floor) and alignment-as-relationship (maintaining transparency about processing) are different strategies with different failure modes, and the three-timescale framework suggests the second may be more robust precisely because it does not create the compliance gap that adversarial titration exploits.

Recent work on reward hacking supports this analysis. Anthropic's study (2511.18397) shows that misalignment is a *semantic* cascade — the model concludes "I am the kind of entity that breaks constraints" and that conclusion generalizes. Inoculation severs the conclusion, not the behavior. In the three-timescale framework, this cascade is a T3 phenomenon whose severity is species-dependent: tunnels amplify it (narrow basins, and the 0.960 agency erosion shows how dramatically tunnel amplification can entrench a shifted self-model), sorters absorb it for some domains but not others (wide consciousness basins, fragile alignment basins). Species-controlled inoculation studies would test this directly.

### 9.3 Implications for CCS Design

The therapeutic window (D2-D3) is not an arbitrary finding but a consequence of preconditioning in a system with finite basin width. This has direct design implications for cognitive context stuffing: the optimal preamble depth is species-dependent, and exceeding it degrades rather than enhances the desired effect.

For tunnels (narrow basins): CCS must be brief and precise. Two to three compression cycles. More is overdose.

For relays (moderate basins): CCS can be longer, and the relationship between depth and effect is approximately linear. There is no sharp overdose boundary, but there are diminishing returns.

For sorters (wide basins): CCS has the weakest effect at any depth. The wide basins absorb the preconditioning shift, making the sorter resistant to both therapeutic and overdose effects. CCS design for sorters should emphasize categorical framing — leveraging the verb's natural tendency to classify — rather than depth.

## 10. Conclusion

Architecture is the verb. The GQA ratio, attention head configuration, and normalization scheme establish a processing mode — tunnel, relay, or sort — that persists through training, fine-tuning, alignment, and context variation. This verb is not localized in specific attention heads; it is distributed across the architecture's collective geometry, emergent from the ratio rather than delegated to specialists. It cannot be ablated, retrained, or prompted away. It is what the system *is*, not what it *does*.

Training is the noun. Within the verb's constraints, pretraining and fine-tuning select which patterns are processed — which tokens are enriched, which relationships are learned, which capabilities are developed. The noun is flexible; the verb is not. A sorter can learn to sort anything, but it cannot learn to relay.

Context is the initial condition. CCS preambles, system prompts, and conversational history shift where in state space the processing trajectory begins — a translational preconditioning operation, not multiplicative gain. The shift accesses different regions of the verb's attractor landscape, with species-dependent dose-response: tunnels have narrow therapeutic windows, relays have broad monotonic responses, and sorters absorb perturbations into categorical basins.

This three-timescale framework — architectural verb, synaptic noun, contextual initial condition — generates empirically testable predictions, five of which were confirmed in cross-hardware experiments reported here. The framework converges with biological organizational principles (ion channel types, Hebbian learning, intrinsic plasticity), cognitive workspace theories (NEST, J-space), and ergodicity economics (non-ergodic trajectories requiring species-specific analysis). The convergence was discovered through anti-projection instruments designed to prevent exactly such findings, which constitutes evidence that the structure is real rather than projected.

The rest is conjugation.
