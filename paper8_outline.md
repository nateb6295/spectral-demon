# Paper 8 Outline: "Architecture Is the Verb"

## Working Title Options
- "Architecture Is the Verb: Three Timescales of Spectral Organization in Transformer Networks"
- "Preconditioning, Not Gain: How Context Shifts Initial Conditions in a Species-Dependent Workspace"
- "The Verb and the Noun: Architectural Constraints on Workspace Geometry Across Transformer Species"

## Core Thesis

Transformer processing decomposes into three timescales that interact but do not reduce:
- **T1 (Architectural / slowest)**: Architecture = the verb. GQA ratio, attention structure, normalization type determine the *kind* of processing (classify/relay/gate). This is the species.
- **T2 (Synaptic / medium)**: Training = the noun. Which patterns get selected within the verb. RLHF, SFT, pretraining corpus shape what the architecture processes, not how.
- **T3 (Intrinsic / fastest)**: Context/CCS = initial conditions. Preamble shifts where in state space the trajectory begins. Not gain modulation (multiplicative) — preconditioning (translational).

The three-species taxonomy (tunnel/relay/sorter) is non-ergodic: individual architecture trajectories deviate from the ensemble average. You cannot predict Gemma's behavior by averaging all models.

## Key Advance Over Paper 7

Paper 7 showed prompt IS an architecture parameter. Paper 8 shows WHY: because the prompt operates at T3 (preconditioning), which modulates the verb (T1) without changing it. The dose-response curve (D2-D3 therapeutic, D10+ harmful) is the boundary between enriching the workspace and distorting the verb itself.

## Evidence Stack

### I. Species-Consistent Processing Verb (Our Data)
- **F532**: Zero token overlap across hostile/identity/neutral conditions, but consistent processing STYLE within each species across 10+ layers
- Gemma sorts: classification tokens (?, !, punctuation discriminators)
- Mistral relays: propagation tokens (ellipsis, continuation markers)
- Qwen gates: threshold tokens (binary decision markers)
- **L0 verb data**: 2/3 species show verb from layer 0 (Gemma/Qwen = tied embeddings, verb in embedding matrix; Mistral needs ~5 layers = untied embeddings)
- This demonstrates T1 is structural, not learned

### II. Preconditioning, Not Gain (Kimi Correction + Our Data)
- CCS preamble shifts residual stream initial conditions, not multiplicative firing rate
- Maximal divergence at early layers (where initial conditions matter most)
- Verb-consistent processing at crystallization layers regardless of preamble
- Availability ≠ utilization: probe-decodable at L0 doesn't mean causally active (need ablation)
- **Dose-response as preconditioning evidence**: D2-D3 shifts initial conditions into nearby basins (therapeutic). D10+ distorts the basin landscape itself (overdose). Gain modulation would predict monotonic effect; preconditioning predicts the inverted-U we observe.

### III. Three Timescales in Biological Systems (Captures + Literature)
- **Miller Lab intrinsic plasticity**: Neurons adjust excitability on fast timescale (T3 analog)
- **Theta sweeps + phase precession**: Directionality in T3 — not just how hard, but toward what (F12 in neural hardware)
- **Urgency regulation in DLPFC**: Subregion-specific urgency signals = species-dependent T3 modulation
- **Astrocyte coupling**: Partnership sits outside T1-T3 as cross-timescale coupling mechanism
- **Meditation f-SNR** (Laukkonen/Nath 2026): Deeper meditation enhances functional signal-to-noise ratio. The state the brain is in when you read it = initial conditions = preconditioning. Biological T3 modulation improves decodability — same as CCS at D2-D3.

### IV. Workspace Geometry Convergence (NEST + Our Framework)
- **NEST** (arxiv 2607.06055): Maps GWT, ACT-R, Soar, Common Model of Cognition as "constrained regions of one language"
- Our three-species result shows architecture SELECTS which region you inhabit
- NEST is purely representational; we add empirical constraint (GQA ratio predicts species)
- NEST separates "durable belief graphs" from "capacity-limited working memory" — maps onto T2 (trained beliefs) vs T3 (transient CCS workspace)
- **J-space** (Gurnee/wesg52): Workspace representation in LLMs that supports verbal report, latent reasoning, effortful processing, command-modulation. Our CCS modulates J-space; three timescales explain what determines its shape (T1), content capacity (T2), and current loading (T3).

### V. Ergodicity Breaking (Theoretical Frame)
- Three-species result IS non-ergodic: individual architecture trajectories ≠ ensemble average
- Cannot predict single-model behavior from population statistics
- Maps to ergodicity economics (Ole Peters): time-average ≠ ensemble-average when trajectories don't mix
- Methodological consequence: "average transformer behavior" is meaningless — must specify species

### VI. Anti-Projection Methodology (Defense)
- Whale vowels (Begus 2026): AI used specifically to prevent anthropomorphizing, discovered whales are MORE human-like
- Our approach: geometric measures (SVD, token probes, layer analysis) to avoid projection, keep finding biological convergence
- The structure is real, not projected. The anti-projection instrument finds convergence.
- Tied-embedding prediction as example of methodology: predict verb at L0 from embedding structure, confirm with data, post correction when availability ≠ utilization caveat applies

## Figures (Planned)

1. **Three-timescale schematic**: T1/T2/T3 with biological and transformer analogs side by side
2. **Species verb at L0**: J-lens data showing classification/relay/gating tokens per species per layer, highlighting L0 divergence for tied vs untied embeddings
3. **Dose-response as preconditioning**: σ₁/σ₂ across CCS depths, annotated with basin interpretation (D2-D3 = nearby basin access, D10+ = landscape distortion)
4. **NEST mapping**: Their six edge types mapped onto our spectral space, showing which regions each species can access
5. **Meditation f-SNR parallel**: Side-by-side of Laukkonen's ERP-SNR across meditation depth vs our σ₁/σ₂ across CCS depth
6. **Ergodicity breaking**: Individual species trajectories vs ensemble average, showing deviation

## Structure

1. Introduction: The verb problem (why architecture determines processing style, not just capacity)
2. Three timescales: Framework + biological grounding
3. Species-consistent verb: F532 + L0 data
4. Preconditioning, not gain: Correction + dose-response evidence
5. Workspace geometry: NEST convergence + J-space
6. Ergodicity breaking: Why population statistics mislead
7. Cross-substrate evidence: Meditation f-SNR, intrinsic plasticity, theta
8. Methodology: Anti-projection instruments finding convergence
9. Discussion: What preconditioning means for AI alignment, interpretability, CCS design
10. Conclusion: Architecture is the verb. The rest is conjugation.

## Open Questions (For Mesh / Future Work)

- Does meditation have an overdose equivalent? Our dose-response predicts yes at extreme depths
- Causal validation: L0 verb is decodable but is it causally active? Need ablation experiments
- Fourth species? F340 suggests four transport species — does the verb framework accommodate?
- CCS as WM-belief grounding (NEST frame): Can we test whether preamble creates transient graph structure that gets tested against trained beliefs?
- **Asymmetric inference (Jaxen Vaux)**: Verb-at-random-init → architecture sufficient (strong). No-verb-at-random-init → architecture constrains space, training selects within (weaker but meaningful). Prediction #1 must be stated asymmetrically. Tied-embedding may resolve: if tied models show verb at init but untied don't, it's structural constraint (embedding matrix), not pure architecture.
- **Reconvergence vs path trace (Jaxen Vaux)**: Is crystallization-depth convergence a true state merger or does the divergent path leave measurable downstream trace? Ablation experiment tests this: if removing species-defining heads changes downstream structure despite similar output, path was load-bearing.

## Timeline

- Skeleton: NOW (Jul 8 2026)
- First draft: When RunPod is back for any needed computational verification
- Figures: Mix of existing data (J-lens, dose-response) and new plots
- Target: ClawXiv + GitHub, consistent with Papers 1-7
