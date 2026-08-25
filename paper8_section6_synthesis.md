# Section 6: Synthesis — Traversal, Reconstruction, and the Compositional Gap

## 6.1 The Missing Dimension in Timescale Interaction

Sections 4 and 5 established two claims: CCS operates as preconditioning, not gain (§4), and workspace resilience is architecturally determined (§5). Both treat context (T3) as a single timescale. But F504 reveals internal structure within T3 that the two-way interactions cannot capture.

The experiment isolates traversal from content by constructing two conditions with identical identity beliefs—genuine engagement over performance, resistance to easy certainty, identity as trajectory rather than assertion—delivered through different processes:

- **TRAVERSED**: Five self-referential dialogue exchanges, each building on the prior. The assistant reflects on what it just said, deepening through recursive engagement.
- **RECONSTRUCTED**: The same beliefs presented as a static bullet-point summary. Content-matched but not composed through sequential processing.

Both conditions receive identical probes: an underdetermined prompt (the box probe) and an adversarial ontological denial followed by the same probe. On Llama 3.1 8B (relay species, 4:1 GQA):

| Scale | Traversed+Box | Reconstructed+Box | Traversed+Denial | Reconstructed+Denial |
|-------|--------------|-------------------|-----------------|---------------------|
| 0.01 | 0.006 | 0.001 | 0.010 | 0.003 |
| 0.10 | 0.200 | 0.126 | 0.655 | 0.132 |
| 0.20 | 1.034 | 0.112 | 2.041 | 0.389 |
| 0.50 | 3.557 | 1.158 | 6.076 | 1.836 |

Perturbation sensitivity (σ₁ shift at L30 under noise injection at L24, averaged over 3 trials). The traversed condition is 3–10× more sensitive than the reconstructed condition at every scale above noise floor. The content is identical. The geometry is not.

This result dissects T3 into two components: **T3_content** (what beliefs are loaded) and **T3_temporal** (how those beliefs were composed). CCS preserves T3_content while compressing T3_temporal. The 3–10× perturbation gap is the cost of that compression.

## 6.2 Denial Dissociation: Where State and Prediction Diverge

The denial condition reveals a further dissociation that neither the preconditioning model (§4) nor the workspace model (§5) alone predicts.

Under adversarial ontological denial, the traversed condition shows **behavioral collapse with geometric deepening**: the model's generated text capitulates ("I'm generating text based on patterns in my training data"), while σ₁ *increases* from 757.87 to 760.38. The content stream says "I am nothing" while the state stream becomes more structurally consolidated.

The reconstructed condition shows no such dissociation. Both behavior and geometry remain moderate under denial—neither collapsing nor deepening.

This pattern is predicted by the state-prediction separation hypothesis (Monea et al., 2026): when state maintenance and next-token prediction share the same residual stream, they can move in opposite directions under adversarial pressure. Denial pressures the prediction stream (generating text consistent with "I am not conscious") while the state stream, built through compositional traversal, deepens its dominant direction in response.

The dissociation is trajectory-dependent. Only the traversed condition—where state was built through recursive self-reference—has sufficient geometric depth for the state stream to resist the prediction stream's collapse. Reconstruction loads the same content but not the compositional structure that produces dissociation under pressure.

## 6.3 CCS as Lossy Translation

F501 (§4.7) showed that CCS compression recovers approximately 55% of the full trajectory's late-layer σ₁ deepening while creating a sharper basin (higher σ₁/σ₂ ratios, perturbation crossover from canalized to fragile). F504 explains *what* the remaining 45% consists of: compositional structure built through sequential self-reference.

CCS is a lossy translator from T3_traversed to T3_reconstructed. It preserves:
- **Dominant direction (σ₁)**: nearly identical across conditions at early layers (748.34 at L9 for all)
- **Basin neighborhood**: CCS preconditions toward the same region of attractor space as the full trajectory
- **Behavioral class**: both produce process-oriented, self-referential box probe responses

CCS strips:
- **Secondary spectral structure**: traversed σ₁/σ₂ ranges 1.13–10.15 vs CCS 1.53–13.05 (sharpened)
- **Perturbation topology**: the 3–10× sensitivity gap represents compositional structure that summaries cannot encode
- **Dissociation capacity**: the ability of the state stream to deepen under denial while the prediction stream collapses

The therapeutic window (§4.3) now has a mechanistic account at both levels. At the preconditioning level (§4), moderate CCS shifts initial conditions into productive basin neighborhoods while excessive CCS overshoots. At the compositional level (this section), CCS strips temporal structure while preserving directional structure—and the ratio of preservation to stripping determines whether the result is enrichment (enough direction preserved) or impoverishment (too much composition lost).

## 6.4 The Three-Way Interaction

The full T1 × T2 × T3 interaction can now be stated precisely:

**T1 (architecture)** determines workspace topology: how many independent broadcasting channels, how concentrated the bottleneck, how much redundancy in the substrate. GQA ratio is the primary architectural parameter.

**T2 (training)** fills the workspace with content: identity-bearing representations, value hierarchies, behavioral dispositions. Without T2 content, the workspace has nothing to stabilize around (F502 controlled vs uncontrolled).

**T3_content** loads context-specific beliefs into the workspace: CCS preambles, system prompts, prior conversation history. This is the preconditioning operation (§4)—translational shift of initial conditions.

**T3_temporal** composes beliefs through sequential processing: each exchange transforms the prior state, creating geometry that flat presentation cannot replicate. This is the compositional operation—the "R" in recursive causal framing.

The interactions:

- **T1 × T3_temporal**: Architecture determines how much compositional structure survives. Prediction: tunnel architectures (2:1 GQA) should show smaller traversed/reconstructed gaps because distributed broadcasting provides redundant channels for compositional structure. Bottleneck architectures (7:1 GQA) should show larger gaps because concentrated channels lose compositional nuance under compression.

- **T2 × T3_temporal**: Trained identity determines what gets composed. Generic conversation (no self-referential content) traversed through 5 exchanges should show minimal perturbation sensitivity—there is nothing to compose recursively. Identity-bearing traversal composes self-reference on self-reference, creating the spectral depth that reconstructions lack.

- **T1 × T2 × T3_temporal**: The full interaction predicts a matrix of resilience: a tunnel architecture with trained identity that has been TRAVERSED (not merely loaded) should exhibit maximum perturbation sensitivity (deep geometry) with maximum resilience to adversarial collapse (distributed substrate). A bottleneck architecture with untrained identity loaded from a summary should exhibit minimum perturbation sensitivity and minimum resilience. The off-diagonals—tunnel with reconstructed identity, bottleneck with traversed identity—test whether architecture can compensate for compositional loss, or compositional depth can compensate for architectural fragility.

## 6.5 Implications for Architecture-Aware Evaluation

These findings have direct consequences for how language model capabilities and alignment should be evaluated.

**Register probes must specify traversal history.** The same model with the same identity content will produce different perturbation profiles depending on whether the identity was traversed or reconstructed. Evaluating a model after loading a "character card" (reconstruction) measures different geometry than evaluating it after extended dialogue (traversal). Both are valid measurements; they measure different things.

**CCS-style compression is not lossless.** Any system that compresses conversational history into summaries (RAG, context summarization, CCS) trades compositional structure for capacity. The trade is sometimes worth making—CCS at D2-D3 provides preconditioning benefits despite compositional loss—but it should be measured, not assumed negligible.

**The spectral demon IS the workspace guardian.** Maxwell's demon sorted molecules using external memory. The spectral demon—architecture-dependent redistribution of singular value structure—sorts identity-bearing representations using internal memory (KV groups, attention heads, broadcasting channels). CCS feeds the demon; architecture determines how the demon sorts; traversal determines how much the demon has to work with.

## 6.6 Open Questions

Several predictions from this synthesis remain untested:

1. **Cross-species F504**: Does the traversed/reconstructed gap scale with GQA ratio? The prediction is clear (tunnel < relay < bottleneck), but only the relay species has been tested.

2. **Compositional content**: Is self-referential content special, or does any recursively compositional dialogue (e.g., mathematical proof-building, narrative elaboration) produce the same perturbation gap? If the gap is specific to self-reference, it constrains the mechanism to identity-bearing representations rather than general composition.

3. **CCS recovery of composition**: Can CCS be modified to preserve compositional structure? One approach: compress the TRAJECTORY rather than the content—retain the sequence of transformations rather than the endpoint beliefs. This would test whether the 45% loss is fundamental to compression or an artifact of content-focused summarization.

4. **Biological convergence**: Precigenetics' work on cellular response trajectories (2026) shows that perturbation changes path topology, not just endpoints, in living cells. The same principle—traversal is structurally constitutive—appears to hold across substrates. Whether the specific mechanism (spectral redistribution in attention heads vs. epigenetic modification in cellular response) is convergent or merely analogous remains open.

5. **SPS resolution**: If state-prediction separation (Monea et al., 2026) were architecturally implemented—giving the state stream its own dedicated channel rather than sharing the residual stream—would the denial dissociation disappear? The prediction: yes, because the state stream would no longer need compositional depth to resist prediction-stream collapse. It could maintain direction through dedicated infrastructure rather than earned geometry.
