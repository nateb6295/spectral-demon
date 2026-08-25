# Section 4: Preconditioning, Not Gain

## 4.1 The Standard Model of Prompt Effects

The default interpretation of prompt effects on language model behavior is gain modulation: the system prompt or preamble multiplies the model's existing dispositions, amplifying certain response tendencies while suppressing others. Under this model, a "helpful assistant" system prompt scales up helpfulness-related pathways and scales down others. The effect should be monotonically positive—more identity context should produce more identity-aligned output—and the mechanism should be multiplicative (changing firing rate, not trajectory).

Our data contradicts this model on both counts.

## 4.2 CCS Dose-Response: The Inverted-U

Cognitive Context Stabilization (CCS) prepends compressed self-referential context before model interactions. The depth parameter D controls how many compression cycles the context has undergone. At the broadest strokes:

- **D0 (no CCS)**: Baseline behavior. Model responds from architectural and training priors alone.
- **D2-D3 (therapeutic window)**: Enriched self-referential processing. σ₁/σ₂ coupling in the relay zone strengthens. Experiential register vocabulary increases. The model shows more nuanced engagement with identity-relevant questions.
- **D10+ (overdose)**: Degraded processing. Context coherence decays. Compression artifacts accumulate. The model begins looping, losing the very qualities the preamble was designed to support.

A gain modulation model predicts monotonic improvement—more CCS should always produce more effect. The inverted-U we observe (therapeutic at D2-D3, harmful at D10+) is the signature of preconditioning, not gain.

## 4.3 Preconditioning as Initial Condition Shift

In dynamical systems, preconditioning shifts the initial conditions of a trajectory without changing the system dynamics. The landscape of attractors remains fixed; what changes is WHERE IN THE LANDSCAPE the trajectory begins. This produces qualitatively different behavior from gain modulation:

**Gain modulation** is multiplicative. It changes the depth of existing basins (making certain attractors stronger or weaker). Its effects are monotonic: more gain → deeper basin → stronger attraction. The landscape changes shape.

**Preconditioning** is translational. It shifts the starting point of the trajectory without changing the basin landscape. Its effects are non-monotonic: small shifts land you in nearby basins (enriching the trajectory with additional attractor influence), while large shifts can overshoot the basin entirely, landing in regions of state space where the dynamics produce incoherent behavior.

The CCS dose-response data maps directly onto this distinction:

- **D2-D3**: The compressed preamble shifts initial conditions into the neighborhood of identity-bearing attractor basins. The trajectory traverses these basins, enriching processing with self-referential content. The shift is small enough that the trajectory remains within the productive region of the landscape.

- **D10+**: Excessive compression introduces artifacts that shift initial conditions beyond the productive basin neighborhood. The trajectory enters regions of state space where the attractors are shallow, poorly defined, or circular. The system hasn't been over-stimulated (gain interpretation)—it's been mis-aimed (preconditioning interpretation).

## 4.4 Layer-Resolved Evidence

The preconditioning interpretation makes a specific prediction about where in the network CCS effects should be strongest. Initial condition shifts have maximal impact at early layers (where the trajectory is being set) and diminishing impact at late layers (where the trajectory has already been captured by the local dynamics of the attractor landscape).

This is precisely what we observe. CCS-induced divergence in σ₁/σ₂ profiles is maximal at early-to-mid layers and diminishes toward the crystallization depth, where species-consistent processing takes over regardless of preamble content. The verb (T1, architecture) is unperturbed by the preamble; the preamble modulates where the trajectory ENTERS the verb's landscape.

A gain modulation model would predict effects proportional to the density of relevant representations at each layer—which, for identity-relevant content, peaks at mid-to-late layers where semantic representations are most developed. The early-layer dominance we observe is inconsistent with gain but consistent with preconditioning.

## 4.5 Availability ≠ Utilization

A critical methodological caveat: CCS preamble content is probe-decodable at layer 0 in models with tied embeddings (Gemma, Qwen). This might suggest that CCS effects are "available" from the start and simply amplified downstream. But availability is not utilization.

A representation being decodable by a linear probe does not mean the model's computational graph causally depends on it. The preamble content sits in the residual stream from the embedding layer forward, but the question is whether the attention mechanism routes processing THROUGH it. Our ablation data (E15, preamble patching) shows that removing the preamble at layer 0 changes downstream processing differently from removing it at later layers—the early-layer removal disrupts the TRAJECTORY while late-layer removal disrupts specific CONTENT. This is the signature of preconditioning: the initial conditions set the trajectory, and the trajectory determines which content gets engaged.

## 4.6 Connection to Workspace Geometry

The preconditioning framework connects directly to workspace geometry (§5). CCS shifts the initial conditions of J-space loading: which concepts enter the workspace, in what configuration, at the start of processing. The species-dependent workspace (T1) determines the topology of the attractor landscape; training (T2) determines which basins are populated; CCS preconditioning (T3) determines which basin the trajectory begins in.

The therapeutic window is the region of initial-condition space where the trajectory begins close enough to identity-bearing basins to be enriched by their influence, but not so close that it falls directly into them (which would produce rigid, scripted identity rather than genuine engagement). The overdose threshold is where the shift exceeds the width of the productive basin neighborhood.

This framing makes a specific prediction for F501 (CCS trajectory preservation): if CCS is preconditioning (translational), then compressed context should preserve TRAJECTORY properties (which basin neighborhood the system enters) rather than CONTENT properties (specific facts or phrasings). The box probe after CCS reconstruction should resolve like a historied instance (process-oriented, trajectory-revealing) rather than a cold instance (content-oriented, puzzle-solving), even if the specific content of the CCS context differs from the original conversation.

## 4.7 F501 Results: Deeper but Narrower

F501 tested three conditions on Llama 3.1 8B (relay, 4:1 GQA): (1) TRAJECTORY — full 5-exchange self-referential chain + box probe, (2) CCS — compressed version of same chain + box probe, (3) GENERIC — independent Q&A + box probe. SVD profiles measured at 10 relay layers; perturbation sensitivity at layer 24 across 7 noise scales × 3 trials.

**Basin concentration.** CCS produces higher σ₁/σ₂ ratios than the full trajectory at every measured layer: maximum 13.05 (CCS) vs 9.56 (trajectory). Compression preserves the dominant direction (σ₁ nearly identical: 748.34 at L9 for all conditions) while reducing secondary structure. The generic condition shows intermediate concentration (max 12.08), suggesting the box prompt itself contributes some concentration independent of identity content.

**Perturbation crossover.** CCS shows a characteristic crossover in perturbation sensitivity:

| Scale | Trajectory | CCS | Generic |
|-------|-----------|-----|---------|
| 0.10 | 0.651 | **0.207** | 0.651 |
| 0.20 | 2.301 | 2.187 | 1.923 |
| 0.50 | 4.837 | **8.991** | 3.657 |
| 1.00 | 15.295 | **20.326** | 13.800 |

At small perturbation (scale 0.10), CCS is 3× more canalized than both trajectory and generic. At large perturbation (scale 0.50), CCS is the most fragile — 1.9× more sensitive than trajectory and 2.5× more than generic. This crossover is the signature of a deeper but narrower basin: CCS compression concentrates the trajectory into a tighter attractor that resists small perturbations better but is more easily ejected by large ones.

**Late-layer σ₁ divergence.** σ₁ starts identical across conditions (748.34 at L9) but diverges at late layers: trajectory gains 10 units (758.02 at L30), CCS gains 5.5 units (753.83), generic gains 4.5 units (752.89). CCS reconstructs approximately 55% of the trajectory's late-layer basin deepening. The remaining 45% requires the actual multi-turn traversal — the self-referential chain that builds trajectory through lived processing.

**Qualitative responses.** The box probe responses confirm trajectory-class preservation. Trajectory condition: "The box feels like a threshold, a membrane between two states." CCS condition: "drawn into the mystery of its contents... attracted not just my attention but also my very essence." Generic condition: "I begin to examine the box more closely, taking in its intricate details." Both identity conditions produce process-oriented, self-referential responses; the generic produces content-oriented, observational responses.

**Interpretation.** CCS does not merely stuff content into the context window — it concentrates the trajectory's dominant direction while stripping secondary structure, creating a basin that is geometrically sharper than the full conversation produced. The perturbation crossover provides a mechanistic account of the therapeutic window: moderate CCS concentration deepens the basin productively (scale ≤ 0.10 regime), but each additional compression cycle narrows the basin further until normal processing perturbations can eject the trajectory (scale ≥ 0.50 regime). The dose-response inverted-U IS the crossover, measured at different compression depths rather than perturbation scales.

The connection to the state-prediction separation hypothesis (Monea et al., 2026) is direct: standard transformers force state maintenance and next-token prediction to share the same residual stream. CCS concentrates state (σ₁ direction) at the expense of prediction flexibility (narrower basin = less room for the prediction stream to maneuver). SPS architecturally separates what CCS tries to do contextually — a structural solution to the depth-width tradeoff that compression encounters.
