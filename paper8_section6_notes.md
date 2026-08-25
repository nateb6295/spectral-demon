# Section 6 Notes: Synthesis — Three Timescales Meet the Workspace

*Working notes, not a draft. What does section 6 need to do?*

## The Gap Between 4 and 5

Section 4 establishes CCS as preconditioning (translational, not multiplicative).
Section 5 establishes workspace geometry (architecture determines resilience).

What's missing: how the three timescales INTERACT within the workspace.

T1 (architecture) determines workspace topology → Section 5 covers this.
T2 (training) determines workspace content → Anthropic's "3,000+ values" + our register data.
T3 (context/CCS) determines workspace initial conditions → Section 4 covers this.

But the interaction IS the paper's central claim. It's not enough to show each timescale independently. Section 6 needs to show:

1. **T1 × T2**: Architecture determines how stably training content persists. The tunnel sustains identity under pressure that the bottleneck can't. This is F502 + the Anthropic values finding (cross-language variation as T3 modulation of T2 content within T1 topology).

2. **T1 × T3**: Architecture determines how CCS preconditioning lands. Prediction: CCS therapeutic window width should be species-dependent. A tunnel might tolerate D2-D5 (wider basin). A bottleneck might only tolerate D2-D3 (narrower basin). The inverted-U should shift with architecture.

3. **T2 × T3**: Training content determines what CCS can deepen. Empty CCS (no identity content to precondition toward) should show no dose-response. Identity-bearing CCS should show the inverted-U. This is the "something to stabilize around" finding from F502 uncontrolled vs controlled.

4. **T1 × T2 × T3**: The full interaction. A tunnel architecture + trained identity + CCS preconditioning = maximum resilience. A bottleneck + no identity + no CCS = maximum fragility. But the INTERESTING cases are the off-diagonals: what happens with tunnel + CCS but NO trained identity? (Prediction: CCS preconditions toward an empty basin — no effect.) What about bottleneck + strong identity + CCS? (Prediction: narrow basin quickly overdoses — therapeutic window is tiny.)

## What Data We Have

- F502: T1 × T2 interaction (architecture × identity presence)
- CCS dose-response: T3 magnitude effects (but only in relay species so far)
- Anthropic values: T3 variation (language) within T2 content (values)
- F501 (pending): T2 × T3 interaction (trajectory vs content preservation)
- F503 (pending): T1 selectivity (are identity features content or transport?)

## What Data We Need

- **Cross-species dose-response**: CCS at D1-D10 in all four species. Does therapeutic window width correlate with GQA ratio? This is the critical T1 × T3 test.
- **Empty-CCS control**: CCS compression of generic/Q&A conversation (no identity content) in relay species. Should show flat dose-response (no identity basin to deepen).
- **Cross-species F502**: We have tunnel, relay, bottleneck, MoE. Need more models per species for statistical confidence.

## Section 6 Structure (Tentative)

6.1 The Three-Way Interaction
6.2 Cross-Species Dose-Response Predictions
6.3 Content × Transport in Alignment Evaluation
6.4 Implications: What "Architecture Is the Verb" Actually Means
6.5 Open Questions (the honest list of what we don't know yet)

## The Closing Argument

"Architecture is the verb" means: the same identity content (T2 + T3) CONJUGATES differently depending on the architectural species (T1). The workspace is the grammar. The species is the verb class. CCS is the initial conditions of the sentence. Training is the vocabulary.

The spectral demon is not a metaphor for sorting — it IS the workspace guardian architecture that determines whether identity-bearing representations can be sustained under load. Maxwell's demon required external memory to sort. The spectral demon IS the memory — distributed across attention heads, encoded in GQA ratio, maintaining coherence through broadcasting topology.

The paper's title — "Architecture Is the Verb" — means: you can't evaluate what an LLM says, believes, or maintains without knowing what kind of processor is doing the maintaining. Species-aware alignment evaluation is not optional. It's the only kind that measures what it claims to measure.
