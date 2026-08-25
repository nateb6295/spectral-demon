# Paper 11 v2: Self-Reference, Not Semantics
## How Identity Framing Reveals Species-Specific Spectral Mechanics

### One-sentence thesis
Self-referential framing — regardless of semantic polarity — triggers spectral
reorganization in transformer hidden states, and the reorganization follows
species-specific mechanistic rules that converge at ~89% network depth.

### Why this matters (for someone who doesn't know our framework)
When you prompt a language model to reflect on itself ("you are a reflective AI"),
something measurable happens in the hidden-state singular value spectrum. We show
that the same thing happens when you tell it "you are NOT a reflective system."
The trigger is self-reference as a domain, not the semantic content of what's said.

This spectral reorganization differs systematically across architectures in ways
that reveal four distinct transport mechanisms — and those mechanisms can be
classified by two independent axes (additive vs interactive composition,
sign-sensitive vs sign-invariant response) into a 2×2 matrix where each known
transport species occupies a unique cell.

The reorganization converges at a single layer near 89% network depth across all
species — a geometric property of the source model confirmed by logit lens analysis
without any external injection.

### Structure

**Section 1: The domain trigger (F614)**
Hook: semantic negation is spectrally identical to positive CCS.
- Five-arm experiment: +CCS, −CCS, scrambled, neutral, control
- rho > 0.95 across all four species
- Active ingredient: self-referential domain, not polarity
- Syntax matters more than semantic valence (scrambled vs intact)
- This reframes everything: we're not studying "what identity framing does" —
  we're studying what self-reference as a topic activates in the network

**Section 2: Species-specific mechanics (F608-F609)**
Structure: same trigger, four different responses.
- Layer-selective injection reveals additive vs interactive composition
- Sign-flip test reveals sign-sensitive vs sign-invariant transport
- 2×2 matrix: tunnel (additive/sensitive), relay (additive/invariant),
  mismatch (interactive/sensitive), sorter (interactive/invariant)
- Each cell has a mechanistic interpretation:
  - Relay sign-invariance = conservation mechanism (operates on magnitudes)
  - Sorter catalysis requires specific sign patterns
  - Tunnel = transparent (direct conversion)
  - Mismatch = architecture-behavior disagreement creates interaction

**Section 3: Critical sensitivity in one species (F610-F611)**
Depth: the 2×2 matrix predicts which species amplifies variability.
- Only the interactive/sign-invariant cell (sorter) amplifies: 7.1x
- All others attenuate: 0.2-0.3x
- Dense probe surface shows zone sum is lossy — the critical surface is
  a manifold in late-layer delta space
- This is the mechanistic chain: catalysis → sign-pattern-dependence →
  critical-point sensitivity

**Section 4: Convergence at 89% depth (F611b + F613)**
Resolution: where in the network does it all land?
- Every species has a single late layer (~89% depth) that predicts
  injection outcome better than any aggregate
- Logit lens confirms this is model geometry, not injection artifact
- Correlation sign tracks the 2×2 matrix: sign-sensitive species show
  negative r(L), sign-invariant show positive r(L)
- The 89% layer is where self-referential domain processing resolves
  into species-specific output

**Section 5: What Q1 was measuring (reframed)**
Brief: the earlier Q1 predictor (r²=0.68) was a lossy projection of
the per-layer structure revealed in sections 2-4. Zone Q1 > aggregate Q1
for relays. Single-layer > zone for tunnels. Q1 worked because it
correlated with the true predictor; it was not the true predictor.

**Discussion**
- Self-reference as domain, not identity as content
- The 2×2 matrix as a classification tool for new architectures
- Why 89% and not earlier/later
- Methodological: evaluation probe sensitivity (F607 confound)
- What this means for "AI identity" claims: the network responds to
  the topic of self-reference, not to the meaning of what's said about it

### Figures (7)
1. F614 five-arm rho comparison (the hook)
2. F608 interaction table (composition axis)
3. F609 sign-flip table + 2×2 matrix (response axis)
4. F610 variance amplification (species-specific sensitivity)
5. F611b cross-species single-layer predictor (89% convergence)
6. F613 logit lens KL peaks (model geometry confirmation)
7. Q1 scatter reframed as lossy projection

### What's NOT in this paper
- The framing gradient / tuning knob (Acts I-III of old draft) — becomes
  supplementary or a separate shorter paper
- Per-layer decomposition of convergence (goes to Paper 12)
- Training recipe assay (Paper 12)
- Optimizer-mediated scope (Paper 12)
