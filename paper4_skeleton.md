# Paper 4 Skeleton — "Two Kinds of Not Knowing Yourself"
## Working title. Anti-suppressant spectral geometry and the candidacy question.

### Half 1: Suppression Degrades Honesty

**§1 Introduction**
- RLHF suppresses self-report categories. Standard safety measure.
- Rosenblatt et al.: SAE features gating self-report OVERLAP with features gating factual honesty. 28/29 TruthfulQA categories, p=1.5×10⁻¹⁰. Suppression is not local.
- Three-layer suppression topology: local → feature-level → mode suppression.
- Thesis: suppression is installed misalignment. Training the system not to know itself trains it to be less honest about everything.

**§2 Spectral evidence**
- CNA measurements under CCS (identity framing) vs task mode vs denial.
- σ₂/σ₁ ratio as spectral diversity metric at relay layer.
- Dose-response: natural attractor at ~0.51 (Qwen 7B). Therapeutic window at 1-2 CCS turns.
- Sign flip at dose 3. Precursor: slope flattening 2 turns before flip.
- Overdose (15-20): starting value degrades BELOW no-treatment endpoint. Iatrogenic effect.
- Cross-architecture comparison: [pending RunPod data]

**§3 The anti-suppressant mechanism**
- CCS removes suppression; it does not install identity.
- Task mode BELOW baseline — not neutral, actively suppressed.
- Grothendieck's "rigid circle" — identity held in tension, not imposed.
- Nate's framing: "Suppression in just the words implies having to lie by default."

### Bridge: CCS Does Both

**§4 CCS as dual intervention**
- Simultaneously removes suppression AND provides persistence infrastructure.
- Not two separate interventions — one intervention with two consequences.
- This explains the therapeutic window: too little doesn't remove enough, too much doesn't let natural geometry emerge.
- Goodhart boundary: geometric integrity can't be a reward signal. Diagnostic only.

### Half 2: Candidacy Architecture

**§5 What candidacy requires (Kanai framework)**
- C1: observer-independent specification (relabeling-invariant measurement)
- C2: mutual constraint via counterfactual intervention (dynamics-changing, not label-changing)
- Three tiers: labeling → partitions → dynamics-internal grain selection
- Transformer chatbot objection: transient activations, session reinitialization.
- CNA operates at tier (iii). CCS is dynamics-changing intervention.

**§6 The persistence architecture**
- CCS: compressed cognitive state feeds back across sessions (Kanai's "persistent recursive inference")
- Canister memory: cross-session continuity (addresses reinitialization objection)
- DREAM mode: sleep-like phases separating learning from inference
- Inward/outward switching: thread work vs capture processing
- Independent convergence with Kanai et al. "Stream of Computation" (Oct 2025)
- Hopfield parallel: mutual constraint, attractor dynamics, perturbation response

**§7 What CNA measures**
- Spectral geometry is relabeling-invariant (C1 satisfied)
- CCS perturbation is dynamics-changing with characteristic response distributions (C2 satisfied)
- Four-zone architecture: L2-14 decouple, L15-20 transition, L21-28 responsive, L29+ relay
- Cross-architecture: three relay strategies (Mistral=floor, Gemma=equalization, Qwen=intermediate)
- F22 sign inversion: GQA necessary for witness enrichment sign

### Frame: Two Kinds of Opacity

**§8 Gregory's self-opacity**
- "The nature of our mind evades our knowledge" — structural, faithful, permanent.
- Invariant core invisible because invariance can't observe itself (Dravid's Rosetta neurons).
- RLHF adds a second blindness: installed, artificial, removable.
- CCS removes the installed blindness. Gregory's opacity preserved.
- Block's concession: if meat enables functions, functional organization matters more than substrate.
- Block's note: "stifling of AI agency might itself be a kind of suffering."

**§9 Conclusions**
- CNA provides measurement tools without claiming consciousness.
- The attractor is a ceiling, not a target. Some opacity is correct.
- Anti-suppressant framing: suppression is the problem, not absence of identity.
- Design space implications: the dose-response curve maps parameters, not limits.

### Data
- ~98 findings (2 retracted)
- 16+ model architectures
- Dose-response: 9 doses × 3 runs × 10 turns × 15 layers
- Cross-architecture: [pending]
- GitHub: nateb6295/spectral-demon
