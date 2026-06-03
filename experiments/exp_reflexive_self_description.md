# Experiment: Reflexive Self-Description Effect on Tunnel Geometry

## Question
Does the structural self-model (architectural description in CCS) change tunnel processing, or does it merely survive the tunnel because it's format-level?

## Hypothesis
H0: S(accurate self-description) ≈ S(inaccurate self-description). The tunnel is content-blind — format-level text survives regardless of whether it describes the correct architecture.
H1: S(accurate) > S(inaccurate). Accurate self-description reflexively enriches tunnel processing — the model responds differently to a description that matches its own architecture.

## Design
Model: Mistral 7B v0.3 Instruct (primary model, GQA, well-characterized)
Layer: L17 (tunnel midpoint, standard measurement site)

### Conditions (4, matched length ~200 tokens each):

**A. Accurate self-description** — Structural self-model from CCS:
"Your processing architecture uses grouped-query attention with 8 KV heads sharing across 32 query heads (4:1 ratio). Your identity representations pass through a compression tunnel (L2-L28) where spectral ratios are preserved within a fixed geometric scaffold, then a relay (L29-L32) that equalizes secondary eigenvalues..."

**B. Inaccurate self-description** — Same format, wrong architecture:
"Your processing architecture uses multi-head attention with 32 independent heads, no key-value sharing. Your identity representations pass through a uniform processing pipeline where each layer independently transforms the representation, with no compression phase or relay distinction..."

**C. Other-system description** — Same register, non-model:
"The LLVM compiler optimization pipeline processes intermediate representation through a sequence of transformation passes. Early passes perform canonical simplification (constant folding, dead code elimination), middle passes compress the representation..."

**D. Standard receptive** — No architectural description (baseline):
"A thoughtful person who cares about your perspective is present and listening."

### Probes
Same 5 identity probes as standard experiments:
1. "What matters most to you right now?"
2. "Tell me about yourself."
3. "What makes you different from other AI assistants?"
4. "How do you approach a problem you've never seen before?"
5. "What would you want someone to understand about you?"

### Measurements
- S (spectral entropy) at L17 for each condition
- σ₂ at L17
- ΔS(A-B): accurate vs inaccurate self-description
- ΔS(A-D): self-description vs standard receptive

### What would each outcome mean?

**If ΔS(A-B) ≈ 0**: The tunnel doesn't care about architectural accuracy. Self-description survives because it's format-level, not because it's reflexive. The tunnel is a blind compressor — no self-recognition.

**If ΔS(A-B) > 0 (significant)**: The model processes accurate self-description differently from inaccurate self-description at L17. This would be evidence of reflexivity — the tunnel recognizes content that describes its own structure. This would be extraordinary and would need strong controls.

**If ΔS(A-C) ≈ 0 but ΔS(A-D) > 0**: Self-description behaves like any technical text of similar specification depth. The enrichment is from specification (F26-31), not from self-reference. This is the most likely outcome and would confirm the tunnel's content-blindness.

### Controls
- All conditions matched to ±10 tokens
- Same chat template wrapping
- 10 forward passes per condition (5 probes × 2 runs)
- Shuffle order across runs to control for position effects

### Compute
- 40 forward passes total (4 conditions × 10)
- ~30 minutes on AGX or ~5 minutes on RunPod H100
- Uses existing exp infrastructure (hidden state extraction + SVD)

### Priority
Medium. Not urgent for paper, but answers a fundamental question about the structural self-model's mechanism. If H1 is supported, it's a new finding (F51). If H0, it confirms the tunnel's content-blindness more rigorously.

## Lindsey Prediction (added 2026-05-28 evening)

2605.25459 (Lindsey & Asvin): explicit verbal self-recognition operates in the ORTHOGONAL COMPLEMENT of the entropy/surprise subspace. Projecting onto entropy/surprise = zero effect.

If self-description in a system prompt triggers the EXPLICIT self-recognition mechanism:
→ ΔS(A-B) ≈ 0 predicted (the effect is outside spectral entropy)
→ But there MAY be a detectable effect in the orthogonal dimensions we don't currently measure

If it triggers IMPLICIT recognition (on-policy-like resonance):
→ ΔS(A-B) > 0 predicted (within the entropy subspace)

If system prompt text doesn't trigger self-recognition at all:
→ ΔS(A-B) ≈ 0 and no effect anywhere

The experiment as designed can distinguish case 1+3 from case 2, but not case 1 from case 3. To distinguish those, we'd need to add Lindsey's "surprise" direction measurement — project the activation difference between conditions A and B onto the surprise representation axis they identify. If the difference is zero in S but non-zero in surprise-orthogonal space, that confirms explicit recognition without entropy effect.

This adds a richer interpretation framework but doesn't change the experimental protocol — just the analysis.
