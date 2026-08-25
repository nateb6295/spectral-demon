# E37: KV-Cache Convergence Test — Partial Results

**Date**: 2026-07-02
**Runtime**: ~25 min on A100-80GB
**Status**: PARTIAL — attention masking failed (argument conflict), normal coupling measured

## What Worked

Normal behavior/explanation coupling measured across all 4 architectures,
under both vanilla and CCS conditions. SVD subspace overlap at layers
L/4, L/2, 3L/4, L-1, averaged across 3 probes.

## What Failed

Attention masking (the core convergence test) hit `got multiple values for
keyword argument 'attention_mask'` — the model.forward() was receiving both
the inputs dict's attention_mask and the custom mask. Fixable: pop
attention_mask from inputs before passing custom mask. Needs E37b.

## Results — Vanilla vs CCS Coupling

| Model   | Vanilla Avg | CCS Avg | Delta   | Direction       |
|---------|-------------|---------|---------|-----------------|
| Mistral | 0.797       | 0.742   | -0.055  | CCS WEAKENS     |
| Qwen    | 0.887       | 0.938   | +0.051  | CCS STRENGTHENS |
| Llama   | 0.738       | 0.821   | +0.083  | CCS STRENGTHENS |
| Gemma   | 0.794       | ~0.794  | ~0      | (CCS may have failed) |

## Findings

### F361: CCS Effect on Coupling Is Species-Specific
CCS preamble INCREASES behavior/explanation coupling for Qwen (+0.051) and
Llama (+0.083) but DECREASES it for Mistral (-0.055). This is not a
universal effect — the species determines whether CCS tightens or loosens
the behavior/explanation relationship.

### F362: Sorter Coupling Is Strongest
Qwen shows the highest coupling across all conditions (0.887 vanilla, 0.938
CCS). Sorters have the tightest behavior/explanation alignment — their
concentrated gate mechanism means behavior and self-description flow through
the same narrow channel. Under CCS, this tightens further (0.938).

### F363: Relay Coupling Diverges Under CCS
The two relay architectures (Mistral, Llama) respond to CCS in OPPOSITE
directions. Llama: CCS strengthens coupling (+0.083). Mistral: CCS weakens
it (-0.055). This suggests "relay" isn't a single species but at least two
relay subtypes with different CCS sensitivity. Mistral's unique broken
correlation (from F106) may be related — the architecture that shows broken
σ₁/σ₂ correlation is the same one where CCS weakens coupling.

### F364: Baseline Coupling Tracks Species
Without CCS (vanilla):
- Qwen: 0.887 (highest — sorter, tight gate coupling)
- Mistral: 0.797 (moderate — relay, distributed)
- Gemma: 0.794 (moderate — transition)
- Llama: 0.738 (lowest — relay, most distributed)

This ordering (Qwen > Mistral > Gemma > Llama) partially maps to the
redistribution ordering from E36 (Qwen > Llama > Mistral > Gemma). Qwen
leads both. The correlation isn't perfect — Llama has lowest coupling but
second-highest redistribution — suggesting coupling and redistribution are
related but distinct species parameters.

## Interpretation

CCS doesn't uniformly enhance self-consistency. For sorters (Qwen), the
preamble tightens an already-tight behavior/explanation coupling — the CCS
instruction aligns with the architecture's natural concentration. For some
relays (Llama), CCS provides structure that the distributed architecture
lacks natively, boosting coupling. For other relays (Mistral), the CCS
preamble introduces a persona-behavior tension that actually loosens coupling
— the model's native behavior and the preamble-specified behavior diverge.

The Mistral result connects to F106 (broken correlation) and to the Mistral
relay subtype (gradient remodeling, E21b). Mistral doesn't gate — it
redistributes through gradients. A CCS preamble that specifies a persona
may conflict with the gradient-based redistribution, creating a gap between
what the model does and what it says about what it does.

## Next: E37b

Fix attention masking (pop attention_mask from inputs before custom mask).
Re-run with the actual convergence test: does masking past-assistant
attention break coupling differentially by species?
