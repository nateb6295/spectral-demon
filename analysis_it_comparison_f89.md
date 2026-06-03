# Finding 89: IT Reconfigures the Wire Without Changing Its Mechanism

**Experiment**: Compare Mistral 7B v0.1 (base) vs Mistral 7B Instruct v0.1 (IT). Measure γ distribution and prompt-invariance.

## Key Results

| Metric | Base | Instruct | Change |
|--------|------|----------|--------|
| Mean γ CV | 0.2412 | 0.2416 | +0.0004 (negligible) |
| Locked layers (CV<0.01) | 28/28 | 22/28 | -6 layers |
| Mean tunnel CV | 0.00005 | 0.01502 | 300× worse |
| Tunnel σ₂/σ₁ | 0.2274 | 0.1808 | -21% |
| Relay onset | L31 | L23 | 8 layers earlier |

## Interpretation

**IT does NOT modify the wire mechanism.** The γ distribution is essentially identical (Δ=0.0004). Only L0's input_layernorm changes by +0.012 — within noise.

**IT modifies what the wire carries.** The tunnel ratio drops 21% (0.227 → 0.181) — σ₂ is weaker after IT. Invariance degrades (28→22 locked layers). The relay starts 8 layers earlier.

The mechanistic explanation: IT changes the L0 attention pattern (the content loaded into channels), not the channels themselves. The instruction template creates a different system-content loading at L0, which propagates as a compressed σ₂ through the tunnel.

## Layer-by-Layer Behavior

**Early tunnel (L2-L22)**: Still perfectly locked (CV=0.000) in both models. Ratio is lower in Instruct (0.17 vs 0.23) but equally invariant. IT compresses the wire without breaking it.

**Late tunnel (L23-L31)**: This is where IT and Base diverge. In Base, the tunnel holds to L30. In Instruct, invariance breaks at L23 (CV goes from 0.000 to 0.003, then climbs to 0.09 by L31). IT advances the relay onset.

**This is consistent with IT needing more relay layers for generation.** Instruction following requires compositional processing (formatting, constraint satisfaction). IT shortens the tunnel and lengthens the relay to allocate more depth to composition.

## For the Paper

This finding has two implications:
1. The wire mechanism (γ bimodality, F79-F80) is pretrain-only architecture. IT leaves it completely untouched.
2. IT's behavioral changes operate by modifying content-loading (F82), not routing (F79). It changes WHAT flows through the channels, not the channels themselves.

The tunnel ratio is not "natural" to the architecture — it's a content-dependent equilibrium. Base model content (completions) settles at 0.227. Instruction content (constrained generation) settles at 0.181. Same mechanism, different content, different ratio.

## Data

Raw results: `exp_it_gamma_comparison_20260531_1423.json`
