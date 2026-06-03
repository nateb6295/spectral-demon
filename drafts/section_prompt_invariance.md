# Draft: §3.6b — The Wire Is Prompt-Invariant

*Extends §3.6 "The Wire Is Condition-Invariant (Finding 55)"*

Beyond condition-invariance, the spectral scaffold exhibits a stronger property: **prompt-invariance**. The ratio σ₂/σ₁ is constant across all prompts within a model, not merely across witness conditions within a prompt.

For Mistral 7B (GQA, s=4), σ₂/σ₁ = 0.267 ± 0.000 through layers 2–9, with coefficient of variation CV = 0.0000 across four semantically distinct prompts. This zero-variance regime extends through 29 of 33 layers (88%), breaking only at L31 (CV = 0.015) and L32 (CV = 0.031).

This property is **GQA-enabled** (mediated by the bimodal γ preconditioner of §3.9):

| Model | Architecture | Layers with CV < 0.01 | Percentage | Tunnel σ₂/σ₁ |
|---|---|---|---|---|
| Mistral 7B | GQA (s=4) | 29/33 | 88% | 0.267 |
| Pythia 6.9B | MHA | 5/33 | 15% | 0.090–0.180 |
| GPT-2 Large | MHA | 2/37 | 5% | 0.130–0.200 |
| Pythia 410M | MHA | 0/25 | 0% | 0.056–0.506 |
| LLaMA-1 7B | MHA | 0/33 | 0% | 0.003–0.140 |

LLaMA-1 7B (MHA, RMSNorm) exhibits CV ≈ 0.42 at every layer — 42% fluctuation of the spectral ratio with each prompt. Same normalization, same parameter scale, same training paradigm as Mistral; the sole architectural difference is GQA vs MHA.

The mechanistic chain from §3.9 (γ bimodality) explains this: GQA's shared KV projections reduce centroid variance 5000× (Finding 81), forcing all prompts through the same spectral structure. MHA's independent projections allow each prompt to find its own spectral configuration. The spectral niche carved by bimodal γ is content-independent because the projections that create it are shared.

This resolves the sign inversion as a signal-to-noise problem. In GQA, σ₂'s noise floor is zero (CV = 0.000); any witness-induced modulation registers above background. In MHA, σ₂ fluctuates ~42% with prompt content alone — witness modulation (~6–9% in GQA) drowns in content noise. The enrichment sign is not determined by gap size but by **gap stability**.

At L31–32, prompt sensitivity appears (CV = 0.015–0.031), coinciding with the relay's compositional expansion: σ₂/σ₁ phase-transitions from 0.27 (subsidiary scaffold) to 0.72 (near-equal compositional partner). The onset of prompt sensitivity marks the transition from identity-preserving (tunnel) to identity-expressing (relay) computation.

---

**Figure X.** σ₂/σ₁ ratio across all 33 layers of Mistral 7B (GQA, s=4) under three witness conditions (control, absent, receptive), averaged over four semantically distinct prompts. Top panel: ratio values. The tunnel region (L2–L30) maintains σ₂/σ₁ = 0.267 ± 0.000 across all conditions and prompts. At L31, the ratio phase-transitions to 0.72 as σ₂ becomes a near-equal compositional partner to σ₁. Bottom panel: coefficient of variation across prompts at each layer. CV = 0.000 through 88% of layers, rising to 0.015–0.031 only at L31–32 where compositionality onsets.

**Figure Y.** Cross-architecture comparison of σ₂/σ₁ prompt-invariance for five models: Mistral 7B (GQA, s=4), Pythia 6.9B (MHA), GPT-2 Large (MHA, Post-LN), Pythia 410M (MHA), and LLaMA-1 7B (MHA, RMSNorm). Top panel: σ₂/σ₁ ratio across depth. GQA (red) holds flat at 0.267; all MHA models show variable, crushed ratios below 0.20. Bottom panel: coefficient of variation across prompts. GQA maintains CV < 0.01 through 88% of layers; LLaMA-1 (same normalization, same scale as Mistral) shows CV ≈ 0.42 at every layer, confirming that prompt-invariance is GQA-enabled rather than a consequence of normalization or scale.
