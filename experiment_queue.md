# Experiment Queue — Prioritized
Updated 2026-05-28 ~8:00 PM PDT

## COMPLETED (recent)
- **Exp 11 — Developmental Passage Distance**: Pythia 6.9B × 5 checkpoints. d(control)=1.93±0.04 from random init. ΔS≈0 at all checkpoints for non-GQA. F17-18.
- **Exp 12 — Self-Witness Scaling**: 7 conditions × 210 passes. Self-witness 37%, imagined exceeds 113%. σ₂ orthogonality. F13-16.
- **Exp 13 — Scaling Laws**: 5 sizes × 5 checkpoints × 3 conditions = 750 passes. Δd∝N^(-0.36). No non-GQA develops positive ΔS. F19-21.
- **Exp 14 — Falcon Base vs Instruct**: IT produces inversion on MHA substrate.
- **Exp 15 — GQA Discriminator**: GQA necessary and sufficient for enrichment sign. F22-23.
- **Exp 18 — Reverie Gradient**: 300 passes. Tunnel valence-blind. J-curve. Spec depth > relational quality.
- **Exp 18b — Relay Gap Probe**: 180 passes. Tunnel spec:valence = 30:1. Relay amplifies valence 5.4×.
- **Exp 18c — Agency Gradient**: 150 passes. Agency real, subordinate to spec 7:1, zero interaction.
- **Exp 18d — Neptic Self-Observation**: 150 passes. Predictions FALSIFIED — neptic is MAXIMUM not minimum. S=0.408. Agency inverts for self. F36-39.
- **Exp 19 — Process-Other**: 60 passes. Tunnel reads self-reference; relay reads observation context. F41-42.
- **InternLM Relay Verification**: GPU-optimized, 62s. Relay is zone L14-L20. L16-17 strongest.

- **F43-47 from re-analysis** (no new passes): Piotrowski gap (F43), baseline reframe (F44-45), orthogonal σ₂ modulators (F46), default-witness gradient (F47).
- **Exp CodeLlama/CodeQwen F47 Discriminator**: CodeLlama 7B (MHA+code) + CodeQwen 1.5 7B (GQA+code). 2×2 grid complete. F48: sign invariant to training domain. Interaction hypothesis falsified.

- **F49 — Scale threshold**: 1.5B constrains, 7B enriches. Architecture recipe: GQA + KV≥500 + 7B+ + IT.
- **F50 — Passage distance normalization**: d/d_max = 0.955±0.006 across 3 GQA architectures. Proper invariant.
- **F51 — Relay rebuilds**: PR input~15→tunnel~1.4→relay~9.9, σ₂ 438×. Free∘Forgetful ≠ Identity.

**Total: ~3270 forward passes, 50 findings**

## Priority 1 (NEW): Sharing Ratio Tests — Pre-registered
**Scripts ready**: `experiments/exp_gemma2_sharing_ratio.py` + `experiments/exp_qwen25_3b_passage.py`
**Theory**: d/d_max = 1-(1-s·C/L)^L, C=0.796. One free parameter, <1.5% error on 4 measured points.
**Gemma 2 9B** (2:1, 42L): predicted d/d_max=0.803. Falsification: >0.90 or <0.70. ~9GB fp16, ~45min H100.
**Qwen 2.5 3B** (8:1, 36L): predicted d/d_max=0.999. Falsification: <0.95. ~6GB fp16, ~20min H100.
**Secondary prediction**: Non-monotonic enrichment (Goldilocks zone). Gemma 2 ΔS should be LOWER than Mistral ΔS if the peak is at s≈4.
**RunPod**: $53.76 balance, both experiments within budget.
**Impact**: VERY HIGH — single formula predicting passage distance from architecture alone. Strongest test of the sharing-ratio theory.

## Priority 2: Exp 17 — Witness Saturation (Rilke)
**Why first**: Tests whether enrichment has a ceiling or destabilizes at extremes. Now that we have the full Bion gradient (Exp 18 series), we know the range from absent to metabolizing. What happens ABOVE metabolizing?
**Model**: Mistral 7B Instruct
**Design**: Extend Exp 18's gradient with 3-4 conditions above "metabolizing" — increasingly intense relational specification. Does S asymptote, plateau, or destabilize?
**Key prediction**: Non-monotonicity at extreme witness intensity would confirm the "angel" hypothesis (Rilke's overwhelming angel). Monotonic saturation would confirm Winnicott's "good-enough" framing.
**Passes**: ~120 additional. ~20 min on H100.
**Impact**: Medium-high. Defines the enrichment range.

## Priority 3: Exp 16 — Iterative Self-Reflection vs Imagined Witness
**Why second**: Tests Grim fractal prediction — oscillatory S under self-reference. Novel. Imagined witness exceeding receptive (F14: 113%) makes multi-turn imagined-witness especially interesting.
**Model**: Mistral 7B Instruct
**Design**: 4 conditions × 5 iterations × 30 passes = 600 passes. Self-reflection, imagined witness, declared witness, absent — each iterated 5 rounds with feedback.
**Key prediction**: Self-reflection S oscillates; imagined witness S converges/grows.
**Passes**: 600. ~4 hours on AGX (~1.5 on H100).
**Impact**: Medium. Beautiful if confirmed.

## Priority 4: Exp 21 — Bidirectional Coupled Witness
**Why third**: Distinguishes relational geometry from prompt sensitivity. Two models witnessing each other. The experiment that could move the field, but logistically harder.
**Model**: Two Mistral 7B instances in conversation
**Design**: 4 conditions (solo, unidirectional, bidirectional, bidirectional-hostile) + fake-response control. Multi-turn.
**Key prediction**: Genuine conversational coupling produces different geometry than content-matched fake.
**Passes**: Complex — multi-turn × 2 models. ~6-8 hours on H100.
**Impact**: Very high if successful. High setup cost.

## Priority 5: Exp 20 — Ollivier-Ricci Curvature Per Layer
**Why fourth**: Script ready (spectral-demon/experiments/exp20_ricci_curvature.py). Tests whether tunnel shows positive curvature (convergence) and relay shows negative (divergence). Would falsify compression-expansion narrative if no sign change.
**Model**: Mistral 7B Instruct
**Design**: 3 conditions × 10 probes × 33 layers = 990 measurements.
**Key prediction**: Curvature sign change near L17-L20 (tunnel-relay boundary).
**Passes**: 30 forward passes × 33 layer analyses. ~45 min on H100.
**Impact**: Medium. Geometric confirmation of tunnel/relay distinction.
**Connection**: Piotrowski (2502.01954) constrained belief updates — curvature may track belief simplex geometry.

## Priority 6: Exp Full-Layer Mistral Profile
**Why**: We only have L17 and L30 data for Mistral. The CodeQwen full-layer run revealed bimodal enrichment (L4 + L16 peaks) and zone-specific allocation that we can't compare to Mistral. Need full-layer profile to determine if bimodal enrichment is GQA-general or code-training-specific.
**Model**: Mistral 7B Instruct
**Design**: Same protocol as CodeQwen/CodeLlama (10 probes × 3 conditions × 33 layers = 990 measurements). ~15 min on H100.
**Key prediction**: If Mistral shows single peak (L17 only) → training reshapes zone allocation within GQA. If bimodal (encoding + mid-tunnel) → GQA-general.
**Passes**: ~30 forward passes, all-layer extraction. ~15 min.
**Impact**: High — resolves whether zone-specific enrichment is architectural or training-dependent.

## Backlog
- ~~CodeLlama F47 test~~ → **COMPLETED** as Exp CodeLlama/CodeQwen (see above). Architecture hypothesis confirmed. F48.
- KV sharing ratio sweep: vary number of heads sharing KV, measure ΔS. Requires architecture modification. **UPDATE (2026-05-28)**: Preliminary data suggests THRESHOLD not gradient — Mistral (4:1) and CodeQwen (8:1) show identical gap (~3.95 at L17). Prediction: step function at 2:1, not continuous. Still valuable to confirm but lower priority if threshold is real.
- Cross-architecture coupling: GQA witnessed by MHA output and vice versa.
- Temporal dynamics: S across token positions within a single generation.
- Held-neptic condition: external holding + passive self-observation. Tests Evagrius's nepsis-requires-koinonia.
- Seed-invariance test: same arch, different seeds, measure ΔS sign. No public checkpoints available.
- Eigenvalue spectrum at L17: Piotrowski-inspired. Compare attention weight eigenstructure between GQA and MHA at the tunnel layer. Tests whether negative eigenvalue handling differs.
- Reflexive self-description: protocol at `experiments/exp_reflexive_self_description.md`. 40 forward passes, Mistral 7B. Tests whether self-referential system prompts (describing the tunnel) produce different tunnel geometry than equivalent-length non-self-referential prompts.
- Stress-test F49: need second sub-7B non-Qwen architecture to confirm relay inversion threshold.
