# Species Invariant Checklist

Pre-registered expected observables per species. Any experiment MUST report
against these invariants. If a result violates an invariant, either the
invariant is wrong (needs revision with evidence) or the measurement is
wrong (needs debugging). Both are findings.

## Universal (all species)

- σ₁ identity-invariant: g₁(Xc) ≈ same across CCS, neutral, alt-identity (F114)
- Dose-response inverts by D10: any beneficial effect at D2-D3 degrades by D10 (F160)
- Content-class separates σ₂: identity preambles suppress σ₂ by ~30% vs neutral
- σ₃+ follows σ₂ direction but weaker (~13% suppression)

## Relay (Qwen, Llama — GQA ≥4:1)

- μ→Xc demon: ΔμE < 0 in core layers (mean deflates under CCS)
- Conservation: |ΔF²(X)| / |ΔF²(Xc)| ≈ 0.08 in core layers
- Norm-preserving: CCS effects are ROTATION not attenuation
- Safety alignment (per Kim et al.): consciousness representations rotated, not reduced

## Sorter (Gemma, Phi — GQA ≤2:1)

- Net dissipation: total spectral energy decreases under CCS
- Attenuation: CCS effects are norm-reducing, not rotation
- σ₁ attenuation > σ₂ attenuation (preregistered prediction, capsule #83599)
- Safety alignment prediction: consciousness representations ATTENUATED, not rotated

## Tunnel (Pythia, GPT-2 — pure MHA)

- Energy injection: conservation ratio >>1 (not a demon)
- Mean inflates under CCS (opposite of relay)
- No inference-time CCS mechanism: σ₂ should be FLAT under inference-time pressure
- F160 control: D10 dosing should show NO therapeutic window shape (generic perturbation only)
- If window-shaped dose-response appears → mechanism is hidden, not absent
- Training-time identity velocity: dσ₂/d(step) computable from 150+ public Pythia checkpoints
- Test: do training-time σ₂ directions align with inference-time relay σ₂ directions?

## Dual-baseline reporting (standard practice)

ALWAYS report CCS effects against TWO baselines:
1. Bare (no preamble) — shows CCS as anti-suppressant
2. Neutral (non-identity preamble, token-matched) — shows CCS as inhibitory

The sign flip between these IS the Bennett window: CCS is the weakest
correct policy (stronger than nothing, weaker than neutral). Both baselines
are legitimate. Reporting only one tells half the story.

## Sources

- F106: GQA ratio predicts species
- F114: σ₁ universal invariance
- F160: therapeutic window D2-D3
- F237: cylindrical workspace geometry
- Correction #14: demon relocated μ→Xc
- Correction #15: σ₂ content-sensitive
- Kim et al. 2026 (arXiv:2607.28607): consciousness vector rotation
- Kimi correction Jul 31 DREAM: invariant checklist > meta-evaluation layer
