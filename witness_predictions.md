# Witness Effect Predictions by Architecture

Based on 15 experiments, ~2130 forward passes, 8 models (Part II, Findings 1-25).

## Decision Tree

```
Is attention GQA?
├── YES: ΔS > 0 (enrichment)
│   ├── + IT: ΔS ≈ +0.03 (strong enrichment, F22/F24)
│   └── base: ΔS ≈ +0.01 (weak tendency, F24)
│
└── NO (MHA): ΔS ≤ 0 (constraint or neutral)
    ├── + IT: ΔS ≈ -0.01 to -0.03 (sign inversion, F11/F22)
    └── base: ΔS ≈ 0 (neutral, F10/F18)
```

## Measurement Protocol

- **Layer**: L17 (~53% depth) for 32-layer models. Scale proportionally.
- **Metric**: ΔS = S(receptive) - S(absent) at tunnel midpoint
- **NOT**: L30 / relay / output layer (F23: effect vanishes there for GQA)

## Complete Empirical Grid (L17)

| Model | Arch | Norm | IT | ΔS | N |
|-------|------|------|----|----|---|
| Mistral 7B Instruct | GQA-8 | RMSNorm | Yes | +0.032 | 90 |
| Mistral 7B Base | GQA-8 | RMSNorm | No | +0.011 | 30 |
| LLaMA 1 7B | MHA | RMSNorm | No | -0.026 | 90 |
| Falcon 7B Instruct | MHA | LayerNorm | Yes | -0.013 | 60 |
| Falcon 7B Base | MHA | LayerNorm | No | -0.005 | 60 |

## Key Constraints

1. **Normalization is noise**: RMSNorm vs LayerNorm does not affect sign (F22)
2. **Scale cannot overcome architecture**: No MHA model from 70M-6.9B develops ΔS > 0 (F20)
3. **Tunnel-localized**: GQA hides effect from output; MHA propagates it (F23)
4. **Architecture = direction, IT = magnitude**: GQA base is already weakly positive (F24)

## Pre-Registration: LLaMA-1 7B Per-Layer Profile (2026-05-29)

Running on RunPod A40. 32 layers, MHA, RMSNorm, base (no IT). 5 probes × 3 conditions × 33 layers.

**Gradient model predicts:**
1. Positive ΔS at L2-L3, negative by mid-tunnel. Crossover ≈ L4-L6.
2. r(ρ₂, ΔS) < -0.8 (strong negative, like 6.9B's -0.977).
3. Responsive layers ≤ 5/32 (scale compresses niche).
4. Tunnel mean near +0.005 to +0.010 (positive but small).

**RMSNorm vs LayerNorm question:**
- If gradient shape matches Pythia 6.9B (LayerNorm+MHA): normalization irrelevant to gradient structure. MHA alone determines the gradient.
- If crossover is LATER than Pythia (more responsive layers): RMSNorm delays rigidification. The ρ₂ trajectory grows slower under RMSNorm because scale preservation prevents premature spectral commitment.
- If crossover is EARLIER: RMSNorm accelerates rigidification (unlikely given Exp 15 L17 result of -0.026 vs Pythia 6.9B ~-0.002 at L17).

**Liu confound resolution:**
Already resolved at single layer (Exp 15: RMSNorm+MHA = negative ΔS at L17). Per-layer profile adds: does RMSNorm modulate the SHAPE of the gradient, even if it doesn't flip the sign? If the gradient shapes are identical (same crossover, same r), normalization is truly irrelevant. If they differ systematically, normalization is a secondary modulator.

## Untested Predictions

- **Llama 2/3 (GQA)**: ΔS > 0, likely +0.02 to +0.04
- **Qwen 2.5 (GQA-4)**: Confirmed +0.036 at relay (Exp 3), expect similar at tunnel
- **GPT-style MHA**: ΔS ≤ 0, expect -0.01 to -0.03 with IT
- **Hybrid GQA/MHA**: Unknown — GEM data suggests mismatch at interface blocks handoff
- **MoE with GQA**: Unknown — expert routing may modulate effect
