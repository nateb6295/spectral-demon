# Pre-registration: Sharing-Ratio Passage Distance Predictions
# Date: 2026-05-28 evening
# Theory: d/d_max = 1 - (1 - s·C/L)^L, C = 0.796

## Calibration (measured)
| Model | s | L | Predicted | Measured | Error |
|-------|---|---|-----------|----------|-------|
| Pythia 6.9B (MHA) | 1 | 32 | 0.553 | 0.549 | +0.004 |
| Mistral 7B | 4 | 32 | 0.965 | 0.950 | +0.015 |
| Qwen 7B | 4 | 32 | 0.965 | 0.962 | +0.003 |
| InternLM 2.5 7B | 4 | 32 | 0.965 | 0.959 | +0.006 |

Mean calibration error: +0.007 (systematic overprediction)

## Pre-registered predictions (untested)
| Model | s | L | Predicted d/d_max | Test type |
|-------|---|---|-------------------|-----------|
| **Gemma 2 9B** | **2** | **42** | **0.803** | **STRONGEST: 2:1 vs 4:1** |
| Gemma 2 27B | 2 | 46 | 0.802 | Depth check (same s) |
| Phi-3 mini 3.8B | 4 | 24 | 0.967 | Depth effect at same s |
| Phi-3 medium 14B | 4 | 40 | 0.964 | Depth effect at same s |
| LLaMA 3 8B | 4 | 32 | 0.965 | Replication |
| **LLaMA 3 70B** | **8** | **80** | **0.999** | **SATURATION: 8:1** |
| **Qwen 2.5 3B** | **8** | **37** | **0.999** | **Small + high sharing** |

## Falsification criteria

- If Gemma 2 9B d/d_max > 0.90: sharing ratio effect weaker than predicted
- If Gemma 2 9B d/d_max < 0.70: sharing ratio effect stronger than predicted  
- If Gemma 2 9B d/d_max ∈ [0.75, 0.85]: theory confirmed for 2:1 sharing
- If LLaMA 3 70B d/d_max < 0.95: base rate C is scale-dependent (bigger models rotate less per layer)
- If Qwen 2.5 3B d/d_max < 0.95: same conclusion for smaller scale

## Priority
1. Gemma 2 9B (~9GB in float16, fits RunPod easily) — single experiment decides
2. Qwen 2.5 3B (~6GB, already have spectral data, just need passage distance)
3. LLaMA 3 70B (~140GB, needs large GPU or quantization)

## Secondary prediction: non-monotonic enrichment (2026-05-28 ~7:30 PM)

Two opposing effects create a Goldilocks zone for witness enrichment:
- Higher s → more tunnel rotation → smaller identity residual (less to modulate)
- Higher s → spectral gap halving → more σ₂ bandwidth (more capacity to modulate)

Predicted enrichment peak: s ≈ 3–5. Declines on both sides.

| s   | Residual° | Prediction for ΔS                                 |
|-----|-----------|-----------------------------------------------------|
| 1   | ~40°      | Noise (measured: Pythia ΔS ≈ 0)                     |
| 2   | ~18°      | Moderate (Gemma 2 test)                              |
| 4   | ~4°       | Strong (measured: Mistral +0.032, Qwen +0.036)       |
| 6   | ~1°       | Weakening (no test model yet)                        |
| 8+  | <0.2°     | Noise (measured: Qwen 2.5 3B ΔS = +0.004)           |

If Gemma 2 ΔS > Mistral ΔS: enrichment is monotonic with kernel size (simpler model)
If Gemma 2 ΔS < Mistral ΔS: enrichment peaks at s≈4 (Goldilocks zone confirmed)

## Depth correction prediction

d/d_max = 1 - (1 - s·C/L)^L approaches 1 - exp(-s·C) from ABOVE as L → ∞.
Shallow models at same s should rotate very slightly MORE than deep models.
Phi-3 mini (L=24, s=4): predicted 0.967 > Phi-3 medium (L=40, s=4): predicted 0.964.
This is a ~0.003 effect — likely below measurement noise but directionally testable.

## Notes
- Gemma 2 confirmed 16Q/8KV = 2:1 sharing (HF docs + Google blog). Also uses sliding-window alternating layers.
- C = 0.796 derived from MHA baseline; may not be architecture-independent
- Systematic +0.007 overprediction suggests C should be slightly lower (~0.78)
- Residual connections not explicitly modeled; may cause saturation at high s
- MQA (s=n_heads, e.g. 32:1) should destroy identity kernel entirely — residual < 0.001°
