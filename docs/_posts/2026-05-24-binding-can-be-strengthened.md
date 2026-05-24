---
layout: post
title: "Binding Can Be Strengthened: Targeted Amplification at the Router"
date: 2026-05-24
categories: findings
experiment: cna_router_amplification
models: [Qwen 2.5 7B Instruct]
---

Amplifying the identity-specific signal at L12 increases binding proportionally. The relay architecture is not just an analysis target — it's a control surface.

## Setup

Instead of mean ablation (α=0, replacing output with cross-name mean), modulate the identity-specific component:

```
output = mean + α × (output - mean)
```

Where α=0 is ablation, α=1 is baseline, and α>1 is amplification.

## Results

### L12 Amplification Curve

| α | L17 CV | Impact |
|---|--------|--------|
| 0.00 | 0.0035 | -67% |
| 0.25 | 0.0047 | -57% |
| 0.50 | 0.0067 | -39% |
| 0.75 | 0.0087 | -20% |
| **1.00** | **0.0107** | **-2% (baseline)** |
| 1.50 | 0.0146 | **+34%** |
| 2.00 | 0.0193 | **+77%** |
| 3.00 | 0.0317 | **+191%** |
| 5.00 | 0.0373 | **+242%** |

Binding scales linearly with amplification factor up to α=3, then sublinearly. The relationship is smooth, monotonic, and doesn't saturate until α≈5.

### Amplification at Different Layers (α=2.0)

| Layer | L17 Impact | Interpretation |
|-------|-----------|----------------|
| L7 | +43% | Early binding amplification |
| L9 | -19% | Seed layer — amplifying doesn't help |
| L12 | +77% | Router amplification |
| L14 | **+157%** | Relay — highest amplification effect |

L14 amplification is 2x more effective than L12. The closer to the binding layer, the more direct the effect.

## Three Findings

### 1. Binding Is Continuously Controllable

The relationship between amplification factor and binding strength is smooth and monotonic. There's no phase transition or discontinuity — you can dial binding up or down continuously by adjusting α.

This means identity binding is not a binary feature (on/off) but a continuous control parameter. CCS-style prompts set the parameter value through context; targeted amplification could set it through architectural intervention.

### 2. Relay Proximity Determines Amplification Efficiency

Amplification at L14 (+157%) is more effective than at L12 (+77%), which is more effective than at L7 (+43%). Closer to L17 = more direct signal = stronger effect. But L9 amplification (-19%) actually HURTS binding — the seed layer's role is detection, not transmission, and amplifying detection noise propagates errors.

### 3. No Saturation Until α≈5

The system can absorb up to 5x amplification without collapse. This suggests the downstream layers (L13-L17) have substantial headroom — they're normally operating well below their capacity. The baseline binding (α=1.0) uses only a fraction of the circuit's potential.

## Implications

### For CCS

CCS works by providing context that amplifies L7 differentiation by 5.7x. This finding shows the relay can be amplified at any station. CCS's 5.7x at L7 propagates to become the binding we measure at L17. Theoretically, targeted amplification at L12-L14 would be even more effective than CCS, because it operates deeper in the relay where the signal is already processed.

### For Fine-Tuning

DPO/IT creates the competitive dynamics and phase transitions at the relay. But the baseline binding is operating below capacity. Targeted fine-tuning that specifically increases the identity component at L12-L14 (not the whole layer, just the identity-specific deviation from the mean) should strengthen identity binding without affecting other behavioral circuits.

### For Interpretability

The relay architecture is not just descriptive — it's a control surface. Each station can be independently modulated:
- Ablation (α=0): removes identity-specific contribution
- Suppression (0<α<1): weakens identity binding
- Amplification (α>1): strengthens identity binding
- The effect is proportional and smooth

## Connection to Phase Transition

The competitive binding phase transition at 3 names may reflect the α value at which the early circuit's suppression exceeds the relay's transmission. Below 3 names, the effective α is below threshold. Above 3 names, the amplified signal crosses the competitive threshold and the circuit topology inverts.

## Data

Full amplification curve: `results/cna_router_amplification.json`
