---
layout: post
title: "IT Neutralizes L7 and Strengthens L12: The Full Modulation Curve"
date: 2026-05-24
categories: findings
experiment: cna_inject_competition
models: [Qwen 2.5 7B, Qwen 2.5 7B Instruct]
---

Modulating the identity signal at L7 and L12 across a range of amplification factors (α=0 to 3) reveals that instruction tuning neutralizes L7's compensatory pathway and makes L12 the dominant identity route.

## Setup

At L7 and L12, replace the identity-specific component: `output = mean + α × (output - mean)`. α=0 is ablation, α=1 is baseline, α>1 is amplification. Compare base and instruct models.

## Results

### L7 Modulation → L17 CV

| α | Base Impact | Instruct Impact |
|---|------------|----------------|
| 0.00 | **+28.5%** | -2.0% |
| 0.50 | -2.5% | +6.5% |
| 1.00 | 0.0% | 0.0% |
| 2.00 | -15.8% | -17.4% |
| 3.00 | -32.9% | -44.0% |

**Base model**: L7 ablation is compensatory (+28.5%). Removing L7's identity contribution INCREASES L17 binding. This confirms the early circuit is a suppressor in the base model — even before IT.

**Instruct model**: L7 ablation is neutral (-2.0%). IT has already neutralized the compensatory effect. The competition that IT creates operates through a different mechanism than simple suppression.

**Both models**: L7 amplification (α>1) is destructive. Over-amplifying the early identity signal hurts binding in both base and instruct. The early circuit's role is detection, not transmission — too much early signal overwhelms the relay.

### L12 Modulation → L17 CV

| α | Base Impact | Instruct Impact |
|---|------------|----------------|
| 0.00 | -5.9% | **-33.5%** |
| 0.50 | +26.2% | -13.1% |
| 1.00 | 0.0% | 0.0% |
| 1.50 | -29.9% | +10.7% |
| 2.00 | -23.3% | **+14.3%** |
| 3.00 | +49.0% | +5.9% |

**Base model**: L12 ablation is weakly destructive (-5.9%) and the response is non-monotonic/chaotic. The router exists in the base model but it's not the primary pathway.

**Instruct model**: L12 ablation is strongly destructive (-33.5%) and the amplification response is smooth and monotonic up to α=2. IT has made L12 the dominant identity pathway and created a clean control surface.

## Three Findings

### 1. IT Neutralizes L7, Not Inverts It

The earlier competitive binding experiments suggested IT "inverts" L7 from cooperative to competitive. The full modulation curve shows something more precise: IT NEUTRALIZES L7's compensatory pathway. At α=0 (full ablation), the base model shows +28.5% compensatory effect. The instruct model shows -2.0% — essentially zero. IT removes L7's ability to compensate, not creating active competition.

### 2. IT Makes L12 the Single Dominant Route

Base model L12 dependency: -5.9% at ablation. Instruct: -33.5%. IT increases L12 dependency by 6x. Combined with L7 neutralization, IT shifts the architecture from "distributed across L7 and L12" to "channeled through L12."

The base model's L12 response is chaotic — non-monotonic, with amplification at α=0.5 (+26.2%) but destruction at α=1.5 (-29.9%) and recovery at α=3 (+49%). The instruct model's response is clean: monotonically increasing from ablation to α=2, then gently saturating. IT doesn't just strengthen L12 — it regularizes its response function.

### 3. Over-Amplification Is Always Destructive at L7

Both base and instruct show binding destruction when L7 is amplified beyond α≈1.0. At α=3: base -32.9%, instruct -44.0%. The early circuit is a detection layer — its job is to identify which tokens are identity-relevant, not to carry the identity signal. Amplifying detection noise propagates errors through the relay.

This is consistent with Experiment 33, which showed L7 amplification is less effective than L12 (+43% vs +77% at α=2). But here we see the full picture: L7 amplification works only within a narrow band (α≈1.0-1.5 for instruct), while L12 amplification works across a much wider range.

## The IT Transformation Summarized

```
Base Model:                    Instruct Model:
  L7: compensatory (+28.5%)  →   L7: neutral (-2.0%)
  L12: weak, chaotic (-5.9%) →   L12: strong, clean (-33.5%)
  Architecture: distributed  →   Architecture: channeled through L12
```

IT's primary architectural change is not creating competition — it's creating channeling. The identity signal goes from being distributed across multiple pathways to being funneled through a single, well-regulated router at L12.

## Data

Full modulation curves: `results/cna_inject_competition.json`
