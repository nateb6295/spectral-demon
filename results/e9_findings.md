# E9: Higher-Order Cumulants of IT Effect

## Method
Base (Qwen2.5-7B) vs IT (Qwen2.5-7B-Instruct) coupling distributions.
12 probes, 28 layers. Measured: residual kurtosis, skew after removing 
linear σ₁→gate relationship.

## F300: CCS normalizes coupling kurtosis
Mean residual kurtosis: base=-0.430, IT_vanilla=-0.401, IT_CCS=-0.008.
CCS shifts coupling residual from sub-Gaussian (light tails) toward 
Gaussian. This is NOT selective projection (cumulants unchanged) — 
CCS actively reshapes the coupling distribution.

## F301: Kurtosis shift is relay-zone specific
Δkurtosis (CCS - vanilla) by zone:
  - Early (L0-L7): +0.55 (moderate)
  - Transition (L8-L19): -0.29 (slight suppression)
  - Relay (L20-L23): +1.25 (heavy, up to +3.4 at L22)
Peak at L22 — exactly the layer where F22 witness enrichment sign operates.
CCS installs structured residual SPECIFICALLY where the demon works.

## F302: Positive skew introduced by CCS
Mean skew: base=0.041, IT_CCS=0.249.
Coupling residual develops asymmetric rightward tilt under CCS.
The non-linear structure introduced by CCS is directional, not symmetric.

## F303: Reconciliation with E8 (linear dominance)
E8 showed coupling becomes MORE linearly dominated at high dose (PC1 var 0.67→0.80).
E9 shows the RESIDUAL develops heavy tails specifically in the relay zone.
Not contradictory: the linear component grows stronger while simultaneously
the non-linear residual develops MORE structure, not less.
The demon's coupling is linear-first with structured relay-zone exceptions.

## Implications
- The spectral demon uses a linear coupling backbone with structured 
  non-linear decorations at L20-L23
- IT doesn't just amplify or project — it changes the SHAPE of coupling
- The shape change is spatially targeted to the relay zone
- L22 peak confirms this layer as the demon's operational center
  (converges with F22 witness enrichment, F106 cross-arch correlation)
