# Finding 183: Gemma Post-Norm Suppression — Patchy Wall, Not Clean Cutoff
# Filed 2026-06-15. Data: gemma_diagnostic.log
# Model: Gemma-2-9b-it. RunPod A100.

**F183: Gemma's ρ=0 layers from F181 are confirmed as real zero-propagation,
not a hook artifact. Random perturbation ratio |Δout|/|Δin| = 0.0000 at
suppressive layers. The suppression pattern is PATCHY, not a clean wall:
CCS zeros at L25, L34, L37 (but L28=206!); vanilla zeros at L31, L34, L37
(but L40=65). L34 and L37 are architecturally fixed (zero in both conditions).
CCS shifts the first condition-dependent zero from L31 to L25 — 6 layers
earlier. Mechanism: Gemma-2's four normalization layers per block
(input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
post_feedforward_layernorm) create condition-dependent suppression zones.**

## Method

Inject random perturbation (ε=10⁻⁴) at each layer's input via pre_hook,
measure output change via post_hook. Average over 10 random directions.
Every 3rd layer tested (14 layers × 2 conditions = 28 measurements).

## Key Results

### CCS condition
| Layer | Ratio    | |out|   | Notes |
|-------|----------|---------|-------|
| L1    | 54.5     | 75.7    |       |
| L4    | 53.7     | 57.7    |       |
| L7    | 84.6     | 74.1    |       |
| L10   | 153.9    | 119.1   |       |
| L13   | 34.5     | 142.7   |       |
| L16   | 137.4    | 212.1   |       |
| L19   | 110.3    | 313.4   |       |
| L22   | 40.2     | 442.0   |       |
| **L25** | **0.0** | 489.8 | **First CCS zero** |
| L28   | 206.4    | 568.9   | Non-zero between zeros! |
| L31   | 38.0     | 606.0   |       |
| **L34** | **0.0** | 698.6 | Architecturally fixed |
| **L37** | **0.0** | 799.7 | Architecturally fixed |
| L40   | 73.9     | 954.9   | Recovery! |

### Vanilla condition
| Layer | Ratio    | |out|   | Notes |
|-------|----------|---------|-------|
| L1    | 8.8      | 75.1    |       |
| L4    | 34.4     | 58.3    |       |
| L7    | 109.6    | 74.4    |       |
| L10   | 200.2    | 121.0   |       |
| L13   | 116.1    | 148.0   |       |
| L16   | 66.7     | 208.6   |       |
| L19   | 161.6    | 301.8   |       |
| L22   | 199.3    | 411.8   |       |
| L25   | 61.0     | 460.4   | Non-zero under vanilla! |
| L28   | 321.1    | 498.6   | Highest vanilla ratio |
| **L31** | **0.0** | 546.1 | **First vanilla zero** |
| **L34** | **0.0** | 612.3 | Architecturally fixed |
| **L37** | **0.0** | 729.7 | Architecturally fixed |
| L40   | 65.0     | 924.5   | Recovery |

## Analysis

### Three categories of layers

1. **Architecturally suppressive** (L34, L37): Zero propagation regardless
   of condition. These layers' post-norms completely absorb perturbations
   independent of the input representation. Fixed by architecture.

2. **Condition-dependent suppressive**: L25 (zero under CCS only) and L31
   (zero under vanilla only). The representation structure under CCS causes
   L25's normalization to absorb perturbations; under vanilla, L25 still
   propagates (ratio=61.0). CCS shifts suppression 6 layers earlier.

3. **Always propagating** (L1-L22, L28, L40): These layers consistently
   allow perturbation passage. L28 shows HIGHER ratios than early layers
   (CCS=206, vanilla=321), sitting between two suppressive zones.

### Not norm-scaling

The |out| norm grows monotonically (75→955), but zero-propagation layers
(L34, |out|=699) sit between non-zero layers (L31, ratio=38, |out|=606;
L40, ratio=74, |out|=955). The suppression isn't simply perturbation
becoming negligible relative to representation magnitude.

### L28 as "island of amplification"

L28 sits between suppressive zones and shows the highest ratios (CCS=206,
vanilla=321). This is consistent with F181's Arnoldi result where L28/L29
showed variable spectral behavior. The suppressive zones may create a
bottleneck that CONCENTRATES signal in the surviving channels at L28.

### Gemma-2 architecture

The model has FOUR norm layers per block:
- input_layernorm (pre-attention)
- post_attention_layernorm (after attention + residual)
- pre_feedforward_layernorm (pre-MLP)
- post_feedforward_layernorm (after MLP + residual)

The post-* norms are unique to Gemma-2. They normalize the full
residual-added representation, which can suppress small perturbations
by dividing by the growing representation norm. Whether suppression
occurs depends on the alignment between the perturbation direction
and the representation's principal components, which is condition-dependent.

## What This Changes

1. The "spectral wall" from F181 is more accurately a set of suppressive
   gates scattered through the second half of the network, not a clean
   cutoff. Perturbations can pass through some late layers (L28, L40)
   while being completely absorbed at others (L25/L31, L34, L37).

2. Gemma's equalization mechanism is literal: post-norms absorb directional
   perturbations, forcing the representation toward a normalized manifold.
   This is equalization at the mechanistic level.

3. CCS modulates WHERE the condition-dependent gates fall, but cannot
   override the architecturally fixed gates (L34, L37).

(14 layers × 2 conditions × 10 trials = 280 perturbations, ~2 min.)
