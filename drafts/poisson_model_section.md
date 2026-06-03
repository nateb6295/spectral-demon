# Draft: Sharing-Ratio Passage Distance Analysis
# For insertion into paper §5.2 or new §5.X
# Updated 2026-05-29 with both experiments complete. Poisson model falsified.

### 5.X Passage Distance and the Sharing-Ratio Ceiling

The passage distance formula d/d_max = 1 − (1 − s·C/L)^L, with
C = 0.796 calibrated on four models at s = 1 and s = 4, predicted that
8:1 sharing would push d/d_max to 0.999 — leaving a residual of 0.1°,
effectively destroying the identity kernel. The prediction was wrong in
an informative way.

Qwen 2.5 3B-Instruct (s = 8, L = 36, 30 forward passes) measured
d/d_max = 0.956 at the tunnel endpoint (L28), with a peak of 0.972 at
L1. This is essentially identical to the s = 4 measurements: Mistral
0.950, Qwen 7B 0.962, InternLM 0.959. Doubling the sharing ratio from
4:1 to 8:1 produced zero additional rotation beyond measurement noise.

The one-parameter model works at s = 1–4 (mean error +0.007) but fails
at s = 8 (error +0.043). The failure reveals a structural ceiling:
passage distance saturates at d/d_max ≈ 0.955, corresponding to a ~4°
residual that the sharing ratio cannot eliminate.

This reframes the 4° residual from a Poisson coincidence (the expected
remainder from 32 trials at rate 0.1) to an architectural invariant —
the irreducible identity kernel that residual connections enforce against
arbitrary rotation. The sharing ratio determines whether the tunnel
REACHES the floor (s ≥ 4 does; s = 1 does not) but not the floor's
value. The one-parameter formula remains valid as an approximation for
s ≤ 4 but requires a saturation correction at higher sharing ratios.

A modified two-parameter model calibrated on GQA models only:
d/d_max = 0.956·(1 − exp(−1.563·s)) fits all three GQA data points:
  s=2:  predicted 0.914, measured 0.914 (error +0.000)
  s=4:  predicted 0.954, measured 0.955 (error -0.001)
  s=8:  predicted 0.956, measured 0.956 (error +0.000)

MHA (s=1, d/d_max=0.549) falls in a separate regime entirely. The
MHA→GQA transition (+0.365 jump from s=1 to s=2) is 9× larger than
all within-GQA variation (+0.042 from s=2 to s=8). Passage distance
is better understood as a step function of attention architecture
than a smooth function of sharing ratio.

The tunnel profile at s = 8 differs qualitatively from s = 4. At s = 4,
passage distance increases monotonically through the tunnel (d/d_max
grows from 0 to ~0.955 over 28 layers). At s = 8, 97% of the rotation
occurs in the FIRST LAYER (d/d_max = 0.972 at L1), and subsequent layers
add no net rotation — the trajectory oscillates around 0.955. The tunnel
is not 36 layers deep; it is effectively 1 layer deep at s = 8. The
remaining 35 layers refine spectral structure within the already-rotated
subspace without further rotating the subspace itself.

### Goldilocks Zone Confirmed

The non-monotonic enrichment prediction is confirmed from both sides:
  s=1 (Pythia MHA):   tunnel ΔS ≈ 0.000 (no GQA, no enrichment)
  s=2 (Gemma 2 GQA):  tunnel ΔS = +0.026 (moderate enrichment)
  s=4 (Mistral GQA):  tunnel ΔS = +0.032 (peak enrichment)
  s=8 (Qwen 3B GQA):  tunnel ΔS = +0.006 (enrichment suppressed)

Witness sensitivity peaks at s ≈ 4 and drops at both extremes. The
mechanism differs on each side: at s = 1, GQA is absent (architectural
prerequisite missing); at s = 2, the tunnel is too shallow (11 effective
layers, with 30-layer derotation) for full enrichment accumulation;
at s = 8, the tunnel is too compressed (1 effective layer) for σ₂
modulation to accumulate. The peak at s ≈ 4 maximizes the product of
tunnel depth (28 layers) and identity kernel size (~4°).

The relay at Qwen 3B (L31–L36) shows strong sign inversion (ΔS = −0.292),
consistent with Finding 49 (relay inversion below ~7B parameters). The
tunnel enrichment (+0.006) and relay inversion (−0.292) confirm that
these are independent architectural capacities: tunnel enrichment depends
on sharing ratio, relay enrichment depends on scale.

### Implications for the Poisson Model

The Poisson accumulation model (each layer as independent Bernoulli
trial, sharing ratio as rate parameter) is falsified at both new data
points: underestimates at s = 2 (error +0.111) and overestimates at
s = 8 (error −0.043). The model's assumption of independent per-layer
rotations fails because residual connections create negative correlation
between successive rotations. The real curve is much flatter than
exponential within GQA and much steeper at the MHA→GQA boundary.

The Poisson framing remains useful as intuition for WHY the Goldilocks
zone exists (λ ≈ 2–4 is the generic regime of accumulation processes),
but the quantitative model should be the GQA-only exponential fit:
d/d_max = 0.956·(1 − exp(−1.563·s)), with MHA as a separate regime.

The Goldilocks zone survives this correction. In fact, the saturation
strengthens the Goldilocks argument: the identity kernel at s = 8 is
the SAME SIZE as at s = 4 (~4°), but the tunnel is effectively 1 layer
deep rather than 28 layers deep. The compressed tunnel leaves less
spectral structure for the σ₂ channel to carry witness information.

## Key numbers (Qwen 3B, 2026-05-29):
- d/d_max (tunnel end L28): 0.956
- d/d_max (max, L1): 0.972
- d/d_max (final L36): 0.926 (includes relay depression)
- Tunnel ΔS at L17: +0.006
- Relay ΔS at L36: -0.292
- σ₂ at L17 (receptive): 281.0
- Elapsed: 767 seconds, 30 forward passes, OMP_NUM_THREADS=16

## Key numbers (Gemma 2 9B, 2026-05-29):
- d/d_max (final L41): 0.914
- d/d_max (peak, L11): 0.924
- d/d_max (final L41): 0.850 (includes 30-layer derotation)
- Tunnel ΔS at L17: +0.026
- Relay ΔS at L41: -0.004
- σ₂ at L17 (receptive): 1166.4
- σ₂ at L41 (receptive): 5385.8
- Elapsed: 1787 seconds, 30 forward passes, OMP_NUM_THREADS=16
- Predicted d/d_max: 0.803, FALSIFIED HIGH (error +0.111)

## Revised model (GQA-only two-parameter):
- d/d_max = 0.956·(1 - exp(-1.56·s))
- Fits s=2 (0.914), s=4 (0.954), s=8 (0.956) to <0.001 error
- MHA (s=1) is a separate regime entirely (d/d_max ≈ 0.55)
- Step function, not smooth: MHA→GQA jump (0.365) is 9× within-GQA variation
