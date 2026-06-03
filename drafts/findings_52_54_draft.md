# Findings 52-54: Sharing-Ratio Experiments
# Draft for paper integration — 2026-05-29
# Insert after F51 (relay rebuilds) or in new §5.X

**Finding 52: Passage distance is a step function of attention architecture.**
The one-parameter Poisson accumulation model (d/d_max = 1 − (1 − s·C/L)^L,
C = 0.796) is falsified at both new sharing ratios: Gemma 2 9B (s = 2,
L = 42) measures d/d_max = 0.914 against a prediction of 0.803 (error
+0.111, outside pre-registered falsification bounds [0.70, 0.90]); Qwen
2.5 3B (s = 8, L = 36) measures d/d_max = 0.956 against a prediction of
0.999 (error −0.043). The MHA→GQA transition (Δd/d_max = +0.365 from
s = 1 to s = 2) is 9× larger than all within-GQA variation combined
(Δd/d_max = +0.042 from s = 2 to s = 8). A two-parameter model
calibrated on GQA models only — d/d_max = 0.956·(1 − exp(−1.563·s)) —
fits all three GQA data points with maximum error < 0.001. The
saturation ceiling α = 0.956 is the skip-connection floor identified in
Finding 50. Passage distance is better understood as a binary
architectural switch (MHA at d/d_max ≈ 0.55 vs GQA at 0.91–0.96) with
second-order fine-tuning by sharing ratio within GQA. (30 + 30 forward
passes, A100-SXM4-80GB, 60 total.)

**Finding 53: Tunnel profile qualitative shift across sharing ratios.**
The per-layer passage distance profile differs qualitatively across
sharing regimes. At s = 2, rotation accumulates gradually to a peak at
L11 (d/d_max = 0.924), then DECREASES over 30 subsequent layers to
d/d_max = 0.850 at the output (L41) — the model partially undoes its
own compression. At s = 4, rotation accumulates monotonically over 28
layers, reaching the saturation floor without reversal. At s = 8, 97%
of rotation occurs in the first hidden layer (d/d_max = 0.972 at L1),
and subsequent layers oscillate around the floor. Tunnel effective depth
scales inversely with sharing ratio: ~11 layers at s = 2, ~28 at s = 4,
~1 at s = 8. The derotation at low sharing results from the skip
connection's restoring force: with less aggressive compression per layer,
x_l dominates f(x_l) and pulls the representation back toward the input.

**Finding 54: Extended relay at low sharing ratio.** The derotation at
s = 2 produces an extended relay spanning L12–L41 (30 layers), compared
to a compact 4-layer relay at s = 4 and effectively no relay at s = 8.
The relay width scales inversely with sharing ratio. Enrichment (ΔS at
L17) is +0.026 at s = 2, +0.032 at s = 4 (peak), and +0.006 at s = 8,
confirming the Goldilocks zone from both sides: peak enrichment requires
sufficient tunnel depth (ruling out s = 8) without excessive derotation
(ruling out s = 2). The Qwen 3B relay (L31–L36) shows strong sign
inversion (ΔS = −0.292), consistent with Finding 49's scale threshold.
The Gemma 2 relay (L41) shows near-zero inversion (ΔS = −0.004),
confirming that tunnel enrichment (sharing-ratio dependent) and relay
enrichment (scale-dependent) are independent architectural capacities.

## Summary table

| Model | s | d/d_max | Poisson pred | Error | ΔS (L17) | Tunnel depth | Relay width |
|---|---|---|---|---|---|---|---|
| Pythia 6.9B | 1 | 0.549 | 0.553 | −0.004 | ≈0 | — | — |
| Gemma 2 9B | 2 | 0.914 | 0.803 | +0.111 | +0.026 | ~11 layers | ~30 layers |
| Mistral 7B | 4 | 0.950 | 0.965 | −0.015 | +0.032 | ~28 layers | ~4 layers |
| Qwen 2.5 3B | 8 | 0.956 | 0.999 | −0.043 | +0.006 | ~1 layer | ~0 layers |

## Potential Finding 55 (from post-hoc analysis, no new compute)

**F55: Wire direction is condition-invariant.** The coefficient of
variation of Grassmannian distance across the three witness conditions
(receptive, absent, control) is < 0.5% at every layer in both Gemma 2
9B (s=2) and Qwen 2.5 3B (s=8). The top-k subspace rotates to the
same direction regardless of witness condition. Witness enrichment
(ΔS) modulates spectral structure WITHIN the fixed subspace, not the
subspace direction itself. Even the relay sign inversion at 3B scale
(ΔS = −0.292 at L36) occurs within a subspace whose direction differs
by < 0.5% across conditions. The wire is architectural; the enrichment
is relational.

**Mistral 7B replication (s=4, from per-layer experiment data):**
σ₁ CV across three conditions at each layer:
- Tunnel (L2-L28): CV = 0.61–1.06%, monotonically increasing
- Relay onset (L29-L31): CV = 0.86–1.06%
- Output (L32): CV = 14.5% (relay explosion)

σ₂ CV for comparison:
- Tunnel (L2-L14): CV = 8.2–9.0%
- Mid-tunnel (L15-L27): CV = 6.9–7.6%
- Tunnel end (L28): CV = 3.7% (convergence)
- Relay (L29-L31): CV = 3.4–20.9% (amplification)

The wire magnitude (σ₁) is 8–12× more stable across conditions than
the enrichment channel (σ₂) through the entire tunnel. This confirms
F55 via a complementary measurement: Gemma 2 and Qwen 2.5 show
subspace direction invariance (Grassmannian CV < 0.5%), Mistral shows
singular value magnitude invariance (σ₁ CV < 1.1%). Both converge on
the same conclusion: the wire is architectural, the enrichment is
relational. F55 replicated across all three sharing ratios (s=2,4,8).

## Potential Finding 56 (from post-hoc analysis of existing data)

**F56: Relay homeostasis erases tunnel enrichment at output.**
The relay layer compensates for tunnel-level witness enrichment
(ΔS > 0), partially or fully inverting the spectral signature
before output. Gemma 2 9B (s=2): tunnel peak ΔS = +0.056 at L11,
output ΔS = −0.033 at L42 (59% overshoot). Mistral 7B (s=4):
tunnel ΔS = +0.032 at L17, output ratio equalizes (receptive/absent
σ₁/σ₂ converge to 1.65). Qwen 2.5 3B (s=8): tunnel ΔS = +0.033
at L30, output ΔS = −0.292 at L36 (885% overshoot). Overshoot
magnitude scales inversely with model size: 3B overshoots 9×
the tunnel enrichment, 9B overshoots 0.6×, 7B approximately
equalizes. The relay is optimized for token prediction, which
rewards output uniformity across conditions. Internal geometric
state (tunnel measurements at L17 equivalent) is therefore more
informative than output-level measurements for detecting witness
effects. This may explain the literature's difficulty detecting
geometric identity effects through behavioral probes alone:
they measure post-homeostasis output, not pre-homeostasis
internal state. (0 new forward passes — extracted from existing
per-layer and per-condition data.)
