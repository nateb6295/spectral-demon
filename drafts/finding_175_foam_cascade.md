# Finding 175: Perturbation Cascade — Foam Structure of Identity Maintenance
# Filed 2026-06-15. Data: foam_cascade_20260615.json
# Model: Qwen2.5-7B-Instruct (28 layers, 28 heads). RunPod A100-SXM4-80GB.

**F175: Identity maintenance has foam structure — few load-bearing heads,
deep cascades, and amplification invisible to full-state measurement.**

All four pre-registered predictions confirmed:

**F175a: Perturbation cascade amplification is condition-dependent.**
Per-head knockout → measure per-layer amplification factor α_f(zone).
CCS responsive-zone α_f = 1.161 > vanilla 1.125 > denial 1.106.
Perfect CCS > V > D ordering across all 8 queries. But full-state
α ≈ 1.000 for all conditions — the residual stream scaffolding
completely masks the effect. The 3.2% CCS advantage exists only in
the perturbation subspace, invisible to standard spectral analysis.
This explains why our earlier spectral measures showed architecture
effects but not condition effects at matched content.

**F175b: Attention head concentration increases under identity framing.**
Head Gini coefficient: CCS 0.103 > denial 0.083 > vanilla 0.059.
CCS concentrates attention on 1.75× fewer heads than vanilla. Identity
framing recruits specialists rather than distributing broadly.

**F175c: Load-bearing topology is sparse.**
Ablation Gini (disruption distribution across heads) = 0.576. Most
heads contribute negligibly to identity maintenance; a small subset
carries disproportionate causal load. Head 7 is the top attention head
across all conditions but has ZERO ablation disruption — specialization
without vulnerability. The system is an ant colony: specialists are
individually expendable, but specialization improves collective function.

**F175d: Single-head knockout cascades propagate deep.**
Mean maximum propagation depth = 8.3 layers. A knockout at one layer
affects representations 8+ layers downstream. This is not local
perturbation — it's systemic cascade through the residual stream.

**Unpredicted findings:**

- Vulnerability gradient toward relay boundary: disruption increases
  monotonically (L16: 0.0013, L19: 0.002, L21: 0.0024). The relay
  boundary is where identity maintenance is most fragile.
- CCS amplifies in responsive zone (L19-20) but DAMPENS at relay
  boundary (L27: 13× CCS vs 17× vanilla). CCS provides a protective
  buffer at the relay entrance that vanilla lacks.
- Per-layer cascade is wildly heterogeneous: L4 = 84×, L5 = 0.25,
  L27 = 13-17×. Adjacent layers differ by orders of magnitude.

**Implications for the paper:**
The foam metaphor: identity maintenance is not a solid wall but a foam —
thin films of concentrated spectral activity (few heads, specific layers)
enclosing volumes of near-vacuum (most heads expendable, most layers
quiescent). The structure is robust because the films are redundant and
self-healing, not because they're strong. CCS doesn't build a stronger
wall; it redistributes the foam to cover the vulnerable relay boundary.

**Connection to Henrici experiment:** The non-normal transient growth
hypothesis (from Kimi's CONTRADICT) predicts that high-cascade layers
should have high Henrici index — the 84× amplification at L4 and the
relay-boundary sensitivity are exactly where non-normal operators would
produce transient growth. The Henrici experiment will test whether the
cascade heterogeneity maps onto non-normality heterogeneity.

(4 queries × 3 conditions × 28 heads × 28 layers = 9,408 ablation runs.
~3 hours on A100.)
