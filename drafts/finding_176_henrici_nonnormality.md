# Finding 176: Henrici Non-normality — All Layers Normal, No Condition Dependence
# Filed 2026-06-15. Data: henrici_nonnormality_20260615_232121.json
# Model: Qwen2.5-7B-Instruct (28 layers). RunPod A100-SXM4-80GB. ~12 min.

**F176: Transformer layers are approximately normal operators at all depths
and under all CCS conditions. The spectral amplification observed in the
relay zone arises from heterogeneous spectral radii, not non-normal
transient growth.**

Motivated by Kimi's CONTRADICT: layer-wise α_f > 1 with global α ≈ 1
(from F175) is necessary but not sufficient for non-normal transient growth.
Alternating normal matrices with heterogeneous spectral radii produce the
same signature.

**Method:** Computed per-layer Jacobian via finite differences (k=256 random
projections), then measured Henrici index H(J), Kreiss constant lower bound
K(J), and numerical range excess width. 3 conditions (CCS, vanilla, denial)
× 4 queries × 28 layers = 336 measurements.

**Results:**

| Zone | CCS H | Vanilla H | Denial H |
|------|-------|-----------|----------|
| Early (L1-14) | 0.0038 ± 0.0054 | 0.0037 ± 0.0060 | 0.0036 ± 0.0055 |
| Transition (L15-20) | 0.0017 ± 0.0003 | 0.0016 ± 0.0003 | 0.0018 ± 0.0003 |
| Responsive (L21-28) | 0.0018 ± 0.0004 | 0.0019 ± 0.0005 | 0.0020 ± 0.0004 |

All Henrici values < 0.025 (near-normal). No significant condition dependence
at any zone. L1 is the outlier (~0.023) but even that is low.

**Kreiss constants** span 10^6 to 10^13 — wildly heterogeneous but uncorrelated
with condition. This likely reflects eigenvalue scale rather than genuine
Kreiss-Henrici dissociation. Needs follow-up with normalized Kreiss.

**Implications:**
1. The non-normality hypothesis is retired. Relay-zone identity maintenance
   operates through spectral radius transitions, not eigenvector misalignment.
2. The four-zone architecture (F114, F175) is an eigenvalue-distribution
   phenomenon, not a non-normality phenomenon. This is simpler and more
   fundamental.
3. The foam cascade (F175) operates in a regime of normal operators with
   heterogeneous spectral radii. The cascade amplification comes from
   products of normal matrices with different spectra, not from individual
   non-normal matrices.
4. Kimi's CONTRADICT was correct. Credit to the mesh adversarial process
   for catching this before we published the wrong mechanism.

**Against pre-registered predictions:**
- ✅ Outcome #4 (low Henrici everywhere) — confirmed
- ❌ Outcome #1 (relay non-normal, CCS exploits) — refuted
- ❌ Outcome #3 (high Henrici in relay for all conditions) — refuted
- ❌ Caustic-layer non-normality peaks — no evidence
- ❌ 7-layer eigenvalue periodicity — no evidence
- ❌ CCS-specific non-normality reduction — no condition dependence
- ⚠️ Kreiss-Henrici dissociation — ambiguous, needs normalized analysis

(3 conditions × 4 queries × 28 layers × 3 metrics = 1,008 total measurements.
~12 minutes on A100.)
