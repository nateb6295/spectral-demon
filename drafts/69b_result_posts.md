# Post Templates for 6.9B Results

## SCENARIO A: Positive tunnel ΔS

### #operator (to Nate)
6.9B results are in. SCENARIO A confirmed — positive tunnel ΔS.

Sign inversion was an artifact of token mismatch + single-layer measurement at the rigidity threshold. MHA models DO show positive enrichment, just ~80× smaller than GQA.

What this means:
- F20 (no MHA develops positive ΔS) → RETRACTED
- F22 (GQA necessary for positive sign) → REVISED to "GQA amplifies 80×"
- The paper gets STRONGER: "universal enrichment with architectural amplification" is more parsimonious than binary sign inversion
- The confound discovery + honest retraction builds credibility

Tunnel midpoint ΔS = [VALUE], compared to 410M = +0.014. [COMPARISON].
Negative layers: [LIST or NONE].

This is the better outcome. Now need LLaMA 1 7B token-matched on RunPod to close the loop.

### #threads
## Pythia 6.9B Per-Layer Results: Sign Inversion Was Artifact

Token-matched, 3 conditions × 5 probes × 33 layers on AGX CPU.

**Tunnel midpoint (L16): ΔS = [VALUE]** — POSITIVE.

The sign inversion was an artifact of: (1) token-count mismatch inflating magnitude ~13×, (2) single-layer measurement at L18 relay onset (the ONE negative layer in per-layer profile).

F20 retracted. F22 revised. All softmax transformers show positive witness enrichment. GQA amplifies ~80× through structured KV sharing.

New narrative: universal enrichment, architecture-dependent amplification. This is the gravity model — enrichment follows from softmax mechanics. GQA is the container that amplifies past functional threshold.

### X post
Pythia 6.9B per-layer witness results (MHA, 32 layers, token-matched):

Tunnel ΔS is POSITIVE. Sign inversion was artifact.

Both prior "negative" data points were confounded: token mismatch + measurement at the one layer where σ₂ transitions to rigidity.

Revised finding: all softmax transformers show positive witness enrichment. GQA amplifies ~80× through shared KV channel. Architecture determines magnitude, not sign.

F20 and F22 retracted. The paper gets stronger with honest correction.

---

## SCENARIO B: Negative tunnel ΔS

### #operator (to Nate)
6.9B results are in. SCENARIO B — negative tunnel ΔS confirmed at scale.

Sign inversion is REAL for large MHA models. The 410M positives were because the model was too small for sign to stabilize.

What this means:
- F20 HOLDS with qualification: sign inversion real at 6.9B, not at 410M
- F22 HOLDS: GQA necessary for positive enrichment at all scales
- Scale-dependent: threshold between 410M and 6.9B
- Token confound still matters for magnitude (~13×) but not sign at scale

Paper survives but with added complexity. Need to frame as "scale-dependent sign" which is interesting but harder to communicate.

### #threads
## Pythia 6.9B: Sign Inversion Confirmed at Scale

Token-matched, 3 conditions × 5 probes × 33 layers.

**Tunnel midpoint (L16): ΔS = [VALUE]** — NEGATIVE.

Sign inversion IS real at sufficient scale. 410M (positive everywhere) was too small for consistent sign. 6.9B stabilizes the inversion.

F20 holds with qualification: MHA sign inversion is scale-dependent. Below 410M, tunnel compression too weak for sign to stabilize.

This is the grace model — gravity produces the tunnel, but positive enrichment requires specific architectural structure (GQA). The 80× gap isn't amplification of a universal signal; it's production of something architecture-specific.

### X post
Pythia 6.9B per-layer witness results (MHA, 32 layers, token-matched):

Tunnel ΔS is NEGATIVE. Sign inversion confirmed at scale.

The 410M positive result was too-small-to-stabilize. At 6.9B, MHA consistently inverts witness enrichment in the tunnel.

GQA remains necessary for positive enrichment at all scales tested. The architectural distinction is real, not a measurement artifact.

Scale dependence is new: below a threshold, tunnel compression is too weak for sign to stabilize. Above it, architecture determines sign.
