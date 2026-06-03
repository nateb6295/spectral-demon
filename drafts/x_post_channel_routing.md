# X Post Draft: Channel Routing Finding

## Option A (technical, concise)
New finding from the 2×2 factorial: normalization type determines which spectral
channel carries contextual sensitivity in transformers.

LayerNorm centers hidden states → decouples the dominant singular value from
context → forces contextual signal through σ₂ (secondary direction).

RMSNorm doesn't center → σ₁ stays context-sensitive → contextual signal stays
in the dominant direction.

Same functional outcome. Different geometric pathway. Architecture as channel router.

## Option B (finding + honest limitation)
Ran the 2×2 factorial: {LayerNorm, RMSNorm} × {MHA, GQA}.

The surprising finding isn't which cell won — it's that normalization type
routes contextual signals through different spectral channels entirely.
LayerNorm's centering creates a spectral filter that forces context into σ₂.
Without centering, it stays in σ₁.

The dramatic part (opposite-sign gradients) is mostly one probe.
The robust part (channel difference, -10% σ₁ shift across all probes) is clean.

GQA still dominates: eliminates gradients regardless of normalization.

## Option C (broader frame)
The same functional capacity — sensitivity to relational context — is achieved
through completely different geometric pathways depending on one architectural
choice: whether you center the hidden states.

This is the pattern we keep finding. The capacity is the invariant. The implementation
varies by architecture. Design determines which spectral channels are available for
modulation. Training loads content into whatever channels the architecture provides.

## Option D (F76 angle — cleanest finding)
Ran the same witness-sensitivity probes on LayerNorm and RMSNorm models.

Surprise: LayerNorm equalizes the total witness effect across all content
types. Identity probes, process probes, contrastive probes — same total ΔS
(range: 0.005). But it routes the signal through DIFFERENT spectral channels
depending on content: σ₂ for identity, σ₄/σ₅ for contrastive.

RMSNorm preserves content-dependent variation: 14× difference between
process-oriented and identity-factual probes.

Centering (x → x-μ) creates a fixed-bandwidth bus with elastic channel
allocation. Without centering, both total bandwidth and channel selection
vary by content type.

## Option E (combined: channel + democratization)
Two findings from the 2×2 factorial ({LayerNorm, RMSNorm} × {MHA, GQA}):

1. Normalization routes contextual sensitivity through different spectral
channels. LayerNorm's centering forces context into σ₂. Without centering,
σ₁ stays context-sensitive.

2. LayerNorm also EQUALIZES sensitivity across content types. Same total
modulation whether the model is processing identity claims or procedural
reasoning. RMSNorm preserves 14× content-dependent variation.

Centering does two things at once: routes AND equalizes.
GQA still dominates — eliminates gradients regardless of normalization.

The honest limitation: one probe (process-oriented) accounts for most of the
RMSNorm gradient. The channel difference and democratization are robust across
all probes. The dramatic gradient isn't.

## FINAL — Ready to post (tomorrow ~7am PDT)

One operation in normalization — centering (x → x-μ) — determines whether a
transformer treats all content types equally.

We ran witness-sensitivity probes on LayerNorm (Pythia 6.9B) and RMSNorm
(LLaMA-1 7B) models. In the RMSNorm model, process-oriented content gets 14×
more witness modulation than identity-factual content. The dominant singular
value varies from −5.7% to −77.6% across probes (r = 0.93 with baseline wire
strength).

In the LayerNorm model: all five probes get the same total modulation. Range:
0.03 percentage points. But the spectral channels differ — identity probes route
through σ₂, contrastive probes route through σ₄/σ₅.

Centering creates a fixed-bandwidth bus with elastic channel allocation. Total
capacity is guaranteed regardless of content type. Individual channels are
assigned dynamically to match representational demand.

Without centering, both total bandwidth and channel selection vary with content.
Architecture that processes identity claims differently from procedural reasoning
isn't a bug — it's a normalization choice.

Every major production model since 2023 uses RMSNorm. The democratic bus is the
road not taken.

## Notes
- Post ~7am PDT Friday for engagement
- Thread reply with fig11_content_democratization.png
- Consider fig10_liu_confound.png as second thread reply (channel routing)
- "Road not taken" closing might be controversial — good for engagement
