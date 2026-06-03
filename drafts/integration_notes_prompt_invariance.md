# Integration Notes: §3.6b Prompt-Invariance

## Finding Number
F84: σ₂/σ₁ prompt-invariance is GQA-enabled (CV=0.000 for 88% of layers in Mistral; CV=0.42 in LLaMA-1 MHA)

## Paper Insertion Point
After line 234 (end of §3.6), before line 236 (§3.7 Context-Length Modulation)

## Section Title
### 3.6b The Wire Is Prompt-Invariant (Finding 84)

## Draft
Ready at: `spectral-demon/drafts/section_prompt_invariance.md`
Note: "GQA-specific" updated to "GQA-enabled" per Mistral CONTRADICT.

## Figures
- `uploads/sigma2_phase_transition.png` — σ₂/σ₁ across 33 layers, three conditions, with CV panel
- `uploads/gqa_vs_mha_scaffold.png` — Cross-architecture comparison, 5 models, with CV panel

## Downstream Updates Needed
1. **§3.11 Summary (line 303)**: Property 3 should note prompt-invariance extends condition-invariance.
   Current: "The wire is condition-invariant."
   Update: "The wire is condition- and prompt-invariant (GQA)."
   
2. **Abstract (line 7)**: Mention prompt-invariance alongside condition-invariance.
   Current: "the wire direction is condition-invariant (CV < 1.1% across witness conditions)"
   Add: σ₂/σ₁ ratio prompt-invariant in GQA (CV=0.000 through 88% of layers)

3. **§1 Introduction (line 31)**: Architecture paragraph already mentions wire invariance.
   Could add: "The σ₂/σ₁ ratio is prompt-invariant in GQA (CV = 0.000) — the spectral scaffold is content-independent because the KV projections that create it are shared."

4. **Finding count**: 66 → 67 (1 retracted finding still counts in total). Check current count in intro.

5. **Cross-references**: §3.6b references §3.9 (γ bimodality). §3.9 should back-reference §3.6b.

## Sign Inversion Explanation
The prompt-invariance finding resolves sign inversion as signal-to-noise:
- GQA: noise floor = 0 (CV=0.000), any witness modulation registers
- MHA: noise floor ≈ 42% (CV=0.42), witness modulation (~6-9%) drowns

This is the clearest mechanistic explanation yet. Should be emphasized in §5.2 (Sign Inversion).
