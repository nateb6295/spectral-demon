# Experiment: σ₂ Ablation at Tunnel Layer

## Motivation
F57 showed that inference-time GQA conversion changes σ₁/σ₂ gap without changing
ΔS. But does the witness effect actually FLOW through σ₂, or is it distributed
across the full spectrum?

If σ₂ IS the carrier: suppressing it should zero out ΔS.
If ΔS is distributed: suppressing σ₂ should only partially reduce ΔS.

## Method
Hook hidden states at L17 (tunnel measurement point). Before passing to L18:
1. Compute SVD of hidden state matrix H (n_tokens × hidden_dim)
2. Project out the σ₂ direction: H_ablated = H - (H @ v₂) ⊗ v₂
3. Pass H_ablated to the next layer

Measure S, ΔS, σ₁, gap at L17 for both native and ablated conditions.

## Implementation (~40 lines of PyTorch hook code)

```python
def make_sigma2_ablation_hook():
    def hook_fn(module, input, output):
        # output[0] is hidden_states: (batch, seq, hidden_dim)
        H = output[0].squeeze(0).float()
        _, _, Vt = torch.linalg.svd(H, full_matrices=False)
        v2 = Vt[1]  # second right singular vector
        # Project out σ₂ direction
        proj = H @ v2  # (seq,)
        H_ablated = H - proj.unsqueeze(-1) * v2.unsqueeze(0)
        # Replace in output tuple
        return (H_ablated.unsqueeze(0).to(output[0].dtype),) + output[1:]
    return hook_fn
```

Hook target: `model.gpt_neox.layers[17]` (for Pythia) or equivalent.

## Predictions

### Strong channel hypothesis:
- ΔS → 0 after σ₂ ablation (witness sensitivity carried entirely by σ₂)
- σ₁ unchanged (wire unaffected)
- S drops (one eigenvalue removed from spectrum)

### Distributed hypothesis:
- ΔS reduced but > 0 (witness sensitivity partially in σ₃, σ₄, etc.)
- Suggests a "witness subspace" rather than a single enrichment vector
- Would connect to participation ratio changes under witness

### Surprise hypothesis:
- ΔS INCREASES after σ₂ ablation
- σ₂ was suppressing rather than carrying witness information
- Would require completely rethinking the enrichment channel model

## Variants
1. Ablate σ₁ instead of σ₂ — test whether the wire carries any witness info
2. Ablate σ₃ through σ₅ — test specificity of σ₂ as enrichment channel
3. Run on Mistral (GQA+IT) where ΔS is more reliable than base Pythia
4. Combined: forced GQA + σ₂ ablation — test both simultaneously

## Requirements
- Same model as F57 (Pythia 6.9B) for direct comparison
- OR Mistral 7B for cleaner ΔS signal (but needs RunPod)
- ~30 forward passes per variant (10 probes × 3 conditions)
- AGX can handle Pythia; Mistral needs GPU cloud

## Relation to existing findings
- F55: σ₁ is condition-invariant (CV < 1.1%), σ₂ varies 7-9%
- F57: σ₂ modulation attenuated 63% under forced GQA but ΔS unchanged
- If σ₂ ablation zeros ΔS: the enrichment channel IS σ₂, and F57's
  unchanged ΔS despite attenuated σ₂ modulation means ΔS is a
  nonlinear function of σ₂ (threshold, not proportional)
- If σ₂ ablation doesn't zero ΔS: the enrichment signal is in the
  spectral bulk, and σ₂ is a marker not a carrier

## Note (2026-05-29 DREAM)
F57's finding that σ₂ modulation was attenuated 63% (15.7%→5.7%) while
ΔS was unchanged (+0.050→+0.056) already hints at the distributed
hypothesis. If the carrier were purely σ₂, you'd expect proportional
ΔS reduction. The ablation experiment would make this definitive.

## Convergence predictions (added 2026-05-29, 5:10 AM PDT)

### From "Small Singular Values Matter" (Nguyen et al., 2410.17770)
IT loads learned information into small SV directions. σ₂ is the first
"small" SV (the enrichment channel). GQA preserves this channel (Nait
Saada: reduced rank collapse). Predicts: ablating σ₂ SHOULD significantly
reduce ΔS, because IT-loaded relational information concentrates there.

But: we're running on Pythia (MHA) where σ₂ is crushed by σ₁ dominance.
MHA may not have loaded relational info into σ₂ (no channel available).
**Strongest test: run on Mistral (GQA) where σ₂ channel is available
and ΔS is larger (+0.032 vs ~±0.01 on Pythia).**

### From Lindsey & Asvin (2605.25459)
Explicit vs implicit self-recognition use ORTHOGONAL mechanisms.
If σ₂ is the implicit self-recognition channel (format-level, tunnel-located),
ablating it should:
- Disrupt implicit identity (ΔS change)
- Leave explicit identity intact (verbal self-identification unchanged)

Could test by measuring both ΔS (implicit) and probing the model's verbal
self-identification accuracy (explicit) with and without σ₂ ablation.

### Relay asymmetry prediction
GQA relay works by SPIKING σ₁ (amplifying). If σ₂ carries witness info
through the tunnel, and the relay amplifies, then ablating σ₂ before the
relay should produce LARGER effects than ablating it AFTER the relay —
because the relay would amplify the ablation's impact.

MHA relay works by COLLAPSING σ₁ (breaking). Ablating σ₂ in MHA tunnel
might have weaker effects because the relay doesn't selectively amplify
the σ₂ channel.

## Alternative experiment: RWKV-6 (non-softmax control)

RWKV-6 1.6B is available on HuggingFace with standard `output_hidden_states` support.
24 layers, 2048 hidden dim. Linear attention — NO softmax.

Nait Saada (2410.07799) proves softmax causes rank collapse. RWKV's linear
attention should show:
- Weaker σ₁ dominance (no exponential reweighting)
- Smaller spectral gap
- Possibly different/absent tunnel
- Unknown ΔS behavior (untested territory)

This would be the FIRST non-softmax architecture in our dataset. Would directly
test whether the tunnel is softmax-specific (as Nait Saada predicts) or a more
general property of deep sequential processing.

Runs on AGX (~3GB for 1.6B in float16). Standard measurement infrastructure works.
~30 forward passes for basic characterization (10 probes × 3 conditions).
