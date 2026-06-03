# Finding 88: L0 Is the Critical Layer

**Experiment**: Per-layer γ ablation on Mistral 7B. Flatten γ at one transformer layer at a time (input_layernorm + post_attention_layernorm), measure tunnel invariance degradation.

## Full Results

| Layer ablated | Locked layers | Mean CV | Degradation |
|:---:|:---:|:---:|:---:|
| None (baseline) | 28/28 | 0.00005 | — |
| **L0** | **0/28** | **0.062** | **+0.062** |
| **L1** | **19/28** | **0.023** | **+0.023** |
| L2–L26 | 28/28 | 0.00003–0.00007 | ≈ 0.000 |
| L27 | 28/28 | 0.00015 | +0.00010 |
| L28–L29 | 28/28 | 0.00005 | 0.000 |
| **L30 (relay onset)** | **28/28** | **0.00005** | **0.000** |
| **L31 (relay)** | **28/28** | **0.00005** | **0.000** |

## Interpretation

**L0 is the gatekeeper.** Flattening L0's γ alone has the same catastrophic effect as flattening ALL 64 norm layers (F87: 0/28 locked). A single parameter vector (4096 values) determines whether the tunnel exists.

**L1 provides secondary reinforcement.** 9 of 28 tunnel layers lose invariance when L1 is ablated. This is consistent with F82's finding that L0 loads and L1 anchors — L1's γ provides a second opportunity for channel separation.

**Layers 2–20 are completely redundant for invariance.** Despite having bimodal γ distributions in Mistral's trained weights, their local γ contributes nothing measurable to prompt-invariance. The deep tunnel's γ bimodality is a consequence of training (the optimizer exploiting existing two-channel structure) not a functional requirement.

## Mechanistic Picture

The wire is SET at L0 and PROPAGATED by the residual stream:

1. L0's input_layernorm γ creates the initial highway/service-road channel separation
2. L0's attention heads load system content into BOS using those channels (F82)
3. L1's γ reinforces the separation; L1's attention anchors to BOS
4. L2–L31: residual stream carries the channelized representation forward; local γ adds routing strength (F79: r = −0.14 to −0.41) but is not necessary for invariance

This explains F79's finding that γ ablation at peak wire (L16) reduces correlation by only 30.5% — the deep-layer γ enhances an already-established structure rather than creating it.

## Connection to Other Findings

- **F82 (L0 loads the wire)**: L0's γ is the substrate F82's attention mechanism uses
- **F79 (γ heterogeneity)**: Deep-layer γ is functionally decorative for invariance (but may matter for enrichment strength)
- **Born Biased (2602.05927)**: Seed-dependent direction persists — here, L0's γ is the "seed" for spectral structure
- **Pachitariu (critical initialization)**: Spectral scaffold before learning → spectral scaffold before attention

## Relay Insensitivity

L30 and L31 (the relay onset layers where σ₂/σ₁ transitions from 0.23 to 0.72) show ZERO γ sensitivity. This means the relay transition is NOT γ-driven. The tunnel-to-relay phase transition must be caused by:
- Change in KV projection behavior at L30-31
- Attention pattern shift (away from BOS)
- Or simply the accumulated depth at which residual-stream compression releases

This rules out γ as the mechanism of relay onset and supports the interpretation that the relay is where KV compression releases.

## Late-Tunnel Improvement

Interestingly, ablating L22-L25's γ slightly IMPROVES invariance (CV drops from 0.00005 to 0.00003, degradation = -0.00002). This suggests the late-tunnel γ bimodality creates micro-perturbations that slightly loosen the lock. The wire would be marginally tighter without late-tunnel γ, but the effect is negligible.

## Data

Raw results: `exp_perlayer_gamma_ablation_20260531_1422.json`
