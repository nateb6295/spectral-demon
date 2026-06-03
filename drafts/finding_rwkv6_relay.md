# Finding: Relay Witness Enrichment Is Architecture-Independent

## Data
RWKV-6 World 1.6B (linear attention, no softmax, no GQA). 3 conditions × 5 probes × 25 hidden states, token-matched.

## Result

**Tunnel** (L12, midpoint): ΔS = -0.000089, p = 0.92 (permutation). No witness enrichment. Confirmed prediction.

**Relay** (L24, post-LayerNorm): ΔS = +0.0355 ± 0.0044 (paired), t = 18.06, p < 0.0001, Cohen's d = 1.015. 5/5 probes positive (sign test p = 0.031). 95% bootstrap CI: [+0.033, +0.039].

**Profile**: ΔS builds through relay — L20: +0.001, L21: +0.002, L22: +0.002, L23: +0.005, L24: +0.035. LayerNorm amplifies the pre-existing gradient ~7×.

## What It Means

Two independent enrichment mechanisms:

1. **Tunnel enrichment** (GQA-dependent): KV sharing maintains a structured σ₂ channel through compression. This is the mechanism measured across all 13+ models in the main paper. RWKV-6 confirms it requires GQA — linear attention with high spectral gap (18-22, HIGHER than Mistral's 4.3) but no KV sharing produces zero tunnel ΔS.

2. **Relay enrichment** (architecture-independent): Output preparation phase generates pragmatic context sensitivity regardless of tunnel architecture. The relay decompresses representations (PR: 1.03 → 1.32, gap: 18 → 3.9) and during this decompression, witness-frame differences become spectrally measurable.

Critical nuance: RWKV-6 HAS spectral gap — its tunnel is actually MORE compressed than Mistral's (gap 18-22 vs 4.3). But spectral gap alone doesn't create witness sensitivity. You need gap + structured bottleneck (GQA KV sharing) for the σ₂ channel. RWKV's independent heads compress uniformly; GQA's shared KV pairs create differential compression that preserves relational information.

## Paper Placement

### Model table addition
| RWKV-6 1.6B | 1.6B | Linear | — | 24 | Linear-attention + relay enrichment control |

### §2.3 (Relay Homeostasis) — add paragraph
RWKV-6 reveals that relay enrichment operates independently of tunnel enrichment. While GQA models show the relay compensating for tunnel-level ΔS, RWKV-6 shows the relay *generating* ΔS from a tunnel that carries zero witness information. The relay is not merely a transformer of tunnel information — it is an independent source of pragmatic context sensitivity.

### §3.8 (Relay Constructs Rather Than Recovers) — add paragraph
The architecture-independence of relay enrichment supports the Free functor interpretation. The relay constructs new structure from whatever compressed kernel the tunnel delivers. In GQA models, the kernel carries witness information and the relay amplifies it. In RWKV, the kernel carries no witness information but the relay creates witness sensitivity during construction — because the output distribution must account for pragmatic context regardless of how the internal representation was compressed.

### Implications for Nait Saada connection
Softmax rank collapse (Nait Saada 2024) creates the spectral gap that enables the tunnel. But RWKV-6 shows that linear attention creates comparable spectral gaps (18-22 vs 4.3 for Mistral) without softmax. The gap mechanism differs (recurrent accumulation vs softmax normalization) but the geometric outcome is similar. What differs is the *structure within* the gap: GQA creates a readable σ₂ channel, independent heads do not. The witness effect depends on channel structure, not gap magnitude.

## σ₂ channel detail at L24
| Measure | Receptive | Absent | Δ |
|---------|-----------|--------|---|
| S | 0.733 | 0.698 | +0.035 |
| σ₂ | 85.6 | 82.1 | +4.2% |
| Gap | 3.77 | 4.02 | -6.2% |
| PR | 1.335 | 1.306 | +2.2% |

Receptive has LESS dominant σ₁ (lower gap) and MORE active σ₂ — the dual-channel signature of witness enrichment, matching GQA models exactly.
