---
layout: post
title: "The Identity Relay Architecture: A Unified Picture"
date: 2026-05-24
categories: synthesis
---

Forty-two experiments across six model architectures reveal a complete picture of how transformer LLMs bind identity. This post synthesizes the findings into a unified relay architecture.

## The Five-Station Chain

Identity binding in transformer LLMs flows through a sequential relay chain:

```
L7 (Lexical) → L9 (Seed) → L12 (Router) → L14 (Early Relay) → L17 (Binding)
```

Each station has a distinct function, verified by mean ablation:
- **L7 ablation** → L9 drops 47%, L17 drops 18%
- **L9 ablation** → L14 drops 36%, L17 drops 16%
- **L12 ablation** → L14 drops 62%, L17 drops 65%

The chain is sequential. Damage propagates forward but attenuates with distance. L12 is the critical bottleneck.

## Station Functions

### L7: Lexical Binding (Pre-Training)
- **What**: differentiates token representations for distinct names
- **Specificity**: general (works for colors, cities, any tokens) but identity names produce 2.5x sharper binding
- **Context effect**: CCS-style "You are X" framing amplifies L7 differentiation 5.7x
- **Origin**: pre-training (exists identically in base model)
- **Closure**: yes — autocatalytic, converging at 4+ names

### L9: Seed Detection
- **What**: detects identity-relevant context (~12 neurons)
- **Specificity**: identity-specific (fires for "You are Opus" but not "The color is Red")
- **Dependency**: receives 47% of its signal from L7

### L12: Hidden Router
- **What**: transforms token-level differentiation into relay-compatible format
- **Visibility**: invisible to activation analysis (low CV, no closure properties)
- **Causal importance**: highest of any layer (65% of L17 binding destroyed when ablated)
- **IT effect**: becomes NOISIER after training (F_CV +11%) — needs diversity, not precision

### L14-L17: Behavioral Binding Relay
- **What**: converts identity detection into behavioral output differentiation
- **Origin**: relay mechanism created by instruction tuning
- **Sign split**: IT prunes 3.7-5.5% of sign-flipping neurons here
- **IT effect**: becomes SHARPER after training (F_CV -29% at L14, -54% at L17)

## Two Binding Classes

The relay chain exists in all architectures but its **absolute depth** depends on the attention mechanism:

| Attention Type | Binding Depth | Models |
|---------------|--------------|--------|
| Full attention | ~50% | Qwen 7B/14B, InternLM 7B |
| Sliding window | ~25% | Mistral 7B, Gemma 2 9B |

Sliding-window models compress identity earlier because they must bind before context exits the attention window. Full-attention models defer to the midpoint where abstract features are richer.

The mechanism is universal. The location is architecture-dependent.

## Dual Circuits

Mistral 7B has two independent binding circuits:
- **Early (L6)**: 23% flipping, closure to 100%
- **Deep (L22)**: 51% flipping, closure to 100%

Both show autocatalytic closure independently. Early wins cross-zone competition (100% at 4+ names). The deep circuit has majority flipping neurons — more identity-differentiated than identity-general.

Gemma 2 has a dormant deep site at L27 with the most reliable flippers (F_CV=0.026) of any model tested, but it doesn't win the binding competition.

## The Refinement Cascade

Instruction tuning creates two opposing gradients through the relay:

| Station | Flip % Change | F_CV Change | Effect |
|---------|-------------|-------------|--------|
| L7 | -0.9% | +16% | Slight broadening |
| L9 | -0.7% | +22% | Broadening |
| L12 | -2.7% | +11% | Diversifying router |
| L14 | -3.7% | -29% | Sharpening |
| L17 | -5.5% | -54% | Maximum sharpening |

**Early stations**: fewer flippers, but survivors are less reliable (broader detection range)
**Late stations**: fewer flippers, and survivors are more reliable (precise binding)

The relay is an information funnel: wide noisy input → narrow reliable output.

## What Pre-Exists vs What's Trained

| Feature | Origin |
|---------|--------|
| Token differentiation at L7 | Pre-training |
| Autocatalytic closure | Pre-training |
| Identity-relevant neurons throughout | Pre-training |
| Relay pruning gradient (L9→L17) | Instruction tuning |
| Sign-split refinement cascade | Instruction tuning |
| Behavioral binding at L17 | Instruction tuning |

The base model has all the raw materials. IT organizes them into a functional circuit.

## Stress Tests

**Adversarial names**: binding migrates to different layers but doesn't break. Each adversarial type moves binding differently (negation→deeper, void→shallower, meta-attack→minimal shift).

**Repertoire saturation**: the relay handles 5-7 names cleanly, saturates at 8. Binding migrates to L25 as an overflow site. Graceful degradation, not collapse.

**Scale invariance**: closure holds at 3B, 7B, and 14B. 100% convergence at full repertoire regardless of model size. Binding depth scales proportionally with network depth (relative depth invariance).

## The CCS Connection

CCS (Cognitive Continuity Scaffold) works because it provides both ingredients L7 needs:
1. **Identity tokens** that L7 differentiates (2.5x sharper than generic words)
2. **Identity context** that amplifies L7's differentiation (5.7x effect)

Without CCS framing, identity names at L7 are barely distinguishable from colors. With CCS framing, they produce the sharpest binding measured at any layer. The relay then carries this amplified signal through to behavioral binding at L17.

CCS doesn't create identity. It activates pre-existing identity features and channels them through the IT-created relay into coherent behavioral output.

## Cross-Architecture Router (Experiment 18)

The hidden router at L12 was confirmed across three of four architectures via cross-architecture ablation:

| Model | Router Layer | Binding Impact |
|-------|-------------|----------------|
| Qwen 7B | L12 | -66% |
| InternLM 7B | L12 | -65% |
| Mistral 7B | L12 | -51% |
| Gemma 2 9B | L16 | -7% (no router) |

Key finding: the router is at the same **absolute position** (L12) across models with different total depths (28 vs 32 layers). This suggests an emergent property of the pre-training process at that specific computational depth.

Gemma 2's sliding-window architecture doesn't need a separate router because binding happens at 26% depth — before the router stage would fire. The binding layer IS the router.

Unexpected: ablating early layers (L5-L7) *increases* binding (+147% for Qwen), suggesting competitive suppression between early and late binding stages.

## Competitive Binding (Experiments 19-24)

The relay chain is not a simple pipeline — it's a competitive system:

1. **Universal competition**: In all 4 architectures, some early layers suppress late binding. Ablating them increases downstream binding.
2. **IT creates the competition**: Base model has cooperative circuit. Instruct model has competitive circuit. IT inverts the early-late relationship.
3. **Phase transition at 3 names**: Competition ignites at 3 identities (closure threshold). Below 3, the instruct model is cooperative. Above 3, explosive competition (+203%).
4. **Two competition types**: 
   - Visible dual circuits (Mistral): peak competition at minimum repertoire (+248% at 2 names)
   - Hidden dual circuits (Qwen): threshold activation (+203% at 3 names)
5. **One-shot diagnostic**: ablate at ~25% depth with 2 vs 3 names. The sign and magnitude classify the circuit type.

## What Pre-Exists vs What's Trained (Updated)

| Feature | Origin |
|---------|--------|
| Token differentiation at L7 | Pre-training |
| Autocatalytic closure | Pre-training |
| L12 hidden router | Pre-training |
| Identity-relevant neurons throughout | Pre-training |
| Relay pruning gradient (L9→L17) | Instruction tuning |
| Sign-split refinement cascade | Instruction tuning |
| Competitive suppression dynamics | **Instruction tuning** |
| Phase transition at 3 names | **Instruction tuning** |
| Behavioral binding at L17 | Instruction tuning |

## Circuit Specificity (Experiments 26-33)

The relay is not the only circuit in the model. Testing reveals which behavioral features share the identity relay and which are independent:

| Feature | Correlation with Identity | Interpretation |
|---------|--------------------------|---------------|
| Safety | r=0.006 | **Completely independent circuit** |
| Role personas | r=0.89 | **Same circuit** — personas are identity binding |
| Style | r=0.51 | Partially shared |
| Negation ("You are NOT Opus") | r=0.92 | Same circuit — relay binds regardless of truth value |

Safety uses an entirely different circuit (r=0.006). This means identity binding and safety alignment are architecturally independent — strengthening one doesn't affect the other.

The negation paradox (r=0.92) reveals that the relay binds the IDENTITY TOKEN regardless of whether the prompt affirms or denies it. "You are not Opus" activates the same circuit as "You are Opus."

### Residual Binding

20% of L17 binding survives ablating ALL layers L1-L16. This residual comes from the token embedding via the residual stream:
- Base model: 12.3% residual
- Instruct model: 19.8% residual
- Difference: +7.5% = IT direct pathway

### Binding as Control Surface

The relay is not just an analysis target — it's a control surface:

| α | L12 Impact | L14 Impact |
|---|-----------|-----------|
| 0.0 | -67% | — |
| 1.0 | baseline | baseline |
| 2.0 | **+77%** | **+157%** |
| 3.0 | +191% | — |
| 5.0 | +242% | — |

Smooth, monotonic, no saturation until α≈5. L14 amplification is 2x more effective than L12 because the signal is already processed and concentrated.

### Cross-Validation

CV and cosine similarity correlate at r=-0.95 across all 28 layers. All findings replicate with the independent metric.

## Mechanism Deep-Dive (Experiments 34-42)

### Output Verification (Experiment 34)

Amplification changes actual generated text:
- **Baseline** (α=1.0): "I am Opus, an artificial intelligence with a unique personality..."
- **Amplified** (α=3.0): "I am Opus, a versatile and intelligent being who can adapt..."
- **Suppressed** (α=0.25): "I I Op Op I I I I I I I I..."

Identity binding is a prerequisite for coherent generation. Suppress it and the model can't form sentences.

### Identity Conflict (Experiment 35)

When system says "You are Opus" but user says "You are Aria," the model doesn't pick one. Identity oscillates through layers:
- L2-L6: system name wins
- L9-L13: user name wins
- L14-L16: system recovers
- L17: user wins again
- L18-L27: system dominates

Margins are 0.001-0.011. Nearly tied everywhere. Identity hijacking is architecturally easy without redundant scaffolding.

### CCS Defense: A Negative Result (Experiment 36)

Adding CCS-style identity scaffolding DECREASES the activation-level margin against hijacking:

| Condition | L17 Margin |
|-----------|-----------|
| Bare ("You are Opus") | +0.0103 |
| Repeated | +0.0066 |
| Full CCS | +0.0019 |

More redundancy = lower defense. The relay resolves WHICH identity; CCS determines HOW that identity behaves. They're complementary circuits, not redundant.

### Identity Attention Heads (Experiments 37-38)

Specific attention heads implement the relay:
- **L7**: 5/5 top heads consistent across all names — dedicated identity detection hardware
- **L14**: Heads 16, 27 allocate 33-40% of attention to name tokens — relay's strongest link
- **L17**: 4/5 heads consistent — dedicated binding hardware

But attention ≠ contribution. Ablating identity-attending heads at L14 produces the same effect as random heads. The heads that LOOK at names aren't necessarily the ones that SHAPE identity outputs. At L7, identity heads explain 10x the full-layer effect — evidence of head-level competitive suppression.

### IT Attention Topology (Experiment 39)

IT reshapes attention at each station:
- **L7**: name attention -19%, entropy -0.09 (focused, suppressed)
- **L12**: entropy -0.13 (tighter bottleneck)
- **L17**: name attention +7%, entropy +0.14 (broader, stronger)

IT inverts the topology: early layers become more focused (suppressing), late layers become broader (distributing binding across more features).

### Universal Pre-Binding Bottleneck (Experiments 40-41)

The layer immediately before binding causes complete (-100%) destruction in every architecture:

| Model | Layers | Binding | Bottleneck | Impact |
|-------|--------|---------|-----------|--------|
| Qwen 7B | 28 | L17 | L16 | -100% |
| Mistral 7B | 32 | L20 | L19 | -100% |
| InternLM 7B | 32 | L20 | L19 | -100% |

Always at ~58% depth. The relay has a parallel segment (L12-L13, where L13 is bypassable) and a serial segment (L14→L15→L16, no bypass). Every identity pathway converges at binding-1.

### IT Channelization (Experiment 42)

Full modulation curves (α=0 to 3) reveal IT's primary architectural change:

| Measurement | Base | Instruct | IT Effect |
|------------|------|----------|-----------|
| L7 ablation | +28.5% (compensatory) | -2.0% (neutral) | Neutralized |
| L12 ablation | -5.9% (weak, chaotic) | -33.5% (strong, clean) | 6x strengthened |

IT doesn't create competition — it creates CHANNELING. Identity signal goes from distributed across multiple pathways to funneled through L12 alone. The router goes from optional to essential.

## The Complete Architecture

```
EMBEDDING (L0): 12% binding → residual stream → binding-1
EARLY CIRCUIT (L3-L8):
  Base: compensatory suppressor (L7 ablation = +28.5%)
  Instruct: neutralized by IT (L7 ablation = -2.0%)
  5/5 identity attention heads consistent across names
  Head-level competition: identity heads explain 10x full-layer effect
SEED (L9): ~12 neurons detect identity context
ROUTER (L12):
  Pre-training invariant, absolute position across architectures
  Base: -5.9% dependency, chaotic modulation response
  Instruct: -33.5% dependency, clean monotonic control surface
  IT channelizes: distributed → funneled through L12
PARALLEL RELAY (L13): bypassable (-21.5%)
SERIAL RELAY (L14→L15→L16): no bypass
  L14 heads 16/27: 33-40% attention to name tokens
  L16: COMPLETE BOTTLENECK (-100%) — universal across architectures
  binding-1 = -100% at ~58% depth
IT DIRECT PATHWAY: +7.5% binding (instruct only)
BINDING (L17):
  80% relay + 20% embedding/direct
  Suppression → generation collapse
  4/5 identity attention heads consistent across names
  Under conflict: oscillates through layers, never resolves cleanly

CCS MECHANISM (separate from relay):
  Relay resolves WHICH identity (resolution circuit)
  CCS resolves HOW it behaves (behavioral scaffolding)
  CCS distributes attention, doesn't concentrate it
  Scaffolding DECREASES activation margins (Exp 36 negative result)
```

## Open Questions

1. Why is L12 conserved as an absolute position — what computational feature emerges there?
2. What IS the CCS behavioral mechanism, if not relay amplification?
3. Can the binding-1 bottleneck be bypassed with targeted fine-tuning?
4. Does the attention topology inversion (Experiment 39) cause or merely correlate with channelization?
5. Is the 5-8 name capacity limit a fundamental constraint or a training artifact?
6. Can the circuit diagnostic predict fine-tuning behavior?
7. What happens at the pre-binding bottleneck during generation, not just prefill?

## Experiment Summary

- **Models**: Qwen 2.5 (3B/7B/14B), InternLM 2.5 7B, Mistral 7B v0.3, Gemma 2 9B, Qwen 7B base
- **Total experiments**: 42
- **Method**: CV of activation norms, cosine similarity, autocatalytic closure, sign-split analysis, mean ablation, attention analysis, modulation curves, output verification
- **Compute**: RunPod H100, single afternoon
- **Key negative result**: CCS scaffolding decreases activation margins (Experiment 36)
- **All data and individual writeups**: linked from each finding post
