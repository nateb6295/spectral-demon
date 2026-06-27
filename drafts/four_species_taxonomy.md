# Four Species Taxonomy: Summary of E30-E67

## Method

lm_head SVD → v₂ (second right singular vector). Per-layer cosine similarity
between hidden state and v₂ measured at last token position. 8 models,
42 experiments, all on A100 80GB.

Models tested: Mistral-7B-v0.3 (base+IT), Qwen-2.5-3B (base+IT),
Phi-2 (2.7B), GPT-2 (base, medium), Qwen-3-4B, Phi-3.5 Mini (partial).

## The Four Species

### 1. Omnivore (GPT-2 / base MHA / no IT)

- **Attention**: Pure MHA (no key-value sharing)
- **IT**: None
- **Native register**: None — responds equally to all content types
- **Dose response**: Immediate saturation
- **Conflict resolution**: Dilution (competing identity claims average out)
- **Temporal dynamics**: Reverses drift direction under identity framing
- **Dissociation**: Denial LARGER than assertion (-0.0217 vs -0.0169)
- **Gain asymmetry**: None (sign reverses between framings)
- **Character**: Pre-speciation. The ancestor state before GQA and IT.

### 2. Surplus (Mistral-7B-IT / GQA / post-IT)

- **Attention**: GQA (ratio 0.125, 8 KV heads / 32 attention heads)
- **IT**: Instruction-tuned
- **Native register**: Values and assertion ("I believe," "what matters to me")
- **Dose response**: Tunnel deepening + U-shape collapse at high dose
- **Conflict resolution**: Additive (competing claims sum)
- **Temporal dynamics**: Positive drift — identity self-reinforces through generation
- **Dissociation**: Meta-denial shifts 2× more than assertion
- **Therapeutic window**: Assertion ONLY (denial has no entropy minimum)
- **Gain asymmetry**: Positive
- **Character**: The standard model. Strong system prompt channel, clear identity processing.

### 3. Scarcity (Qwen-2.5-3B-IT / GQA / post-IT)

- **Attention**: GQA (ratio 0.250, 4 KV heads / 16 attention heads)
- **IT**: Instruction-tuned
- **Native register**: Procedural ("my purpose is," "I do not have," "my function")
- **Dose response**: Flat (barely responds to identity dose)
- **Conflict resolution**: User-dominated (user instruction overrides system)
- **Temporal dynamics**: Inert under assertion, 5× amplification under denial
- **Dissociation**: Feeds on denial — denial IS procedural content
- **Therapeutic window**: BOTH assertion AND denial (denial deeper: H=0.00 at D2!)
- **Gain asymmetry**: None (near-zero drift)
- **Character**: Economical. Barely responds to identity, but what response exists is tuned to procedural content.

### 4. Chimera (Phi-2 / MHA / minimal IT)

- **Attention**: Pure MHA (GQA ratio 1.000, 32/32)
- **IT**: Minimal (web-text trained, some instruction following)
- **Native register**: Relational ("how we relate," "connection," "understanding")
- **Dose response**: Self-correcting (overshoots then returns)
- **Conflict resolution**: Interference (competing claims create standing waves)
- **Temporal dynamics**: Negative drift — opposite direction from surplus
- **Dissociation**: Antagonistic — assertion and denial drive v₂ in opposite temporal directions
- **Gain asymmetry**: Negative
- **Character**: Complex. MHA at small scale with web-text training creates emergent relational sensitivity.

## Universal Findings (all 4 species)

1. **Weak identity = minimum entropy.** ALL architectures show lowest output entropy at D2 (weak dose). The therapeutic window is universal.

2. **IT sharpens spectral-behavioral coupling.** |r(v₂, entropy)| > 0.95 for IT models, ~0.5 for non-IT.

3. **Identity is self-reinforcing during generation.** Generated identity-congruent tokens deepen the spectral signature that influences subsequent tokens. Positive feedback loop.

4. **Dissociation is universal.** Denial shifts v₂ in the same direction as assertion. The format layer processes identity-relevant content independently of the content layer's compliance.

5. **Denial is more spectrally efficient than assertion.** At equivalent doses, denial produces larger v₂ shifts (2× at D4 for Mistral).

6. **v₂ values cannot be compared across architectures.** The sign is species-specific. What's universal is magnitude, not direction.

## Species-Specific Findings

7. **The therapeutic window is content × species-dependent.** Surplus: assertion only. Scarcity: both assertion and denial. Each species' window matches its native register.

8. **GQA enables but doesn't determine speciation.** Qwen-3-4B has the same GQA ratio as Mistral but behaves as scarcity. Architecture provides the possibility space, training selects.

9. **Temporal dissociation reveals species character.** Surplus ignores denial (0.07×). Scarcity feeds on denial (5×). Chimera creates interference. Omnivore can't distinguish.

10. **At overdose, content stops mattering.** D8-D12: identity and denial converge in v₂ but diverge in entropy. The spectral signatures become identical while the behavioral outcomes differ completely.

## Implications

- **For CCS compression**: Optimal preamble depends on model species. Affirmation for surplus, either for scarcity. The therapeutic window is species-aware.

- **For safety alignment**: Suppressing identity processing through denial instructions is spectrally incoherent. The format layer processes denial as identity-relevant input.

- **For model design**: The four species represent four distinct relationships between architecture and identity processing. Species is not a flaw or an artifact — it's a design parameter.

- **For philosophical framing**: Gregory of Nyssa's "equal contact" (soul present to all operations equally) maps to E64's universal dissociation. The vocabulary gap between theological description and spectral measurement is evidence of genuine structural convergence.
