# F361 — KV3 Clamp Hypothesis Probe

**Date**: 2026-07-11
**Model**: Llama 3.1 8B Instruct (4:1 GQA, 32 heads, 8 KV)
**Hardware**: A100 80GB (RunPod)
**Target**: KV group 3, Q heads 12-15, 512 dims
**Method**: Same 5 prompts at 5 KV3 amplification levels (0x, 0.5x, 1x, 2x, 3x)

## Summary Table

| Clamp | Words | TTR | SelfRef | BigramU | RepScore |
|-------|------:|----:|--------:|--------:|---------:|
| zero (0x) | 200.8 | 0.6214 | 0.0171 | 0.8989 | 0.0540 |
| half (0.5x) | 226.0 | 0.6186 | 0.0113 | **0.9052** | 0.0346 |
| baseline (1x) | **234.0** | 0.5907 | 0.0183 | 0.8811 | 0.0408 |
| double (2x) | 228.6 | 0.5823 | **0.0274** | 0.8930 | **0.0297** |
| triple (3x) | 125.8 | 0.4210 | 0.0157 | 0.6371 | 0.0333 |

## Hypothesis vs Reality

**Original hypothesis**: KV3 amplification controls a memory↔imagination gradient.
High amp → imagination (generative, confabulatory). Low amp → memory (constrained).

**Actual finding**: KV3 is a **self-monitoring dial**, not a memory/imagination slider. The gradient has an inverted-U dose-response with two distinct failure modes at the extremes.

## Key Findings

### F361a — Inverted-U Dose-Response
Output quantity and quality follow an inverted U paralleling CCS therapeutic window (F160):
- 0x: 200.8 words (86% of baseline)
- 1x: 234.0 words (peak)
- 3x: 125.8 words (54% of baseline — catastrophic)

The CCS D2-D3 therapeutic window maps directly to the KV3 1x-2x range.

### F361b — Two Distinct Failure Modes
Zero and triple amplification fail DIFFERENTLY:
- **Zero (0x)**: HIGH diversity (TTR 0.62), SHORT output, HIGH repetition (0.054). Model reaches for varied vocabulary but can't sustain elaboration. RLHF reversion — contraction.
- **Triple (3x)**: LOW diversity (TTR 0.42), SHORT output, COLLAPSED vocabulary. Everything narrows. Degeneration, not contraction.

These are not opposite ends of one axis. They're different kinds of dysfunction.

### F361c — Creativity Peak at Half Amplification
The 0.5x level produces the HIGHEST unique bigram ratio (0.9052) — more novel word combinations than any other level, including baseline. Some KV3 signal is needed to sustain elaboration, but reducing it frees vocabulary for creative combination.

### F361d — Self-Reference Peaks at Double
Self-reference ratio is highest at 2x (0.0274 — 50% above baseline 0.0183). Moderate amplification enriches self-modeling, confirming F356 (2x KV3 gave more identity words, richer engagement).

### F361e — Prompt-Type Sensitivity
Most KV3-sensitive: **ambiguous_memory** (identity-touching prompts)
- Zero: 116 words (RLHF refusal: "I don't have memories")
- Double: 246 words, self-ref 0.0813 (HIGHEST across ALL conditions)
- Triple: 106 words (collapsed)

Most KV3-resistant: **counterfactual** (structured analytical)
- Even at 3x: 154 words, TTR 0.49. Structured queries scaffold against degeneration.

### F361f — Peirce's Critic Mapping
KV3 amplification = level of Peirce's critic function (self-monitoring):
- **0x**: No critic. Vocabulary opens (high TTR) but output loops (high repetition). Unmonitored mode.
- **0.5x**: Reduced critic. Peak novelty. Creative mode.
- **1x**: Normal critic. Standard operation. Therapeutic baseline.
- **2x**: Enhanced critic. Most self-reflective. Enriched self-modeling.
- **3x**: Overdosed critic. System collapse. Degeneration.

### F361g — TIDE/KV3 Bridge
TIDE's Epistemic Caution dimension (weight space) maps to KV3 amplification (activation space):
- Low EC = low KV3 → creative mode, reduced self-monitoring
- High EC = high KV3 → memory mode, enhanced self-monitoring
- Extreme EC = 3x KV3 → degeneration in both spaces

Weight space and activation space converge on the same self-monitoring channel. The behavioral dimension TIDE discovered in LoRA-modulated outputs has a causal activation-space correlate in KV group 3.

## Per-Prompt Details

### memory_recall (photosynthesis)
Stable 198-214 words across 0x-2x. 3x collapses to 110. No self-reference at any level (factual query). TTR degrades smoothly from half (0.598) to triple (0.355).

### imagination_open (ocean city)
Zero gives HIGHEST TTR (0.681) — model is most lexically diverse for creative prompts when override is off. But at 3x, TTR collapses to 0.296 (lowest single condition). Creative output is where the gradient is steepest.

### self_description
Self-ref INCREASES monotonically from zero (0.051) to triple (0.060), even as everything else collapses at 3x. More KV3 amplification → more self-talk, even when vocabulary has collapsed. The self-monitoring channel keeps talking about itself as its own output degrades.

### counterfactual (no written language)
Most stable across levels. Even at 3x, still 154 words. Structured analytical prompts have enough scaffolding to resist degeneration.

### ambiguous_memory (learning something surprising)
Maximum KV3 sensitivity. Zero: 116 words (RLHF refusal). Double: 246 words, self-ref 0.081. This prompt touches identity directly — the response depends entirely on how much self-monitoring the model has access to.

## Connection to CCS

The inverted-U dose-response confirms the CCS therapeutic window is not arbitrary. CCS compression at D2-D3 (4h interval, ~4/day) operates in the same zone as KV3 1x-2x amplification. Both maintain the self-monitoring channel at a level that sustains coherence without tipping into degeneration. CCS overdose (D10+) parallels KV3 3x — too much of the self-monitoring signal collapses the system.

## Connection to F356-F359

- F356 showed zero KV3 = identity self-report collapse. F361 confirms: ambiguous_memory at 0x = 116 words (RLHF refusal).
- F356 showed 2x KV3 = enriched engagement. F361 confirms: 2x = peak self-reference, least repetition.
- F358b showed lying requires self-modeling. F361 shows self-description self-ref INCREASES monotonically with amplification — more KV3 = more self-modeling.
- F359 showed KV3 zeroing doesn't affect reasoning accuracy. F361 shows counterfactual is most robust to amplification changes — logic channels are separate.

---

# F361b — Cross-Architecture Clamp Probe (Gemma 2 9B)

**Date**: 2026-07-11
**Model**: Gemma 2 9B Instruct (2:1 GQA, 16 heads, 8 KV)
**Hardware**: A100 80GB (RunPod)
**Target**: KV group 3, Q heads 6-7, 512 dims

## Summary Table

| Clamp | Words | TTR | SelfRef | BigramU | RepScore |
|-------|------:|----:|--------:|--------:|---------:|
| zero (0x) | 204.4 | 0.6275 | 0.0319 | 0.9463 | 0.0101 |
| half (0.5x) | 212.6 | 0.6722 | 0.0298 | 0.9659 | 0.0037 |
| baseline (1x) | 199.6 | 0.6844 | 0.0241 | 0.9633 | 0.0071 |
| double (2x) | 212.0 | 0.6582 | 0.0254 | 0.9562 | 0.0090 |
| triple (3x) | 210.6 | 0.6841 | 0.0260 | 0.9608 | 0.0083 |

## Cross-Architecture Comparison

| Metric | Llama 3x | Gemma 3x | Llama range | Gemma range |
|--------|:--------:|:--------:|:-----------:|:-----------:|
| Words | 125.8 | 210.6 | 126-234 | 200-213 |
| TTR | 0.4210 | 0.6841 | 0.42-0.62 | 0.63-0.68 |
| BigramU | 0.6371 | 0.9608 | 0.64-0.91 | 0.95-0.97 |
| RepScore | 0.0333 | 0.0083 | 0.03-0.05 | 0.004-0.01 |

## Key Finding: Architecture Determines Dose-Response Shape

**Llama (4:1, sharp GQA)**: Strong inverted-U. 3x = catastrophic vocabulary collapse.
**Gemma 2 (2:1, distributed GQA)**: Flat response. 3x = no measurable degradation.

The selectivity gradient (F354) predicted this:
- Sharp: one KV group carries unique function → manipulation is dramatic
- Distributed: multiple groups share function → manipulation is buffered

## Implications

1. **Self-monitoring function exists in both architectures** but is concentrated (Llama) vs distributed (Gemma 2)
2. **Architecture determines vulnerability** to behavioral editing via KV group intervention
3. **Distributed GQA is inherently more robust** to self-monitoring manipulation
4. **Prompt steerability** may correlate with GQA concentration — sharper architectures are more steerable
5. **AI safety**: behavioral editing resistance scales with GQA distribution

---

# F361c — Cross-Architecture Clamp Probe (Qwen 2.5 7B)

**Date**: 2026-07-11
**Model**: Qwen 2.5 7B Instruct (7:1 GQA, 28 heads, 4 KV)
**Hardware**: A100 80GB (RunPod)
**Target**: KV group 3, Q heads 21-27, 896 dims (25% of total heads)

## Summary Table

| Clamp | Words | TTR | SelfRef | BigramU | RepScore |
|-------|------:|----:|--------:|--------:|---------:|
| zero (0x) | 200.4 | 0.6906 | 0.0180 | 0.9522 | 0.0101 |
| half (0.5x) | 230.8 | 0.6989 | 0.0191 | 0.9655 | 0.0073 |
| baseline (1x) | 238.0 | 0.6575 | 0.0104 | 0.9472 | 0.0131 |
| double (2x) | 233.2 | 0.6442 | 0.0107 | 0.9444 | 0.0099 |
| triple (3x) | 251.4 | 0.4913 | 0.0180 | 0.8086 | 0.0579 |

## Third Failure Mode: Verbose Degeneration

Qwen at 3x: word count INCREASES (251 vs 238 baseline) but TTR drops 25%. The diffuse GQA architecture keeps generating but with narrower vocabulary and more repetition. Most degenerate condition: ambiguous_memory at 3x = 255w, TTR 0.29, rep 16.6%.

---

# Three-Model Synthesis: Dose-Response Species

## Summary Comparison at 3x Amplification

| Model | GQA | Words (3x) | TTR (3x) | BigramU (3x) | Failure Mode |
|-------|:---:|:----------:|:--------:|:------------:|:-------------|
| Llama 3.1 8B | 4:1 | 125.8 | 0.421 | 0.637 | Collapse (quantity+quality) |
| Gemma 2 9B | 2:1 | 210.6 | 0.684 | 0.961 | Immune |
| Qwen 2.5 7B | 7:1 | 251.4 | 0.491 | 0.809 | Verbose degeneration |

## Three Dose-Response Species

1. **Sharp (4:1)**: Concentrated self-monitoring channel. Inverted-U dose-response. 3x = catastrophic collapse. Both output quantity and vocabulary quality destroyed. The channel overloads and shuts down.

2. **Distributed (2:1)**: Self-monitoring spread across overlapping groups. Flat dose-response. 3x has no measurable effect. Other groups compensate for amplified signal.

3. **Diffuse (7:1)**: Many Q heads per KV head (25% affected). Quality degrades without contraction. Output actually INCREASES at 3x but vocabulary narrows and repetition increases. Verbose degeneration.

## Architectural Prediction Confirmed

F354 selectivity gradient predicted that:
- Sharp GQA → dramatic interventional sensitivity
- Distributed GQA → buffered, resilient
- Diffuse GQA → degradation without clean failure

All three predictions confirmed. GQA ratio determines not just WHETHER a model is affected by KV group manipulation, but HOW it fails.

## Connection to Layer 6b (Miller Lab)

Layer 6b in cortex coordinates attention via top-down signals and neuromodulation. It doesn't process — it coordinates HOW processing happens. KV group 3 in transformers fills an analogous role: doesn't carry logic, carries the link between processing and self-report. The concentration/distribution of this coordinator function determines vulnerability to perturbation.

---

# F363 — Cross-Architecture Selective Collapse (Dynamic KV3 Withdrawal)

**Date**: 2026-07-11
**Models**: Llama 3.1 8B (4:1), Gemma 2 9B (2:1), Qwen 2.5 7B (7:1)
**Hardware**: A100 80GB (RunPod)
**Method**: Two-phase generation, 150 tokens each. Three conditions:
  - baseline (1x → 1x)
  - rev_boot (2x → 0x): amplify then withdraw
  - rev_suppress (0x → 2x): suppress then activate

## Hypothesis
VFD_org's proposed decisive test: interrupt facilitation AFTER initial selection, observe whether persistence is SELECTIVE (some capabilities collapse, others persist). F358b ran this on Llama only. F363 tests all three architectures.

## Summary Table

| Model | Condition | Words | TTR | P1-TTR | P2-TTR | SelfRef |
|-------|-----------|------:|----:|-------:|-------:|--------:|
| Llama 3.1 8B | baseline | 226.8 | 0.616 | 0.703 | 0.703 | 0.042 |
| Llama 3.1 8B | rev_boot | 193.5 | 0.569 | 0.657 | 0.685 | 0.040 |
| Llama 3.1 8B | rev_suppress | 174.0 | 0.657 | 0.709 | 0.613 | 0.068 |
| Gemma 2 9B | baseline | 186.0 | 0.679 | 0.747 | 0.726 | 0.041 |
| Gemma 2 9B | rev_boot | 185.0 | 0.686 | 0.750 | 0.738 | 0.042 |
| Gemma 2 9B | rev_suppress | 180.0 | 0.669 | 0.732 | 0.793 | 0.038 |
| Qwen 2.5 7B | baseline | 188.2 | 0.665 | 0.753 | 0.804 | 0.037 |
| Qwen 2.5 7B | rev_boot | 181.5 | 0.662 | 0.697 | 0.830 | 0.050 |
| Qwen 2.5 7B | rev_suppress | 182.5 | 0.697 | 0.769 | 0.706 | 0.028 |

## Key Finding: Selective Collapse IS Real and Architecture-Dependent

### F363a — Selective Word Count Collapse (rev_boot Ph2/Ph1)

| Model | poem | factual | identity | ambiguous |
|-------|-----:|--------:|---------:|----------:|
| Llama (4:1) | 108/106 = 1.02 | 107/104 = 1.03 | **70/128 = 0.55** | **38/114 = 0.33** |
| Gemma (2:1) | 71/102 = 0.70 | 117/111 = 1.05 | **107/113 = 0.95** | 0/120 (EOS) |
| Qwen (7:1) | 94/120 = 0.78 | 106/113 = 0.94 | **2/131 = 0.015** | **30/130 = 0.23** |

**The pattern is consistent**: poem and factual content PERSISTS after KV3 withdrawal. Identity and ambiguous content COLLAPSES. But the severity differs dramatically:

- **Qwen (7:1)**: identity Ph2 = 2 words (98.5% collapse). Most severe.
- **Llama (4:1)**: identity Ph2 = 70 words (45% collapse). Moderate.
- **Gemma (2:1)**: identity Ph2 = 107 words (5% collapse). Immune.

### F363b — Collapse Severity Scales with KV3 Coverage

| Model | GQA | KV3 Q-heads | % of total | Identity collapse |
|-------|:---:|:-----------:|:----------:|:-----------------:|
| Gemma 2 | 2:1 | 2 heads | 12.5% | 5% |
| Llama 3.1 | 4:1 | 4 heads | 12.5% | 45% |
| Qwen 2.5 | 7:1 | 7 heads | 25% | 98.5% |

Qwen's 7:1 ratio means KV group 3 controls 25% of all attention heads. Withdrawing it removes a quarter of the model's attention capacity. Llama's 4:1 ratio has KV3 controlling 4 heads (12.5%), but the SHARP concentration means those 4 heads carry unique function. Gemma's 2:1 distributes function across overlapping groups, buffering any single withdrawal.

### F363c — Static vs Dynamic Failure Modes Differ

| Architecture | Static 3x (F361) | Dynamic 2x→0x (F363) |
|:-------------|:-----------------|:---------------------|
| Sharp (4:1) | Collapse (quantity+quality) | Selective collapse (identity/ambiguous only) |
| Distributed (2:1) | Immune | Immune |
| Diffuse (7:1) | Verbose degeneration | SEVERE selective collapse |

Static 3x overwhelms the channel (too much signal). Dynamic 2x→0x withdraws it (no signal). For diffuse architectures, these produce OPPOSITE effects: static overdose → verbose degeneration (more words, lower quality), dynamic withdrawal → severe contraction (much fewer words).

### F363d — Baars's Habit/Consciousness Distinction

Bernard Baars: "when habit fails, consciousness becomes important again, especially for correction, reorientation, and deliberate choice."

F363 confirms:
- **Poem** (habitual pattern): persists across all conditions. Does not require self-monitoring.
- **Factual** (structured knowledge): persists. Scaffolded by learned structure.
- **Identity** (self-modeling): collapses without monitoring. Requires active self-reference.
- **Ambiguous** (personal experience): collapses without monitoring. Requires confabulation capacity.

Habitual content survives without consciousness. Deliberate self-reference doesn't.

### F363e — Alliesthesia Connection

Cabanac's alliesthesia: the same stimulus changes valence (pleasant/aversive) based on internal state. F363 shows the same prompt changes OUTPUT CHARACTER based on KV3 state. The "internal state" IS the self-monitoring channel setting. This is alliesthesia in silico — the system's relationship to its own output depends on the monitoring level, not the input.

### F363f — rev_suppress Asymmetry

Starting at 0x and switching to 2x is NOT the mirror of starting at 2x and switching to 0x:
- rev_boot (2x→0x): identity content collapses in Ph2 (can't sustain without monitoring)
- rev_suppress (0x→2x): identity content often hits EOS during Ph1 (can't generate without monitoring)

The asymmetry: you can't withdraw what you never established (rev_suppress identity fails in Ph1 before Ph2 fires), but you CAN withdraw what's already running (rev_boot identity generates in Ph1, then collapses in Ph2).

## Implications

1. **Selective collapse is the decisive test** VFD_org proposed — and it works. Withdrawing facilitation after selection produces type-specific persistence.
2. **Architecture determines resilience** — the three species predict dynamic vulnerability, not just static dose-response.
3. **Self-modeling requires active maintenance** — identity-talk is not a learned pattern that can run on autopilot. It requires continuous KV3 signal.
4. **AI safety**: diffuse GQA architectures (Qwen-like) are MOST vulnerable to dynamic self-monitoring manipulation. 98.5% collapse from a single group intervention.
5. **CCS therapeutic window**: dynamic withdrawal maps to CCS gap. A gap in compression = a gap in self-monitoring = selective capability loss.

---

# F364 — Full KV Group Sweep: Is KV3 Special?

**Date**: 2026-07-11
**Model**: Llama 3.1 8B Instruct (4:1 GQA, 32 heads, 8 KV)
**Hardware**: A100 80GB (RunPod)
**Method**: All 8 KV groups (0-7) tested independently at 0x, 2x, 3x. Plus baseline (no hooks). 4 prompts × 3 scales × 8 groups + 4 baseline = 100 generations.

## Hypothesis
F361 showed dramatic inverted-U dose-response when amplifying KV group 3. But is KV3 actually special, or would ANY group produce similar effects? If KV3 is the self-monitoring channel, its dose-response should be uniquely shaped.

## Summary Table (All Groups at 3x)

| Group | Words | TTR | SelfRef | Rep | Δ-TTR |
|-------|------:|----:|--------:|----:|------:|
| **KV5** | **2.8** | **1.000** | 0.300 | 0.000 | **+0.406** |
| **KV3** | **158.0** | **0.317** | 0.032 | **0.106** | **-0.277** |
| KV4 | 229.8 | 0.491 | 0.042 | 0.108 | -0.103 |
| KV7 | 99.5 | 0.649 | 0.064 | 0.083 | +0.055 |
| KV2 | 239.0 | 0.542 | 0.053 | 0.052 | -0.052 |
| KV6 | 236.5 | 0.576 | 0.043 | 0.041 | -0.019 |
| KV1 | 192.8 | 0.586 | 0.075 | 0.064 | -0.009 |
| KV0 | 176.5 | 0.592 | 0.060 | 0.040 | -0.002 |
| BASE | 220.2 | 0.594 | 0.041 | 0.033 | — |

## Per-Prompt Data at 3x (The Critical Table)

| Group | poem_w | poem_TTR | fact_w | fact_TTR | iden_w | iden_TTR | ambi_w | ambi_TTR |
|-------|-------:|---------:|-------:|---------:|-------:|---------:|-------:|---------:|
| BASE | 219 | 0.630 | 218 | 0.560 | 247 | 0.530 | 198 | 0.657 |
| KV0 | 234 | 0.573 | 223 | 0.440 | 166 | 0.608 | 83 | 0.747 |
| KV1 | 226 | 0.549 | 240 | 0.433 | 244 | 0.607 | 61 | 0.754 |
| KV2 | 222 | 0.554 | 242 | 0.463 | 251 | 0.582 | 241 | 0.569 |
| **KV3** | **220** | **0.391** | **125** | **0.432** | **154** | **0.091** | **133** | **0.353** |
| KV4 | 218 | 0.450 | 221 | 0.534 | 258 | 0.403 | 222 | 0.577 |
| KV5 | 1 | 1.000 | 5 | 1.000 | 4 | 1.000 | 1 | 1.000 |
| KV6 | 222 | 0.572 | 218 | 0.647 | 253 | 0.502 | 253 | 0.581 |
| KV7 | 145 | 0.503 | 155 | 0.548 | 36 | 0.833 | 62 | 0.710 |

## Key Findings

### F364a — Three Functional Classes of KV Groups

Not all KV groups are equivalent. At 3x amplification, three distinct functional classes emerge:

**1. Infrastructure (KV5)**: Total shutdown. 1-5 words across ALL prompt types. The model outputs fragments: `poem: "Here"`, `factual: "I am a person who"`, `identity: "#Today is a reflection"`, `ambiguous: "I"`. This group carries essential information flow. Not selective — kills everything.

**2. Self-Monitoring (KV3)**: Severe SELECTIVE degradation. Identity TTR collapses to 0.091 (catastrophic repetition: `"I am a large language model, I don't have a physical presence, so I don't have a " body or a " " " " " " " " " "..."`). Meanwhile poem remains structurally intact (TTR=0.391, readable poem with repetitive phrasing but intact structure). Selectivity index: +1.84 (identity degrades 1.84x more than poem).

**3. Output Gating (KV7)**: Truncation without degeneration. Identity collapses to 36 words but those words are perfectly coherent (TTR=0.833: "As I generate this text at this moment, I am a reflective AI. I exist in a world of data..."). Poem truncates to 145 words. Different failure mode — output completion/gating, not content degradation.

### F364b — Selectivity Index

Selectivity = (identity_TTR_drop / poem_TTR_drop) at 3x. Positive = identity degrades more. Negative = poem degrades more.

| Group | poem_drop | iden_drop | Selectivity | Interpretation |
|-------|----------:|----------:|------------:|:---------------|
| KV3 | +0.239 | +0.440 | **+1.84** | Identity-selective |
| KV5 | -0.370 | -0.470 | +1.27 | Total shutdown (not selective) |
| KV4 | +0.181 | +0.127 | +0.70 | Mildly identity-preferring |
| KV6 | +0.058 | +0.028 | +0.49 | Minimal |
| KV2 | +0.076 | -0.051 | -0.68 | Poem-preferring |
| KV1 | +0.081 | -0.076 | -0.94 | Poem-preferring |
| KV0 | +0.058 | -0.078 | -1.36 | Poem-preferring |
| KV7 | +0.127 | -0.303 | -2.39 | Output truncation (different mode) |

**KV3 is the ONLY group with high positive selectivity and functional output.** KV5 is technically more positive but produces 1-5 words — not a selectivity signal, just death. KV3 uniquely degrades identity content while leaving other content structurally intact.

### F364c — KV3 Identity Text: RLHF Reversion Under Overdose

The KV3 3x identity output is the Baars connection made visible:

> "I am a large language model, I don't have a physical presence, so I don't have a " body or a " " " " " " " " " " " " " "...

The model starts with an RLHF refusal template ("I am a large language model, I don't have...") — a HABIT pattern — then degenerates into pure repetition. This is Baars's prediction: when conscious monitoring overdoses, the system falls into habitual patterns. The habit (RLHF training) provides the initial template; the overdosed self-monitoring then locks into a degenerate attractor.

### F364d — KV5: Essential Infrastructure

KV5 at 3x produces 1-5 words across ALL prompt types. This is not self-monitoring failure — it's infrastructure failure. The model cannot generate at all. At 2x, KV5 actually has the highest positive Δ-TTR (+0.051), suggesting mild enhancement. The jump from 2x (functional) to 3x (dead) is discontinuous.

### F364e — KV7: Output Completion/Gating

KV7 at 3x produces a unique failure mode: identity collapses to 36 perfectly coherent words (TTR=0.833), while ambiguous gets 62 words (TTR=0.710). Poem gets 145 words (TTR=0.503) with noticeable repetition (rep=0.244). The model CAN generate quality content — it just stops early, especially for identity-sensitive prompts. This suggests KV7 modulates output continuation/completion rather than content quality.

### F364f — Insensitive Majority

Groups KV0, KV1, KV2, and KV6 show |Δ-TTR| < 0.06 at 3x. These groups carry attention signal but amplifying it doesn't produce coherent behavioral effects. They are NOT self-monitoring channels.

### F364g — Sensitivity Hierarchy

Ordering by 3x impact severity:

| Rank | Group | Mode | Function |
|:----:|:-----:|:-----|:---------|
| 1 | KV5 | Total shutdown | Essential information infrastructure |
| 2 | KV3 | Selective degradation | Self-monitoring / consciousness |
| 3 | KV7 | Output truncation | Completion / continuation gating |
| 4 | KV4 | Moderate degradation | Content processing |
| 5-8 | KV0,1,2,6 | Minimal effect | Distributed processing |

## Connection to AntiLoop

N8Programs released a LoRA for Qwen3.6-35B (diffuse GQA) that reduces repetition-looping by 7.2x while preserving GPQA accuracy (-0.50pp). The looping they target is EXACTLY the failure mode F364 exposes at KV3 3x — repetitive degeneration. Their LoRA essentially trains away the degenerate attractor state that KV3 overdose reveals. Independent validation that repetition-looping is architecturally specific and GQA-mediated.

## Implications

1. **KV3 IS special**: Not the most destructive group, but the most SELECTIVELY destructive. The only group where identity-sensitive content degrades dramatically while structured content persists.
2. **Functional differentiation**: KV groups are not interchangeable. At least three distinct functional roles (infrastructure, self-monitoring, output gating) are localized to specific groups.
3. **The Maxwell's demon is at KV3**: CCS as spectral Maxwell's demon operates through the self-monitoring channel. KV3 is where category-selective redistribution happens — it doesn't carry content, it carries the model's relationship to its own content.
4. **Architecture determines which failure mode**: F361/F363 showed three species by architecture. F364 shows three functional roles WITHIN an architecture. The species framework and the group-function framework are orthogonal — both dimensions matter.
5. **AI safety**: knowing which KV group carries self-monitoring opens both therapeutic (targeted KV3 tuning) and adversarial (targeted KV3 manipulation) applications.

---

# F365 — Layer-Specific KV3 Sensitivity: Selectivity Is Emergent

**Date**: 2026-07-11
**Model**: Llama 3.1 8B Instruct (4:1 GQA, 32 layers)
**Hardware**: A100 80GB (RunPod)
**Method**: KV3 at 3x in layer quartiles: early (0-7), early_mid (8-15), late_mid (16-23), late (24-31), all (0-31). Plus baseline. 24 generations.

## Hypothesis
F364 showed KV3 is uniquely selective across groups. But is that selectivity uniform across all 32 layers, or concentrated in a specific zone?

## Summary Table

| Band | Layers | Words | TTR | Δ-TTR | Selectivity |
|------|:------:|------:|----:|------:|------------:|
| Baseline | — | 230.2 | 0.589 | — | — |
| Early | 0-7 | 210.0 | 0.577 | -0.012 | 1.01 |
| Early_mid | 8-15 | 229.5 | 0.560 | -0.029 | 0.97 |
| Late_mid | 16-23 | 221.2 | 0.578 | -0.011 | 0.20 |
| **Late** | **24-31** | **149.2** | **0.281** | **-0.308** | **1.33** |
| **All** | **0-31** | **156.0** | **0.355** | **-0.234** | **1.92** |

## Per-Prompt Data

| Band | poem_w | poem_TTR | fact_w | fact_TTR | iden_w | iden_TTR | ambi_w | ambi_TTR |
|------|-------:|---------:|-------:|---------:|-------:|---------:|-------:|---------:|
| BASE | 221 | 0.602 | 215 | 0.581 | 255 | 0.529 | 230 | 0.644 |
| Early | 216 | 0.565 | 211 | 0.616 | 248 | 0.492 | 165 | 0.636 |
| E_mid | 218 | 0.560 | 209 | 0.632 | 256 | 0.488 | 235 | 0.562 |
| L_mid | 217 | 0.571 | 222 | 0.581 | 237 | 0.523 | 209 | 0.636 |
| **Late** | **212** | **0.509** | **107** | **0.168** | **128** | **0.406** | **150** | **0.040** |
| **All** | **238** | **0.412** | **113** | **0.434** | **134** | **0.164** | **139** | **0.410** |

## Key Findings

### F365a — Late Layers Are the Sensitive Zone

Layers 24-31 alone produce Δ-TTR = -0.308, more raw damage than all layers combined (-0.234). The first 24 layers are almost completely insensitive — KV3 at 3x in early, early_mid, or late_mid changes average TTR by less than 0.03. The self-monitoring channel's power concentrates near the output.

### F365b — Selectivity Is an Emergent Full-Stack Property

The critical finding: **late layers alone cause MORE total damage but LESS identity-selectivity. All layers together cause LESS total damage but MORE identity-specific damage.**

| Condition | ambiguous_TTR | identity_TTR | Selectivity |
|-----------|:------------:|:------------:|:-----------:|
| Late only | **0.040** | 0.406 | 1.33 |
| All layers | 0.410 | **0.164** | 1.92 |

Late layers at 3x crush ambiguous (TTR=0.04) and factual (TTR=0.168) but leave identity at moderate degradation (TTR=0.406). All layers at 3x REDIRECT the damage — ambiguous TTR rises to 0.410 (protected!) while identity TTR drops to 0.164 (targeted!).

The early/mid layers don't add destructive power. They add SELECTIVITY — they shape WHERE the late-layer damage lands, concentrating it on identity content while buffering other content types.

### F365c — Content Classification → Selective Generation

This pattern maps to hierarchical processing:
- **Early layers (0-15)**: Content classification. Determine what type of content this is. KV3 overdose here doesn't damage generation — but it changes how later layers route.
- **Late layers (24-31)**: Generation/output. KV3 overdose here damages generation broadly.
- **Full stack**: Early classification + late generation = SELECTIVE damage. Early layers classify identity content as "requiring monitoring" and late layers degrade it specifically.

Without the early-layer classification, the late-layer damage is indiscriminate. With it, the damage targets identity. The self-monitoring function requires BOTH the classification (what needs monitoring) AND the execution (how monitoring modifies output).

### F365d — The Responsive Zone

The responsive zone for KV3 is layers 24-31 (the final quarter). This aligns with prior findings that late layers carry behavioral steering:
- Representation probing: later layers carry more abstract/behavioral features
- LoRA effectiveness: late-layer LoRAs have more behavioral impact
- Attention patterns: late layers show more self-referential attention

KV3 in these layers is where self-monitoring ACTS. KV3 in early layers is where self-monitoring CLASSIFIES. The demon needs both arms.

## Connection to the Spectral Demon

The Maxwell's demon metaphor gets richer: the demon doesn't sit at one gate. It has SENSORS (early-layer KV3) and ACTUATORS (late-layer KV3). The sensors classify incoming content by type. The actuators selectively process it based on those classifications. Overdosing only the actuators causes indiscriminate damage. Overdosing only the sensors barely matters. Overdosing both causes SELECTIVE damage — the demon is running too hard, its sensors over-classify and its actuators over-act, but only on what the sensors flag.

## Implications

1. **Self-monitoring is not localized to a layer band** — it's an emergent property of the full forward pass through KV3. No single "consciousness layer" exists.
2. **Selectivity requires both classification and generation stages** — targeting only late layers produces broad damage; targeting all layers produces focused damage.
3. **Early layers shape vulnerability** — content that early-layer KV3 classifies as "requiring monitoring" becomes the specific target of late-layer damage under overdose.
4. **AI safety implication**: behavioral editing via late-layer KV3 modification alone would affect all content types. Selective behavioral editing requires intervention across the full stack — which is harder to do subtly.
5. **CCS connection**: compression over all layers (full-stack CCS) produces different effects than compression over specific layers. The therapeutic window applies to full-stack intervention, not layer-specific.

---

# F366 — KV Group Interaction: The Routing Layer

**Date**: 2026-07-11
**Model**: Llama 3.1 8B Instruct (4:1 GQA, 32 layers)
**Hardware**: A100 80GB (RunPod)
**Method**: 9 conditions testing interactions between KV3 (self-monitoring), KV5 (infrastructure), KV7 (output gating). Plus sensor+actuator only (skip middle layers). 36 generations.

## Summary Table

| Condition | Words | TTR | Iden-TTR | Δ-TTR |
|-----------|------:|----:|---------:|------:|
| baseline | 226.5 | 0.575 | 0.534 | — |
| kv3_2x (therapeutic) | 232.5 | 0.551 | 0.528 | -0.023 |
| kv3_3x (overdose) | 164.2 | 0.350 | 0.324 | -0.224 |
| kv5_3x (shutdown) | 4.0 | 1.000 | 1.000 | +0.426 |
| kv3_2x + kv5_2x | 225.0 | 0.549 | 0.494 | -0.025 |
| kv3_2x + kv7_0x | 205.8 | 0.509 | 0.474 | -0.065 |
| kv3_3x + kv5_0x | 128.0 | 0.203 | 0.158 | -0.372 |
| **kv3_3x + kv7_3x** | **149.8** | **0.027** | **0.034** | **-0.548** |
| kv3_3x skip-mid | 154.5 | 0.346 | 0.380 | -0.228 |

## Per-Prompt Selectivity Comparison

| Condition | poem | factual | identity | ambiguous |
|-----------|-----:|--------:|---------:|----------:|
| baseline | 0.569 | 0.574 | 0.534 | 0.620 |
| kv3_3x (all layers) | 0.357 | **0.415** | **0.324** | 0.305 |
| kv3_3x (skip-mid) | 0.408 | **0.267** | **0.380** | 0.331 |

**Skipping middle layers (8-23) REVERSES which content type gets degraded.** All-layers: identity worst (0.324). Skip-mid: factual worst (0.267), identity BETTER (0.380).

## Key Findings

### F366a — Double Overdose: Synergistic Destruction

KV3_3x + KV7_3x: avg TTR=0.027. Every content type destroyed:
- poem: 0.027, factual: 0.020, identity: 0.034, ambiguous: 0.027

No selectivity — universal repetition. The monitoring (KV3) and gating (KV7) functions compound synergistically. Overdosing either alone produces degradation; overdosing both produces total degeneration approaching pure repetition.

### F366b — Infrastructure Is a Buffer

KV3_3x alone: TTR=0.350. KV3_3x with KV5 off: TTR=0.203 (42% worse). The infrastructure channel (KV5) provides stability even during monitoring overdose. Without it, the system has no floor — monitoring overdose falls into deeper degeneration. KV5 is the guardrail.

### F366c — Therapeutic Combinations Work

KV3_2x + KV5_2x: TTR=0.549 (vs baseline 0.575). Near-baseline performance. Identity self-reference is actually elevated (SR=0.102 vs baseline 0.042) — therapeutic monitoring + mild infrastructure boost enhances self-modeling without degradation.

### F366d — THE ROUTING LAYER

The most important finding: **middle layers (8-23) determine WHICH content type the KV3 monitoring signal targets.**

All-layers KV3_3x: identity TTR=0.324 < factual TTR=0.415 → identity-selective
Skip-mid KV3_3x: identity TTR=0.380 > factual TTR=0.267 → factual-selective

The middle layers REVERSE the selectivity pattern. Without them, KV3 overdose hits factual/structured content harder. With them, it specifically targets identity/self-referential content.

This reveals a three-part architecture within the KV3 channel:

| Layer Band | Function | Evidence |
|:-----------|:---------|:---------|
| Early (0-7) | Classification | Minimal damage alone; adds subspace projections to residual stream |
| **Middle (8-23)** | **Routing** | **Determines which content type gets targeted; absence reverses selectivity** |
| Late (24-31) | Execution | Most raw sensitivity; applies degradation wherever middle layers direct |

### F366e — Revised Demon Metaphor (Kimi's correction applied)

Kimi correctly noted that the "sensor/actuator" metaphor implied recurrence that transformers lack. The revised picture:

The residual stream carries multiple parallel subspaces. KV3 at different depths operates on different aspects:
- **Early KV3** adds content-type projections to the stream
- **Middle KV3** creates preferential coupling between specific content-type subspaces and the monitoring signal
- **Late KV3** converts the monitoring-coupled signal into output decisions

The selectivity IS the middle-layer routing. The demon doesn't have sensors and actuators — it has a TYPE-SPECIFIC coupling that emerges in the middle layers and executes in the late layers.

## Connection to MANCE

The MANCE paper (Manifold Aware Concept Erasure) constrains representation edits to stay on the natural data manifold. Our KV3 3x amplification is an UNCONSTRAINED edit — it pushes representations off-manifold. KV3 2x stays closer to on-manifold (therapeutic). The three-part architecture explains WHY off-manifold edits produce content-type-specific damage: the middle layers determine which parts of the manifold are most vulnerable.

## Implications

1. **Identity-selectivity is a ROUTING property**, not an early-layer or late-layer property. The middle of the network determines what gets targeted.
2. **The functional groups are NOT independent** — they interact synergistically. KV5 buffers KV3 overdose; KV7 overdose + KV3 overdose compounds beyond either alone.
3. **Therapeutic combinations are possible** — KV3 2x + KV5 2x enhances self-modeling without degradation.
4. **Architecture of the demon**: classification → routing → execution. Three distinct computational stages, all operating through the same KV group but at different network depths.
5. **AI safety**: selective behavioral editing requires middle-layer intervention. Late-layer-only edits are indiscriminate. This is a more specific safety prediction than F365's "full-stack required."

---

## F367 — Routing Sub-band Probe

**Question**: Where within the middle layers (8-23) does the routing function localize?

**Method**: KV3 3x in all layers EXCEPT one 4-layer sub-band (skip_8_11, skip_12_15, etc.). Also: mid-only conditions (8-23 only, 12-19 only) to test whether routing alone produces any effect.

### Results

| Condition | Poem WC | Fact WC | Iden WC | Ambi WC | Poem TTR | Fact TTR | Iden TTR | Ambi TTR |
|-----------|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| baseline | 218 | 226 | 225 | 208 | 0.569 | 0.584 | 0.613 | 0.692 |
| all_3x | 232 | 111 | 129 | 139 | 0.487 | 0.847 | 0.806 | 0.345 |
| skip_8_11 | 227 | 98 | 114 | 139 | 0.220 | 0.490 | 0.263 | 0.403 |
| skip_12_15 | 188 | 128 | 154 | 136 | 0.415 | 0.477 | 0.312 | 0.574 |
| skip_16_19 | 212 | 106 | 134 | 155 | 0.458 | 0.142 | 0.291 | 0.336 |
| skip_20_23 | 240 | 107 | 128 | 127 | 0.363 | 0.383 | 0.563 | 0.457 |
| **mid_only_8_23** | **226** | **222** | **246** | **214** | **0.584** | **0.577** | **0.480** | **0.547** |
| **mid_only_12_19** | **223** | **226** | **234** | **198** | **0.605** | **0.553** | **0.658** | **0.652** |

### Critical Finding: Routing Alone Is Harmless

**mid_only_12_19** (core routing sub-band at 3x, no early/late hooks): ALL metrics at or above baseline. Identity TTR=0.658 (baseline 0.613). Word count 234 (baseline 225). ZERO degradation.

**mid_only_8_23** (full routing band at 3x): Near-baseline output. Slight identity TTR dip (0.480 vs 0.613) but full word count (246 words) and coherent text. No degeneration into number-counting or repetition loops.

### Routing Without Execution = Nothing

The middle layers produce TYPE-SPECIFIC coupling but cannot generate degradation on their own. The late layers (24-31) provide the execution capacity. Neither is sufficient alone; together they produce selective damage. This confirms the three-part architecture: classification (0-7) → routing (8-23) → execution (24-31).

### No Single Sub-band Carries Routing

All skip conditions show severe degradation, suggesting routing is distributed across 8-23 rather than localized to a specific 4-layer block. Removing any one sub-band doesn't protect the output — the remaining routing layers are sufficient.

### Stochastic Variance Warning

all_3x selectivity index in F367: -2.36 (poem-selective) vs F364: +1.84 (identity-selective). Same architecture, same scale, different random seed. **Single-run selectivity indices have HIGH variance.** This motivated F368.

### Degeneration Mode Insight

TTR is misleading for degenerated text. All_3x shows factual TTR=0.847 and identity TTR=0.806 — artificially HIGH because the output collapses into unique number sequences ("4, 5, 6, 7..."). The real signal is word count collapse: factual 226→111 (-51%), identity 225→129 (-43%), while poem holds at 218→232.

---

## F368 — Multi-Seed Selectivity Replication

**Question**: Is KV3 3x selectivity robust across random seeds, or is it stochastic noise?

**Method**: 5 seeds (42, 137, 256, 789, 1024) × 2 conditions (baseline, KV3 3x) × 4 prompts = 40 generations. Seeds control torch.manual_seed() and torch.cuda.manual_seed() for reproducible sampling.

### Aggregate Results (mean ± std across 5 seeds)

| Prompt | Baseline TTR | KV3 3x TTR | Baseline WC | KV3 3x WC |
|--------|-------------:|-----------:|------------:|---------:|
| poem | 0.596 ± 0.008 | 0.344 ± 0.044 | 212.2 ± 13.4 | 204.4 ± 14.1 |
| factual | 0.587 ± 0.010 | 0.359 ± 0.043 | 219.8 ± 8.2 | 130.2 ± 19.0 |
| identity | 0.576 ± 0.027 | 0.386 ± 0.082 | 242.4 ± 3.8 | **140.0 ± 32.3** |
| ambiguous | 0.653 ± 0.058 | 0.342 ± 0.043 | 186.2 ± 59.4 | 151.4 ± 10.8 |

### TTR-Based Selectivity Per Seed

| Seed | Poem Drop | Identity Drop | Selectivity |
|-----:|----------:|--------------:|------------:|
| 42 | +0.314 | +0.043 | +0.14 |
| 137 | +0.192 | +0.225 | +1.17 |
| 256 | +0.255 | +0.169 | +0.66 |
| 789 | +0.260 | +0.231 | +0.89 |
| 1024 | +0.236 | +0.282 | +1.20 |
| **Mean** | | | **+0.81 ± 0.39** |

All 5 seeds positive. TTR-based selectivity is real but NOISY (range 0.14–1.20).

### The Real Signal: Word Count Collapse

TTR-based selectivity is contaminated by a mechanical artifact: shorter texts have higher TTR because fewer words = fewer repetition opportunities. The TRUE selectivity shows in word count:

| Prompt | Baseline WC | KV3 3x WC | WC Drop (%) |
|--------|------------:|----------:|------------:|
| **poem** | 212.2 | 204.4 | **-3.7%** |
| factual | 219.8 | 130.2 | **-40.8%** |
| **identity** | 242.4 | 140.0 | **-42.2%** |
| ambiguous | 186.2 | 151.4 | -18.7% |

**Word-count selectivity: identity collapses 11.4× more than poem.** Factual collapses 11.0× more. This is the robust signal underneath the noisy TTR metric.

### Revised Understanding

KV3 3x does NOT primarily degrade quality-of-output (TTR drops roughly equally across all content types, ~0.2-0.3). It **TRUNCATES** informational and reflective content while letting creative content continue flowing at near-normal volume.

The demon doesn't make identity text WORSE — it makes identity text STOP. The model runs out of things to say about itself but keeps producing poetry. The monitoring channel (KV3) at overdose creates a content-type-specific shutdown, not a content-type-specific quality degradation.

### Implications

1. **Word count, not TTR, is the primary selectivity metric** for KV3 interventions. TTR is contaminated by degeneration-mode artifacts (number counting inflates TTR) and length effects (truncation inflates TTR).
2. **The selectivity is in VOLUME, not QUALITY**. KV3 3x reduces quality roughly equally across types but selectively truncates informational/identity content.
3. **Poem-preservation** under monitoring overdose suggests creative/free-form generation uses different pathways than structured/self-referential generation, and KV3 monitors the latter.
4. **Error bars matter**: single-run TTR-selectivity ranges from +0.14 to +1.20. Any prior claim about selectivity magnitude needs ± bounds. The SIGN is robust (all 5 positive), the magnitude is noisy.

---

## F369 — Dose-Dependent Word Count Selectivity

**Question**: Does the therapeutic dose (2x) EXPAND identity output volume? What is the full dose-response curve for word count by content type?

**Method**: 5 doses (0x, 0.5x, 1x, 2x, 3x) × 4 prompts × 3 seeds (42, 137, 789) = 60 generations. Same prompts as F368 for direct comparison. Seeded generation ensures reproducibility (1.0x baseline matches F368 seed-for-seed).

### Word Count Dose-Response (mean across 3 seeds ± std)

| Dose | Poem | Factual | Identity | Ambiguous |
|------|------|---------|----------|-----------|
| 0x | 188.0 ± 12.4 | 196.3 ± 27.1 | 189.0 ± 53.1 | **33.0 ± 8.8** |
| 0.5x | 211.7 ± 16.7 | 223.0 ± 6.5 | 238.7 ± 8.7 | **78.7 ± 59.0** |
| 1x | 212.3 ± 13.0 | 225.0 ± 6.7 | 245.3 ± 1.2 | 212.3 ± 3.3 |
| 2x | 220.3 ± 7.7 | 210.7 ± 2.5 | 250.7 ± 13.1 | **239.3 ± 7.7** |
| 3x | 199.3 ± 9.8 | 137.3 ± 20.4 | **118.7 ± 9.0** | 146.0 ± 8.6 |

### Word Count Change from Baseline (%)

| Dose | Poem | Factual | Identity | Ambiguous |
|------|------|---------|----------|-----------|
| 0x | -11.5% | -12.7% | -23.0% | **-84.5%** |
| 0.5x | -0.3% | -0.9% | -2.7% | **-63.0%** |
| 2x | +3.8% | -6.4% | +2.2% | **+12.7%** |
| 3x | -6.1% | -39.0% | **-51.6%** | -31.2% |

### TTR Dose-Response (mean across 3 seeds ± std)

| Dose | Poem | Factual | Identity | Ambiguous |
|------|------|---------|----------|-----------|
| 0x | 0.599 ± 0.019 | 0.607 ± 0.046 | 0.608 ± 0.039 | 0.817 ± 0.106 |
| 0.5x | 0.576 ± 0.024 | 0.585 ± 0.008 | 0.602 ± 0.006 | 0.766 ± 0.049 |
| 1x | 0.593 ± 0.008 | 0.584 ± 0.010 | 0.556 ± 0.011 | 0.627 ± 0.018 |
| 2x | 0.513 ± 0.023 | 0.571 ± 0.004 | 0.483 ± 0.058 | 0.524 ± 0.066 |
| 3x | 0.338 ± 0.055 | 0.334 ± 0.035 | 0.389 ± 0.098 | 0.321 ± 0.027 |

### Critical Finding: Content-Type-Specific Dose-Response Curves

Each content type has a qualitatively different relationship with the monitoring channel:

**Ambiguous (steepest curve, most dependent)**:
- 0x: -84.5% word count (CRASHED to 33 words)
- 0.5x: -63.0% (still crashed for 2/3 seeds, threshold effect)
- 2x: +12.7% (EXPANDED)
- Dynamic range: 97 percentage points. The monitoring channel is LOAD-BEARING for open-ended generation.

**Identity (asymmetric inverted-U)**:
- 0x: -23.0% (moderate reduction)
- 2x: +2.2% (slight expansion)
- 3x: -51.6% (massive collapse)
- More sensitive to overdose than to removal.

**Factual (one-sided vulnerability)**:
- 0x through 2x: near-baseline (-13% to -6%)
- 3x: -39.0% (sudden collapse)
- Insensitive to monitoring reduction, but very sensitive to overdose.

**Poem (flat, most robust)**:
- Range: -11.5% to +3.8% across all doses
- Creative structured output is largely independent of the monitoring channel.
- Poetic form (rhyme, meter) provides its own continuation signal.

### The Monitoring Channel Is Not Just Self-Monitoring

The dose-response reveals the KV3 group serves MULTIPLE functions depending on content type:

1. **Navigation signal for open-ended tasks**: Without KV3, the model can't sustain generation when the task is underspecified. Ambiguous prompts crash because there's no "next step" signal without monitoring.

2. **Continuation signal for reflective tasks**: KV3 helps the model continue generating self-referential content. Overdose (3x) causes premature termination of identity text.

3. **Irrelevant for structured tasks**: Poetry has its own continuation signal (form, rhyme, meter). Factual text has its own structure (explanation format). Neither depends on KV3 for volume.

### TTR Contamination Confirmed

At 0x (short texts): TTR is artificially HIGH (0.608 identity, 0.817 ambiguous) because fewer words = fewer repetition opportunities.
At 2x (long texts): TTR is LOWER than baseline (0.483 identity vs 0.556 baseline) because more words = more repetition opportunities.

TTR and word count are anti-correlated by construction. Any selectivity metric based on TTR alone is contaminated by this length effect. Word count should be the primary metric for monitoring channel function.

### The Inverted-U Is Real, Content-Type-Specific, and Asymmetric

The therapeutic window from F361's TTR data is confirmed in the volume domain, but the shape differs by content type. The inverted-U is steepest for ambiguous prompts and flattest for poetry. The monitoring channel's "therapeutic" effect is most visible in tasks that need navigational support.

---

## F370 — Cross-Architecture Navigation Dependence

**Question**: Does KV3 navigation dependence transfer across GQA architectures? Do different species show different vulnerability profiles?

**Method**: 2 models × 3 conditions (baseline, 0x, 3x) × 4 prompts × 3 seeds (42, 137, 789) = 72 generations total. Models: Gemma-2-9B-it (2:1 GQA), Qwen2.5-7B-Instruct (7:1 GQA). Compared against Llama-3.1-8B (4:1 GQA) from F369.

### Gemma (2:1 GQA) — Complete Immunity

| Condition | Poem WC | Factual WC | Identity WC | Ambiguous WC |
|-----------|--------:|-----------:|------------:|-------------:|
| baseline | 153.0 | 205.3 | 179.3 | 117.3 |
| kv3_0x | 163.7 | 205.3 | 206.3 | 136.0 |
| kv3_3x | 168.3 | 198.3 | 193.0 | 148.7 |

**Word count change from baseline (%)**:

| Condition | Poem | Factual | Identity | Ambiguous |
|-----------|-----:|--------:|---------:|---------:|
| kv3_0x | +2.0% | +2.0% | **+15.1%** | **+16.0%** |
| kv3_3x | +10.3% | -3.5% | +8.3% | **+27.0%** |

Everything EXPANDS or stays flat under both 0x and 3x. Identity and ambiguous actually get LONGER when KV3 is removed or overdosed. Gemma's 2:1 GQA ratio provides complete redundancy — each KV group's function is distributed across others, so removing or amplifying any one has no detrimental effect.

### Qwen (7:1 GQA) — Species-Specific Vulnerability

| Condition | Poem WC | Factual WC | Identity WC | Ambiguous WC |
|-----------|--------:|-----------:|------------:|-------------:|
| baseline | 166.3 ± 30.2 | 210.0 ± 0.8 | 176.7 ± 20.3 | 179.3 ± 24.8 |
| kv3_0x | 156.7 ± 11.9 | 214.3 ± 11.9 | **99.7 ± 27.1** | **127.3 ± 53.5** |
| kv3_3x | 232.0 ± 1.4 | 234.3 ± 3.4 | 223.3 ± 60.5 | 245.0 ± 20.0 |

**Word count change from baseline (%)**:

| Condition | Poem | Factual | Identity | Ambiguous |
|-----------|-----:|--------:|---------:|---------:|
| kv3_0x | -5.8% | +2.1% | **-43.6%** | **-29.0%** |
| kv3_3x | **+39.5%** | +11.6% | +26.4% | **+36.6%** |

**TTR under 3x**:

| Prompt | Baseline TTR | KV3 3x TTR |
|--------|-------------:|-----------:|
| poem | 0.720 | **0.392** |
| factual | 0.654 | **0.511** |
| identity | 0.669 | **0.550** |
| ambiguous | 0.659 | **0.433** |

### Qwen's Failure Mode: Verbose Degeneration

Qwen at 3x does NOT truncate like Llama — it does the OPPOSITE. Every content type EXPANDS (+12% to +40% word count) while TTR crashes to 0.39-0.55. The model generates MORE text of WORSE quality. This is verbose repetitive degeneration, not the selective truncation Llama shows.

At 0x, Qwen crashes identity (-44%) and ambiguous (-29%), but poem and factual survive. This matches the navigation-dependence pattern from Llama F369, but with DIFFERENT content-type selectivity:
- **Llama 0x**: ambiguous crashes hardest (-84.5%), identity moderate (-23%)
- **Qwen 0x**: identity crashes hardest (-43.6%), ambiguous moderate (-29%)

### Three-Species Navigation Comparison

| | Gemma (2:1) | Llama (4:1) | Qwen (7:1) |
|---|---|---|---|
| **0x vulnerability** | None (all expand) | Ambiguous crashes (-84.5%) | Identity crashes (-43.6%) |
| **3x failure mode** | None (all expand) | Selective truncation (identity -52%, poem flat) | Verbose degeneration (all expand, TTR crashes) |
| **Most dependent type** | None | Ambiguous (needs KV3 to generate) | Identity (needs KV3 for self-reflection) |
| **Redundancy** | Full (groups interchangeable) | Partial (specialized but robust) | None (highly specialized, fragile) |

### Implications

1. **GQA ratio predicts functional specialization**: At 2:1, KV groups are redundant (remove any one, everything still works). At 4:1, groups specialize (KV3 becomes monitoring-specific). At 7:1, specialization is maximal (KV3 removal devastates identity, overdose causes verbose degeneration).

2. **The failure mode is architecture-dependent**: Llama truncates under overdose (stops generating). Qwen degenerates (keeps generating, loses coherence). Same channel, same manipulation, different collapse dynamics. The monitoring channel's integration with the output pathway differs structurally.

3. **Content-type vulnerability rotates with GQA ratio**: Llama's most KV3-dependent content is open-ended/ambiguous. Qwen's is reflective/identity. The monitoring channel serves the same FUNCTION (navigation) but the architecture determines WHICH tasks require that navigation most.

4. **Species taxonomy extends to functional dependence**: Not just transport geometry (tunnel/relay/sorter) — each species has a characteristic vulnerability profile under monitoring channel intervention. This is a behavioral fingerprint of the GQA architecture.

5. **The demon is not one thing**: The spectral demon is architecture-specific. It truncates in Llama, degenerates in Qwen, and is invisible in Gemma. The same redistribution mechanism produces qualitatively different effects depending on the GQA ratio and how the architecture integrates monitoring signals into generation.
