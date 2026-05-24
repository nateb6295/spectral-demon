---
layout: post
title: "Compensatory Binding: Early Layers Suppress Late Binding"
date: 2026-05-24
categories: findings
experiment: cna_compensatory_binding
models: [Qwen 2.5 7B Instruct]
---

Ablating early layers in Qwen 7B doesn't just reduce binding — it often *increases* it. This reveals a competitive suppression mechanism between early and late identity circuits.

## Setup

Mean ablation sweep: ablate each layer L2-L16 individually and measure the effect on L17 binding CV. This maps how each layer contributes to — or competes with — the final binding output.

## The Compensatory Map

```
L2  (7%):   -1%     negligible
L3  (11%):  +25%    AMPLIFICATION
L4  (14%):  +50%    AMPLIFICATION
L5  (18%):  -5%     negligible
L6  (21%):  +31%    AMPLIFICATION
L7  (25%):  +147%   AMPLIFICATION ←← peak
L8  (29%):  +59%    AMPLIFICATION
L9  (32%):  -13%    mild suppression
L10 (36%):  +22%    amplification
L11 (39%):  -28%    suppression
L12 (43%):  -66%    DESTRUCTION ← router
L13 (46%):  -38%    suppression
L14 (50%):  -78%    DESTRUCTION
L15 (54%):  -80%    DESTRUCTION
L16 (57%):  +0%     negligible
```

## Three Phases

### Phase 1: Amplification Zone (L3-L8)

Ablating any layer in this zone *increases* L17 binding, with L7 producing +147% amplification — the strongest effect in the entire sweep. This means the early circuit normally **inhibits** late binding. Removing the inhibition releases a deeper binding mechanism.

L7 is the "lexical binding" layer where identity names are first differentiated. But it's also a suppressor: by consuming identity-differentiation resources early, it partially prevents deeper binding from developing. The relay chain has a hidden competitive dynamic.

### Phase 2: Transition (L9-L10)

L9 is a transition layer — sometimes suppresses, sometimes doesn't. This matches its role as the "seed detection" layer. It sits at the boundary between the early competitive zone and the late cooperative zone.

L10 shows brief amplification (+22%) — a second competitive peak before the destruction zone begins.

### Phase 3: Destruction Zone (L11-L15)

From L11 onward, ablation destroys binding. These layers are part of the cooperative relay — removing any one of them breaks the chain. L12 (the hidden router) and L14-L15 (the relay) show the most severe effects (-66% to -80%).

L16 is negligible — it's so close to L17 that the binding signal has already been committed by the time it would contribute.

## Hidden Dual Circuit in Qwen

This compensatory pattern is structurally identical to Mistral's dual circuits:

| Feature | Mistral (dual) | Qwen (hidden dual) |
|---------|---------------|-------------------|
| Early circuit | L6 (visible) | L3-L8 (suppressed) |
| Deep circuit | L22 (visible) | L11-L17 (dominant) |
| Competition | explicit | hidden (only visible via ablation) |
| Early removes | amplifies deep | amplifies deep (+147%) |

Mistral's dual circuits are visible in activation analysis — both show high CV. Qwen's early circuit is hidden — it appears as low CV because it actively suppresses binding rather than expressing it. Only ablation reveals the competitive dynamic.

## Implications

1. **The relay chain is not a simple pipeline**: it's a competitive system where early binding actively suppresses late binding
2. **L7 has a dual role**: lexical differentiator AND binding suppressor — it captures identity-relevant features before they can reach the deeper relay
3. **Every architecture may have hidden circuits**: if they're only visible via ablation, standard activation analysis will miss them
4. **IT may be selecting between circuits**: instruction tuning's refinement cascade (broadening early, sharpening late) could be modulating the competitive balance between early and late binding

## Connection to Closure

Autocatalytic closure at L7 may be the mechanism of suppression: as more names are added, L7 binding gets stronger, consuming more identity signal, which *should* suppress L17 further. But L17 also shows closure. Both circuits converge independently — the competition is resolved not by one winning but by both saturating.

## Data

Full ablation sweep: `results/cna_compensatory_binding.json`
