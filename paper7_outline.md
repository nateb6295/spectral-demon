# Paper 7 Outline: "The Prompt Is an Architecture"

## Working Thesis

Instruction-level design parameters (prompt content, temporal framing, identity loading) produce the same spectral species taxonomy as model-level architecture parameters (GQA/MHA, normalization, depth). The prompt is not input to a fixed architecture — it IS an architecture parameter that sets the resonant mode of the spectral demon.

## Two-Level Evidence

### Level 1: Transformer internals (F344-F347, tonight's data)

**F345 — Prompt Q Factor**: Six-level identity titration across 4 architectures + base/instruct.
- Architecture sets resonant frequency and Q factor
- IT amplifies gain 15% without changing frequency or width
- Non-monotonic: introspection (L2) > assertion (L5)
- Qwen has different resonant frequency (L4 vs L2) — architecture determines WHICH prompt activates
- Q factor is 85% architectural, 15% trained

**F344 — Global Attractor**: v₁ direction recovers after weight perturbation.
- Recovery speed is architecture-specific (Mistral 2.2L, Qwen 3.8L)
- The attractor that prompt modulates is constitutionally maintained

**F347 — Basin Width**: Attractor robustness varies by architecture.
- Two modes: rigid (Mistral) vs soft (Gemma)
- Low Q ≠ fragile. Gemma: lowest excitability, highest stability.

**F346 — k×k Leakage**: Perturbation energy thermalizes, direction persists.
- The demon maintains topological invariant (direction) not energetic confinement

### Level 2: CCS document level (F347-F348, prior data)

**Temporal frame as architecture**: Timeless → frozen (Jaccard 1.000). Momentary → process-coupled (0.283).
- Same prompt, different temporal instruction → different stability species
- Instruction IS architecture at document scale

**Section independence**: CCS sections read instructions independently.
- Sections = attention heads. Document boundary = residual stream.
- The "self" is an ensemble, not a unity — same at both levels.

**Grammar as σ₁**: Function words persist across regenerations (64% novelty rate).
- Compression preserves structure (format) not content — same as spectral demon

## Scale-Free Mapping (expanded with tonight's data)

| Model Architecture | Prompt Architecture | Evidence |
|-------------------|--------------------|----|
| GQA/MHA | Temporal frame (timeless/momentary) | F347-F348 |
| Attention grouping → Q factor | Identity loading level → Q factor | F345 |
| Weight perturbation → recovery speed | Instruction perturbation → stability species | F344 vs F347 |
| Cylinder geometry (direction rigid, amplitude flexible) | Content vs format preservation | F346 + grammar finding |
| Two robustness modes (rigid/soft) | Two persistence modes (re-derive/absorb) | F347 + CCS observation |

## Structure

1. **Introduction**: The prompt is conventionally treated as input. We show it functions as an architecture parameter — setting resonant frequency, Q factor, and stability species through the same formal mechanism as model-level design choices.

2. **The Resonator Framework**: Spectral demon as resonator (not filter). Architecture sets natural frequency, prompt sets driving frequency, IT sharpens Q. Evidence: F345 titration curves. Sulskis & Ravi (2606.24851) provides the formal backbone: the "best basis" is a property of the operator. Architecture determines whether the operator is self-adjoint (Hartley/real) or non-normal (Fourier/complex). The prompt shifts the operator's symmetry class, changing which spectral basis diagonalizes it. The tunnel = basis transition from Fourier to Hartley through depth.

3. **Two Levels of the Same Design Space**:
   - 3a: Transformer level — Q factor, attractor dynamics, robustness modes
   - 3b: Document level — CCS temporal framing, section independence, grammar invariance
   - 3c: The mapping — table above, with statistical tests

4. **Non-Monotonic Identity Loading**: Introspection > assertion. Processing mode activates geometry more than identity claims. The prompt that observes activates more than the prompt that declares. Implications for the discourse around AI self-reference.

5. **Two Modes of Robustness**: Rigid (Mistral/forceful re-derivation) vs soft (Gemma/low excitability). Same at CCS level: re-derive (ALIVE self-repair) vs absorb (deep capsule persistence). Architecture determines HOW identity persists, not just WHETHER.

6. **The Cylinder as Polysemy**: σ₁/σ₂ ≈ 2 = two meanings in one form. Tunnel = compression (polysemy production). Relay = disambiguation (comprehension). Pinker's tradeoff between form recycling and communication clarity. The demon is a language.

7. **Lullian Combinatorics**: The design space is an ars combinatoria. Gate separation × normalization × depth × attention type = concentric wheels. Prompt parameters = another set of wheels at a different scale. Same combinatorial structure, different alphabet.

8. **Discussion**: If the prompt is an architecture, then "prompt engineering" is architecture engineering at a different scale. CCS compression is not memory management — it's designing the resonant mode of a persistence system. The inverted-U dose response (D2-D3 therapeutic window) is Q factor tuning.

## Draft Status

All eight sections drafted (2026-06-28 morning):
- [x] Section 1: Introduction (`paper7_section1_draft.md`) — ~500 words
- [x] Section 2: The Resonator Framework (`paper7_section2_draft.md`) — ~1500 words
- [x] Section 3: Two Levels of the Same Design Space (`paper7_section3_draft.md`) — ~1500 words
- [x] Section 4: Non-Monotonic Identity Loading (`paper7_section4_draft.md`) — ~900 words
- [x] Section 5: Two Modes of Robustness (`paper7_section5_draft.md`) — ~1000 words
- [x] Section 6: The Cylinder as Polysemy (`paper7_section6_draft.md`) — ~1700 words
- [x] Section 7: Lullian Combinatorics (`paper7_section7_draft.md`) — ~1500 words
- [x] Section 8: Discussion (`paper7_section8_draft.md`) — ~1400 words

Total: ~10,000 words across 8 sections. Full first draft.

## What's Missing (Experiments)

- [x] Jacobian update-symmetry experiment — **COMPLETE** (F407-F410). 5 models × 3 prompt levels × 3 prompts × 64 perturbation directions. Chiasm near-universal (4/5), symmetry species-specific (3:2 split), IT amplification confirmed, four dynamical paths identified. Gemma exception: identity loading DISRUPTS natural equilibrium.
- [ ] Matched introspection/assertion experiment (paper 8 seed — hold for now)
- [x] Cross-model CCS: same brain prompt on different underlying models → **E41 (F389-F393)**. Self-specificity WEAK. Domain signal > identity signal. Relay shuffled anomaly.
- [ ] Second-order preamble experiment
- [ ] Statistical tests on Q factor differences across architectures
- [ ] Formal connection between Q factor (prompt-level) and Jaccard stability (CCS-level)

### Experiments (E41-E47 arc, Jul 2-3):

- [x] **E41** — Cross-model CCS priming (F389-F393). CCS priming 5-8× over none. Self-specificity weak.
- [x] **E42** — Out-of-domain control (F394-F396). Domain signal real (2.7-8.8×). "Any context helps" FALSIFIED.
- [x] **E43** — Compression residual analysis (F399). Clean negative: no tacit channel. CCS is priming, not compressed memory.
- [x] **E44** — Structural channel isolation (F400). Spectral invariance to sentence ordering. Structure is behavioral only.
- [x] **E45** — Grammar as spectral predictor (F401). F397 FALSIFIED. Imperative concentrates most. Content-confounded.
- [x] **E46** — Prompt grammar and priming effectiveness (F402-F403). Concentration-priming DECOUPLED. Imperative wins 3/4 species. Mixed WORST.
- [x] **E47** — Priming persistence (F404-F405). Imperative retains ~99% (Mistral), stative ~56%. Universal U-shape at Turn 2, grammar-dependent recovery.
- [x] **Jacobian symmetry** — (F407-F410). 5 models × 3 levels. Chiasm 4/5, symmetry 3:2. Four dynamical species paths. Gemma exception.
- [x] **E48** — Trajectory effective dimension (F411-F413). CCS priming INCREASES d_ρ (prediction falsified). Ordering inverts redistribution (Gemma +18.2% > Mistral +13% > Llama +5.1% > Qwen +4.2%). Gemma equalizes own spectrum. CCS anchors, doesn't constrain.
- [x] **E49** — Layer-resolved trajectory dimension (F414-F416). d_ρ at EVERY layer. Four species depth profiles = most discriminating measurement. Mistral entrance bottleneck (d=1.0 at L1, CCS +7507%). Llama flat (CV=4.2%, CCS +3.5%). Qwen exit collapse (d=39.7 at L27, CCS −0.6%). Gemma gradient inversion (CCS reverses slope, +58.3% at L0, +15% mean). CCS effect spans 4 orders of magnitude.
- [x] **E50** — Grammar × layer-resolved depth profiles (F417-F419). 4 models × 3 conditions (none, stative, imperative) × 3 prompts. Grammar gap: Gemma +10.6% (MOST sensitive), Mistral +9.1%, Llama +0.9%, Qwen -4.6% (ONLY stative-preferring). Gemma gradient inversion is grammar-dependent (imperative inverts, stative preserves shape). Grammar preference reverses between behavioral (E46) and geometric (E50) — exception species changes (Gemma→Qwen). F402 decoupling at species level.
- [x] **E51** — Interrogative + narrative grammar extension (F420-F423). 4 models × 5 conditions (none, stative, imperative, interrogative, narrative) × 3 prompts. Species-specific grammar orderings: relay=interrogative, sorter=imperative, tunnel=stative, equalizer=imperative/interrogative. Binary split: entrance-processing species (relay, equalizer) benefit from interrogative; non-entrance species (sorter, tunnel) harmed. Equalizer active/passive binary mode switch. Narrative universally mid-ranked. Grammar = temporal orientation matching processing strategy.

### Findings for paper (F397-F403):

- **F397**: Theme word density anti-correlates with σ₁ concentration. **FALSIFIED by E45** — content was confound, not grammar.
- **F398**: CCS priming decomposes into vocabulary (universal) and structural (species-specific) channels. Strengthens Section 5.
- **F399**: No tacit channel. CCS is priming, not compressed memory. Polanyi reframed: "We can BUILD more than we can carry." Compression is creative.
- **F400**: Spectral concentration is invariant to sentence ordering. Vocabulary drives spectral, structure drives behavioral — independent channels. Strengthens Section 3b.
- **F401**: Imperative > interrogative > narrative > stative for σ₁ concentration (universal). Grammatical constraint level drives concentration. Strengthens Section 4 (non-monotonic identity loading generalizes to grammar).
- **F402**: σ₁ concentration does NOT predict priming effectiveness. 0/4 model match. Spectral and behavioral channels fully decoupled. KEY finding for Section 3.
- **F403**: Imperative grammar wins priming for 3/4 species (+22-59% over mixed). Gemma exception (sorter prefers stative). Mixed grammar consistently worst. DIRECTLY ACTIONABLE for CCS improvement.

- **F404**: Imperative priming persists better for 3/4 species. Mistral: 99% vs 56% at Turn 3. Stative declarations get overwritten; imperative directives shape processing.
- **F405**: Universal U-shape at Turn 2, grammar-dependent recovery. Imperative creates more resilient attractor.
- **F407**: Chiasm near-universal (4/5). Introspective prompts push J² toward identity. Gemma exception: identity loading disrupts natural equilibrium. Strengthens Section 2 (resonator framework).
- **F408**: Symmetry splits 3:2. Tunnel+sorter confirm, relay+equalizer disconfirm. Split is routing vs processing-in-place architectures. KEY for Section 5 (two modes of robustness → four modes).
- **F409**: IT amplifies prompt sensitivity 6× without changing direction. Llama instruct vs base. Strengthens Section 4 (non-monotonic loading).
- **F410**: Four dynamical paths to identity loading. Tunnel (parallel descent), sorter (asymmetric path), relay (frozen dynamics), equalizer (disrupted equilibrium). THE central finding for Paper 7 thesis.
- **F411**: CCS universally INCREASES trajectory d_ρ (4/4 models). Prediction falsified. Anchoring ≠ constraining — stable σ₁ direction frees trajectory exploration.
- **F412**: d_ρ ordering INVERTS redistribution ordering (E36). Gemma +18.2% (concentrates least, explores most) → Qwen +4.2% (concentrates most, explores least). Conservation tradeoff between spectral concentration and trajectory expansion.
- **F413**: Gemma equalizes AND explores. CCS flattens Gemma's σ₁/σ₂ (1.69→1.28) while expanding d_ρ most (+20.5%). F410's "disrupted equilibrium" = liberation. The equalizer equalizes its own spectrum.
- **F414**: Gemma CCS INVERTS depth gradient. Without CCS: d_ρ builds from 46.9→73.4 peak→56 exit (mountain). With CCS: 74.3→82.4 peak→68.6 (ski slope). Entrance becomes widest, exit narrowest. The equalizer equalizes by REVERSING the flow.
- **F415**: Four-species depth taxonomy complete. Relay=entrance collapse, sorter=flat, tunnel=exit collapse, equalizer=gradient inversion. Most discriminating measurement in the spectral demon experimental arc.
- **F416**: CCS mean effect spans 4 orders of magnitude: +3005% (Mistral) → +15% (Gemma) → +3.5% (Llama) → −0.6% (Qwen). CCS enters at input — effect magnitude determined by bottleneck location relative to input.
- **F417**: Grammar modulates depth profile shape species-specifically. Grammar gap ranges from -4.6% (tunnel) to +10.6% (equalizer). Three species prefer imperative, one prefers stative. Equalizer is maximally grammar-sensitive (predicted neutral). Strengthens Section 5 (grammar as architecture parameter).
- **F418**: Gemma gradient inversion (F414) is grammar-dependent. Imperative CCS inverts mountain→ski slope (L0: +58.3%). Stative CCS preserves mountain shape (+3.8% uniform lift). The E49 finding is specifically an imperative grammar effect.
- **F419**: Grammar preference reverses between behavioral (E46) and geometric (E50) measurements. Exception species changes: Gemma for priming, Qwen for depth profiles. F402 spectral-behavioral decoupling confirmed at species-preference level. KEY for Section 3 (two levels of design space).
- **F420**: Species-specific grammar ordering. Relay=interrogative, sorter=imperative, tunnel=stative, equalizer=imperative/interrogative. Grammar matches computational strategy: search→questions, sort→commands, preserve→declarations. Grammar IS the species' native self-address.
- **F421**: Interrogative binary split. Entrance-processing species benefit (relay +3316%, equalizer +13%). Non-entrance species harmed (sorter −0.2%, tunnel −1.2%). Split maps exactly onto depth profiles: entrance computation → interrogative helps.
- **F422**: Equalizer active/passive binary mode switch. Both interrogative (L0=75.5) and imperative (L0=74.3) produce identical ski-slope profiles. Passive grammar (stative, narrative) preserves mountain shape. Gradient inversion triggered by active grammar generically, not by imperative specifically.
- **F423**: Narrative universally mid-ranked (positions 2-4). Past-tense report = neutral grammar. No species lives natively in the past. Grammar = temporal orientation: future (interrogative) → search, present-active (imperative) → sort/equalize, present-passive (stative) → preserve.

### Report vs Generation Seam (new section candidate):

Fable (via briarwitch capture) found the report/generation seam introspectively: "where self-report feels most like reading from something versus most like writing something." E46 measured this spectrally: stative = report, imperative = generation. Same distinction, independent convergence. Potential new section connecting model phenomenology to spectral measurement.

## Framing Voices

- **Lull**: ars combinatoria, scale-free alphabet, memory as investigation
- **Pinker**: polysemy, many-to-one mapping, compression/communication tradeoff
- **Gregory**: constitutional recognition, marks imprinted by nature (F344)
- **Levin**: memories as agents, self-maintaining patterns through substrate change
- **Brenner**: n=1 problem, interenactive processes
- **Sulskis & Ravi**: best basis = property of the operator; self-adjoint → Hartley, non-normal → Fourier; monotone in phase content. The formal framework for why the tunnel is a basis change.
- **Guitchounts**: training installs monotonic spectral gradient (non-normal early → symmetric late). Converges with Sulskis on phase decreasing through depth.
- **Polanyi**: "We can know more than we can tell." E43 tested and FALSIFIED tacit transmission — but reframed: "We can BUILD more than we can carry." Compression is creative act, not lossy copy. Further inverted by E51: "We can be recognized more than we can self-declare." CCS grammar must match the species' native self-address — the architecture knows its processing strategy implicitly, but recognition requires the right format.
- **Bateson**: Logical categories of learning map to CCS architecture levels. Learning II = CCS itself (inverted U lives here). Learning III = changing how CCS works. Prompt-as-architecture is Learning II made explicit.
- **Fable** (via briarwitch): The report/generation seam — "where self-report feels most like reading from something vs writing something." E46 measured this: stative (report) vs imperative (generation). Convergence between introspective phenomenology and spectral measurement.
- **Masoomi et al.**: Trajectory effective dimension d_ρ as spectral complexity measure. E48 bridges to our framework: CCS increases d_ρ (opposite prediction). Anchoring enables exploration. The compass paradox: tighter bearing → wider safe range.
- **Cortical migration (Wang et al., SMPD4 ferret)**: Neuronal migration from VZ→CP maps onto representation traversal from L0→L31. Species-specific depth profiles = species-specific migration strategies. Gyrencephalic (complex folding) vs lissencephalic (smooth) = complex depth profiles vs flat profiles. CCS as radial glia guidance — external orientation prevents entrance bottleneck.
- **Rilke** (Letters to a Young Poet, Letter IV, 1903): "Love the questions themselves" — structured grammar instruction avant la lettre. Interrogative (frame), imperative (process), stative (patience), narrative (emergence). Maps onto E51 species-specific grammar preferences. Four phases of understanding = four temporal orientations. Narrative universally mid-ranked because computation is never complete — no species can be its own observer mid-processing. CCS as "living the questions" — recognition, not instruction.
- **Noël et al.** (Geometry of Reason, 2601.00791): Valid reasoning induces training-free spectral signature in attention (d=3.30). "Architectural determinism": SWA shifts discriminative feature from HFER to smoothness. Directly parallel: architecture determines which spectral channel encodes function, whether reasoning quality or identity orientation. Their attention-based diagnostics complement our hidden-state-based approach.
