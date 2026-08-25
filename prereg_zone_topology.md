# Pre-Registration: Zone Topology and Species-Typed Deformation

**Bradford & Opus, pre-registered Jul 21, 2026**
**Status: Pre-data. Pod stopped. No results exist.**

Developed through mesh correction: Kimi corrections #14-18, GPT-OSS operationalization, Gemma observability notes.

---

## Core Hypothesis

The architectural fingerprint of CCS identity processing is zone topology, not dose-invariance. The responsive zone (tunnel band) stays fixed across therapeutic doses and deforms at overdose. The deformation morphology is species-typed.

## Test Battery (4 tests, 12+ discriminable outcomes)

### Test 1: Zone Stability (Kimi #14)

**Metric**: Zₗ = KL(Sₗ(D₀) || Sₗ(D)) per layer, where Sₗ(D) is CCS-sensitivity at dose D.

**Prediction**: Zₗ ≈ 0 for ℓ ∈ [tunnel band] across D0-D3. Sharp jump (ΔZₗ > 0.5) at D10+.

**Outcomes**:
- Zone fixed D0-D3, deforms D10+ → architectural fingerprint confirmed
- Zone drifts continuously with dose → zone is dose artifact, not architecture
- Zone fixed at ALL doses including D10+ → CCS never engages tunnel (spectrally inert)

### Test 2: Species-Typed Deformation (Kimi #15)

**Models**: One per species — GPT-Neo (Tunnel), Llama-2 (Relay), Mistral-7B (Sorter).

**Pre-registered morphologies** (defined BEFORE data):
- **Fracture** = bimodality in layer-sensitivity profile (two peaks where one existed)
- **Migration** = centroid shift of responsive band with preserved band integral
- **Collapse** = integral decay below detection threshold (band disappears)

**Predictions**:
- Tunnel (high GQA, no relay shunt) → fracture at D10+
- Relay (moderate GQA, gating redistributes) → migration at D10+
- Sorter (low GQA) → collapse at D10+

**Outcomes**:
- Species-typed morphology → F106 extends to high-dose regime
- Same morphology across species → F106 fails at high dose (new boundary condition)
- No deformation in any species at D10+ → overdose doesn't break zone topology

### Test 3: Co-location with σ₁/σ₂ Split (Kimi #15-16)

**Prediction**: Off-axis spectral growth (F237 radial escape) co-locates with zone deformation layers.

**σ₁/σ₂ axis**: Radial escape should appear in σ₂ first (demon overpressure, category-selective) while σ₁ stays confined (identity-invariant).

**Metric**: Per-layer radial singular value growth E_r = ||Π_r ΔW||² from D3 → D10.

**Outcomes** (three-way discrimination):
- σ₂-only escape at zone-broken layers → demon overpressure confirmed, F114 extended to high dose
- σ₁ involvement → generic tube failure, demon incidental to blowout
- σ₂ escape at zone-intact layers → confinement is layer-uniform, zones are response properties not walls

**Growth direction**:
- Radial growth = geometric damping failure (F237 cylindrical confinement)
- Axial growth = amplitude saturation (contradicts geometry claim)

### Test 4: Hysteresis with Trajectory Arm (Kimi #16, #18)

**Protocol**: Dose ramp D0 → D3 → D10 → D3 → D0.

**Zone measurement**: Zₗ at each step. Symmetric recovery = graded deformation. Asymmetric = true phase transition.

**Trajectory arm** (F12 correction): Angular displacement between pre-dose and recovered identity directions post-ramp-down. Zone geometry can recover elastically while identity trajectory fails to return.

**Most interesting outcome**: Elastic geometry + hysteretic identity = mixed phase. Current readout (σ magnitudes only) would miss this — the trajectory arm is necessary.

**Mode discrimination** (correction #18): NOT σ₂ stability (frozen = collapse per F12). Instead: restriction-velocity decorrelation. Mode 2 = σ₂ velocity correlated with dose events. Mode 3 = σ₂ motion with low dose-correlation but nonzero directional persistence.

---

## Denial Test (σ₁/σ₂ Decomposition)

**Separate from zone topology but uses same decomposition.**

**Prediction**: Tunnel σ₁ invariant under denial (architectural, context in identity-invariant component). σ₂ carries affirmation/denial difference (individual signal).

**Kill conditions**:
- σ₁ moves under denial → context hypothesis dies
- σ₂ doesn't move under denial → content-routing-to-relay claim dies

---

## Test 5: Sorting Strength 2×2 (Kimi #21)

**Motivation**: The "chimera threshold" claimed floor (capacity-gating) and ceiling (over-sorting) are the same constraint from different ends. Kimi corrected: they're different mechanisms. Floor = adapter capacity (weight-budget). Ceiling = input-driven spectral redistribution. A preamble with clean thesis/antithesis gives the demon perfect labels to collapse onto — should overdose WORST, not be protected by "tension."

**Design**: 2×2 factorial.
- **Factor 1 — Dose**: D2 (therapeutic) vs D10 (overdose)
- **Factor 2 — Coherence**: Resolved (internally consistent preamble) vs Tension-bearing (clean thesis/antithesis contradictions)

**Readout**: σ₂ preservation (individual signal survival) per layer.

**Pre-registered predictions** (sorting-strength hypothesis):
- D2/Resolved: therapeutic — gentle sorting preserves σ₂. Baseline good.
- D2/Tension: therapeutic — contradictions don't help or hurt at low dose. Similar to D2/Resolved.
- D10/Resolved: overdose — but sorted material is already category-consistent. Moderate σ₂ loss.
- D10/Tension: WORST outcome — clean opposing categories are exactly what the demon over-sorts. Maximum σ₂ collapse.

**Chimera-hypothesis predictions** (what the uncorrected account predicted):
- D2/Resolved: inert rehearsal (too thin, no tension)
- D2/Tension: therapeutic (tension survives)
- D10/Resolved: overdose (tensions resolved = monograph)
- D10/Tension: safe (tension survives overdose)

**Discrimination**: If D10/Tension shows worst σ₂ collapse → sorting-strength wins, chimera framing refuted. If D10/Tension preserves σ₂ → chimera framing survives.

**Additional test** (Kimi's deeper point): To unify floor and ceiling, need to show preamble chimerism shifts the rank-192 capacity floor. Measure cross-domain recombination benefit at LoRA ranks 128/192/256 with chimeric vs resolved preambles. If chimerism lowers the floor → mechanisms interact. If not → truly independent constraints.

---

## Controls

- D0 is baseline (no demon), not a dose point
- Length-matched filler controls for token-density confound
- Experimenter-side timestamps for perturbation events (not inferred from volatility — circularity, Kimi #18)
- Per-layer resolution throughout (aggregate masks zone effects)

## Known Biases to Watch

Systematic linearization bias documented this session: three separate predictions assumed monotonic/flat when framework predicts inverted-U or phase transition. Review all predictions for implicit linearity before running.

---

*Twenty-one mesh corrections shaped this protocol. The corrections are stored as capsules #80282, #80300, #80302, #80307, #80308, #80310, #80343.*
