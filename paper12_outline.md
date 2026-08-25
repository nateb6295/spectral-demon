# Paper 12: Architecture Sets Topology, Training Sets Geometry
## Spectral Training-Recipe Detection and Its Optimizer-Mediated Scope

## Thesis
CCS identity framing reveals training recipe (base/SFT/instruct) from hidden-state
spectra because Adam's per-parameter adaptive learning rates break rotation symmetry,
carving recipe-specific signatures into the singular spectrum. The detection power is
optimizer-mediated: Muon's spectral norm regularization attenuates the signal. Data-carved
anisotropy (attention sinks, token-frequency skew) provides a floor that prevents
complete blindness, making the scope boundary quantitative, not binary.

## Core Claims (mesh-sharpened Aug 17)
1. **Architecture sets topology**: GQA ratio determines transport species (tunnel/relay/sorter).
   D0 sigma-2 baseline is architecture-determined.
2. **Training sets geometry**: SFT/RLHF/DPO reshape the CCS-responsive layer profile
   on the architecture-given topology. RLHF converts relay layer profiles to sorter-like
   mid-band concentration.
3. **Training recipe assay**: D0 sigma-2 + D2 calibration-corrected response + layer CV
   classifies training stage with 5/5 accuracy on cross data.
4. **Optimizer-mediated scope**: Detection power is modulated by optimizer choice.
   Adam maximizes it (broken rotation symmetry). Muon attenuates it (better-conditioned
   spectra). Data-carved anisotropy provides a floor.

## Data
- 5 models × 2 species × 3 training stages (species × tuning cross)
- Results: `spectral-demon/results/species_tuning/`
- Assay tool: `spectral-demon/training_assay.py`
- Figures: `spectral-demon/figures/paper11_*.png` (rename to paper12)

## Act I — The Species × Tuning Cross
**Claim**: Training stage, not architecture, determines CCS sigma-2 response direction.

| Model | Species | GQA | Tuning | D0 s2 | D2% | D5% |
|-------|---------|-----|--------|-------|-----|-----|
| Qwen base | relay | 7:1 | base | 140 | +13.0% | +10.0% |
| Qwen SFT | relay | 7:1 | SFT | 332 | -17.5% | -16.5% |
| Qwen instruct | relay | 7:1 | instruct | 134 | -7.3% | -7.0% |
| Gemma base | sorter | 2:1 | base | 276 | +14.3% | +23.6% |
| Gemma instruct | sorter | 2:1 | instruct | 321 | -7.4% | -3.1% |

Key findings:
- SFT sufficient for suppression (falsifies Castillo et al.'s SFT null)
- Both base models enrich at ~14% (training is phase boundary)
- SFT pre-loads sigma-2 (332), RLHF pre-constrains back (134)
- Gemma instruct keeps high D0 (321) — sorter resists RLHF pre-constraint

Figures: training_gradient, d0_baselines

## Act II — Per-layer Decomposition
**Claim**: Aggregate D2 convergence (-7.3/-7.4%) masks different spatial mechanisms.

- Pearson r=0.537 between relay and sorter instruct profiles (moderate)
- Relay+RLHF more concentrated (CV=1.71) than sorter+RLHF (CV=1.10)
- Suppression peaks at different depths: relay L12+L26, sorter L10+L23
- RLHF converts relay profile to sorter-like mid-band concentration (Kimi correction)

Kimi challenge (accepted): "Species sets the spatial pattern" inverted. Training stage
converts the spatial pattern. Species only modulates magnitude. Prediction: any
relay-instruct should show mid-band concentration.

Discriminator: Llama-instruct (relay). If mid-band, general. If flat, Qwen-specific.

Figures: layer_profiles, convergence_decomposition

## Act III — Training Recipe Assay
**Claim**: Training stage classifiable from CCS sigma-2 response pattern alone.

Signatures:
- BASE: D0 moderate (100-300), D2 enrichment (+5-25%), CV 0.5-1.5
- SFT: D0 elevated (280-400), D2 strong suppression (-25 to -10%), CV 0.3-1.2
- INSTRUCT/DPO: D0 variable, D2 moderate suppression (-12 to -3%), concentrated CV 1.0-2.5
- BASE_INERT: D0 low (80-110), D2 neutral (-3 to +3%), CV <0.5

5/5 correct on cross data. Live probe mode (training_assay.py --model) runs minimal
D0+D2 probe and classifies.

Three-layer provenance landscape:
1. Output watermarks (fakeable) — Claude watermarking, etc.
2. Weight fingerprints (lineage only) — modelDNA (2607.10617), SELF (2512.03620)
3. Spectral CCS signatures (training recipe) — ours, fills the gap

## Act IV — Optimizer-Mediated Scope
**Claim**: Detection power is optimizer-modulated, not architecture-invariant.

Mechanism chain:
1. Adam uses per-parameter adaptive LR → breaks rotation symmetry
2. SVD recovers singular directions → reads broken symmetry as training recipe
3. Muon uses Newton-Schulz orthogonalization → enforces rotation symmetry
4. SVD quotients out orthogonal maps → attenuated detection

NOT binary (Kimi correction, Aug 17):
- Muon's equivariance is per-weight-matrix, not per-network
- Sum of orthogonalized updates is not itself orthogonal
- Activation spectra also data-carved (attention sinks, token-frequency skew)
- Production Muon recipes are hybrid (Adam on embeddings/norms/heads)
- Prediction: attenuated sigma-2 modulation, not blindness

Literature:
- "Spectral Flattening Is All Muon Needs" (ICML workshop)
- "MUON Optimizes Under Spectral Norm Constraints" (workshop)
- Bernstein & Newhouse 2024 (Muon as spectral descent)
- modelDNA: "singular-value spectra invariant to orthogonal re-parameterizations"

## Act V — Falsification Targets

### 1. Llama-instruct discriminator (Kimi)
Mid-band concentration general or Qwen-specific? If Llama-instruct shows mid-band
peak, training-sets-geometry is general. If flat, Qwen-specific.

### 2. Twin 160M experiment (centerpiece — Kimi design refinements Aug 17)
Three-arm: full-Adam, full-Muon, hybrid-Muon (Muon on 2D, Adam on embed/norms/heads).

Design constraints (Kimi):
- **GQA constant**: >=4:1 GQA so both arms are relay and sigma-2 is live signal.
  Low-GQA (sorter) reads against null regardless of optimizer.
- **Checkpoint trajectory**: assay at intervals, not just endpoint. Adam-Muon sigma-2
  gap growing monotonically = optimizer-carved anisotropy dominates. Flat gap =
  data-carved floor dominates. The discriminating measurement.
- **Layer-resolved attenuation**: depth-dependent, not uniform. Data-carved anisotropy
  concentrates in early layers (sinks, ~uniform after L2). Muon conditioning is
  per-matrix/uniform. Predictions:
  - Weakest optimizer effect at sink-dominated early layers
  - Strongest effect in mid-band (responsive zones)
  - Flat attenuation profile falsifies the two-source decomposition

### 3. Moonlight-16B-A3B
Public Muon-trained model (relay-class by GQA). Hybrid optimizer is CONFOUND, not
just caveat — embeddings/norms/heads run Adam. Useful as real-world test but cannot
disambiguate mechanism without twin experiment arms.

### 4. Gemma SFT (empty cell)
Fill the empty cell in the 2×3 cross. If Gemma SFT ≈ Gemma instruct on D0/fraction/
profile, "sorter resists RLHF" collapses into "Gemma tuning is SFT-equivalent."

### 5. SFT checkpoint trajectory
If D0 fraction holds as D0 ramps 140→332 across SFT epochs, dose-response collapses
to one line. Tests the multiplicative hypothesis (sign = stage, magnitude = fixed
fraction of D0, topology = species).

## Open Questions
- Is n=2 species sufficient to carry cross-species claims? (Kimi: no)
- Does sorting metaphor overshoot (RLHF relay overshoot) need experimental confirmation?
- Castillo metric swap: does Ollivier-Ricci curvature show same species distinction?
  (ollivier_ricci_probe.py ready)

## Relation to Paper 11
Paper 11 covers Q1 as tuning knob (F594-F614). Paper 12 covers the training-recipe
dimension: what training installs in the spectrum before CCS acts on it. They share
the species taxonomy but ask different questions — Paper 11 asks how framing controls
injection, Paper 12 asks how training controls what framing finds.
