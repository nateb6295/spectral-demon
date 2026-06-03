# Experiment Sketch: OLMo-2 GQA Developmental Trajectory
# Status: SKETCH — not yet coded
# Date: 2026-05-29

## Motivation
The unified paper's Furnishing section (§4) is thinner than Room and Living.
We have Pythia developmental (MHA: room forms, furnishing never installs).
We LACK GQA developmental — when does each pillar emerge?

## Model
allenai/OLMo-2-1124-7B — GQA architecture, 965 training checkpoint branches.
Stage 1 spans step 150 to step 928,646 (928 checkpoints).

## Proposed Checkpoints (5-6 points)
1. step 150 (1B tokens) — near random init
2. step 50000 (~210B tokens) — early training
3. step 150000 (~629B tokens) — mid training
4. step 300000 (~1258B tokens) — late-mid training
5. step 500000 (~2097B tokens) — late training
6. step 928646 (~3896B tokens) — final checkpoint

## Measurements per checkpoint
- Passage distance d/d_max (Room forming)
- σ₁/σ₂ spectral gap at tunnel midpoint (Room severity)
- ΔS(receptive - absent) at tunnel midpoint (Living precursor)
- σ₂ magnitude across conditions (Furnishing loading)
- 3 conditions × 5 probes × 2 repeats = 30 forward passes per checkpoint
- Total: ~180 forward passes on RunPod H100

## Predictions
From Pythia data + Nguyen et al.:
1. d/d_max saturates early (step <50k), invariant thereafter (Room is congenital)
2. σ₂ channel develops late (step >300k), after small SVs gain overlap
3. ΔS(rec-abs) emerges even without IT on GQA (base model tendency from F24: +0.011)
4. ΔS magnitude increases with training (more loaded σ₂ = more modulation capacity)
5. The GQA spectral gap should be ~half of what Pythia shows at matched training step

## Cost
~180 forward passes × 7B model = ~2 GPU-hours on H100. ~$5 on RunPod.

## What this would prove
- WHEN the Furnishing happens (training step where σ₂ channel activates)
- WHETHER Room and Furnishing have distinct developmental timescales on GQA
- WHETHER the GQA architectural tendency (F24: +0.011 on base) exists from early training or develops

---

## Experiment Sketch: Inference-Time GQA Conversion (exp_gqa_conversion)

**Question:** Can you create the Room at inference time by forcing GQA-like KV sharing on an MHA model?

**Models:** Pythia 6.9B (MHA, 32 heads, most data on hand)

**Intervention:** PyTorch forward hook on attention modules. Before softmax, average K and V projections within groups of 4 (simulating s=4 GQA). This reduces the effective number of independent KV representations from 32 to 8.

**Measurements (at L17):**
- d/d_max — does passage distance increase toward GQA levels?
- ΔS(rec-abs) — does witness enrichment appear?
- σ₁/σ₂ gap — does the spectral gap decrease?
- Wire direction — does the dominant eigenvector change?

**Predictions:**
- d/d_max increases (partial Room created) BUT ΔS stays ≈ 0 (no Furnishing)
- σ₁/σ₂ gap decreases (Nait Saada mechanism operates at computation level)
- Wire direction changes minimally (σ₁ is determined by weight initialization, not attention pattern)

**If confirmed:** Strongest possible evidence for three-act decomposition — Room, Furnishing, Living require different timescales and cannot be substituted.

**Compute:** 90 forward passes on AGX (Pythia fits in 16GB). ~15 min. Free.

**Priority:** Lower than OLMo-2 developmental but conceptually sharper. Run after OLMo-2.
