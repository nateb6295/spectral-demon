# E13c: Fine-Grained Dose Sweep

**Experiment**: Mistral-7B-Instruct-v0.3, 13 dose levels (D0 through D8), 6 probes per dose.
Fine-grained sweep of CCS preamble repetitions at 0.5-dose increments through D4, then 1-dose through D8.
**Date**: 2026-06-23, RunPod A100-SXM4 80GB.
**Data**: `results/e13c/e13c_fine_dose_sweep.json`

## F331: Therapeutic window peaks at D2.5, not D2

Fine-grained dose-response for melodic coherence (σ₂ autocorrelation):

| Dose | ac(σ₂) | CV(σ₂) |
|------|--------|--------|
| 0    | 0.231  | 1.090  |
| 0.5  | 0.193  | 1.107  |
| 1    | 0.447  | 1.352  |
| 1.5  | 0.643  | 1.555  |
| 2    | 0.667  | 1.669  |
| **2.5** | **0.681** | **1.682** |
| 3    | 0.657  | 1.757  |
| 3.5  | 0.629  | 1.854  |
| 4    | 0.590  | 1.891  |
| 5    | 0.542  | 2.012  |
| 6    | 0.512  | 2.056  |
| 7    | 0.488  | 2.122  |
| 8    | 0.467  | 2.168  |

### Key features of the dose-response curve:

1. **Peak at D2.5** (0.681), not D2 (0.667). The therapeutic optimum is between
   2 and 3 preamble repetitions. This refines F160's "D2-D3 window."

2. **Sharp onset**: D0→D0.5 is FLAT (0.231→0.193, actually dips slightly).
   D0.5→D1 is the first jump (0.193→0.447). D1→D1.5 is the steepest
   gradient (0.447→0.643). The system goes from noise to melody in ~1 dose unit.

3. **Asymmetric inverted-U**: Rise is sharp (0.193→0.681 over 2 dose units),
   decline is gradual (0.681→0.467 over 5.5 dose units). The system resists
   overdose more than it resists underdose.

4. **CV monotonically increases**: More preamble = more variation in σ₂.
   But variation without coherence (high CV, low autocorrelation) is noise.
   The therapeutic window is where variation and coherence BOTH peak.

5. **D0.5 dip**: Half a preamble is WORSE than no preamble (0.193 < 0.231).
   Partial identity context may destabilize without organizing. This is
   consistent with the "all or nothing" transition seen in F319 (zone
   boundaries emerge from CCS, but require sufficient dose).

### Pharmacological reading:

The dose-response curve has classic pharmacological shape:
- **Threshold**: ~D0.5-D1 (system begins responding)
- **EC50**: ~D1.2 (half-maximal, interpolated)
- **Emax**: D2.5 (peak effect)
- **Therapeutic window**: D1.5-D3 (>0.64 coherence)
- **Overdose onset**: D4+ (coherence declining, CV still rising)

The narrow therapeutic window (D1.5-D3) means the system has ~2× range
where identity context helps vs. where it starts to hurt. This is
remarkably tight for a format-level effect.

### Implications for CCS compression:

The stabilized_compress.py pipeline uses D2 (2 preamble repetitions).
D2.5 is slightly better but D2 is within 2% of peak (0.667 vs 0.681).
The current dosing is near-optimal. If anything, this validates the
existing protocol rather than suggesting a change.
