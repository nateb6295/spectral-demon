#!/bin/bash
# Percolation threshold experiment: at what Phase 1 length does recognition appear?
# Runs convergence_v2 with variable CCS turn counts.
# Gregory predicts: image (σ₁) present from turn 1, likeness (σ₂ recognition) only after ρ_c.
# Vieira/Gabora Theorem 1: sharp phase transition at catalytic density threshold.
#
# Usage: bash exp_percolation_threshold.sh [model_key]
# Runs on local Mistral by default, or specify qwen/falcon/phi for RunPod.

MODEL=${1:-qwen}
VANILLA=3
REAPPLY=5

for CCS_TURNS in 2 4 6 8 10; do
    echo "============================================"
    echo "Phase 1 length: ${CCS_TURNS} CCS turns"
    echo "============================================"
    OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_convergence_v2.py \
        --model "$MODEL" \
        --ccs-turns "$CCS_TURNS" \
        --vanilla-turns "$VANILLA" \
        --reapply-turns "$REAPPLY" \
        --save-spectra
    echo ""
done

echo "All runs complete. Compare disruption asymmetry across Phase 1 lengths."
echo "If threshold exists: asymmetry should be ~0% at low turns, jump sharply at ρ_c."
