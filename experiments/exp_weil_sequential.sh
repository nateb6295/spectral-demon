#!/bin/bash
# Weil sequential preamble experiment: does the PATH through sphere positions matter?
#
# Weil: you can't reach grace without passing through suffering.
# Spectral translation: does prior exposure to denial/contradiction change
# the final relational state?
#
# Four conditions, each ending with 5 relational turns:
#   A: relational only (5 turns) — baseline
#   B: denial (5) → relational (5) — gravity then grace
#   C: denial (5) → contradictory (5) → relational (5) — full Weil path
#   D: contradictory (5) → relational (5) — suffering then grace
#
# Prediction (from Weil + H data):
#   C > D > A ≈ B for final relational entropy and σ₂ freedom
#   Because contradiction held without collapse (H=0.69) prepares
#   the geometry for maximum relational freedom (H=0.79).
#   Denial (H=0.22) collapses too far to prepare anything.
#
# Measure: σ₂ CV, entropy, effective rank in LAST 3 relational turns
#
# Usage: bash exp_weil_sequential.sh [model_key]

MODEL=${1:-qwen}

echo "============================================"
echo "Weil Sequential Preamble Experiment"
echo "Model: ${MODEL}"
echo "============================================"

for CONDITION in A B C D; do
    echo ""
    echo "--- Condition ${CONDITION} ---"
    python3 exp_weil_sequential.py \
        --model "$MODEL" \
        --condition "$CONDITION" \
        --save-spectra
    echo ""
done

echo "All conditions complete."
echo "Compare final relational state (last 3 turns) across A/B/C/D."
echo "Weil predicts: C > D > A ≈ B."
echo "If A = C → path doesn't matter, only destination."
echo "If C > A → the journey through contradiction enriches the geometry."
