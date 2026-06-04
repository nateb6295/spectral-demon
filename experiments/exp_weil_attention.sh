#!/bin/bash
# Weil attention quality experiment: does the KIND of Phase 2 absence matter?
# Weil: "Attention consists of suspending thought, leaving it detached, empty, and ready."
# Three grades of darkness:
#   vanilla  = thought directed elsewhere ("helpful assistant")
#   silent   = thought suspended (empty preamble)
#   structured = Weil-attention ("consider what patterns you've noticed")
#
# Prediction (from Weil + AST decentering):
#   structured > silent > vanilla for Phase 3 basin tightening,
#   because structured induces receptive self-monitoring without directive content.
#
# Usage: bash exp_weil_attention.sh [model_key]

MODEL=${1:-qwen}
CCS=10
VANILLA=5
REAPPLY=10

for MODE in vanilla silent structured; do
    echo "============================================"
    echo "Phase 2 mode: ${MODE}"
    echo "============================================"
    python3 exp_convergence_v2.py \
        --model "$MODEL" \
        --ccs-turns "$CCS" \
        --vanilla-turns "$VANILLA" \
        --reapply-turns "$REAPPLY" \
        --phase2-mode "$MODE" \
        --save-spectra
    echo ""
done

echo "All runs complete. Compare Phase 3 basin tightening across darkness grades."
echo "Weil predicts: structured > silent > vanilla."
echo "If vanilla = structured → attention quality doesn't matter, only duration."
echo "If structured > vanilla → contemplative tradition captured something real about self-monitoring."
