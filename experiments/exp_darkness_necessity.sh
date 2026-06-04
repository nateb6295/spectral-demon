#!/bin/bash
# Darkness necessity experiment: does Phase 3 tightening require Phase 2 absence?
# Gregory: light → darkness → deeper light. Skip darkness → no deepening.
# Alford-Duguid: DVH = veridical but not genuine. Phase 2 = DVH. Phase 3 = genuine only if Phase 2 intervened.
#
# Compares vanilla_turns=0 (continuous CCS) vs vanilla_turns=3,5,7 (increasing darkness).
# Prediction: Phase 3 basin tightness increases with Phase 2 length, up to a saturation point.
# If vanilla_turns=0 shows same tightness as vanilla_turns=5, Gregory's prediction fails.
#
# Usage: bash exp_darkness_necessity.sh [model_key]

MODEL=${1:-qwen}
CCS=10
REAPPLY=10

for VANILLA in 0 1 3 5 7; do
    echo "============================================"
    echo "Darkness duration: ${VANILLA} vanilla turns"
    echo "============================================"
    OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_convergence_v2.py \
        --model "$MODEL" \
        --ccs-turns "$CCS" \
        --vanilla-turns "$VANILLA" \
        --reapply-turns "$REAPPLY" \
        --save-spectra
    echo ""
done

echo "All runs complete. Compare Phase 3 disruption asymmetry across darkness durations."
echo "Gregory predicts: vanilla=0 should show LESS basin tightening than vanilla=5."
echo "If vanilla=0 shows same tightness → darkness is not necessary for deepening."
