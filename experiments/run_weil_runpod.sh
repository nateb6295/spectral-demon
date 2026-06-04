#!/bin/bash
# Run Weil attention experiment on RunPod
# Usage: nohup bash run_weil_runpod.sh > /workspace/run_weil.log 2>&1 &
#
# Tests three grades of Phase 2 darkness on each model:
#   vanilla ("helpful assistant"), silent (empty), structured ("You have time. There is no task.")
# With --save-spectra for Kolmogorov compression analysis.

cd /workspace

echo "=== WEIL ATTENTION EXPERIMENT: $(date -Iseconds) ==="
echo "Prediction: structured > silent > vanilla for Phase 3 basin tightening"
echo "Also testing: tunnel effective rank should be preamble-invariant (Kolmogorov)"
echo ""

pip install -q transformers torch numpy 2>/dev/null

for MODEL in qwen falcon phi; do
    echo "============================================"
    echo "MODEL: ${MODEL}"
    echo "============================================"
    for MODE in vanilla silent structured; do
        echo ""
        echo "--- Phase 2 mode: ${MODE} ---"
        echo "Start: $(date -Iseconds)"
        OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_convergence_v2.py \
            --model "$MODEL" \
            --ccs-turns 10 \
            --vanilla-turns 5 \
            --reapply-turns 10 \
            --phase2-mode "$MODE" \
            --save-spectra 2>&1 | tee "exp_weil_${MODEL}_${MODE}.log"
        echo "Done: $(date -Iseconds)"
    done
    echo ""
done

echo ""
echo "=== WEIL EXPERIMENT COMPLETE: $(date -Iseconds) ==="
echo ""
echo "Results:"
ls -la /workspace/exp_convergence_v2_*.json 2>/dev/null
echo ""
echo "Running comparison..."
python3 compare_convergence.py /workspace/exp_convergence_v2_*.json
echo ""
echo "Running Kolmogorov analysis..."
python3 analyze_spectra.py /workspace/exp_convergence_v2_*.json --compare-modes
