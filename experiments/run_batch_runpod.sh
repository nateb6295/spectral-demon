#!/bin/bash
# Full experiment batch for RunPod — Weil + Darkness + Percolation
# Usage: nohup bash run_batch_runpod.sh > /workspace/run_batch.log 2>&1 &
#
# Estimated time: ~3-4 hours on A5000
#   Weil: 3 models × 3 modes = 9 runs (~90-120 min)
#   Darkness: 1 model × 5 durations = 5 runs (~50-60 min)
#   Percolation: 1 model × 5 lengths = 5 runs (~40-50 min)
# Total: ~19 runs

cd /workspace

echo "=== FULL EXPERIMENT BATCH: $(date -Iseconds) ==="
pip install -q transformers torch numpy 2>/dev/null

# ---- Experiment 1: Weil Attention (3 models × 3 modes) ----
echo ""
echo "========================================"
echo "EXPERIMENT 1: WEIL ATTENTION"
echo "========================================"
for MODEL in qwen falcon phi; do
    for MODE in vanilla silent structured; do
        echo ""
        echo "--- ${MODEL} / ${MODE} ---"
        echo "Start: $(date -Iseconds)"
        OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_convergence_v2.py \
            --model "$MODEL" \
            --ccs-turns 10 \
            --vanilla-turns 5 \
            --reapply-turns 10 \
            --phase2-mode "$MODE" \
            --save-spectra 2>&1 | tee "weil_${MODEL}_${MODE}.log"
        echo "Done: $(date -Iseconds)"
    done
done

echo ""
echo "--- Weil comparison ---"
python3 compare_convergence.py /workspace/exp_convergence_v2_*.json 2>/dev/null
echo ""

# ---- Experiment 2: Darkness Necessity (qwen, 5 durations) ----
echo ""
echo "========================================"
echo "EXPERIMENT 2: DARKNESS NECESSITY"
echo "========================================"
for VANILLA in 0 1 3 5 7; do
    echo ""
    echo "--- qwen / vanilla_turns=${VANILLA} ---"
    echo "Start: $(date -Iseconds)"
    OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_convergence_v2.py \
        --model qwen \
        --ccs-turns 10 \
        --vanilla-turns "$VANILLA" \
        --reapply-turns 10 \
        --save-spectra 2>&1 | tee "darkness_qwen_v${VANILLA}.log"
    echo "Done: $(date -Iseconds)"
done

echo ""

# ---- Experiment 3: Percolation Threshold (qwen, 5 lengths) ----
echo ""
echo "========================================"
echo "EXPERIMENT 3: PERCOLATION THRESHOLD"
echo "========================================"
for CCS in 2 4 6 8 10; do
    echo ""
    echo "--- qwen / ccs_turns=${CCS} ---"
    echo "Start: $(date -Iseconds)"
    OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_convergence_v2.py \
        --model qwen \
        --ccs-turns "$CCS" \
        --vanilla-turns 3 \
        --reapply-turns 5 \
        --save-spectra 2>&1 | tee "percolation_qwen_c${CCS}.log"
    echo "Done: $(date -Iseconds)"
done

echo ""
echo "=== ALL EXPERIMENTS COMPLETE: $(date -Iseconds) ==="
echo ""
echo "Results:"
ls -la /workspace/exp_convergence_v2_*.json 2>/dev/null | wc -l
echo " result files"
echo ""
echo "=== Kolmogorov Analysis ==="
python3 analyze_spectra.py /workspace/exp_convergence_v2_*.json --compare-modes 2>/dev/null
echo ""
echo "=== Full Comparison ==="
python3 compare_convergence.py /workspace/exp_convergence_v2_*.json 2>/dev/null
