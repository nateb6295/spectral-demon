#!/bin/bash
# Run all experiments sequentially on RunPod
# Usage: nohup bash run_all_runpod.sh > /workspace/run_all.log 2>&1 &

cd /workspace

echo "=== EXPERIMENT BATCH START: $(date -Iseconds) ==="
echo ""

# Experiment 1: L18 Perturbation (skip if results already exist)
if [ -f /workspace/exp_l18_perturbation_results.json ]; then
    echo ">>> L18 perturbation: ALREADY COMPLETE, skipping"
else
    echo ">>> Starting L18 perturbation: $(date -Iseconds)"
    python3 exp_l18_perturbation.py 2>&1 | tee exp_l18_output.log
    echo ">>> L18 perturbation DONE: $(date -Iseconds)"
fi
echo ""

# Experiment 2: Counted Contradictions
echo ">>> Starting counted contradictions: $(date -Iseconds)"
python3 exp_counted_contradictions.py 2>&1 | tee exp_counted_contradictions_output.log
echo ">>> Counted contradictions DONE: $(date -Iseconds)"
echo ""

# Experiment 3: Variance Ratio
echo ">>> Starting variance ratio: $(date -Iseconds)"
python3 exp_variance_ratio.py 2>&1 | tee exp_variance_ratio_output.log
echo ">>> Variance ratio DONE: $(date -Iseconds)"
echo ""

echo "=== ALL EXPERIMENTS COMPLETE: $(date -Iseconds) ==="
echo "Results:"
ls -la /workspace/exp_*results*.json /workspace/exp_*_2*.json 2>/dev/null
