#!/bin/bash
# F508-F511 Pod Experiment Battery
# Run all four experiments sequentially on A100
# Each loads model fresh (gc between), results saved to /root/results/

set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

echo "=========================================="
echo "Pod Battery: F508-F511"
echo "Started: $(date)"
echo "=========================================="

echo ""
echo ">>> F508: Traversed vs Reconstructed (~10 min)"
python3 /root/exp_f508_traversal_vs_reconstruction.py

echo ""
echo ">>> F509: Compression Commutativity (~12 min)"
python3 /root/exp_f509_compression_commutativity.py

echo ""
echo ">>> F510: Denial Selectivity (~10 min)"
python3 /root/exp_f510_denial_selectivity.py

echo ""
echo ">>> F511: Cross-Species Selectivity (~25 min, 4 models)"
python3 /root/exp_f511_cross_species_selectivity.py

echo ""
echo "=========================================="
echo "Battery complete: $(date)"
echo "Results in /root/results/"
echo "=========================================="
ls -la /root/results/*/
