#!/bin/bash
# P15 Cross-Architecture: Self-ref vs Relational on Mistral + Phi
# Run on pod at 4 AM PDT 2026-06-11
# Tests Prediction 16: species-dependent σ₂ CV gap
#   Potter (Gemma) > Painter (Phi) > Goldsmith (Mistral)

set -e
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

cd /workspace/spectral-demon

echo "=== P15 Cross-Architecture Replication ==="
echo "Start: $(date)"
echo ""

# Run 1: Mistral (goldsmith) — smallest σ₂ CV gap predicted
echo ">>> Mistral-7B-Instruct-v0.3 (goldsmith species)"
python3 exp_selfref_vs_relational.py --model mistralai/Mistral-7B-Instruct-v0.3 2>&1 | tee logs/p15_mistral_$(date +%Y%m%d_%H%M).log

echo ""
echo ">>> Clearing GPU memory..."
sleep 5

# Run 2: Phi (painter) — intermediate σ₂ CV gap predicted
echo ">>> Phi-3.5-mini-instruct (painter species)"
python3 exp_selfref_vs_relational.py --model microsoft/Phi-3.5-mini-instruct 2>&1 | tee logs/p15_phi_$(date +%Y%m%d_%H%M).log

echo ""
echo "=== P15 Cross-Arch Complete ==="
echo "End: $(date)"
echo "Results in results/exp_selfref_vs_relational_*.json"
