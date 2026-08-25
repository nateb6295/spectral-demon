#!/bin/bash
# Pod runner: E22b (pooled basis) → E22d (cylindrical) → E25 (contextual robustness)
# 4h pod session. Sequential per model to avoid OOM.
# Usage: bash pod_run_e22d_e25.sh [model_key|all]
#
# DEPENDENCY: E22b and E22c import from e22_mlp_pathway_alignment.py — must be co-located.
# Upload entire experiments/ directory to pod.

set -e
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export E22_RESULTS_DIR="/workspace/e22b_results"

MODEL=${1:-mistral}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p /workspace/e22b_results /workspace/e22d_results /workspace/e25_results

echo "=== Pod session: $MODEL ==="
echo "Start: $(date)"
echo ""

# E22b: Pooled basis (~5 min per model)
echo ">>> E22b: Pooled Basis ($MODEL)"
python3 "$SCRIPT_DIR/e22b_pooled_basis.py" "$MODEL"
echo ""
echo "E22b done: $(date)"

# E22d: Cylindrical decomposition (~5 min per model)
echo ">>> E22d: Cylindrical Decomposition ($MODEL)"
python3 "$SCRIPT_DIR/e22d_cylindrical_decomposition.py" "$MODEL"
echo ""
echo "E22d done: $(date)"

# E25: Contextual robustness (~10 min per model)
echo ">>> E25: Contextual Robustness ($MODEL)"
python3 "$SCRIPT_DIR/e25_contextual_robustness.py" "$MODEL"
echo ""
echo "E25 done: $(date)"

echo "=== All done: $(date) ==="
