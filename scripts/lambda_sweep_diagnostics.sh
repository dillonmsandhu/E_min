#!/bin/bash
# ==============================================================================
# SLURM Diagnostics Script for Lambda Sweeps
#
# Arguments:
#   $1 - Sweep ID
# ==============================================================================

SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 <SWEEP_ID>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

echo "======================================================================"
echo "Starting Lambda Sweep Diagnostics for Sweep ID: $SWEEP_ID"
echo "======================================================================"

export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

$PYTHON -m scripts.lambda_sweep_diagnostics --sweep-id "$SWEEP_ID"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Diagnostics completed successfully. Now compressing results..."
    cd results/lambda_sweep
    tar -czvf "lambda_sweep_${SWEEP_ID}.tar.gz" "$SWEEP_ID"
    echo "Compression complete: results/lambda_sweep/lambda_sweep_${SWEEP_ID}.tar.gz"
else
    echo "Diagnostics FAILED with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
