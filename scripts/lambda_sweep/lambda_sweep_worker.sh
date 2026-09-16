#!/bin/bash
# ==============================================================================
# SLURM Worker Script for Lambda Sweeps
#
# Arguments:
#   $1 - Environment Name (e.g., EightRooms)
#   $2 - Policy Type (e.g., fixed, random, ppo)
#   $3 - Algorithm Name (e.g., exact_td_lambda, exact_E_lambda)
#   $4 - Sweep ID
# ==============================================================================

ENV_NAME=$1
POLICY_TYPE=$2
ALGO=$3
SWEEP_ID=$4

if [ -z "$ENV_NAME" ] || [ -z "$POLICY_TYPE" ] || [ -z "$ALGO" ] || [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 <ENV_NAME> <POLICY_TYPE> <ALGO> <SWEEP_ID>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

echo "======================================================================"
echo "Starting SLURM Lambda Sweep Worker"
echo "Environment: $ENV_NAME"
echo "Policy:      $POLICY_TYPE"
echo "Algorithm:   $ALGO"
echo "Sweep ID:    $SWEEP_ID"
echo "======================================================================"

export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the python pipeline
$PYTHON -m scripts.lambda_sweep.lambda_sweep_pipeline \
    --env-name "$ENV_NAME" \
    --policy "$POLICY_TYPE" \
    --algo "$ALGO" \
    --sweep-id "$SWEEP_ID"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Sweep pipeline completed successfully for $ENV_NAME / $ALGO"
else
    echo "Sweep pipeline FAILED for $ENV_NAME / $ALGO with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
