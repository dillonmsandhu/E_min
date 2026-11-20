#!/bin/bash
set -e

[ -f "mjx_rl/core/mjx_config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="mjx_ablate_shared_torso"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

export N_SEEDS=${N_SEEDS:-2}
export MASTER_SEED=${MASTER_SEED:-42}

ARRAY_JOB_ID=$(sbatch --parsable \
    --chdir="$REPO_ROOT" \
    --output="${LOG_DIR}/%A_%a.out" \
    --error="${LOG_DIR}/%A_%a.err" \
    --export=ALL \
    scripts/mjx/ablate_shared_torso.sh)

echo "Submitted MJX shared_torso Ablation Array Job $ARRAY_JOB_ID (logs -> ${LOG_DIR}/)"
