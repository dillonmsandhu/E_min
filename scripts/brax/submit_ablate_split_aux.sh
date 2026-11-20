#!/bin/bash
set -e

[ -f "brax_rl/core/brax_config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="brax_ablate_split_aux"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

export N_SEEDS=${N_SEEDS:-1}
export MASTER_SEED=${MASTER_SEED:-42}

ARRAY_JOB_ID=$(sbatch --parsable \
    --chdir="$REPO_ROOT" \
    --output="${LOG_DIR}/%A_%a.out" \
    --error="${LOG_DIR}/%A_%a.err" \
    --export=ALL \
    scripts/brax/ablate_split_aux_opt.sh)

echo "Submitted Brax split_aux_opt Ablation Array Job $ARRAY_JOB_ID (logs -> ${LOG_DIR}/)"
