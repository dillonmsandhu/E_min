#!/bin/bash
# ==============================================================================
# E_min/scripts/submit_tune_ddpg.sh
#
# Submits GPU tuning jobs for standard DDPG TD(lambda=0.8) using
# scripts/tune_ddpg_defaults.py, writing organized logs to:
#   slurm/<DATE>/tune_ddpg_defaults/<JOBID>_<BACKEND>.out
#
# Usage:
#   bash scripts/submit_tune_ddpg.sh gymnax coordinate
#   bash scripts/submit_tune_ddpg.sh brax coordinate
# ==============================================================================

set -e

[ -f "gymnax_rl/core/ddpg_config.py" ] || cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
BACKEND=${1:-gymnax}
MODE=${2:-coordinate}
N_SEEDS=${3:-3}
OUT_DIR=${4:-"results/tune_ddpg_defaults"}
LOG_DIR="slurm/${DATE}/tune_ddpg_defaults"
mkdir -p "$LOG_DIR"

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

echo "================================================================"
echo "Submitting DDPG TD(lambda=0.8) Default Config Tuning Job"
echo "  Backend:      $BACKEND"
echo "  Mode:         $MODE"
echo "  Seeds:        $N_SEEDS"
echo "  Output Dir:   $OUT_DIR"
echo "  Log Dir:      $LOG_DIR"
echo "  Python:       $PYTHON"
echo "================================================================"

JOB_ID=$(sbatch --parsable \
    --job-name="tune_ddpg_${BACKEND}" \
    --output="${LOG_DIR}/%j_${BACKEND}.out" \
    --error="${LOG_DIR}/%j_${BACKEND}.err" \
    --time=06:00:00 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --chdir="$REPO_ROOT" \
    --wrap="export PYTHONPATH='$REPO_ROOT:\${PYTHONPATH}' XLA_PYTHON_CLIENT_PREALLOCATE=false; $PYTHON -u scripts/tune_ddpg_defaults.py --backend $BACKEND --mode $MODE --n_seeds $N_SEEDS --out_dir '$OUT_DIR'")

echo "Submitted Job ID: $JOB_ID"
echo "Logs:             ${LOG_DIR}/${JOB_ID}_${BACKEND}.out"
