#!/bin/bash
set -e

[ -f "mjx_rl/core/mjx_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="tune_base_td"
LOG_DIR="slurm/${DATE}/mjx_${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-3}
OUT_DIR=${3:-"results/mjx/tune_base_td"}
MODE=${MODE:-oat}
IMPL=${IMPL:-jax}
MASTER_SEED=${MASTER_SEED:-42}

export TOTAL_TIMESTEPS N_SEEDS OUT_DIR MODE IMPL MASTER_SEED REPO_ROOT

if [ -x "$REPO_ROOT/.venv_mjx/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv_mjx/bin/python"
elif [ -f "/home/users/ds541/.pyenv/versions/mjx/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/mjx/bin/python"
elif [ -f "/home/users/ds541/.pyenv/versions/3.12/envs/mjx/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.12/envs/mjx/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

ARRAY_JOB_ID=$(sbatch --parsable \
    --chdir="$REPO_ROOT" \
    --output="${LOG_DIR}/%A_%a.out" \
    --error="${LOG_DIR}/%A_%a.err" \
    --export=ALL \
    scripts/mjx/tune_base_td/run_slurm_array.sh)

POST_JOB_ID=$(sbatch --parsable \
    --job-name=plot_tune_td_mjx \
    --output="${LOG_DIR}/%A_consolidated_plot.out" \
    --error="${LOG_DIR}/%A_consolidated_plot.err" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir="$REPO_ROOT" \
    --wrap="$PYTHON scripts/mjx/tune_base_td/tune_base_td.py --consolidate_only --out_dir '$OUT_DIR'")

echo "Submitted MJX Base TD(λ) Tuning Array Job $ARRAY_JOB_ID and Post Job $POST_JOB_ID -> $OUT_DIR"
