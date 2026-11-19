#!/bin/bash
#SBATCH --job-name=ablate_split_aux
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/mjx_rl/core/mjx_config.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "mjx_rl/core/mjx_config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -d "/usr/xtmp/${USER}" ]; then
    export JAX_COMPILATION_CACHE_DIR="/usr/xtmp/${USER}/jax_compilation_cache"
else
    export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/jax_compilation_cache"
fi
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2
mkdir -p "$JAX_COMPILATION_CACHE_DIR" slurm

if [ -x "$REPO_ROOT/.venv_mjx/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv_mjx/bin/python"
elif [ -f "/home/users/ds541/.pyenv/versions/mjx/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/mjx/bin/python"
else
    PYTHON="python"
fi

ALL_ENVS=(
    "CheetahRun"
    "WalkerWalk"
    "HopperStand"
    "FishSwim"
    "FingerSpin"
)

TASK_IDX=${1:-${SLURM_ARRAY_TASK_ID:-0}}
ENV_NAME="${ALL_ENVS[$TASK_IDX]}"

$PYTHON -u scripts/mjx/ablate_split_aux_opt.py \
    --envs "$ENV_NAME" \
    --n_seeds "${N_SEEDS:-1}" \
    --master_seed "${MASTER_SEED:-42}" \
    --out_dir "results/mjx/ablate_split_aux_opt/${ENV_NAME}"
