#!/bin/bash
#SBATCH --job-name=brax_split_aux
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/brax_rl/core/brax_config.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "brax_rl/core/brax_config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -d "/usr/xtmp/${USER}" ]; then
    export JAX_COMPILATION_CACHE_DIR="/usr/xtmp/${USER}/jax_compilation_cache"
else
    export JAX_COMPILATION_CACHE_DIR="${HOME}/.cache/jax_compilation_cache"
fi
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

ALL_ENVS=(
    "hopper"
    "walker2d"
    "halfcheetah"
    "ant"
    "inverted_pendulum"
)

TASK_IDX=${1:-${SLURM_ARRAY_TASK_ID:-0}}
ENV_NAME="${ALL_ENVS[$TASK_IDX]}"

$PYTHON -u scripts/brax/ablate_split_aux_opt.py \
    --envs "$ENV_NAME" \
    --n_seeds "${N_SEEDS:-1}" \
    --master_seed "${MASTER_SEED:-42}" \
    --out_dir "results/brax/ablate_split_aux_opt/${ENV_NAME}"
