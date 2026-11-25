#!/bin/bash
#SBATCH --job-name=octax_td_diag
#SBATCH --output=slurm/%j_octax_td_diag.out
#SBATCH --error=slurm/%j_octax_td_diag.err
#SBATCH --time=02:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/octax_rl/core/octax_config.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "octax_rl/core/octax_config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

ENV_NAME=${1:-"Octax-blinky"}
TOTAL_TIMESTEPS=${2:-2000000}
N_SEEDS=${3:-3}

$PYTHON -u scripts/octax/run_td_contraction_single_env.py \
    --env_name "$ENV_NAME" \
    --lambdas 0.0 0.5 0.8 0.95 \
    --include_e_lambda 0.8 \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --n_seeds "$N_SEEDS"
