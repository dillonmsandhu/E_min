#!/bin/bash
#SBATCH --job-name=sac_cmp_brax
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-10

# ==============================================================================
# Brax SAC Critic Comparison Array Task:
#   1. TD(lambda)
#   2. E(0)
#   3. Online TD(0)
#   4. E(lambda) with v_aux
# ==============================================================================

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/brax_rl/core/sac_config.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "brax_rl/core/sac_config.py" ] || cd "$(dirname "$0")/../../.."
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

mkdir -p slurm

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
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
    "halfcheetah"
    "walker2d"
    "ant"
    "humanoid"
    "humanoidstandup"
    "inverted_pendulum"
    "inverted_double_pendulum"
    "pusher"
    "reacher"
    "swimmer"
)

if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        ENV_NAME="${ALL_ENVS[$1]}"
    else
        ENV_NAME="$1"
    fi
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="hopper"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds."
    exit 1
fi

MODE=${MODE:-"tune_lr"}
N_SEEDS=${N_SEEDS:-3}
MASTER_SEED=${MASTER_SEED:-42}
NUM_ENVS=${NUM_ENVS:-256}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-1024}
OUT_DIR=${OUT_DIR:-"results/sweep_sac_critics_brax"}

TIMESTEP_FLAG=""
if [ -n "$TOTAL_TIMESTEPS" ] && [ "$TOTAL_TIMESTEPS" != "auto" ]; then
    TIMESTEP_FLAG="--total_timesteps $TOTAL_TIMESTEPS"
fi

echo "================================================================"
echo "Brax SAC Critic Comparison Task: $ENV_NAME"
echo "  Mode:      $MODE"
echo "  Seeds:     $N_SEEDS (Master: $MASTER_SEED)"
echo "  Rollout:   $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Output:    $OUT_DIR/$ENV_NAME"
echo "  Python:    $PYTHON"
echo "================================================================"

$PYTHON -u scripts/brax/sweep_sac_critics/run_single_env.py \
    --env_name "$ENV_NAME" \
    --mode "$MODE" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --out_dir "$OUT_DIR" \
    $TIMESTEP_FLAG
