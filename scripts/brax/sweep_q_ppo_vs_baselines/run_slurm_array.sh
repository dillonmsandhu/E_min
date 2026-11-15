#!/bin/bash
#SBATCH --job-name=qppo_cmp_brax
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-10

# ==============================================================================
# Brax Comparison & Hyperparameter Tuning Array Task:
#   1. TD(lambda) (PPO Fitted)
#   2. Q-PPO E(0) (Bilinear + Hybrid GAE)
#   3. E(lambda) (v_aux PPO)
# ==============================================================================

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "brax_rl/core/q_ppo_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm

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

MODE=${MODE:-"tune_coordinate"}
N_SEEDS=${N_SEEDS:-3}
MASTER_SEED=${MASTER_SEED:-42}
NUM_ENVS=${NUM_ENVS:-512}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-1024}
OUT_DIR=${OUT_DIR:-"results/sweep_q_ppo_vs_baselines_brax"}

TIMESTEP_FLAG=""
if [ -n "$TOTAL_TIMESTEPS" ] && [ "$TOTAL_TIMESTEPS" != "auto" ]; then
    TIMESTEP_FLAG="--total_timesteps $TOTAL_TIMESTEPS"
fi

echo "================================================================"
echo "Brax Q-PPO vs Baselines Tuning Task: $ENV_NAME"
echo "  Mode:      $MODE"
echo "  Seeds:     $N_SEEDS (Master: $MASTER_SEED)"
echo "  Rollout:   $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Output:    $OUT_DIR/$ENV_NAME"
echo "  Python:    $PYTHON"
echo "================================================================"

$PYTHON -u scripts/brax/sweep_q_ppo_vs_baselines/run_single_env.py \
    --env_name "$ENV_NAME" \
    --mode "$MODE" \
    $TIMESTEP_FLAG \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --out_dir "$OUT_DIR"
