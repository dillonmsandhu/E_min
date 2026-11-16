#!/bin/bash
#SBATCH --job-name=q_ppo_jumanji
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/jumanji_rl/core/jumanji_config.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "jumanji_rl/core/jumanji_config.py" ] || cd "$(dirname "$0")/../../.."
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
    "Sokoban-v0"
    "Maze-v0"
    "SlidingTilePuzzle-v0"
    "RubiksCube-partly-scrambled-v0"
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
    ENV_NAME="Sokoban-v0"
fi

OUT_DIR="${OUT_DIR:-results/jumanji/sweep_q_ppo}"
N_SEEDS="${N_SEEDS:-3}"
MASTER_SEED="${MASTER_SEED:-42}"
CRITIC_LRS="${CRITIC_LRS:-0.0001 0.0003 0.001}"
CRITIC_EPOCHS="${CRITIC_EPOCHS:-4 8}"
LAMBDAS="${LAMBDAS:-0.4 0.8 0.95}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
GAE_LAMBDA="${GAE_LAMBDA:-0.8}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-128}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-512}"

TIMESTEP_FLAG=""
if [ -n "$TOTAL_TIMESTEPS" ] && [ "$TOTAL_TIMESTEPS" != "auto" ]; then
    TIMESTEP_FLAG="--total_timesteps $TOTAL_TIMESTEPS"
fi

$PYTHON -u scripts/jumanji/sweep_q_ppo/run_single_env.py \
    --env_name "$ENV_NAME" \
    --critic_lrs $CRITIC_LRS \
    --critic_epochs $CRITIC_EPOCHS \
    --lambdas $LAMBDAS \
    --actor_lr "$ACTOR_LR" \
    --gae_lambda "$GAE_LAMBDA" \
    $TIMESTEP_FLAG \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --out_dir "$OUT_DIR"
