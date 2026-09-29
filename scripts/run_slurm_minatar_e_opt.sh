#!/bin/bash
#SBATCH --job-name=sweep_minatar_e_opt
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

# ==============================================================================
# MinAtar E Optimization Sweep: WEIGHT_DECAY x EPOCHS x NUM_ENVS
#
# Grid:
#   - WEIGHT_DECAY: [0.0, 0.01]
#   - epochs: [4, 16]
#   - num_envs: [64, 512]
# Total: 2 x 2 x 2 = 8 configurations per environment.
#
# Environments: 4 MinAtar Games
#   0: Asterix-MinAtar
#   1: Breakout-MinAtar
#   2: Freeway-MinAtar
#   3: SpaceInvaders-MinAtar
#
# Usage:
#   sbatch scripts/run_slurm_minatar_e_opt.sh
#   ./scripts/run_slurm_minatar_e_opt.sh Asterix-MinAtar (local run)
# ==============================================================================

set -e
mkdir -p slurm

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python resolution fallback
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/shims/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/shims/python"
else
    PYTHON="python"
fi

ALL_ENVS=(
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
)

# Select environment from argument or SLURM_ARRAY_TASK_ID
if [ -n "$1" ]; then
    ENV_NAME="$1"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="Asterix-MinAtar"
fi

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
N_SEEDS=${N_SEEDS:-8}
CRITIC_LR=${CRITIC_LR:-0.001}
ACTOR_LR=${ACTOR_LR:-0.003}
CRITIC_LOSS_TYPE=${CRITIC_LOSS_TYPE:-mse}
HUBER_DELTA=${HUBER_DELTA:-1.0}
NUM_VALUE_HEADS=${NUM_VALUE_HEADS:-1}

echo "======================================================================"
echo "LAUNCHING MINATAR E OPTIMIZATION SWEEP"
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON"
echo "Seeds: $N_SEEDS | Total Timesteps: $TOTAL_TIMESTEPS"
echo "Loss Type: $CRITIC_LOSS_TYPE (delta=$HUBER_DELTA) | Value Heads: $NUM_VALUE_HEADS"
echo "Grid: NUM_ENVS=[64, 512] x EPOCHS=[4, 16] x WEIGHT_DECAY=[0.0, 0.01]"
echo "======================================================================"

$PYTHON scripts/sweep_minatar_e_opt.py \
    --env-name "$ENV_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-seeds "$N_SEEDS" \
    --critic-lr "$CRITIC_LR" \
    --actor-lr "$ACTOR_LR" \
    --critic-loss-type "$CRITIC_LOSS_TYPE" \
    --huber-delta "$HUBER_DELTA" \
    --num-value-heads "$NUM_VALUE_HEADS"

echo "Sweep completed for $ENV_NAME"
