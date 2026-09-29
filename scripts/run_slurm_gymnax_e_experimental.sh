#!/bin/bash
#SBATCH --job-name=e_exp_gymnax
#SBATCH --output=slurm/%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --array=0-11

# ==============================================================================
# Comprehensive Gymnax Suite Sweep: E_experimental Multi-Dimensional Tuning
#
# Tunes the new E_experimental objective across non-MinAtar Gymnax environments:
#   - Critic Epochs: [4, 16]
#   - Weight Decay: [0.001, 0.01]
#   - Critic Loss: MSE (Huber removed due to poor performance on sparse rewards)
#   - Value Heads: [1, 4]
# Total: 8 configurations per environment evaluated over 8 seeds.
#
# Environments: 12 Non-MinAtar, Non-Bandit Gymnax Environments:
#   Classic Control: CartPole-v1, Pendulum-v1, Acrobot-v1, MountainCar-v0, MountainCarContinuous-v0
#   BSuite: DeepSea-bsuite, DiscountingChain-bsuite
#   Misc: FourRooms-misc, PointRobot-misc, Reacher-misc, Swimmer-misc, Pong-misc
#
# Usage:
#   sbatch scripts/run_slurm_gymnax_e_experimental.sh
#   ./scripts/run_slurm_gymnax_e_experimental.sh CartPole-v1   (Single environment test)
# ==============================================================================

# Ensure working directory is submission directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
else
    cd "$(dirname "$0")/.." || exit 1
fi

mkdir -p slurm

# Python interpreter discovery
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

ALL_ENVS=(
    # Classic Control
    "CartPole-v1"
    "Pendulum-v1"
    "Acrobot-v1"
    "MountainCar-v0"
    "MountainCarContinuous-v0"

    # BSuite
    "DeepSea-bsuite"
    "DiscountingChain-bsuite"

    # Misc / Navigation / Continuous
    "FourRooms-misc"
    "PointRobot-misc"
    "Reacher-misc"
    "Swimmer-misc"
    "Pong-misc"
)

# Select environment from CLI argument or SLURM_ARRAY_TASK_ID
if [ -n "$1" ]; then
    ENV_NAME="$1"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="CartPole-v1"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS (size ${#ALL_ENVS[@]})."
    exit 1
fi

TOTAL_TIMESTEPS=${CUSTOM_TIMESTEPS:-2048000}
N_SEEDS=${CUSTOM_SEEDS:-8}
GRID_MODE=${GRID_MODE:-full}

echo "======================================================================"
echo "Starting E_experimental Multi-Dimensional Sweep"
echo "  Environment: $ENV_NAME"
echo "  Timesteps:   $TOTAL_TIMESTEPS"
echo "  Seeds:       $N_SEEDS"
echo "  Grid Mode:   $GRID_MODE (16 configs: epochs x wd x loss x heads)"
echo "  Python:      $PYTHON"
echo "  Node / Host: $(hostname)"
echo "  Date:        $(date)"
echo "======================================================================"

$PYTHON scripts/sweep_gymnax_e_experimental.py \
    --env-name "$ENV_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-seeds "$N_SEEDS" \
    --grid-mode "$GRID_MODE"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Sweep successfully finished for $ENV_NAME at $(date)"
else
    echo "Sweep failed with exit code $STATUS for $ENV_NAME at $(date)"
fi
exit $STATUS
