#!/bin/bash
#SBATCH --job-name=cmp_td_vs_e
#SBATCH --output=slurm/%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --array=0-15

# ==============================================================================
# Comprehensive Gymnax Suite Sweep: Classic TD vs. E_experimental
#
# Standard Head-to-Head Comparison over:
#   - Critic Learning Rate: [0.0003, 0.001, 0.003]
#   - Critic Epochs: [4, 16, 32]
#   - Weight Decay: [0.001, 0.01]
#   - Value Heads: [1, 4]
# Total: 36 configurations per algorithm (72 total per environment) evaluated over 8 seeds.
#
# Environments: 16 Gymnax Environments:
#   Classic Control: CartPole-v1, Pendulum-v1, Acrobot-v1, MountainCar-v0, MountainCarContinuous-v0
#   MinAtar: Asterix-MinAtar, Breakout-MinAtar, Freeway-MinAtar, SpaceInvaders-MinAtar
#   BSuite: DeepSea-bsuite, DiscountingChain-bsuite
#   Misc: FourRooms-misc, PointRobot-misc, Reacher-misc, Swimmer-misc, Pong-misc
#
# Usage:
#   sbatch scripts/run_slurm_td_vs_e_experimental.sh
#   ./scripts/run_slurm_td_vs_e_experimental.sh MountainCarContinuous-v0  (Single env)
# ==============================================================================

# Ensure we are in the repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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

    # MinAtar
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"

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
    ENV_NAME="MountainCarContinuous-v0"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS (size ${#ALL_ENVS[@]})."
    exit 1
fi

# Timestep determination
if [ -n "$CUSTOM_TIMESTEPS" ]; then
    TOTAL_TIMESTEPS="$CUSTOM_TIMESTEPS"
elif [[ "$ENV_NAME" == *"MinAtar"* ]]; then
    TOTAL_TIMESTEPS=10000000
else
    TOTAL_TIMESTEPS=2048000
fi

N_SEEDS=${CUSTOM_SEEDS:-8}
CRITIC_LRS=${CUSTOM_CRITIC_LRS:-"0.0003 0.001 0.003"}
EPOCHS=${CUSTOM_EPOCHS:-"4 16 32"}
WDS=${CUSTOM_WDS:-"0.001 0.01"}
HEADS=${CUSTOM_HEADS:-"1 4"}

echo "======================================================================"
echo "LAUNCHING HEAD-TO-HEAD COMPARISON: TD vs E_EXPERIMENTAL"
echo "  Environment:   $ENV_NAME"
echo "  Timesteps:     $TOTAL_TIMESTEPS"
echo "  Seeds:         $N_SEEDS"
echo "  Critic LRs:    $CRITIC_LRS"
echo "  Critic Epochs: $EPOCHS"
echo "  Weight Decays: $WDS"
echo "  Value Heads:   $HEADS"
echo "  Python:        $PYTHON"
echo "  Node / Host:   $(hostname)"
echo "  Date:          $(date)"
echo "======================================================================"

$PYTHON scripts/sweep_td_vs_e_experimental.py \
    --env-name "$ENV_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-seeds "$N_SEEDS" \
    --critic-lr-grid $CRITIC_LRS \
    --epochs-grid $EPOCHS \
    --wd-grid $WDS \
    --heads-grid $HEADS

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Sweep successfully finished for $ENV_NAME at $(date)"
else
    echo "Sweep failed with exit code $STATUS for $ENV_NAME at $(date)"
fi
exit $STATUS
