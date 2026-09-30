#!/bin/bash
#SBATCH --job-name=td_vs_e_lambda
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-15

# ==============================================================================
# SLURM ARRAY SWEEP: E(lambda) vs TD(lambda) Head-to-Head Comparison
#
# Compares E(lambda) (algos/E_lambda_experimental.py) vs TD(lambda) (algos/td.py)
# across the lambda and critic optimization grid:
#   - Lambda: [0.0, 0.8, 0.95]
#   - Critic Epochs: [4, 16, 32]
#   - Critic LR: [0.0003, 0.001]
#   - Weight Decay: [0.001, 0.01]
#   - Value Heads: [1, 4]
#
# Environments: 16 Gymnax Environments (Array 0-15):
#   Classic Control: CartPole-v1, Pendulum-v1, Acrobot-v1, MountainCar-v0, MountainCarContinuous-v0
#   MinAtar: Asterix-MinAtar, Breakout-MinAtar, Freeway-MinAtar, SpaceInvaders-MinAtar
#   BSuite: DeepSea-bsuite, DiscountingChain-bsuite
#   Misc: FourRooms-misc, PointRobot-misc, Reacher-misc, Swimmer-misc, Pong-misc
#
# Usage:
#   sbatch scripts/td_vs_e_lambda/run_slurm_td_vs_e_lambda.sh
#   ./scripts/td_vs_e_lambda/run_slurm_td_vs_e_lambda.sh MountainCarContinuous-v0  (Single env)
# ==============================================================================

# Ensure we are in the repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."

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

# Grid options (defaults match the Experiment 2 plan)
LAMBDAS=${CUSTOM_LAMBDAS:-"0.0 0.8 0.95"}
CRITIC_LRS=${CUSTOM_CRITIC_LRS:-"0.0003 0.001"}
EPOCHS=${CUSTOM_EPOCHS:-"4 16 32"}
WDS=${CUSTOM_WDS:-"0.001 0.01"}
HEADS=${CUSTOM_HEADS:-"1 4"}
N_SEEDS=${CUSTOM_SEEDS:-8}

# Unified suite directory: all tasks in the SLURM array share SLURM_ARRAY_JOB_ID
SWEEP_ID="${SWEEP_ID:-${SLURM_ARRAY_JOB_ID:+cmp_td_vs_e_lambda_${SLURM_ARRAY_JOB_ID}}}"
SWEEP_ID="${SWEEP_ID:-cmp_td_vs_e_lambda_${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}}"

echo "======================================================================"
echo "LAUNCHING HEAD-TO-HEAD COMPARISON: E(lambda) vs TD(lambda)"
echo "  Suite / Sweep: $SWEEP_ID"
echo "  Environment:   $ENV_NAME"
echo "  Seeds:         $N_SEEDS"
echo "  Lambda Grid:   $LAMBDAS"
echo "  Critic LRs:    $CRITIC_LRS"
echo "  Critic Epochs: $EPOCHS"
echo "  Weight Decays: $WDS"
echo "  Value Heads:   $HEADS"
echo "  Python:        $PYTHON"
echo "  Node / Host:   $(hostname)"
echo "  Date:          $(date)"
echo "======================================================================"

$PYTHON scripts/td_vs_e_lambda/sweep_td_vs_e_lambda.py \
    --env-name "$ENV_NAME" \
    --sweep-id "$SWEEP_ID" \
    --n-seeds "$N_SEEDS" \
    --lambda-grid $LAMBDAS \
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
