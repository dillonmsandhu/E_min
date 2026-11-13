#!/bin/bash
#SBATCH --job-name=double_s_gymnax
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-15

# ==============================================================================
# Double-Sampling Critic Comparison Sweep: Gymnax Per-Environment Array Task
# Runs the 4 double sampling critic algorithms on the designated Gymnax environment:
#   1. Fitted (GAE regression baseline)
#   2. MSBE (Baird double-sampled Bellman error)
#   3. E-Min (Double-sampled bilinear energy minimization <e, bar_delta>_D)
#   4. MSPBE (Tangent space projection with CG & exact Hessian-vector products)
#
# Environments: 16 Deterministic Discrete, Continuous & MinAtar Gymnax Environments:
#   MinAtar (10M steps):
#     0: Asterix-MinAtar (10M steps)
#     1: Breakout-MinAtar (10M steps)
#     2: Freeway-MinAtar (10M steps)
#     3: SpaceInvaders-MinAtar (10M steps)
#   MountainCar (1M steps):
#     4: MountainCar-v0 (1M steps)
#     5: MountainCarContinuous-v0 (1M steps)
#   Classic Control, BSuite & Misc (300k steps):
#     6: CartPole-v1 (300k steps)
#     7: Pendulum-v1 (300k steps)
#     8: Acrobot-v1 (300k steps)
#     9: DeepSea-bsuite (300k steps)
#     10: DiscountingChain-bsuite (300k steps)
#     11: FourRooms-misc (300k steps)
#     12: PointRobot-misc (300k steps)
#     13: Reacher-misc (300k steps)
#     14: Swimmer-misc (300k steps)
#     15: Pong-misc (300k steps)
# ==============================================================================

set -e

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "gymnax_rl/core/gymnax_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Ensure fallback log folder exists if run directly via sbatch without submit_array.sh
mkdir -p slurm

# Python interpreter discovery
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
    # MinAtar (10M steps)
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
    # MountainCar (1M steps)
    "MountainCar-v0"
    "MountainCarContinuous-v0"
    # Classic Control
    "CartPole-v1"
    "Pendulum-v1"
    "Acrobot-v1"
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
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        ENV_NAME="${ALL_ENVS[$1]}"
    else
        ENV_NAME="$1"
    fi
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="CartPole-v1"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS."
    exit 1
fi

if [ -z "$TOTAL_TIMESTEPS" ] || [ "$TOTAL_TIMESTEPS" = "auto" ]; then
    if [[ "$ENV_NAME" == *"MinAtar"* ]]; then
        ENV_TIMESTEPS=10000000
    elif [[ "$ENV_NAME" == *"MountainCar"* ]]; then
        ENV_TIMESTEPS=1000000
    else
        ENV_TIMESTEPS=300000
    fi
else
    ENV_TIMESTEPS="$TOTAL_TIMESTEPS"
fi

OUT_DIR="${OUT_DIR:-results/sweep_double_sample_gymnax}"
N_SEEDS="${N_SEEDS:-3}"
MASTER_SEED="${MASTER_SEED:-42}"
CRITIC_LRS="${CRITIC_LRS:-0.0003 0.001 0.003}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-128}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-512}"

echo "=================================================================="
echo "Starting Gymnax Double-Sampling Critic Suite for: $ENV_NAME"
echo "  SLURM Job:         ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "  Host:              $(hostname)"
echo "  Date:              $(date)"
echo "  Python:            $PYTHON"
echo "  Timesteps:         $ENV_TIMESTEPS"
echo "  Seeds:             $N_SEEDS (Master: $MASTER_SEED)"
echo "  Algorithms:        Fitted, MSBE, E-Min, MSPBE"
echo "  Critic LRs:        $CRITIC_LRS"
echo "  Actor LR:          $ACTOR_LR"
echo "  Rollout:           $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Output Directory:  $OUT_DIR"
echo "=================================================================="

$PYTHON -u scripts/gymnax/sweep_double_sample/run_single_env.py \
    --env_name "$ENV_NAME" \
    --critic_lrs $CRITIC_LRS \
    --actor_lr "$ACTOR_LR" \
    --total_timesteps "$ENV_TIMESTEPS" \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --out_dir "$OUT_DIR"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Successfully completed double-sampling sweep for $ENV_NAME at $(date)"
else
    echo "Sweep failed with exit code $STATUS for $ENV_NAME at $(date)"
fi
exit $STATUS
