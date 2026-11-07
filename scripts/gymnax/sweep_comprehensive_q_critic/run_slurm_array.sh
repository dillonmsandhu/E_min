#!/bin/bash
#SBATCH --job-name=comp_q_critic
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-9

# ==============================================================================
# Comprehensive Q-PPO Critic Suite: Per-Environment Array Task
# Runs all Q-critic algorithms on the designated deterministic discrete / MinAtar environment:
#   1. E(lambda)-Q sweep (lambdas: 0.0, 0.4, 0.8, 0.95; critic LRs: 1e-4, 3e-4, 1e-3)
#   2. SARSA(lambda) sweep (lambdas: 0.0, 0.4, 0.8, 0.95; critic LRs: 1e-4, 3e-4, 1e-3)
#   3. E(0)-Q Dirichlet energy minimization
#   4. MC-Q rollout return regression
#
# Environments: 10 Deterministic Discrete-Action & MinAtar Gymnax Environments:
#   0: CartPole-v1
#   1: Acrobot-v1
#   2: MountainCar-v0 (1M steps)
#   3: DeepSea-bsuite
#   4: DiscountingChain-bsuite
#   5: Pong-misc
#   6: Asterix-MinAtar (10M steps)
#   7: Breakout-MinAtar (10M steps)
#   8: Freeway-MinAtar (10M steps)
#   9: SpaceInvaders-MinAtar (10M steps)
# ==============================================================================

set -e

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "core/config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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
    # Classic Control
    "CartPole-v1"
    "Acrobot-v1"
    "MountainCar-v0"

    # BSuite
    "DeepSea-bsuite"
    "DiscountingChain-bsuite"

    # Misc
    "Pong-misc"

    # MinAtar (10M samples)
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
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
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS (size ${#ALL_ENVS[@]})."
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

N_SEEDS="${N_SEEDS:-3}"
MASTER_SEED="${MASTER_SEED:-42}"
LAMBDAS="${LAMBDAS:-0.0 0.4 0.8 0.95}"
CRITIC_LRS="${CRITIC_LRS:-0.0001 0.0003 0.001}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
NUM_ENVS="${NUM_ENVS:-32}"
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-512}"
OUT_DIR="${OUT_DIR:-results/sweep_comprehensive_q_critic}"

echo "=================================================================="
echo "Starting Comprehensive Q-PPO Critic Suite for: $ENV_NAME"
echo "  SLURM Job:        ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "  Host:             $(hostname)"
echo "  Date:             $(date)"
echo "  Python:           $PYTHON"
echo "  Timesteps:        $ENV_TIMESTEPS"
echo "  Seeds:            $N_SEEDS (Master: $MASTER_SEED)"
echo "  Lambdas:          $LAMBDAS"
echo "  Critic LRs:       $CRITIC_LRS"
echo "  Output Directory: $OUT_DIR"
echo "=================================================================="

$PYTHON scripts/gymnax/sweep_comprehensive_q_critic/run_single_env.py \
    --env_name "$ENV_NAME" \
    --lambdas $LAMBDAS \
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
    echo "Successfully completed comprehensive Q-critic sweep for $ENV_NAME at $(date)"
else
    echo "Sweep failed with exit code $STATUS for $ENV_NAME at $(date)"
fi
exit $STATUS
