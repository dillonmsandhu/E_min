#!/bin/bash
#SBATCH --job-name=cont_q_gymnax
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

# ==============================================================================
# Continuous Q-Critic (DDPG) Comparison Sweep: Gymnax Per-Environment Array Task
# Runs all 5 Continuous Q-Critic analysis pipelines on the designated continuous environment:
#   1. E(0) Variants (Naive, v_aux, Forward-Only)
#   2. E(lambda=0.8) Variants (Naive, v_aux, Forward-Only)
#   3. Expected SARSA(lambda) Sensitivity Grid (0.0, 0.4, 0.8, 0.95, 1.0)
#
# Environments: 4 Deterministic Continuous Gymnax Environments:
#   0: Pendulum-v1 (300k steps)
#   1: MountainCarContinuous-v0 (500k steps)
#   2: MountainCarDenseContinuous-v0 (500k steps)
#   3: PointRobot-misc (300k steps)
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
    "Pendulum-v1"
    "MountainCarContinuous-v0"
    "MountainCarDenseContinuous-v0"
    "PointRobot-misc"
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
    ENV_NAME="Pendulum-v1"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS."
    exit 1
fi

if [ -z "$TOTAL_TIMESTEPS" ] || [ "$TOTAL_TIMESTEPS" = "auto" ]; then
    if [[ "$ENV_NAME" == *"MountainCar"* ]]; then
        ENV_TIMESTEPS=500000
    else
        ENV_TIMESTEPS=300000
    fi
else
    ENV_TIMESTEPS="$TOTAL_TIMESTEPS"
fi

OUT_DIR="${OUT_DIR:-results/sweep_continuous_q_critic_gymnax}"
N_SEEDS="${N_SEEDS:-3}"
MASTER_SEED="${MASTER_SEED:-42}"
CRITIC_LRS="${CRITIC_LRS:-0.0003 0.001 0.003}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
SARSA_LAMBDAS="${SARSA_LAMBDAS:-0.0 0.4 0.8 0.95 1.0}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_STEPS="${NUM_STEPS:-64}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-512}"
EXPLORATION_NOISE="${EXPLORATION_NOISE:-0.2}"

echo "=================================================================="
echo "Starting Gymnax Continuous Q-Critic (DDPG) Suite for: $ENV_NAME"
echo "  SLURM Job:         ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "  Host:              $(hostname)"
echo "  Date:              $(date)"
echo "  Python:            $PYTHON"
echo "  Timesteps:         $ENV_TIMESTEPS"
echo "  Seeds:             $N_SEEDS (Master: $MASTER_SEED)"
echo "  Critic LRs:        $CRITIC_LRS"
echo "  Actor LR:          $ACTOR_LR"
echo "  SARSA Lambdas:     $SARSA_LAMBDAS"
echo "  Rollout:           $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Exploration Noise: $EXPLORATION_NOISE"
echo "  Output Directory:  $OUT_DIR"
echo "=================================================================="

$PYTHON -u scripts/gymnax/sweep_continuous_q_critic/run_single_env.py \
    --env_name "$ENV_NAME" \
    --critic_lrs $CRITIC_LRS \
    --actor_lr "$ACTOR_LR" \
    --sarsa_lambdas $SARSA_LAMBDAS \
    --total_timesteps "$ENV_TIMESTEPS" \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --exploration_noise "$EXPLORATION_NOISE" \
    --out_dir "$OUT_DIR"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Successfully completed continuous Q-critic sweep for $ENV_NAME at $(date)"
else
    echo "Sweep failed with exit code $STATUS for $ENV_NAME at $(date)"
fi
exit $STATUS
