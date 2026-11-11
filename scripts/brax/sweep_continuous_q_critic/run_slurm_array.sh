#!/bin/bash
#SBATCH --job-name=cont_q_brax
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5

# ==============================================================================
# Continuous Q-Critic (DDPG) Comparison Sweep: Brax Per-Environment Array Task
# Runs all 5 Continuous Q-Critic analysis pipelines on the designated continuous environment:
#   1. E(0) Variants (Naive, v_aux, Forward-Only)
#   2. E(lambda=0.8) Variants (Naive, v_aux, Forward-Only)
#   3. Expected SARSA(lambda) Sensitivity Grid (0.0, 0.4, 0.8, 0.95, 1.0)
#
# Environments: 6 Deterministic Continuous Brax Environments:
#   0: hopper (2M steps)
#   1: halfcheetah (3M steps)
#   2: walker2d (2M steps)
#   3: ant (3M steps)
#   4: inverted_pendulum (500k steps)
#   5: reacher (1M steps)
# ==============================================================================

set -e

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "brax_rl/core/brax_config.py" ] || cd "$(dirname "$0")/../../.."
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
    "hopper"
    "halfcheetah"
    "walker2d"
    "ant"
    "inverted_pendulum"
    "reacher"
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
    ENV_NAME="hopper"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS."
    exit 1
fi

OUT_DIR="${OUT_DIR:-results/sweep_continuous_q_critic_brax}"
N_SEEDS="${N_SEEDS:-3}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-64}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-512}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
EXPLORATION_NOISE="${EXPLORATION_NOISE:-0.1}"

echo "================================================================"
echo "Job ID:           ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Environment:      $ENV_NAME"
echo "Host:             $(hostname)"
echo "Output Directory: $OUT_DIR"
echo "Seeds:            $N_SEEDS"
echo "Python:           $PYTHON"
echo "================================================================"

$PYTHON -u scripts/brax/sweep_continuous_q_critic/run_single_env.py \
    --env_name "$ENV_NAME" \
    --actor_lr "$ACTOR_LR" \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --n_seeds "$N_SEEDS" \
    --exploration_noise "$EXPLORATION_NOISE" \
    --out_dir "$OUT_DIR"

echo "Environment $ENV_NAME finished successfully."
