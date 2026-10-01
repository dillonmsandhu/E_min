#!/bin/bash
#SBATCH --job-name=brax_e_lambda_sweep
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8

# ==============================================================================
# Full Brax Continuous Control Suite Sweep:
# Symmetrized E(lambda) (Dirichlet geometric jumps) vs. Baseline PPO
#
# Lambdas Fixed:
#   - E_LAMBDA:      0.9 (geometric lookahead jump Dirichlet error)
#   - RETURN_LAMBDA: 0.9 (regression return target lambda)
#
# Critic Sweep Grid:
#   - Critic Learning Rate: [1e-4, 3e-4, 1e-3]
#   - Critic Epochs:        [4, 8, 16]
# Total: 9 configurations for E(lambda) + 1 un-swept Baseline PPO reference per environment.
# Evaluated across independent random seeds (default: 5) using JAX vmap.
#
# Environments:
#   0: hopper
#   1: walker2d
#   2: halfcheetah
#   3: ant
#   4: swimmer
#   5: humanoid
#   6: inverted_pendulum
#   7: inverted_double_pendulum
#   8: reacher
#
# Usage:
#   sbatch scripts/brax_e_lambda/run_slurm_brax_e_lambda_sweep.sh             (Submits all 9 envs as an array)
#   ./scripts/brax_e_lambda/run_slurm_brax_e_lambda_sweep.sh hopper           (Runs hopper locally or as single job)
# ==============================================================================

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
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
    "hopper"
    "walker2d"
    "halfcheetah"
    "ant"
    "swimmer"
    "humanoid"
    "inverted_pendulum"
    "inverted_double_pendulum"
    "reacher"
)

# Select environment from CLI argument or SLURM_ARRAY_TASK_ID
TARGET_ARG=${1:-$SLURM_ARRAY_TASK_ID}

if [ -z "$TARGET_ARG" ]; then
    echo "No task index provided; running all tasks sequentially..."
    for task_idx in $(seq 0 $((${#ALL_ENVS[@]} - 1))); do
        "$0" "$task_idx"
    done
    exit 0
fi

if [[ "$TARGET_ARG" =~ ^[0-9]+$ ]]; then
    ENV_NAME="${ALL_ENVS[$TARGET_ARG]}"
else
    ENV_NAME="$TARGET_ARG"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. Task ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS (size ${#ALL_ENVS[@]})."
    exit 1
fi

TOTAL_TIMESTEPS=${CUSTOM_TIMESTEPS:-50000000}
N_SEEDS=${CUSTOM_SEEDS:-5}
NUM_ENVS=${CUSTOM_NUM_ENVS:-1024}
NUM_STEPS=${CUSTOM_NUM_STEPS:-128}

CRITIC_LRS=${CUSTOM_CRITIC_LRS:-"0.0001 0.0003 0.001"}
EPOCHS=${CUSTOM_EPOCHS:-"4 8 16"}
E_LAMBDA=${CUSTOM_E_LAMBDA:-0.9}
RETURN_LAMBDA=${CUSTOM_RETURN_LAMBDA:-0.9}

# Unified suite directory across all tasks in the SLURM array
SWEEP_ID="${SWEEP_ID:-${SLURM_ARRAY_JOB_ID:+brax_e_lambda_${SLURM_ARRAY_JOB_ID}}}"
SWEEP_ID="${SWEEP_ID:-brax_e_lambda_${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}}"

echo "======================================================================"
echo "LAUNCHING BRAX E(LAMBDA) SWEEP & BASELINE PPO"
echo "  Suite / Sweep:  $SWEEP_ID"
echo "  Environment:    $ENV_NAME"
echo "  Timesteps:      $TOTAL_TIMESTEPS"
echo "  Seeds:          $N_SEEDS"
echo "  Rollout:        $NUM_ENVS envs x $NUM_STEPS steps"
echo "  E_LAMBDA:       $E_LAMBDA"
echo "  RETURN_LAMBDA:  $RETURN_LAMBDA"
echo "  Critic LRs:     $CRITIC_LRS"
echo "  Critic Epochs:  $EPOCHS"
echo "  Python:         $PYTHON"
echo "  Node / Host:    $(hostname)"
echo "  Date:           $(date)"
echo "======================================================================"

$PYTHON scripts/brax_e_lambda/sweep_brax_e_lambda_vs_baseline.py \
    --env-name "$ENV_NAME" \
    --sweep-id "$SWEEP_ID" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-seeds "$N_SEEDS" \
    --num-envs "$NUM_ENVS" \
    --num-steps "$NUM_STEPS" \
    --e-lambda "$E_LAMBDA" \
    --return-lambda "$RETURN_LAMBDA" \
    --critic-lr-grid $CRITIC_LRS \
    --epochs-grid $EPOCHS

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Experiment successfully finished for $ENV_NAME at $(date)"
else
    echo "Experiment failed with exit code $STATUS for $ENV_NAME at $(date)"
fi
exit $STATUS
