#!/bin/bash
#SBATCH --job-name=brax_sweep_all
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-10

# ==============================================================================
# Full Brax Continuous Control Suite Sweep (11 Environments, Excl. Fast):
# 4-Way Critic Benchmark:
#   1. TD(0)      - Classic 1-step online TD MSE critic
#   2. E(0)       - Adjacent-state Dirichlet smoothness error minimization
#   3. TD(lambda) - Standard PPO Fitted Value Iteration MSE (lambda = 0.9)
#   4. E(lambda)  - Geometric jumping Dirichlet error minimization (lambda = 0.9)
#
# Grid per Algorithm:
#   - Critic Learning Rate: [1e-4, 3e-4, 1e-3]
#   - Critic Epochs:        [4, 8, 16]
# Total: 9 configurations per algorithm = 36 configurations per environment.
# Evaluated across independent random seeds (default: 5) using JAX vmap.
#
# Environments:
#   0:  hopper
#   1:  walker2d
#   2:  halfcheetah
#   3:  ant
#   4:  swimmer
#   5:  humanoid
#   6:  inverted_pendulum
#   7:  inverted_double_pendulum
#   8:  reacher
#   9:  humanoidstandup
#   10: pusher
#
# Usage:
#   sbatch scripts/brax_sweep_all/run_slurm_brax_all.sh              (Submits all 11 envs as a job array)
#   sbatch --array=9 scripts/brax_sweep_all/run_slurm_brax_all.sh   (Submits single env index 9: humanoidstandup)
#   ./scripts/brax_sweep_all/run_slurm_brax_all.sh hopper           (Runs hopper locally or as single task)
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
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/3.10.9/envs/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/3.10.9/envs/purejaxrl/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/3.9.1/envs/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/3.9.1/envs/gymnax/bin/python"
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
    "humanoidstandup"
    "pusher"
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
N_SEEDS=${CUSTOM_SEEDS:-3}
NUM_ENVS=${CUSTOM_NUM_ENVS:-1024}
NUM_STEPS=${CUSTOM_NUM_STEPS:-128}

CRITIC_LRS=${CUSTOM_CRITIC_LRS:-"0.0001 0.0003 0.001"}
EPOCHS=${CUSTOM_EPOCHS:-"4 16"}
LAMBDA_VAL=${CUSTOM_LAMBDA:-0.9}
RETURN_LAMBDA=${CUSTOM_RETURN_LAMBDA:-1.0}
ALGOS=${CUSTOM_ALGOS:-"TD_0 E_0 TD_lambda E_lambda"}

# Unified suite directory across all tasks in the SLURM array
SWEEP_ID="${SWEEP_ID:-${SLURM_ARRAY_JOB_ID:+brax_sweep_all_${SLURM_ARRAY_JOB_ID}}}"
SWEEP_ID="${SWEEP_ID:-brax_sweep_all_${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}}"

echo "======================================================================"
echo "LAUNCHING BRAX 4-WAY CRITIC SWEEP"
echo "  Suite / Sweep:  $SWEEP_ID"
echo "  Environment:    $ENV_NAME"
echo "  Timesteps:      $TOTAL_TIMESTEPS"
echo "  Seeds:          $N_SEEDS"
echo "  Rollout:        $NUM_ENVS envs x $NUM_STEPS steps"
echo "  Critic LRs:     $CRITIC_LRS"
echo "  Critic Epochs:  $EPOCHS"
echo "  Lambda (trace): $LAMBDA_VAL"
echo "  Return Lambda:  $RETURN_LAMBDA"
echo "  Algorithms:     $ALGOS"
echo "  Python:         $PYTHON"
echo "  Node / Host:    $(hostname)"
echo "  Date:           $(date)"
echo "======================================================================"

$PYTHON scripts/brax_sweep_all/sweep_brax_all.py \
    --env-name "$ENV_NAME" \
    --sweep-id "$SWEEP_ID" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-seeds "$N_SEEDS" \
    --num-envs "$NUM_ENVS" \
    --num-steps "$NUM_STEPS" \
    --critic-lr-grid $CRITIC_LRS \
    --epochs-grid $EPOCHS \
    --lambda-val "$LAMBDA_VAL" \
    --return-lambda "$RETURN_LAMBDA" \
    --algos $ALGOS

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Experiment successfully finished for $ENV_NAME at $(date)"
else
    echo "Experiment failed with exit code $STATUS for $ENV_NAME at $(date)"
fi
exit $STATUS
