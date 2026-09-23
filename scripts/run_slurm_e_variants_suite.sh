#!/bin/bash
#SBATCH --job-name=sweep_e_variants
#SBATCH --output=slurm/%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-15

# ==============================================================================
# Comprehensive Gymnax Suite Sweep: Comparison of Four Variants of E
#
# Compares the four variants of the Symmetrized Value Objective E:
#   1. E (1-step E(0), Tang & Munos baseline)
#   2. E_lambda (Method 1: FVI Stop-Gradient Form with forward/backward traces)
#   3. E_lambda_diff (Method 2: Full Autodiff Moment Scan)
#   4. E_lambda_geom (Method 3: Geometric Jump Sampling)
#
# Fixed Parameters:
#   - RETURN_LAMBDA: 0.99 (fixed baseline return anchor for all variants)
#   - ACTOR_LR: 0.0003 (policy learning rate held strictly constant)
#   - GAE_LAMBDA: 0.9 (policy advantage estimation held constant)
#
# Swept Parameters:
#   - Critic learning rate (LR): [0.003, 0.001, 0.0003, 0.0001]
#   - E_LAMBDA: [0.0, 0.5, 0.9] (swept for all three multi-step E(lambda) variants)
#   - RECOMPUTE_TARGETS_EACH_EPOCH: [false, true] (swept for E_lambda only)
#
# Environments: 19 Non-Bandit Gymnax Environments (Excluding Catch-bsuite & Bandits)
#
# Usage:
#   sbatch scripts/run_slurm_e_variants_suite.sh
#   ./scripts/run_slurm_e_variants_suite.sh CartPole-v1 (for single env run)
# ==============================================================================

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Python environment
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

# Environment Array: 19 Non-Bandit Gymnax environments (indices 0 to 18)
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

    # BSuite (excluding Catch-bsuite)
    "DeepSea-bsuite"
    "DiscountingChain-bsuite"

    # Misc / Navigation / Continuous
    "FourRooms-misc"
    "PointRobot-misc"
    "Reacher-misc"
    "Swimmer-misc"
    "Pong-misc"
)

# Select environment from argument or SLURM_ARRAY_TASK_ID
if [ -n "$1" ]; then
    ENV_NAME="$1"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="CartPole-v1"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS (size ${#ALL_ENVS[@]})."
    exit 1
fi

N_SEEDS=8

# Environment-specific horizon: 10M for MinAtar games, 2M (2,048,000) for all other environments
if [ -n "$CUSTOM_TIMESTEPS" ]; then
    TOTAL_TIMESTEPS="$CUSTOM_TIMESTEPS"
elif [[ "$ENV_NAME" == *"MinAtar"* ]]; then
    TOTAL_TIMESTEPS=10000000
else
    TOTAL_TIMESTEPS=2048000
fi

RANK_BY="final_window"
WINDOW_SIZE=100
METRIC="returned_discounted_episode_returns"

# Fixed hyperparameters
FIXED_ACTOR_LR=0.0003
FIXED_GAE_LAMBDA=0.9
FIXED_RETURN_LAMBDA=0.99

# Swept hyperparameters
CRITIC_LR_GRID="0.003 0.001 0.0003 0.0001"
E_LAMBDA_GRID="0.0 0.5 0.9"

# Base configuration with Slurm tracking
CONFIG="{\"NUM_ENVS\": 64, \"NUM_STEPS\": 256, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_EPOCHS\": 4, \"GAE_LAMBDA\": $FIXED_GAE_LAMBDA, \"RETURN_LAMBDA\": $FIXED_RETURN_LAMBDA, \"VF_CLIP\": 1000000.0, \"SLURM_JOB_ID\": \"${SLURM_JOB_ID:-local}\", \"SLURM_ARRAY_JOB_ID\": \"${SLURM_ARRAY_JOB_ID:-local}\", \"SLURM_ARRAY_TASK_ID\": \"${SLURM_ARRAY_TASK_ID:-0}\"}"

mkdir -p slurm

# Unified suite directory:
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-${SUITE_TAG:-e_variants_$(date +"%Y%m%d_%H%M%S")}}}"
SWEEP_SUITE_DIR="results/ppo/sweeps/suite_${SUITE_ID}"
SWEEP_ROOT_DIR="${SWEEP_SUITE_DIR}/${ENV_NAME}"
mkdir -p "$SWEEP_ROOT_DIR"

echo "======================================================================"
echo "STARTING GYMNAX SWEEP: COMPARISON OF FOUR VARIANTS OF E"
echo "Start Time: $START_TIME"
echo "Environment: $ENV_NAME"
echo "Seeds: $N_SEEDS | Total Timesteps: $TOTAL_TIMESTEPS"
echo "Fixed Return Lambda: $FIXED_RETURN_LAMBDA"
echo "Fixed Actor LR: $FIXED_ACTOR_LR | Fixed GAE Lambda: $FIXED_GAE_LAMBDA"
echo "Critic LR Grid: $CRITIC_LR_GRID"
echo "E(lambda) Grid: $E_LAMBDA_GRID"
echo "Ranking Metric: $METRIC ($RANK_BY)"
echo "Output Directory: $SWEEP_ROOT_DIR"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Sweep E (1-step E(0) baseline) across critic LR
# ------------------------------------------------------------------------------
echo ""
echo "--> [1/4] Sweeping E (1-step baseline) across critic LR: $CRITIC_LR_GRID..."
$PYTHON scripts/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos E \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --config "$CONFIG" \
    --n-seeds $N_SEEDS \
    --total-timesteps $TOTAL_TIMESTEPS \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE \
    --higher-is-better \
    --sweep-root-dir "$SWEEP_ROOT_DIR" \
    --no-log-scale

# ------------------------------------------------------------------------------
# 2. Sweep E_lambda_fixed (Method 1: FVI stop-grad) across LR x E_LAMBDA x RECOMPUTE
# ------------------------------------------------------------------------------
echo ""
echo "--> [2/4] Sweeping E_lambda_fixed (Method 1) across LR x E_LAMBDA x RECOMPUTE..."
$PYTHON scripts/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos E_lambda_fixed \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --e-lambda-grid $E_LAMBDA_GRID \
    --recompute-targets-grid false true \
    --config "$CONFIG" \
    --n-seeds $N_SEEDS \
    --total-timesteps $TOTAL_TIMESTEPS \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE \
    --higher-is-better \
    --sweep-root-dir "$SWEEP_ROOT_DIR" \
    --no-log-scale

# ------------------------------------------------------------------------------
# 3. Sweep E_lambda_differentiable (Method 2: Autodiff Moment Scan) across LR x E_LAMBDA
# ------------------------------------------------------------------------------
echo ""
echo "--> [3/4] Sweeping E_lambda_differentiable (Method 2) across LR x E_LAMBDA: $E_LAMBDA_GRID..."
$PYTHON scripts/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos E_lambda_differentiable \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --e-lambda-grid $E_LAMBDA_GRID \
    --config "$CONFIG" \
    --n-seeds $N_SEEDS \
    --total-timesteps $TOTAL_TIMESTEPS \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE \
    --higher-is-better \
    --sweep-root-dir "$SWEEP_ROOT_DIR" \
    --no-log-scale

# ------------------------------------------------------------------------------
# 4. Sweep E_lambda_geometric (Method 3: Geometric Jump Sampling) across LR x E_LAMBDA
# ------------------------------------------------------------------------------
echo ""
echo "--> [4/4] Sweeping E_lambda_geometric (Method 3) across LR x E_LAMBDA: $E_LAMBDA_GRID..."
$PYTHON scripts/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos E_lambda_geometric \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --e-lambda-grid $E_LAMBDA_GRID \
    --config "$CONFIG" \
    --n-seeds $N_SEEDS \
    --total-timesteps $TOTAL_TIMESTEPS \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE \
    --higher-is-better \
    --sweep-root-dir "$SWEEP_ROOT_DIR" \
    --no-log-scale

# Email recipient for completion notifications and PDF attachments
EMAIL_RECIPIENT="${EMAIL_RECIPIENT:-ds541@cs.duke.edu}"

# ------------------------------------------------------------------------------
# 5. Dedicated 4-Way Comparison Plot (Learning Curves + E(lambda) Scaling)
# ------------------------------------------------------------------------------
echo ""
echo "--> Generating 4-Way E-Variants Comparison Figure (Learning Curves + E(λ) Scaling)..."
$PYTHON notebooks/plot_e_variants_comparison.py \
    --sweep-dir "$SWEEP_ROOT_DIR" \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE

# ------------------------------------------------------------------------------
# 6. Check suite progress and compile full multi-environment PDF
# ------------------------------------------------------------------------------
COMPLETED_COUNT=0
for E in "${ALL_ENVS[@]}"; do
    if [ -f "${SWEEP_SUITE_DIR}/${E}/comparison/comparison_e_variants.png" ] || [ -f "${SWEEP_SUITE_DIR}/${E}/comparison/comparison_summary.csv" ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    fi
done

echo ""
echo "Suite Progress: $COMPLETED_COUNT / ${#ALL_ENVS[@]} environments completed."

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$COMPLETED_COUNT" -eq "${#ALL_ENVS[@]}" ]; then
    echo "======================================================================"
    echo "ALL ENVIRONMENTS COMPLETE! Compiling and emailing complete suite PDF..."
    echo "======================================================================"
    $PYTHON scripts/generate_e_variants_suite_pdf.py \
        --suite-dir "$SWEEP_SUITE_DIR" \
        --metric "$METRIC" \
        --rank-by "$RANK_BY" \
        --window-size $WINDOW_SIZE \
        --email "$EMAIL_RECIPIENT"
else
    # Update local suite PDF without sending email
    $PYTHON scripts/generate_e_variants_suite_pdf.py \
        --suite-dir "$SWEEP_SUITE_DIR" \
        --metric "$METRIC" \
        --rank-by "$RANK_BY" \
        --window-size $WINDOW_SIZE
fi

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "E-Variants Sweep for $ENV_NAME Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
