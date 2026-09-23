#!/bin/bash
#SBATCH --job-name=sweep_minatar_e_vs_td
#SBATCH --output=slurm/%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-3

# ==============================================================================
# MinAtar Suite Sweep (10M Timesteps, k=32):
# Comparison of E-Variants vs. TD(lambda)
#
# Compares the four variants of E with TD(lambda) (PPO baseline):
#   1. E (1-step E(0), Tang & Munos baseline)
#   2. E_lambda_fixed (Method 1: FVI Stop-Gradient Form with forward/backward traces)
#   3. E_lambda_differentiable (Method 2: Full Autodiff Moment Scan)
#   4. E_lambda_geometric (Method 3: Geometric Jump Sampling)
#   5. ppo (TD(lambda) baseline)
#
# Representation Dimensionality:
#   - k: 32 (final hidden representation dimension, doubled from default 16)
#
# Fixed Parameters:
#   - RETURN_LAMBDA: 0.99 (fixed baseline Monte Carlo return anchor for E variants)
#   - VALUE_LAMBDA: 0.99 (used by E to compute return anchor G_t)
#   - ACTOR_LR: 0.0003 (policy learning rate held strictly constant)
#   - GAE_LAMBDA: 0.9 (policy advantage estimation held constant)
#   - Horizon: 10,000,000 timesteps (610 update steps of 64 envs x 256 steps)
#
# Swept Parameters:
#   - Critic learning rate (LR): [0.003, 0.001, 0.0003, 0.0001]
#   - E_LAMBDA: [0.0, 0.5, 0.9] (swept for multi-step E(lambda) variants)
#   - RECOMPUTE_TARGETS_EACH_EPOCH: [false, true] (swept for E_lambda_fixed only)
#   - TD(lambda) VALUE_LAMBDA: [0.9, 0.95, 0.99, 1.0] (swept for ppo baseline)
#
# Environments: 4 MinAtar Games
#   0: Asterix-MinAtar
#   1: Breakout-MinAtar
#   2: Freeway-MinAtar
#   3: SpaceInvaders-MinAtar
#
# Usage:
#   sbatch scripts/run_slurm_minatar_e_vs_td.sh
#   ./scripts/run_slurm_minatar_e_vs_td.sh Asterix-MinAtar (for single env run)
# ==============================================================================

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Python environment
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

# MinAtar Environments (indices 0 to 3)
ALL_ENVS=(
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
)

# Select environment from argument or SLURM_ARRAY_TASK_ID
if [ -n "$1" ]; then
    ENV_NAME="$1"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="Asterix-MinAtar"
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS (size ${#ALL_ENVS[@]})."
    exit 1
fi

N_SEEDS=8
TOTAL_TIMESTEPS=10000000  # 10M env steps (610 updates of 64 envs x 256 steps)

RANK_BY="final_window"
WINDOW_SIZE=100
METRIC="returned_episode_returns"

# Fixed hyperparameters
FIXED_ACTOR_LR=0.003
FIXED_GAE_LAMBDA=0.8
FIXED_RETURN_LAMBDA=0.999
K_DIM=32

# Swept hyperparameters
CRITIC_LR_GRID="0.005 0.001 0.0005"
E_LAMBDA_GRID="0.0 0.8 0.95 0.99"
TD_LAMBDA_GRID="0.0 0.8 0.95 0.99"

# Base configuration with k=32 and Slurm tracking
CONFIG="{\"NUM_ENVS\": 128, \"NUM_STEPS\": 64, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_EPOCHS\": 4, \"GAE_LAMBDA\": $FIXED_GAE_LAMBDA, \"RETURN_LAMBDA\": $FIXED_RETURN_LAMBDA, \"VALUE_LAMBDA\": $FIXED_RETURN_LAMBDA, \"k\": $K_DIM, \"ENT_COEF\": 0.001,\"LAYER_NORM\": \"True\", \"SLURM_JOB_ID\": \"${SLURM_JOB_ID:-local}\", \"SLURM_ARRAY_JOB_ID\": \"${SLURM_ARRAY_JOB_ID:-local}\", \"SLURM_ARRAY_TASK_ID\": \"${SLURM_ARRAY_TASK_ID:-0}\", \"ACTOR_LR_END\": 0.0001}"

mkdir -p slurm

# Unified suite directory:
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-${SUITE_TAG:-minatar_k32_$(date +"%Y%m%d_%H%M%S")}}}"
SWEEP_SUITE_DIR="results/ppo/sweeps/suite_${SUITE_ID}"
SWEEP_ROOT_DIR="${SWEEP_SUITE_DIR}/${ENV_NAME}"
mkdir -p "$SWEEP_ROOT_DIR"

echo "======================================================================"
echo "STARTING MINATAR 10M SWEEP (k=$K_DIM): E-VARIANTS VS TD(LAMBDA)"
echo "Start Time: $START_TIME"
echo "Environment: $ENV_NAME"
echo "Seeds: $N_SEEDS | Total Timesteps: $TOTAL_TIMESTEPS (10M)"
echo "Representation Dimension k: $K_DIM"
echo "Fixed Return Lambda: $FIXED_RETURN_LAMBDA"
echo "Fixed Actor LR: $FIXED_ACTOR_LR | Fixed GAE Lambda: $FIXED_GAE_LAMBDA"
echo "Critic LR Grid: $CRITIC_LR_GRID"
echo "E(lambda) Grid: $E_LAMBDA_GRID"
echo "TD(lambda) Grid: $TD_LAMBDA_GRID"
echo "Ranking Metric: $METRIC ($RANK_BY)"
echo "Output Directory: $SWEEP_ROOT_DIR"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Sweep E (1-step baseline) across critic LR
# ------------------------------------------------------------------------------
echo ""
echo "--> [1/5] Sweeping E (1-step baseline) across critic LR: $CRITIC_LR_GRID..."
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
echo "--> [2/5] Sweeping E_lambda_fixed (Method 1) across LR x E_LAMBDA x RECOMPUTE..."
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
echo "--> [3/5] Sweeping E_lambda_differentiable (Method 2) across LR x E_LAMBDA: $E_LAMBDA_GRID..."
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
echo "--> [4/5] Sweeping E_lambda_geometric (Method 3) across LR x E_LAMBDA: $E_LAMBDA_GRID..."
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

# ------------------------------------------------------------------------------
# 5. Sweep TD(lambda) (PPO Baseline) across LR x VALUE_LAMBDA
# ------------------------------------------------------------------------------
echo ""
echo "--> [5/5] Sweeping ppo / TD(lambda) baseline across LR x VALUE_LAMBDA: $TD_LAMBDA_GRID..."
$PYTHON scripts/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos ppo \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --value-lambda-grid $TD_LAMBDA_GRID \
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
# 6. Dedicated 5-Way Comparison Plot (Learning Curves + E(lambda) Scaling)
# ------------------------------------------------------------------------------
echo ""
echo "--> Generating E-Variants vs. TD(λ) Comparison Figure..."
$PYTHON notebooks/plot_e_variants_comparison.py \
    --sweep-dir "$SWEEP_ROOT_DIR" \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE

# ------------------------------------------------------------------------------
# 7. Check suite progress and compile full multi-environment PDF
# ------------------------------------------------------------------------------
COMPLETED_COUNT=0
for E in "${ALL_ENVS[@]}"; do
    if [ -f "${SWEEP_SUITE_DIR}/${E}/comparison/comparison_e_variants.png" ] || [ -f "${SWEEP_SUITE_DIR}/${E}/comparison/comparison_summary.csv" ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    fi
done

echo ""
echo "MinAtar Progress: $COMPLETED_COUNT / ${#ALL_ENVS[@]} environments completed."

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$COMPLETED_COUNT" -eq "${#ALL_ENVS[@]}" ]; then
    echo "======================================================================"
    echo "ALL 4 MINATAR ENVIRONMENTS COMPLETE! Compiling and emailing suite PDF..."
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
echo "MinAtar E-Variants vs TD(λ) Sweep for $ENV_NAME Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
