#!/bin/bash
#SBATCH --job-name=sweep_minatar
#SBATCH --output=slurm/%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-3

# ==============================================================================
# MinAtar Suite Sweep (10M Timesteps): E-Minimization vs. TD(lambda) Spectrum
#
# Compares Sampled E Minimization against the full spectrum of TD(lambda):
#   - TD(lambda) swept over VALUE_LAMBDA: [0.9, 0.95, 0.99, 1.0]
#   - Critic learning rate swept: [0.003, 0.001, 0.0003, 0.0001]
#   - Actor training kept strictly constant: ACTOR_LR=0.0003, GAE_LAMBDA=0.9
#   - Horizon: 10,000,000 timesteps (610 update steps)
#
# Environments (4 MinAtar Games):
#   0: Asterix-MinAtar
#   1: Breakout-MinAtar
#   2: Freeway-MinAtar
#   3: SpaceInvaders-MinAtar
#
# Usage:
#   sbatch scripts/run_slurm_minatar_suite.sh
#   ./scripts/run_slurm_minatar_suite.sh Asterix-MinAtar (for single env run)
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
METRIC="returned_discounted_episode_returns"

# Value learning dynamics: sweep critic LR while keeping policy training fixed
CRITIC_LR_GRID="0.003 0.001 0.0003 0.0001"
FIXED_ACTOR_LR=0.0003
FIXED_GAE_LAMBDA=0.9

# High lambda spectrum for comparing high return anchors
HIGH_LAMBDA_GRID="0.9 0.95 0.99 1.0"

# Rollout batch configuration with Slurm tracking
CONFIG="{\"NUM_ENVS\": 64, \"NUM_STEPS\": 256, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_EPOCHS\": 4, \"GAE_LAMBDA\": $FIXED_GAE_LAMBDA, \"SLURM_JOB_ID\": \"${SLURM_JOB_ID:-local}\", \"SLURM_ARRAY_JOB_ID\": \"${SLURM_ARRAY_JOB_ID:-local}\", \"SLURM_ARRAY_TASK_ID\": \"${SLURM_ARRAY_TASK_ID:-0}\"}"

mkdir -p slurm

# Unified suite directory:
# When running as an sbatch array, all 4 tasks share $SLURM_ARRAY_JOB_ID.
# If running single job, use $SLURM_JOB_ID. If local, use SUITE_TAG or timestamp.
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-${SUITE_TAG:-minatar_$(date +"%Y%m%d_%H%M%S")}}}"
SWEEP_SUITE_DIR="results/ppo/sweeps/suite_${SUITE_ID}"
SWEEP_ROOT_DIR="${SWEEP_SUITE_DIR}/${ENV_NAME}"
mkdir -p "$SWEEP_ROOT_DIR"

echo "======================================================================"
echo "STARTING MINATAR 10M SWEEP: E-MINIMIZATION VS TD(LAMBDA) SPECTRUM"
echo "Start Time: $START_TIME"
echo "Environment: $ENV_NAME"
echo "Seeds: $N_SEEDS | Total Timesteps: $TOTAL_TIMESTEPS (10M)"
echo "Critic LR Grid: $CRITIC_LR_GRID"
echo "Fixed Actor LR: $FIXED_ACTOR_LR | Fixed GAE Lambda: $FIXED_GAE_LAMBDA"
echo "High Lambda Grid: $HIGH_LAMBDA_GRID"
echo "Ranking Metric: $METRIC ($RANK_BY)"
echo "Output Directory: $SWEEP_ROOT_DIR"
echo "======================================================================"

# 1. Sweep Sampled E Minimization across critic LR x high lambda
echo ""
echo "--> [1/2] Sweeping sampled_E across high lambda spectrum: $HIGH_LAMBDA_GRID..."
$PYTHON scripts/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos sampled_E \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --value-lambda-grid $HIGH_LAMBDA_GRID \
    --config "$CONFIG" \
    --n-seeds $N_SEEDS \
    --total-timesteps $TOTAL_TIMESTEPS \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE \
    --higher-is-better \
    --sweep-root-dir "$SWEEP_ROOT_DIR" \
    --no-log-scale

# 2. Sweep TD(lambda) across high lambda spectrum
echo ""
echo "--> [2/2] Sweeping sampled_td_lambda across high lambda spectrum: $HIGH_LAMBDA_GRID..."
$PYTHON scripts/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos sampled_td_lambda \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --value-lambda-grid $HIGH_LAMBDA_GRID \
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

# 3. Dedicated Lambda Spectrum vs E Plot (saves local PNG and PDF)
echo ""
echo "--> Generating Lambda Spectrum vs. E Comparison Figure..."
$PYTHON notebooks/plot_lambda_spectrum_vs_E.py \
    --sweep-dir "$SWEEP_ROOT_DIR" \
    --metric "$METRIC"

# 4. Check suite completion: only email the complete sweep once ALL 4 MinAtar environments finish
COMPLETED_COUNT=0
for E in "${ALL_ENVS[@]}"; do
    if [ -f "${SWEEP_SUITE_DIR}/${E}/comparison/comparison_summary.csv" ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    fi
done

echo ""
echo "MinAtar Progress: $COMPLETED_COUNT / ${#ALL_ENVS[@]} environments completed."

if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$COMPLETED_COUNT" -eq "${#ALL_ENVS[@]}" ]; then
    echo "======================================================================"
    echo "ALL 4 MINATAR ENVIRONMENTS COMPLETE! Compiling and emailing PDF..."
    echo "======================================================================"
    $PYTHON scripts/generate_suite_pdf.py \
        --suite-dir "$SWEEP_SUITE_DIR" \
        --metric "$METRIC" \
        --rank-by "$RANK_BY" \
        --window-size $WINDOW_SIZE \
        --email "$EMAIL_RECIPIENT"
else
    # Update local PDF without sending email
    $PYTHON scripts/generate_suite_pdf.py \
        --suite-dir "$SWEEP_SUITE_DIR" \
        --metric "$METRIC" \
        --rank-by "$RANK_BY" \
        --window-size $WINDOW_SIZE
fi

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "MinAtar Sweep for $ENV_NAME Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
