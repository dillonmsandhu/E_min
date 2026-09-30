#!/bin/bash
#SBATCH --job-name=sweep_minatar_lambdas
#SBATCH --output=slurm/%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-3

# ==============================================================================
# MinAtar Suite Sweep: RETURN_LAMBDA vs. VALUE_LAMBDA / E_LAMBDA
#
# Compares the 3 key algorithms across return and bootstrapping lambdas:
#   1. E (E0 base script): swept over RETURN_LAMBDA (baseline return anchor G_t)
#   2. E_lambda algorithm (E_lambda_geometric or E_lambda_differentiable):
#      Configurable via E_LAMBDA_ALGO env var or 2nd CLI argument (default: E_lambda_geometric).
#      Swept over RETURN_LAMBDA x E_LAMBDA grid.
#   3. ppo (TD(lambda) baseline): swept over VALUE_LAMBDA
#
# Environments: 4 MinAtar Games
#   0: Asterix-MinAtar
#   1: Breakout-MinAtar
#   2: Freeway-MinAtar
#   3: SpaceInvaders-MinAtar
#
# Usage:
#   sbatch scripts/e_variants/run_slurm_minatar_lambda_sweep.sh
#   E_LAMBDA_ALGO=E_lambda_differentiable sbatch scripts/e_variants/run_slurm_minatar_lambda_sweep.sh
#   ./scripts/e_variants/run_slurm_minatar_lambda_sweep.sh Asterix-MinAtar E_lambda_geometric
# ==============================================================================

# Ensure we are in the repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Python environment
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/shims/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/shims/python"
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

# Select E_lambda algorithm variant (default: E_lambda_geometric)
E_LAMBDA_ALGO="${2:-${E_LAMBDA_ALGO:-E_lambda_geometric}}"

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: ENV_NAME is empty. SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds for ALL_ENVS (size ${#ALL_ENVS[@]})."
    exit 1
fi

N_SEEDS=8
TOTAL_TIMESTEPS=10000000  # 10M env steps

RANK_BY="final_window"
WINDOW_SIZE=500
METRIC="returned_discounted_episode_returns"

# Fixed network and policy hyperparameters
FIXED_ACTOR_LR=0.003
ACTOR_LR_END=0.0001
FIXED_GAE_LAMBDA=0.8
K_DIM=64

# Swept lambda grids
RETURN_LAMBDA_GRID="0.9 0.95 0.99 1.0"
E_LAMBDA_GRID="0.0 0.6 0.8 0.95 0.99"
TD_LAMBDA_GRID="0.0 0.6 0.8 0.9 0.95 0.99 1.0"

# Critic learning rate (default 0.001; can be expanded to e.g. "0.003 0.001")
CRITIC_LR_GRID="0.001 0.0005"

# Base configuration with Slurm tracking
CONFIG="{\"NUM_ENVS\": 256, \"NUM_STEPS\": 64, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_EPOCHS\": 4, \"GAE_LAMBDA\": $FIXED_GAE_LAMBDA, \"VF_CLIP\": 1000000.0, \"k\": $K_DIM, \"ENT_COEF\": 0.001, \"LAYER_NORM\": \"True\", \"SLURM_JOB_ID\": \"${SLURM_JOB_ID:-local}\", \"SLURM_ARRAY_JOB_ID\": \"${SLURM_ARRAY_JOB_ID:-local}\", \"SLURM_ARRAY_TASK_ID\": \"${SLURM_ARRAY_TASK_ID:-0}\", \"ACTOR_LR_END\": $ACTOR_LR_END}"

mkdir -p slurm

# Unified suite directory:
# When running as an sbatch array, all 4 tasks share $SLURM_ARRAY_JOB_ID.
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-${SUITE_TAG:-minatar_lambdas_$(date +"%Y%m%d_%H%M%S")}}}"
SWEEP_SUITE_DIR="results/ppo/sweeps/suite_${SUITE_ID}"
SWEEP_ROOT_DIR="${SWEEP_SUITE_DIR}/${ENV_NAME}"
mkdir -p "$SWEEP_ROOT_DIR"

echo "======================================================================"
echo "STARTING MINATAR LAMBDA SWEEP: RETURN_LAMBDA vs. VALUE_LAMBDA / E_LAMBDA"
echo "Start Time: $START_TIME"
echo "Environment: $ENV_NAME"
echo "E_lambda Algorithm: $E_LAMBDA_ALGO"
echo "Seeds: $N_SEEDS | Total Timesteps: $TOTAL_TIMESTEPS (10M)"
echo "Critic LR Grid: $CRITIC_LR_GRID | Fixed Actor LR: $FIXED_ACTOR_LR"
echo "RETURN_LAMBDA Grid (E and $E_LAMBDA_ALGO): $RETURN_LAMBDA_GRID"
echo "E_LAMBDA Grid ($E_LAMBDA_ALGO): $E_LAMBDA_GRID"
echo "VALUE_LAMBDA Grid (TD / PPO): $TD_LAMBDA_GRID"
echo "Output Directory: $SWEEP_ROOT_DIR"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Sweep E (E0 base script) across RETURN_LAMBDA (and critic LR)
# ------------------------------------------------------------------------------
echo ""
echo "--> [1/3] Sweeping E (E0 1-step) across RETURN_LAMBDA: $RETURN_LAMBDA_GRID..."
$PYTHON scripts/pipeline/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos E \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --return-lambda-grid $RETURN_LAMBDA_GRID \
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
# 2. Sweep E_lambda algorithm across RETURN_LAMBDA x E_LAMBDA
# ------------------------------------------------------------------------------
echo ""
echo "--> [2/3] Sweeping $E_LAMBDA_ALGO across RETURN_LAMBDA: $RETURN_LAMBDA_GRID x E_LAMBDA: $E_LAMBDA_GRID..."
$PYTHON scripts/pipeline/sweep_pipeline.py \
    --policy ppo \
    --env-name "$ENV_NAME" \
    --algos "$E_LAMBDA_ALGO" \
    --lr-grid $CRITIC_LR_GRID \
    --actor-lr-grid $FIXED_ACTOR_LR \
    --return-lambda-grid $RETURN_LAMBDA_GRID \
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
# 3. Sweep ppo / TD(lambda) baseline across VALUE_LAMBDA
# ------------------------------------------------------------------------------
echo ""
echo "--> [3/3] Sweeping ppo / TD(lambda) baseline across VALUE_LAMBDA: $TD_LAMBDA_GRID..."
$PYTHON scripts/pipeline/sweep_pipeline.py \
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
# 4. Generate 3-Way Comparison Plot (Learning Curves + Lambda Scaling)
# ------------------------------------------------------------------------------
echo ""
echo "--> Generating Comparison Figure for $ENV_NAME..."
$PYTHON scripts/e_variants/plot_e_variants_comparison.py \
    --sweep-dir "$SWEEP_ROOT_DIR" \
    --metric "$METRIC" \
    --rank-by "$RANK_BY" \
    --window-size $WINDOW_SIZE

# Mark this environment as 100% finished
touch "${SWEEP_ROOT_DIR}/.env_complete"

# ------------------------------------------------------------------------------
# 5. Check Suite Completion: Compile Multi-Page PDF Once All 4 Tasks Finish
# ------------------------------------------------------------------------------
COMPLETED_COUNT=0
for E in "${ALL_ENVS[@]}"; do
    if [ -f "${SWEEP_SUITE_DIR}/${E}/.env_complete" ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    fi
done

echo ""
echo "MinAtar Suite Progress: $COMPLETED_COUNT / ${#ALL_ENVS[@]} environments finished."

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Running in a SLURM array job
    if [ "$COMPLETED_COUNT" -eq "${#ALL_ENVS[@]}" ]; then
        # Use atomic directory lock so only the FIRST task to notice completion sends the email
        if mkdir "${SWEEP_SUITE_DIR}/.suite_email_lock" 2>/dev/null; then
            echo "======================================================================"
            echo "ALL ${#ALL_ENVS[@]} MINATAR ENVIRONMENTS COMPLETE! Compiling and emailing PDF..."
            echo "======================================================================"
            $PYTHON scripts/e_variants/generate_e_variants_suite_pdf.py \
                --suite-dir "$SWEEP_SUITE_DIR" \
                --metric "$METRIC" \
                --rank-by "$RANK_BY" \
                --window-size $WINDOW_SIZE \
                --email "$EMAIL_RECIPIENT"
        else
            echo "Complete suite PDF already compiled and emailed by another finished task."
        fi
    else
        # Waiting for other array tasks to finish: compile local PDF without emailing
        $PYTHON scripts/e_variants/generate_e_variants_suite_pdf.py \
            --suite-dir "$SWEEP_SUITE_DIR" \
            --metric "$METRIC" \
            --rank-by "$RANK_BY" \
            --window-size $WINDOW_SIZE
    fi
else
    # Single environment run from CLI: generate local PDF without emailing
    echo "Single environment run finished. Generating local PDF..."
    $PYTHON scripts/e_variants/generate_e_variants_suite_pdf.py \
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
