#!/bin/bash
#SBATCH --job-name=sweep_exact
#SBATCH --output=slurm/%j.out
#SBATCH --time=4:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Exact Algorithms (TD, MC, E_gd, TD_lambda)
# Runs for both Fixed Policy and Random Policy on MountainCar and FourRooms.
#
# Usage:
#   sbatch scripts/run_slurm_sweep_exact.sh
#   ./scripts/run_slurm_sweep_exact.sh (for local test)
# ==============================================================================
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Python environment (defaults to cluster gymnax environment)
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

# Configuration
N_SEEDS=20
TOTAL_TIMESTEPS=2000
ENVS=("FourRooms-misc" "MountainCar-v0")
POLICIES=("random" "fixed")

EXACT_NON_LAMBDA_ALGOS=("exact_td" "exact_mc" "exact_E_gd" "exact_E_td" "exact_E_sampling_form")
EXACT_LAMBDA_ALGOS=("exact_td_lambda")

LR_GRID="0.01 0.005 0.001 0.0005 0.0001"
VALUE_LAMBDA_GRID="0.5 0.9 0.95 1.0"

# Per-environment evaluation policy placeholders for fixed policy evaluation.
# Replace with your trained policy run directories for each environment (e.g. "ground_truth/20260821_164541" or "short_run").
# If left as PLACEHOLDER, the pipeline will auto-resolve to the latest available trained checkpoint for that environment.
declare -A FIXED_MODEL_DIRS=(
    ["FourRooms-misc"]="ground_truth/20260823_122419"
    ["MountainCar-v0"]="ground_truth/20260823_123519"
)

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM EXACT ALGORITHMS SWEEP"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Policies: ${POLICIES[*]}"
echo "Non-Lambda Algorithms: ${EXACT_NON_LAMBDA_ALGOS[*]}"
echo "Lambda Algorithms: ${EXACT_LAMBDA_ALGOS[*]}"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "LR Grid: $LR_GRID | Value Lambda Grid: $VALUE_LAMBDA_GRID"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    MODEL_DIR="${FIXED_MODEL_DIRS[$env]}"

    for policy in "${POLICIES[@]}"; do
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        SWEEP_ROOT_DIR="results/${policy}/sweeps/${policy}_${env}_${TIMESTAMP}_exact"
        mkdir -p "$SWEEP_ROOT_DIR"

        echo ""
        echo "======================================================================"
        echo "Running Sweep: Policy=$policy | Environment=$env"
        echo "Sweep Root Directory: $SWEEP_ROOT_DIR"
        if [ "$policy" = "fixed" ]; then
            echo "Evaluation Policy Model Dir: $MODEL_DIR"
        fi
        echo "======================================================================"

        # 1. Non-lambda algorithms (sweep LR only)
        echo ""
        echo "--> [1/2] Sweeping non-lambda exact algorithms (${EXACT_NON_LAMBDA_ALGOS[*]})..."
        CMD_NON_LAMBDA="$PYTHON scripts/sweep_pipeline.py \
            --policy $policy \
            --env-name $env \
            --algos ${EXACT_NON_LAMBDA_ALGOS[*]} \
            --lr-grid $LR_GRID \
            --n-seeds $N_SEEDS \
            --total-timesteps $TOTAL_TIMESTEPS \
            --model-dir '$MODEL_DIR' \
            --use-geom-mean \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_greedy_performance \
            --sweep-root-dir $SWEEP_ROOT_DIR"
        echo "Command: $CMD_NON_LAMBDA"
        eval "$CMD_NON_LAMBDA"

        # 2. Lambda algorithm: exact_td_lambda (sweeps LR x VALUE_LAMBDA)
        echo ""
        echo "--> [2/2] Sweeping lambda exact algorithms (${EXACT_LAMBDA_ALGOS[*]})..."
        CMD_LAMBDA="$PYTHON scripts/sweep_pipeline.py \
            --policy $policy \
            --env-name $env \
            --algos ${EXACT_LAMBDA_ALGOS[*]} \
            --lr-grid $LR_GRID \
            --value-lambda-grid $VALUE_LAMBDA_GRID \
            --n-seeds $N_SEEDS \
            --total-timesteps $TOTAL_TIMESTEPS \
            --model-dir '$MODEL_DIR' \
            --use-geom-mean \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_greedy_performance \
            --sweep-root-dir $SWEEP_ROOT_DIR"
        echo "Command: $CMD_LAMBDA"
        eval "$CMD_LAMBDA"

        # 3. Final Cross-Algorithm Comparison
        echo ""
        echo "--> Generating Final Cross-Algorithm Comparison Plot & Summary..."
        $PYTHON notebooks/analyze_sweeps.py \
            --sweep-dir "$SWEEP_ROOT_DIR" \
            --metric nn_greedy_performance \
            --rank-by auc \
            --higher-is-better \
            --linear-scale
    done
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Exact Algorithms Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
