#!/bin/bash
# ==============================================================================
# Multi-GPU SLURM Lambda Sweep Launcher
#
# Loops over environments, policies, and algorithms and submits an independent
# SLURM job for each combination.
# Each job sweeps over multiple VALUE_LAMBDA values.
# Finally, dispatches a diagnostic job that depends on all sweep jobs.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Config
DEFAULT_ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
DEFAULT_POLICIES=("fixed" "random" "ppo")
DEFAULT_ALGOS=("exact_td_lambda" "exact_E_lambda")

PARTITION="compsci-gpu"
TIME_LIMIT="12:00:00"
GPU_GRES="gpu:a5000:1"
WORKER_SCRIPT="scripts/lambda_sweep/lambda_sweep_worker.sh"
DIAGNOSTICS_SCRIPT="scripts/lambda_sweep/lambda_sweep_diagnostics.sh"

mkdir -p slurm

DRY_RUN=false
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --sweep-id=*)
            SWEEP_ID="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--sweep-id=ID]"
            exit 0
            ;;
    esac
done

echo "======================================================================"
echo "LAMBDA SWEEP DISPATCHER"
echo "Sweep ID: $SWEEP_ID"
if [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY-RUN"
else
    echo "Mode: SLURM PARALLEL"
fi
echo "======================================================================"

JOB_IDS=""
SUBMITTED_COUNT=0
MAX_CONCURRENT=10
declare -a SUBMITTED_JOBS

for env in "${DEFAULT_ENVS[@]}"; do
    for policy in "${DEFAULT_POLICIES[@]}"; do
        for algo in "${DEFAULT_ALGOS[@]}"; do
            
            JOB_NAME="lswp_${env}_${policy}_${algo}"
            LOG_OUT="slurm/%j_lswp_${env}_${policy}_${algo}.out"
            
            DEP=""
            if [ $SUBMITTED_COUNT -ge $MAX_CONCURRENT ]; then
                DEP_JOB_IDX=$((SUBMITTED_COUNT - MAX_CONCURRENT))
                DEP_JOB_ID=${SUBMITTED_JOBS[$DEP_JOB_IDX]}
                DEP="--dependency=afterany:$DEP_JOB_ID"
            fi
            
            SBATCH_CMD="sbatch \
                $DEP \
                --job-name=\"$JOB_NAME\" \
                --output=\"$LOG_OUT\" \
                --time=\"$TIME_LIMIT\" \
                --partition=\"$PARTITION\" \
                --gres=\"$GPU_GRES\" \
                \"$WORKER_SCRIPT\" \"$env\" \"$policy\" \"$algo\" \"$SWEEP_ID\""
            
            echo "--> Submitting: Env=$env, Policy=$policy, Algo=$algo"
            
            if [ "$DRY_RUN" = true ]; then
                echo "    [DRY-RUN] Command: $SBATCH_CMD"
            else
                JOB_OUTPUT=$(eval "$SBATCH_CMD")
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 0 ]; then
                    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
                    echo "    Status: SUBMITTED (Job ID: $JOB_ID)"
                    if [ -z "$JOB_IDS" ]; then
                        JOB_IDS="$JOB_ID"
                    else
                        JOB_IDS="${JOB_IDS}:${JOB_ID}"
                    fi
                    SUBMITTED_JOBS+=($JOB_ID)
                    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
                else
                    echo "    Status: FAILED"
                fi
            fi
        done
    done
done

echo "======================================================================"
if [ "$DRY_RUN" = false ]; then
    echo "Successfully dispatched $SUBMITTED_COUNT SLURM jobs."
    
    # Dispatch Diagnostics Job
    if [ ! -z "$JOB_IDS" ]; then
        DIAG_JOB_NAME="lswp_diag_${SWEEP_ID}"
        DIAG_LOG_OUT="slurm/%j_lswp_diag_${SWEEP_ID}.out"
        # Diagnostics doesn't necessarily need a GPU, but if it runs model rollouts it does.
        # We'll request one GPU to be safe since it runs LIGHT_METRICS=False rollout.
        DIAG_CMD="sbatch \
            --dependency=afterany:$JOB_IDS \
            --job-name=\"$DIAG_JOB_NAME\" \
            --output=\"$DIAG_LOG_OUT\" \
            --time=\"04:00:00\" \
            --partition=\"$PARTITION\" \
            --gres=\"$GPU_GRES\" \
            \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\""
        
        echo "--> Submitting Diagnostics Job with dependency on $SUBMITTED_COUNT jobs..."
        DIAG_OUTPUT=$(eval "$DIAG_CMD")
        if [ $? -eq 0 ]; then
            DIAG_ID=$(echo "$DIAG_OUTPUT" | awk '{print $NF}')
            echo "    Diagnostics Job SUBMITTED (Job ID: $DIAG_ID)"
        else
            echo "    Failed to submit Diagnostics job."
        fi
    fi
fi
echo "======================================================================"
