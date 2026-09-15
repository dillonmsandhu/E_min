#!/bin/bash
# ==============================================================================
# Multi-GPU SLURM Sweep Launcher for Exact PPO Control Algorithms
#
# Loops over environments and submits an independent 1-GPU SLURM job per
# environment, enabling high-throughput parallel execution across GPUs.
#
# Default Environments:
#   - EightRooms
#   - FourRooms-misc
#   - Whirlpool
#   - MountainCar-v0
#
# Usage:
#   # 1. Launch 1 GPU per environment across all default environments:
#   ./scripts/launch_slurm_sweep_exact_ppo.sh
#
#   # 2. Launch for specific environments only:
#   ./scripts/launch_slurm_sweep_exact_ppo.sh MountainCar-v0 FourRooms-misc
#
#   # 3. Dry-run to preview sbatch commands without submitting:
#   ./scripts/launch_slurm_sweep_exact_ppo.sh --dry-run
#
#   # 4. Local execution test (runs sequentially on local machine):
#   ./scripts/launch_slurm_sweep_exact_ppo.sh --local MountainCar-v0
# ==============================================================================

# Ensure working directory is repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default environments to sweep
DEFAULT_ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")

# SLURM Configuration
PARTITION="compsci-gpu"
TIME_LIMIT="12:00:00"
GPU_GRES="gpu:a5000:1"
WORKER_SCRIPT="scripts/run_slurm_sweep_exact_ppo.sh"

# Ensure output log directory exists
mkdir -p slurm

# Parse CLI options
DRY_RUN=false
LOCAL_RUN=false
TARGET_ENVS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --local)
            LOCAL_RUN=true
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--local] [env1 env2 ...]"
            echo ""
            echo "Options:"
            echo "  --dry-run     Print sbatch commands without submitting"
            echo "  --local       Run sweeps locally without SLURM (sequential)"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Default environments: ${DEFAULT_ENVS[*]}"
            exit 0
            ;;
        *)
            TARGET_ENVS+=("$arg")
            ;;
    esac
done

# If no specific environments passed, use default list
if [ ${#TARGET_ENVS[@]} -eq 0 ]; then
    TARGET_ENVS=("${DEFAULT_ENVS[@]}")
fi

echo "======================================================================"
echo "EXACT PPO MULTI-GPU SWEEP DISPATCHER"
echo "Target Environments (${#TARGET_ENVS[@]}): ${TARGET_ENVS[*]}"
echo "Mode: $(if [ "$LOCAL_RUN" = true ]; then echo "LOCAL (sequential)"; elif [ "$DRY_RUN" = true ]; then echo "DRY-RUN (preview only)"; else echo "SLURM PARALLEL (1 GPU per env)"; fi)"
if [ "$LOCAL_RUN" = false ]; then
    echo "SLURM Partition: $PARTITION | GPU per Job: $GPU_GRES | Time: $TIME_LIMIT"
fi
echo "Worker Script: $WORKER_SCRIPT"
echo "======================================================================"

SUBMITTED_COUNT=0

for env in "${TARGET_ENVS[@]}"; do
    JOB_NAME="sweep_exact_ppo_${env}"
    LOG_OUT="slurm/%j_${env}.out"
    
    if [ "$LOCAL_RUN" = true ]; then
        echo ""
        echo ">>> [LOCAL] Running exact PPO sweep for: $env"
        bash "$WORKER_SCRIPT" "$env"
    else
        SBATCH_CMD="sbatch \
            --job-name=\"$JOB_NAME\" \
            --output=\"$LOG_OUT\" \
            --time=\"$TIME_LIMIT\" \
            --partition=\"$PARTITION\" \
            --gres=\"$GPU_GRES\" \
            \"$WORKER_SCRIPT\" \"$env\""
        
        echo ""
        echo "--> Submitting SLURM job for environment: $env"
        echo "    Job Name : $JOB_NAME"
        echo "    Log File : $LOG_OUT"
        
        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY-RUN] Command: $SBATCH_CMD"
        else
            JOB_OUTPUT=$(eval "$SBATCH_CMD")
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
                echo "    Status   : SUBMITTED (Job ID: $JOB_ID)"
                SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
            else
                echo "    Status   : FAILED to submit (Exit code: $EXIT_CODE)"
            fi
        fi
    fi
done

echo ""
echo "======================================================================"
if [ "$LOCAL_RUN" = true ]; then
    echo "All local sweep runs completed."
elif [ "$DRY_RUN" = true ]; then
    echo "Dry-run complete. ${#TARGET_ENVS[@]} jobs ready to submit."
else
    echo "Successfully dispatched $SUBMITTED_COUNT / ${#TARGET_ENVS[@]} SLURM jobs."
    echo "Monitor jobs with: squeue -u \$USER"
    echo "View logs in: slurm/<job_id>_<env>.out"
fi
echo "======================================================================"
