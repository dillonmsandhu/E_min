#!/bin/bash
#SBATCH --job-name=sweep_sampled_ppo
#SBATCH --output=slurm/%j.out
#SBATCH --time=12:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Sampled PPO Control Algorithms
# Compares: Sampled E Minimization vs TD(lambda)
# Sweeping over Value lambda: [0.9, 0.99, 1.0] for E min and TD(lambda)
# Note: VALUE_LAMBDA and GAE_LAMBDA are strictly separated across all runs.
#
# Environments: EightRooms, FourRooms-misc, Whirlpool, MountainCar-v0
# Optimization Metric: V_start (AUC, higher is better)
#
# Usage:
#   sbatch scripts/run_slurm_sweep_sampled_ppo.sh
#   ./scripts/run_slurm_sweep_sampled_ppo.sh (for local execution)
# ==============================================================================
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Python environment (defaults to cluster gymnax environment if present)
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

# Configuration
N_SEEDS=8

$RANK_BY="final_window"
$WINDOW_SIZE=500

TOTAL_TIMESTEPS=2048000
ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
SAMPLED_ALGOS=("sampled_E" "sampled_td_lambda")

# Fixed GAE lambda for policy advantage estimation (held separate from value lambda)
FIXED_GAE_LAMBDA=0.9 # higher GAE lambda to help the policy get somehwere... 

# Hyperparameter Grids
LR_GRID="0.005 0.001 0.0003"
ACTOR_LR_GRID="0.001 0.0003"
VALUE_LAMBDA_GRID="0.9 0.99 1.0" # also sweeps the value lambda for sampled E

# Base configuration overrides (VALUE_LAMBDA swept via grid; GAE_LAMBDA kept separate)
CONFIG="{\"NUM_ENVS\": 64, \"NUM_STEPS\": 256, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_EPOCHS\": 4, \"GAE_LAMBDA\": $FIXED_GAE_LAMBDA}"

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM SAMPLED PPO CONTROL SWEEP"
echo "Comparing: Sampled E Minimization vs TD(lambda)"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Algorithms: ${SAMPLED_ALGOS[*]}"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "Critic LR Grid: $LR_GRID | Actor LR Grid: $ACTOR_LR_GRID"
echo "Value Lambda Grid: $VALUE_LAMBDA_GRID"
echo "Fixed GAE Lambda: $FIXED_GAE_LAMBDA"
echo "Base Config: $CONFIG"
echo "Optimization Metric: V_start (AUC, higher is better)"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SWEEP_ROOT_DIR="results/ppo/sweeps/ppo_${env}_${TIMESTAMP}_sampled"
    mkdir -p "$SWEEP_ROOT_DIR"

    echo ""
    echo "======================================================================"
    echo "Running Sampled PPO Control Sweep: Environment=$env"
    echo "Sweep Root Directory: $SWEEP_ROOT_DIR"
    echo "======================================================================"
    
    # 1. Sampled E minimization (sweeps LR x ACTOR_LR x VALUE_LAMBDA)
    echo ""
    echo "--> [1/2] Sweeping sampled_E (LR: $LR_GRID | ACTOR_LR: $ACTOR_LR_GRID | VALUE_LAMBDA: $VALUE_LAMBDA_GRID)..."
    CMD_E="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos sampled_E \
        --lr-grid $LR_GRID \
        --actor-lr-grid $ACTOR_LR_GRID \
        --value-lambda-grid $VALUE_LAMBDA_GRID \
        --config '$CONFIG' \
        --n-seeds $N_SEEDS \
        --total-timesteps $TOTAL_TIMESTEPS \
        --metric V_start \
        --rank-by $RANK_BY \
        --window-size $WINDOW_SIZE \
        --higher-is-better \
        --sweep-root-dir $SWEEP_ROOT_DIR \
        --no-log-scale"
    echo "Command: $CMD_E"
    eval "$CMD_E"

    # 2. Sampled TD(lambda) (sweeps LR x ACTOR_LR x VALUE_LAMBDA)
    echo ""
    echo "--> [2/2] Sweeping sampled_td_lambda (LR: $LR_GRID | ACTOR_LR: $ACTOR_LR_GRID | VALUE_LAMBDA: $VALUE_LAMBDA_GRID)..."
    CMD_TD="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos sampled_td_lambda \
        --lr-grid $LR_GRID \
        --actor-lr-grid $ACTOR_LR_GRID \
        --value-lambda-grid $VALUE_LAMBDA_GRID \
        --config '$CONFIG' \
        --n-seeds $N_SEEDS \
        --total-timesteps $TOTAL_TIMESTEPS \
        --metric V_start \
        --rank-by $RANK_BY \
        --window-size $WINDOW_SIZE \
        --higher-is-better \
        --sweep-root-dir $SWEEP_ROOT_DIR \
        --no-log-scale"
    echo "Command: $CMD_TD"
    eval "$CMD_TD"

    # 3. Final Cross-Algorithm Comparison
    echo ""
    echo "--> Generating Final Cross-Algorithm Comparison Plot & Summary..."
    $PYTHON notebooks/analyze_sweeps.py \
        --sweep-dir "$SWEEP_ROOT_DIR" \
        --metric V_start \
        --rank-by $RANK_BY \
        --final-window $WINDOW_SIZE \
        --higher-is-better \
        --linear-scale
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Sampled PPO Control Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
