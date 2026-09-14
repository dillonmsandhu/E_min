#!/bin/bash
#SBATCH --job-name=sweep_exact_ppo
#SBATCH --output=slurm/%j.out
#SBATCH --time=12:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Exact PPO Control Algorithms
# Compares E minimization vs TD(lambda) for policy improvement.
#
# Environments: FourRooms-misc and MountainCar-v0
# Optimization Metric: AUC of start-state value V_start (higher is better)
#
# Usage:
#   sbatch scripts/run_slurm_sweep_exact_ppo.sh
#   ./scripts/run_slurm_sweep_exact_ppo.sh (for local test)
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
N_SEEDS=16
TOTAL_TIMESTEPS=3000
ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
EXACT_ALGOS=("exact_E" "exact_td_lambda")


FIXED_GAE_LAMBDA=0.1
# Grids (2 critic LRs, 2 actor LRs, fixed lambda=0.9 -> 4 configs per seed)
LR_GRID="0.01 0.005 0.001 0.0003"
ACTOR_LR_GRID="0.001 0.0003 0.0001"
VALUE_LAMBDA_GRID="0.9 0.99 1.0"
RANK_BY="final_window"
WINDOW_SIZE=500

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM EXACT PPO / CONTROL ALGORITHMS SWEEP"
echo "Comparing: E minimization vs TD(lambda)"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Algorithms: ${EXACT_ALGOS[*]}"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "Critic LR Grid: $LR_GRID | Actor LR Grid: $ACTOR_LR_GRID | Lambda: $VALUE_LAMBDA_GRID"
echo "Optimization Metric: V_start (AUC, higher is better)"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SWEEP_ROOT_DIR="results/ppo/sweeps/ppo_${env}_${TIMESTAMP}_exact"
    mkdir -p "$SWEEP_ROOT_DIR"

    echo ""
    echo "======================================================================"
    echo "Running PPO Control Sweep: Environment=$env"
    echo "Sweep Root Directory: $SWEEP_ROOT_DIR"
    echo "======================================================================"
    
    # 1. Exact E minimization (sweeps LR x ACTOR_LR)
    echo ""
    echo "--> [1/2] Sweeping exact_E (LR: $LR_GRID | ACTOR_LR: $ACTOR_LR_GRID)..."
    CMD_E="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos exact_E \
        --lr-grid $LR_GRID \
        --actor-lr-grid $ACTOR_LR_GRID \
        --config '{\"GAE_LAMBDA\": $FIXED_GAE_LAMBDA}' \
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

    # 2. Exact TD(lambda) (sweeps LR x ACTOR_LR x VALUE_LAMBDA)
    echo ""
    echo "--> [2/2] Sweeping exact_td_lambda (LR: $LR_GRID | ACTOR_LR: $ACTOR_LR_GRID | VALUE_LAMBDA: $VALUE_LAMBDA_GRID)..."
    CMD_TD="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos exact_td_lambda \
        --lr-grid $LR_GRID \
        --actor-lr-grid $ACTOR_LR_GRID \
        --value-lambda-grid $VALUE_LAMBDA_GRID \
        --config '{\"GAE_LAMBDA\": $FIXED_GAE_LAMBDA}' \
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
        --window-size $WINDOW_SIZE \
        --higher-is-better \
        --linear-scale
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Exact PPO Control Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
