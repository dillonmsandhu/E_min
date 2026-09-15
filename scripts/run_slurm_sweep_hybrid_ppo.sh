#!/bin/bash
#SBATCH --job-name=sweep_hybrid_ppo
#SBATCH --output=slurm/%j.out
#SBATCH --time=12:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Hybrid PPO Control Algorithms
# Compares E minimization vs TD(lambda) vs MC for exact value function
# optimization while using sampled GAE rollouts for the policy.
#
# Environments: FourRooms-misc and MountainCar-v0
# Optimization Metric: AUC of mean reward (higher is better)
#
# Usage:
#   sbatch scripts/run_slurm_sweep_hybrid_ppo.sh
#   ./scripts/run_slurm_sweep_hybrid_ppo.sh (for local test)
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
N_SEEDS=10
TOTAL_TIMESTEPS=2048000
ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
HYBRID_ALGOS=("hybrid_exact_E" "hybrid_exact_td_lambda")

FIXED_GAE_LAMBDA=0.9
# Grids (2 critic LRs, 2 actor LRs, fixed lambda=0.9 -> 4 configs per seed)
LR_GRID="0.01 0.005 0.001 0.0003"
ACTOR_LR_GRID="0.001 0.0003 0.0001"
VALUE_LAMBDA_GRID="0.9 0.99 1.0"
CONFIG="{\"GAE_LAMBDA\": $FIXED_GAE_LAMBDA, \"NUM_STEPS\": 256, \"NUM_ENVS\": 64, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_EPOCHS\": 4, \"LIGHT_METRICS\": true}"
RANK_BY="final_window"
WINDOW_SIZE=500

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM HYBRID PPO / CONTROL ALGORITHMS SWEEP"
echo "Comparing: Hybrid E minimization vs TD(lambda) vs MC"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Algorithms: ${HYBRID_ALGOS[*]}"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "Critic LR Grid: $LR_GRID | Actor LR Grid: $ACTOR_LR_GRID | Lambda: $VALUE_LAMBDA_GRID"
echo "Optimization Metric: V_start (AUC, higher is better)"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SWEEP_ROOT_DIR="results/hybrid/sweeps/hybrid_${env}_${TIMESTAMP}_hybrid"
    mkdir -p "$SWEEP_ROOT_DIR"

    echo ""
    echo "======================================================================"
    echo "Running Hybrid PPO Control Sweep: Environment=$env"
    echo "Sweep Root Directory: $SWEEP_ROOT_DIR"
    echo "======================================================================"
    
    # 1. Hybrid Exact E minimization (sweeps LR x ACTOR_LR)
    echo ""
    echo "--> [1/2] Sweeping hybrid_exact_E (LR: $LR_GRID | ACTOR_LR: $ACTOR_LR_GRID)..."
    CMD_E="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos hybrid_exact_E \
        --lr-grid $LR_GRID \
        --actor-lr-grid $ACTOR_LR_GRID \
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

    # 2. Hybrid Exact TD(lambda) (sweeps LR x ACTOR_LR x VALUE_LAMBDA)
    echo ""
    echo "--> [2/2] Sweeping hybrid_exact_td_lambda (LR: $LR_GRID | ACTOR_LR: $ACTOR_LR_GRID | VALUE_LAMBDA: $VALUE_LAMBDA_GRID)..."
    CMD_TD="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos hybrid_exact_td_lambda \
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
        --window-size $WINDOW_SIZE \
        --higher-is-better \
        --linear-scale
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Hybrid PPO Control Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
