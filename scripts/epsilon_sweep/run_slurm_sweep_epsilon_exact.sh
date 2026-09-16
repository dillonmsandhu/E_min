#!/bin/bash
#SBATCH --job-name=sweep_epsilon
#SBATCH --output=slurm/%j.out
#SBATCH --time=4:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Epsilon values for Exact Algorithms 
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

N_SEEDS=1
ENVS=("FourRooms-misc" "MountainCar-v0")
EXACT_ALGOS=("exact_td_lambda" "exact_E_gd")
EPSILON_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
CONFIG='{"TOTAL_TIMESTEPS": 5000, "NUM_EPOCHS": 1, "FAIL_PROB": 0.0, "LIGHT_METRICS": true}'

LR_GRID="0.001 0.0005 0.0001"
LAMBDA_GRID="0.8 0.95 1"

mkdir -p slurm

for env in "${ENVS[@]}"; do
    MASTER_DIR="results/fixed/epsilon_sweeps/exact_${env}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$MASTER_DIR"
    
    echo "======================================================================"
    echo "STARTING EXACT EPSILON SWEEP: $env"
    echo "Master Directory: $MASTER_DIR"
    echo "======================================================================"

    for epsilon in "${EPSILON_VALUES[@]}"; do

        EPS_DIR="$MASTER_DIR/eps_$epsilon"
        mkdir -p "$EPS_DIR"

        # 1. exact_E_gd (LR only)
        CMD_E="$PYTHON $REPO_ROOT/scripts/sweep_pipeline.py \
            --policy fixed \
            --env-name $env \
            --algos exact_E_gd \
            --lr-grid $LR_GRID \
            --n-seeds $N_SEEDS \
            --config '$CONFIG' \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_advantage_cossim_uniform \
            --use-greedy-policy \
            --sweep-root-dir $EPS_DIR \
            --policy-epsilon $epsilon"
        eval "$CMD_E"

        # 2. exact_td_lambda (LR + lambda)
        CMD_TD="$PYTHON scripts/sweep_pipeline.py \
            --policy fixed \
            --env-name $env \
            --algos exact_td_lambda \
            --lr-grid $LR_GRID \
            --lambda-grid $LAMBDA_GRID \
            --n-seeds $N_SEEDS \
            --config '$CONFIG' \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_advantage_cossim_uniform \
            --use-greedy-policy \
            --sweep-root-dir $EPS_DIR \
            --policy-epsilon $epsilon"
        eval "$CMD_TD"
    done
    
    echo "Generating plot for $env..."
    $PYTHON scripts/epsilon_sweep/plot_epsilon_sweep.py \
        --results-dir "$MASTER_DIR" \
        --env-name "$env" \
        --metric "nn_advantage_cossim_uniform" \
        --rank-order higher \
        --save-path "$MASTER_DIR/epsilon_sweep.png" \
        --algos ${EXACT_ALGOS[*]}
done
