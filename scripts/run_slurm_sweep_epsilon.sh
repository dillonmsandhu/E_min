#!/bin/bash
#SBATCH --job-name=sweep_epsilon
#SBATCH --output=slurm/%j.out
#SBATCH --time=4:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Epsilon values for Sampled Algorithms 
# ==============================================================================

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

N_SEEDS=5
ENVS=("FourRooms-misc" "MountainCar-v0")
SAMPLED_ALGOS=("td" "unbiased_sampled_E")
EPSILON_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
CONFIG='{"NUM_ENVS": 64, "NUM_STEPS": 256, "TOTAL_TIMESTEPS": 1000000, "MINIBATCH_SIZE": 1024, "NUM_EPOCHS": 1, "FAIL_PROB": 0.0}'

LR_GRID="0.001 0.0005 0.0001 0.00001"
LAMBDA_GRID="0.8 0.95 1"

mkdir -p slurm

for env in "${ENVS[@]}"; do
    MASTER_DIR="results/fixed/epsilon_sweeps/sampled_${env}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$MASTER_DIR"
    
    echo "======================================================================"
    echo "STARTING SAMPLED EPSILON SWEEP: $env"
    echo "Master Directory: $MASTER_DIR"
    echo "======================================================================"

    for epsilon in "${EPSILON_VALUES[@]}"; do
        EPS_DIR="$MASTER_DIR/eps_$epsilon"
        mkdir -p "$EPS_DIR"

        # 1. unbiased_sampled_E (LR only)
        CMD_E="$PYTHON scripts/sweep_pipeline.py \
            --policy fixed \
            --env-name $env \
            --algos unbiased_sampled_E \
            --lr-grid $LR_GRID \
            --n-seeds $N_SEEDS \
            --config '$CONFIG' \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_advantage_cossim_uniform \
            --use-geom-mean \
            --use-greedy-policy \
            --sweep-root-dir $EPS_DIR \
            --policy-epsilon $epsilon"
        eval "$CMD_E"

        # 2. td (LR + lambda)
        CMD_TD="$PYTHON scripts/sweep_pipeline.py \
            --policy fixed \
            --env-name $env \
            --algos td \
            --lr-grid $LR_GRID \
            --lambda-grid $LAMBDA_GRID \
            --n-seeds $N_SEEDS \
            --config '$CONFIG' \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_advantage_cossim_uniform \
            --use-geom-mean \
            --use-greedy-policy \
            --sweep-root-dir $EPS_DIR \
            --policy-epsilon $epsilon"
        eval "$CMD_TD"
    done
    
    echo "Generating plot for $env..."
    $PYTHON scripts/plot_epsilon_sweep.py \
        --results-dir "$MASTER_DIR" \
        --env-name "$env" \
        --metric "nn_advantage_cossim_uniform" \
        --rank-order higher \
        --save-path "$MASTER_DIR/epsilon_sweep.png" \
        --algos ${SAMPLED_ALGOS[*]}
done
