#!/bin/bash
#SBATCH --job-name=is_ppo_seeds
#SBATCH --output=slurm/%j.out
#SBATCH --time=1:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1 

# Ensure working directory is the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Run like:
# sbatch scripts/run_sweep.sh mc
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # start timer
SUFFIX=$1

# Hardcode your list of files here (without the .py extension if using -m)
FILES=(
    "random_policy.subset_td"
    "random_policy.subset_pfqi"
)

# Define batch sizes
BATCH_SIZES=(1 2 4 8 16 32 64 128)

# Base configuration template
BASE_CONFIG='{"TOTAL_TIMESTEPS": 1000, "NUM_ENVS": 1, "NUM_STEPS": 1, "NUM_EPOCHS": 4, "MINIBATCH_SIZE": 1, "ENV_NAME": "FourRooms-misc", "MODEL_LOAD_DIR": "250_steps_layer_norm", "FAIL_PROB": 0.15, "LIGHT_METRICS": true}'

# Nested loop: iterate over each file, then each batch size
for file in "${FILES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        echo "========================================"
        echo "Running: File=$file | Batch Size=$bs"
        echo "========================================"

        # Dynamically update the batch size parameter in the JSON config using jq
        CONFIG=$(echo "$BASE_CONFIG" | jq --argjson bs "$bs" '.BATCH_SIZE = $bs')

        # Update suffix to include both the file context (optional) and batch size
        # Replaces dots in the filename for clean suffix naming (e.g., ppo_ppo_lstd)
        FILE_TAG="${file//./_}"
        CURRENT_SUFFIX="${SUFFIX}_${FILE_TAG}_sweep_b_${bs}"

        CMD="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python -m ${file} --run-suffix ${CURRENT_SUFFIX} --config '${CONFIG}' --save-metrics"
        
        echo "$CMD"
        eval "$CMD"
        echo ""
    done
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
