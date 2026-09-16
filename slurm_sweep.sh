#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=slurm/%j.out
#SBATCH --time=4:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1 

# Run like:
# sbatch slurm_sweep.sh random_policy.exact_td cline_tune
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # start timer
FILE=$1
SUFFIX=$2

CONFIG="{\"TOTAL_TIMESTEPS\": 2000, \"NUM_ENVS\": 1, \"NUM_STEPS\": 1, \"NUM_EPOCHS\": 1, \"MINIBATCH_SIZE\": 1, \"ENV_NAME\": \"FourRooms-misc\", \"LOG_FEATURE_METRICS\": "False"}"

CMD="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python -m ${FILE} --run-suffix ${SUFFIX} --config '${CONFIG}' --sweep --n-seeds 3"
echo $CMD
eval $CMD

# Automatically generate seed plots when training finishes
echo "Generating seed plots..."
/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python plot_cline_tune.py

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
