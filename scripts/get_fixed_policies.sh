#!/bin/bash
#SBATCH --job-name=be
#SBATCH --output=slurm/%j.out
#SBATCH --time=1:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1 

# Ensure working directory is the repository root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$REPO_ROOT"
fi
export PYTHONPATH="${PWD}:${PYTHONPATH}"

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # start timer
FILE=ppo.ground_truth
SUFFIX=fixed_policies

CONFIG='{"TOTAL_TIMESTEPS": 100, "NUM_ENVS": 1, "NUM_STEPS": 1, "NUM_EPOCHS": 1, "MINIBATCH_SIZE": 1, "LOG_FEATURE_METRICS": "False", "LIGHT_METRICS": true, "ACTOR_LR": 0.001}'

CMD="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python -m ${FILE} --run-suffix ${SUFFIX} --config '${CONFIG}' --save-metrics --save-video --save-checkpoint --env-ids FourRooms-misc EightRooms MountainCar-v0 Whirlpool"
echo $CMD
eval $CMD

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
