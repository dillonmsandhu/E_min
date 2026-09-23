#!/bin/bash
#SBATCH --job-name=be
#SBATCH --output=slurm/%j.out
#SBATCH --time=1:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1 

cd "$SLURM_SUBMIT_DIR"

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

FILE=$1
SUFFIX=$2
ENV=$3

CONFIG="{\"NUM_ENVS\": 128, \"NUM_STEPS\": 64, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": 1000000, \"NUM_EPOCHS\": 4, \"GAE_LAMBDA\": 0.8, \"RETURN_LAMBDA\": 0.99, \"VALUE_LAMBDA\": 0.8, \"k\": 64, \"ENT_COEF\": 0.001,\"LAYER_NORM\": \"True\", \"SLURM_JOB_ID\": \"${SLURM_JOB_ID:-local}\", \"SLURM_ARRAY_JOB_ID\": \"${SLURM_ARRAY_JOB_ID:-local}\", \"SLURM_ARRAY_TASK_ID\": \"${SLURM_ARRAY_TASK_ID:-0}\", \"ACTOR_LR_END\": 0.0001, \"ACTOR_LR\": 0.003, \"CRITIC_LR\": 0.003,  \"CRITIC_LR_END\": 0.0001 }"

CMD="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python ${FILE} --run-suffix ${SUFFIX} --config '${CONFIG}' --save-metrics --env-ids ${ENV} --save-video --save-checkpoint"
echo $CMD
eval $CMD

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
