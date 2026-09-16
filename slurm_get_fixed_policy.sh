#!/bin/bash
#SBATCH --job-name=be
#SBATCH --output=slurm/%j.out
#SBATCH --time=1:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1 

# Run like:
# sbatch run_slurm.sh ppo.ppo_lstd mc '{"GAE_LAMBDA": 1.0}'
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # start timer
FILE=ppo.ppo_ground_truth
SUFFIX=partial_four_rooms
# CONFIG=$3

CONFIG='{"TOTAL_TIMESTEPS": 250, "NUM_ENVS": 1, "NUM_STEPS": 1, "NUM_EPOCHS": 1, "MINIBATCH_SIZE": 1, "ENV_NAME": "FourRooms-misc", "MODEL_LOAD_DIR": "ground_truth/20260823_123519", "LOG_FEATURE_METRICS": "False", "ACTOR_LR": 0.001, "LR": 0.001}'

# CONFIG='{"TOTAL_TIMESTEPS": 262144000, "NUM_ENVS": 512, "NUM_STEPS": 512, "NUM_EPOCHS": 1, "MINIBATCH_SIZE": 8192, "ENV_NAME": "FourRooms-misc", "FAIL_PROB": 0.01, "GAE_LAMBDA": 0.0, "VALUE_LAMBDA": 0.0, "MODEL_LOAD_DIR": "250_steps_layer_norm", "LAPLACE_SMOOTHING_COEFF": 10.0}'

CMD="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python -m ${FILE} --run-suffix ${SUFFIX} --config '${CONFIG}' --save-checkpoint --save-video"
echo $CMD
eval $CMD

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
