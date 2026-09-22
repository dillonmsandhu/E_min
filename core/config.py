config = {
    # Environment & Seeds
    "ENV_NAME": "CartPole-v1",
    "N_SEEDS": 1,
    "GAMMA": 0.99,
    "NORMALIZE_OBS": False,

    # Network Architecture
    "NETWORK_TYPE": "mlp",
    "k": 16,
    "LAYER_NORM": False,

    # Optimization
    "LR": 0.0003,
    "LR_END": None,
    "ACTOR_LR": 0.0003,
    "ACTOR_LR_END": None,
    "WARMUP_RATIO": 0.1,
    "MAX_GRAD_NORM": 1.0,

    # Rollout & Training Loop
    "NUM_ENVS": 256,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1000000,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 1024,

    # PPO Hyperparameters
    "GAE_LAMBDA": 0.8,
    "VALUE_LAMBDA": 0.9,
    "E_LAMBDA": 0.8,
    "RETURN_LAMBDA": 0.99,
    "RECOMPUTE_TARGETS_EACH_EPOCH": False,
    "CLIP_EPS": 0.1,
    "VF_CLIP": 0.5,
    "VF_COEF": 0.5,
    "ENT_COEF": 0.01,
    "ADV_STD_FLOOR": 0.1,
    "ADV_CLIP": 3.0,

    # Slurm & Metadata
    "SLURM_JOB_ID": None,
    "SLURM_ARRAY_JOB_ID": None,
    "SLURM_ARRAY_TASK_ID": None,
}
