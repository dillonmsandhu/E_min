# Base configuration for Gymnax RL
config = {
    # Environment & Seeds
    "ENV_NAME": "CartPole-v1",
    "N_SEEDS": 1,
    "GAMMA": 0.99,
    "NORMALIZE_OBS": False,
    "POTENTIAL_SCALE": 30.0,
    "SLIP_PROB": 0.0,
    "SLIP_FORCE_SCALE": 0.0,
    "ACTION_NOISE_STD": 0.0,
    "TRANSITION_NOISE": 0.0,
    "DENSE_REWARD": False,
    "FULLY_OBSERVABLE": True,
    "PENALTY_ON_HIT": False,
    "HIT_PENALTY": 1.0,
    "STICKY_ACTION_PROB": 0.0,
    "INSTANT_BULLETS": False,

    # Network Architecture
    "NETWORK_TYPE": "mlp",
    "SHARED_TORSO": False,
    "USE_VISUAL_OBS": True,
    "k": 16,
    "LAYER_NORM": False,
    "NUM_VALUE_HEADS": 1,
    "VALUE_HEAD_AGG": "sum",

    # Optimization
    "LR": 0.0003,
    "LR_END": None,
    "ACTOR_LR": 0.0003,
    "ACTOR_LR_END": None,
    "CRITIC_LR": 0.0003,
    "CRITIC_LR_END": None,
    "WARMUP_RATIO": 0.1,
    "MAX_GRAD_NORM": 1.0,
    "WEIGHT_DECAY": 0.01,
    "ACTOR_WEIGHT_DECAY": None,
    "CRITIC_WEIGHT_DECAY": None,
    "ADAM_EPS": 1e-5,

    # Rollout & Training Loop
    "NUM_ENVS": 256,
    "NUM_STEPS": 64,
    "TOTAL_TIMESTEPS": 1000000,
    "NUM_EPOCHS": 4,
    "ACTOR_EPOCHS": None,
    "CRITIC_EPOCHS": None,
    "MINIBATCH_SIZE": 1024,
    "ACTOR_MINIBATCH_SIZE": None,
    "CRITIC_MINIBATCH_SIZE": None,

    # Critic & PPO Hyperparameters
    "CRITIC_TYPE": "fitted",  # "fitted", "td_0", "td_lambda", "e_0", "e_0_basic", "e_lambda", "mc"
    "E_VARIANT": "aux",       # "aux" (uses v_aux), "fwd" (forward-only), "naive" (symmetric without v_aux)
    "GAE_LAMBDA": 0.8,
    "VALUE_LAMBDA": 0.9,
    "TD_LAMBDA": 0.0,
    "E_LAMBDA": 0.8,
    "RETURN_LAMBDA": 1.0,
    "AUX_LAMBDA": 1.0,
    "AUX_VF_COEF": 1.0,
    "RECOMPUTE_TARGETS_EACH_EPOCH": False,
    "CLIP_EPS": 0.1,
    "VF_CLIP": 1e6,
    "VF_COEF": 0.5,
    "POLICY_COEFF": 1.0,
    "ENT_COEF": 0.01,
    "ADV_STD_FLOOR": 0.1,
    "ADV_CLIP": 3.0,

    # Slurm & Metadata
    "SLURM_JOB_ID": None,
    "SLURM_ARRAY_JOB_ID": None,
    "SLURM_ARRAY_TASK_ID": None,
}
