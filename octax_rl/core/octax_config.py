# Base configuration for Octax RL
config = {
    # Environment & Seeds
    "ENV_NAME": "Octax-pong",
    "N_SEEDS": 5,
    "GAMMA": 0.99,
    "STICKY_ACTION_PROB": 0.0,
    "MAX_STEPS_IN_EPISODE": None,

    # Network Architecture (Octax paper CNN: 32(8x4, s4x2) -> 64(4x4, s2) -> 64(3x3, s1) -> 256)
    "NETWORK_TYPE": "octax_cnn",
    "SHARED_TORSO": False,
    "k": 256,
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
    "NUM_ENVS": 64,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5_000_000,
    "NUM_EPOCHS": 4,
    "ACTOR_EPOCHS": None,
    "CRITIC_EPOCHS": None,
    "MINIBATCH_SIZE": 1024,
    "ACTOR_MINIBATCH_SIZE": None,
    "CRITIC_MINIBATCH_SIZE": None,

    # Critic & PPO Hyperparameters
    "CRITIC_TYPE": "fitted",  # "fitted", "e_lambda", etc.
    "E_VARIANT": "aux",       # "aux" (uses v_aux), "e_aux", "fwd", "naive"
    "GAE_LAMBDA": 0.8,
    "VALUE_LAMBDA": 0.9,
    "TD_LAMBDA": 0.0,
    "E_LAMBDA": 0.8,
    "BSMC_LAMBDA": 0.8,
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

    # Rotation-Gated E(0) / E(lambda)
    "ROT_OMEGA": 1.0,

    # Contraction & Variance Diagnostics
    "LOG_CONTRACTION_DIAGNOSTICS": True,
    "DIAG_PROBE_ENVS": 16,
    "DIAG_EMA": 0.95,

    # Slurm & Metadata
    "SLURM_JOB_ID": None,
    "SLURM_ARRAY_JOB_ID": None,
    "SLURM_ARRAY_TASK_ID": None,
}
