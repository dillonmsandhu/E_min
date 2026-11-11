"""
gymnax_rl/core/ddpg_config.py

Dedicated default configuration for On-Policy Continuous DDPG on Gymnax environments,
strictly free of PPO-specific settings (CLIP_EPS, ENT_COEF, POLICY_COEFF,
GAE_LAMBDA, VF_CLIP, ADV_CLIP, ADV_STD_FLOOR).
"""

config = {
    # Environment & Seeds
    "ENV_NAME": "Pendulum-v1",
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

    # Network Architecture
    "NETWORK_TYPE": "mlp",
    "SHARED_TORSO": False,
    "k": 128,
    "LAYER_NORM": False,
    "NUM_VALUE_HEADS": 1,
    "VALUE_HEAD_AGG": "sum",

    # Optimization
    "LR": 3e-4,
    "LR_END": None,
    "ACTOR_LR": 3e-4,
    "ACTOR_LR_END": 3e-4,
    "CRITIC_LR": 3e-3,
    "CRITIC_LR_END": 3e-3,
    "WARMUP_RATIO": 0.1,
    "MAX_GRAD_NORM": 1.0,
    "WEIGHT_DECAY": 0.0,
    "ACTOR_WEIGHT_DECAY": None,
    "CRITIC_WEIGHT_DECAY": None,
    "ADAM_EPS": 1e-5,

    # Rollout & Training Loop
    "NUM_ENVS": 64,
    "NUM_STEPS": 64,
    "TOTAL_TIMESTEPS": 300_000,
    "NUM_EPOCHS": 4,
    "ACTOR_EPOCHS": 8,
    "CRITIC_EPOCHS": 24,
    "MINIBATCH_SIZE": 512,
    "ACTOR_MINIBATCH_SIZE": 512,
    "CRITIC_MINIBATCH_SIZE": 512,

    # DDPG Exploration, Actor Trust-Region & Q-Critic Settings
    "CRITIC_TYPE": "sarsa",       # "sarsa", "td_0", "e_0", "e_0_basic", "e_lambda", "mc"
    "E_VARIANT": "naive",         # "aux", "e_aux", "fwd", "naive"
    "EXPLORATION_NOISE": 0.1,
    "ACTION_REG_COEF": 0.25,
    "VALUE_LAMBDA": 0.8,
    "TD_LAMBDA": 0.0,
    "E_LAMBDA": 0.8,
    "RETURN_LAMBDA": 1.0,
    "AUX_LAMBDA": 1.0,
    "AUX_VF_COEF": 1.0,
    "VF_COEF": 0.5,

    # Slurm & Metadata
    "SLURM_JOB_ID": None,
    "SLURM_ARRAY_JOB_ID": None,
    "SLURM_ARRAY_TASK_ID": None,
}
