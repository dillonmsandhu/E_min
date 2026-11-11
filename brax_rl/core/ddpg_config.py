"""
brax_rl/core/ddpg_config.py

Dedicated default configuration for On-Policy Continuous DDPG on Brax environments,
strictly free of PPO-specific settings (CLIP_EPS, ENT_COEF, POLICY_COEFF,
GAE_LAMBDA, VF_CLIP, ADV_CLIP, ADV_STD_FLOOR).
"""

config = {
    # Environment & Rollout
    "ENV_NAME": "hopper",
    "GAMMA": 0.99,
    "TOTAL_TIMESTEPS": 10_000_000,
    "NUM_ENVS": 256,
    "NUM_STEPS": 128,

    # Network Architecture
    "SHARED_TORSO": True,
    "LAYER_NORM": False,
    "NUM_VALUE_HEADS": 1,
    "VALUE_HEAD_AGG": "mean",

    # Optimization
    "ACTOR_LR": 3e-4,
    "ACTOR_LR_END": 3e-5,
    "CRITIC_LR": 1e-3,
    "CRITIC_LR_END": 1e-4,
    "ACTOR_EPOCHS": 4,
    "CRITIC_EPOCHS": 16,
    "ACTOR_MINIBATCH_SIZE": 1024,
    "CRITIC_MINIBATCH_SIZE": 1024,
    "NETWORK_CONFIG": {
        "LR": 3e-4,
        "LR_END": 1e-5,
        "NUM_EPOCHS": 4,
        "MINIBATCH_SIZE": 1024,
        "ACTIVATION": "tanh",
        "MAX_GRAD_NORM": 1.0,
        "NORMALIZE_ENV": True,
        "NORMALIZE_REW": True,
        "LR_SCHEDULE": "linear",
    },

    # DDPG Exploration, Actor Trust-Region & Q-Critic Settings
    "CRITIC_TYPE": "sarsa",
    "E_VARIANT": "naive",
    "EXPLORATION_NOISE": 0.2,
    "ACTION_REG_COEF": 0.1,
    "VALUE_LAMBDA": 0.8,
    "TD_LAMBDA": 0.0,
    "E_LAMBDA": 0.8,
    "RETURN_LAMBDA": 1.0,
    "AUX_LAMBDA": 1.0,
    "AUX_VF_COEF": 1.0,
    "VF_COEF": 0.5,
}
