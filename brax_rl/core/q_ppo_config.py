"""
brax_rl/core/q_ppo_config.py

Dedicated default configuration for Continuous Q-PPO on Brax environments,
initialized from the baseline Brax PPO configuration (brax_config.py).
"""

config = {
    "TOTAL_TIMESTEPS": 5e7,
    "NUM_ENVS": 512,
    "NUM_STEPS": 128,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.8,
    "RETURN_LAMBDA": 1.0,
    "VALUE_LAMBDA": 0.8,
    "E_LAMBDA": 0.8,
    "BSMC_LAMBDA": 0.8,
    "E_VARIANT": "aux",
    "AUX_LAMBDA": 1.0,
    "AUX_VF_COEF": 1.0,
    "CRITIC_TYPE": "e_0",
    "K_ACTIONS": 4,
    "GAE_CONTINUOUS_MODE": "loo",
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.001,
    "VF_COEF": 0.5,
    "ENV_NAME": "hopper",
    "NETWORK_CONFIG": {
        "LR": 3e-4,
        "LR_END": 1e-5,
        "NUM_EPOCHS": 4,
        "MINIBATCH_SIZE": 1024,
        "ACTIVATION": "tanh",
        "MAX_GRAD_NORM": 1.0,
        "VF_COEF": 0.5,
        "NORMALIZE_ENV": True,
        "NORMALIZE_REW": True,
        "LR_SCHEDULE": "linear",
    },
    "DEBUG": True,
}
