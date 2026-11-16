"""Dedicated default configuration for fresh-batch SAC on Brax environments."""

config = {
    # Environment & Rollout
    "ENV_NAME": "hopper",
    "GAMMA": 0.99,
    "TOTAL_TIMESTEPS": 10_000_000,
    "NUM_ENVS": 256,
    "NUM_STEPS": 128,

    # Network Architecture (matches brax_network.py)
    "SHARED_TORSO": False,
    "LAYER_NORM": False,

    # Optimization (mirrors Brax PPO split TrainState)
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

    # SAC Entropy, Adaptive KL Trust Region & Critic Settings
    "CRITIC_TYPE": "e_lambda",
    "TD_LAMBDA": 0.95,
    "RETURN_LAMBDA": 1.0,
    "E_LAMBDA": 0.8,
    "ALPHA": 1.0,
    "AUTOTUNE_ALPHA": True,
    "ALPHA_LR": 3e-3,
    "TARGET_ENTROPY": None,
    "KL_COEFF": 0.5,
    "KL_TARGET": 0.03,
}
