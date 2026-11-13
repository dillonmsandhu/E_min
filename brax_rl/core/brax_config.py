# Default configuration for Brax continuous control
config = {
    "TOTAL_TIMESTEPS": 5e7,
    "NUM_ENVS": 512,
    "NUM_STEPS": 128,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.001,
    "ENV_NAME": "hopper",
    "NETWORK_CONFIG": {
        "LR": 3e-4,
        "LR_END": 1e-5,
        "NUM_EPOCHS": 4, 
        "MINIBATCH_SIZE": 1024,
        "ACTIVATION": 'tanh',
        "MAX_GRAD_NORM": 1.0,
        "VF_COEF": 0.5,
        "NORMALIZE_ENV": True,
        "NORMALIZE_REW": True,
        "LR_SCHEDULE": "linear",
    },
    "DEBUG": True,
}
