# jumanji_rl/core/jumanji_config.py
# Default base configuration for Jumanji discrete environments

config = {
    # Environment & Seeds
    "ENV_NAME": "Sokoban-v0",
    "N_SEEDS": 1,
    "GAMMA": 0.99,
    "NORMALIZE_OBS": False,

    # Network Architecture
    "NETWORK_TYPE": "cnn",
    "k": 64,
    "LAYER_NORM": False,
    "NORM_TYPE": "none",

    # Optimization
    "LR": 0.0003,
    "LR_END": None,
    "ACTOR_LR": 0.0003,
    "ACTOR_LR_END": None,
    "CRITIC_LR": 0.0003,
    "CRITIC_LR_END": None,
    "MAX_GRAD_NORM": 1.0,
    "WEIGHT_DECAY": 0.0,
    "ADAM_EPS": 1e-5,

    # Rollout & Training Loop
    "NUM_ENVS": 256,
    "NUM_STEPS": 128,              # Rollout horizon (128 >= 120 Sokoban limit -> full episode MC)
    "TOTAL_TIMESTEPS": 2_000_000,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 512,
    "ACTOR_MINIBATCH_SIZE": 512,
    "CRITIC_MINIBATCH_SIZE": 512,

    # Critic & Policy Loss Hyperparameters
    "CRITIC_TYPE": "sarsa",        # "sarsa", "e_0", "e_lambda"
    "E_VARIANT": "aux",            # "aux" (v_aux debiased default), "naive", "fwd"
    "GAE_LAMBDA": 0.8,
    "VALUE_LAMBDA": 0.8,           # TD(lambda) target parameter
    "RETURN_LAMBDA": 0.8,          # E-min return lambda parameter
    "E_LAMBDA": 0.8,
    "AUX_LAMBDA": 1.0,             # Auxiliary head target (Monte Carlo by default)
    "AUX_VF_COEF": 1.0,
    "CLIP_EPS": 0.1,
    "VF_COEF": 0.5,
    "ENT_COEF": 0.01,
    "POLICY_COEFF": 1.0,
    "ANNEAL_LR": False,
    "NORMALIZE_ADVANTAGE": True,
}
