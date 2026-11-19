"""Default configuration and Brax-parity presets for MuJoCo Playground (MJX) PPO."""

import copy

# Default configuration for MuJoCo Playground continuous control (Brax-parity defaults)
config = {
    "TOTAL_TIMESTEPS": 60_000_000,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 30,
    "GAMMA": 0.995,
    "GAE_LAMBDA": 0.95,
    "ACTOR_GAE_LAMBDA": 0.95,
    "RETURN_LAMBDA": 1.0,
    "VALUE_LAMBDA": 0.95,
    "E_LAMBDA": 0.95,
    "BSMC_LAMBDA": 0.95,
    "CRITIC_TYPE": "fitted",
    "E_VARIANT": "aux",
    "AUX_LAMBDA": 1.0,
    "AUX_VF_COEF": 1.0,
    "K_ACTIONS": 4,
    "GAE_CONTINUOUS_MODE": "loo",
    "CLIP_EPS": 0.3,
    "VF_CLIP": None,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "REWARD_SCALING": 10.0,
    "ACTION_REPEAT": 1,
    "EPISODE_LENGTH": 1000,
    "IMPL": "jax",
    "FULL_RESET": False,
    "ENV_NAME": "CheetahRun",
    "ACTOR_LR": 1e-3,
    "CRITIC_LR": 1e-3,
    "ACTOR_EPOCHS": 16,
    "CRITIC_EPOCHS": 16,
    "ACTOR_MINIBATCH_SIZE": 1920,
    "CRITIC_MINIBATCH_SIZE": 1920,
    "MINIBATCH_SIZE": 1920,
    "POLICY_OBS_KEY": "state",
    "VALUE_OBS_KEY": "state",
    "SHARED_TORSO": False,
    "SPLIT_AUX_OPTIMIZER": False,
    "LAYER_NORM": False,
    "NETWORK_CONFIG": {
        "LR": 1e-3,
        "LR_END": 1e-5,
        "NUM_EPOCHS": 16,
        "NUM_MINIBATCHES": 32,
        "MINIBATCH_SIZE": 1920,
        "ACTIVATION": "swish",
        "POLICY_HIDDEN_LAYER_SIZES": (32, 32, 32, 32),
        "VALUE_HIDDEN_LAYER_SIZES": (256, 256, 256, 256, 256),
        "DISTRIBUTION_TYPE": "tanh_normal",
        "KERNEL_INIT": "lecun_uniform",
        "OPTIMIZER": "adam",
        "ADAM_EPS": 1e-8,
        "ADVANTAGE_NORM": "mean_std",
        "MAX_GRAD_NORM": 1.0,
        "VF_COEF": 0.5,
        "NORMALIZE_ENV": True,
        "NORMALIZE_REW": False,
        "LR_SCHEDULE": "constant",
    },
    "DEBUG": True,
}

# Five diverse & challenging DM Control Suite benchmark tasks for Phase 1 / Phase 2
DEFAULT_DMC_5_ENVS = [
    "CheetahRun",
    "WalkerWalk",
    "HopperStand",
    "FishSwim",
    "FingerSpin",
]


def get_dmc_brax_parity_config(env_name: str, impl: str = "jax") -> dict:
    """
    Returns a configuration dictionary matching the tuned Brax PPO hyperparameters
    from MuJoCo Playground (`rss2025/mujoco_playground/config/dm_control_suite_params.py`),
    with per-environment `TOTAL_TIMESTEPS` sized to convergence based on the paper's
    PPO learning curves.
    """
    cfg = copy.deepcopy(config)
    cfg["ENV_NAME"] = env_name
    cfg["IMPL"] = impl

    # Per-env convergence budgets (from Playground RSS 2025 PPO curves)
    timesteps_by_env = {
        "CartpoleBalance": 20_000_000,
        "CartpoleBalanceSparse": 20_000_000,
        "PointMass": 20_000_000,
        "ReacherEasy": 20_000_000,
        "ReacherHard": 20_000_000,
        "CartpoleSwingup": 30_000_000,
        "CartpoleSwingupSparse": 40_000_000,
        "FingerTurnEasy": 30_000_000,
        "FingerTurnHard": 30_000_000,
        "WalkerStand": 50_000_000,
        "WalkerWalk": 50_000_000,
        "BallInCup": 60_000_000,
        "FingerSpin": 60_000_000,
        "HopperHop": 60_000_000,
        "HopperStand": 60_000_000,
        "PendulumSwingup": 60_000_000,
        "PendulumSwingUp": 60_000_000,
        "CheetahRun": 80_000_000,
        "FishSwim": 80_000_000,
        "HumanoidRun": 100_000_000,
        "HumanoidStand": 100_000_000,
        "HumanoidWalk": 100_000_000,
        "SwimmerSwimmer6": 100_000_000,
        "AcrobotSwingup": 150_000_000,
        "AcrobotSwingupSparse": 150_000_000,
        "WalkerRun": 150_000_000,
    }
    cfg["TOTAL_TIMESTEPS"] = timesteps_by_env.get(env_name, 60_000_000)

    if env_name in ("BallInCup", "FingerSpin"):
        cfg["GAMMA"] = 0.95
    else:
        cfg["GAMMA"] = 0.995

    if env_name in ("PendulumSwingUp", "PendulumSwingup"):
        cfg["ACTION_REPEAT"] = 4
        cfg["ACTOR_EPOCHS"] = 4
        cfg["CRITIC_EPOCHS"] = 4
        cfg["NETWORK_CONFIG"]["NUM_EPOCHS"] = 4

    return cfg
