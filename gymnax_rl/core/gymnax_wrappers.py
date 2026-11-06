import gymnax
from gymnax.environments import spaces
from envs.wrappers import (
    TerminalInfoWrapper,
    ClipAction,
    FlattenObservationWrapper,
    AddChannelWrapper,
    NormalizeObservationWrapper,
    MountainCarNormalizeWrapper,
    MountainCarSparseRewardWrapper,
)
from envs.log_wrapper import LogWrapper


def make_env(config):
    """
    Constructs and wraps a Gymnax environment according to config.
    Supports classic control, MinAtar, and custom environments (MountainCar, PointRobot, SpaceInvaders).
    """
    env_name = config["ENV_NAME"]
    if env_name in [
        "MountainCarDenseContinuous-v0",
        "MountainCarDenseContinuous",
        "MountainCarSparseContinuous-v0",
        "MountainCarSparseContinuous",
        "continuous_mountain_car",
    ]:
        from envs.continuous_mountain_car import MountainCarDenseContinuous, EnvParams as DenseEnvParams
        env = MountainCarDenseContinuous()
        p_scale = 0.0 if ("Sparse" in env_name or not config.get("DENSE_REWARD", True)) else config.get("POTENTIAL_SCALE", 30.0)
        env_params = DenseEnvParams(
            gamma=config.get("GAMMA", 0.99),
            potential_scale=p_scale,
            slip_prob=config.get("SLIP_PROB", 0.0),
            slip_force_scale=config.get("SLIP_FORCE_SCALE", 0.0),
            action_noise_std=config.get("ACTION_NOISE_STD", 0.0),
            transition_noise_std=config.get("TRANSITION_NOISE", 0.0),
        )
    elif env_name in [
        "MountainCarDenseDiscrete-v0",
        "MountainCarDenseDiscrete",
        "MountainCarSparseDiscrete-v0",
        "MountainCarSparseDiscrete",
    ]:
        from envs.continuous_mountain_car import MountainCarDenseDiscrete, EnvParams as DenseEnvParams
        env = MountainCarDenseDiscrete()
        p_scale = 0.0 if ("Sparse" in env_name or not config.get("DENSE_REWARD", True)) else config.get("POTENTIAL_SCALE", 30.0)
        env_params = DenseEnvParams(
            gamma=config.get("GAMMA", 0.99),
            potential_scale=p_scale,
            slip_prob=config.get("SLIP_PROB", 0.0),
            slip_force_scale=config.get("SLIP_FORCE_SCALE", 0.0),
            action_noise_std=config.get("ACTION_NOISE_STD", 0.0),
            transition_noise_std=config.get("TRANSITION_NOISE", 0.0),
        )
    elif env_name in ["PointRobot-misc", "PointRobot"]:
        from envs.point_robot import PointRobot, EnvParams as PointRobotParams
        env = PointRobot(fully_observable=config.get("FULLY_OBSERVABLE", True))
        dense = config.get("DENSE_REWARD", False) and ("Sparse" not in env_name)
        p_scale = config.get("POTENTIAL_SCALE", 1.0) if dense else 0.0
        env_params = PointRobotParams(
            dense_reward=dense,
            gamma=config.get("GAMMA", 0.99),
            potential_scale=p_scale,
            fully_observable=config.get("FULLY_OBSERVABLE", True),
            slip_prob=config.get("SLIP_PROB", 0.0),
            slip_force_scale=config.get("SLIP_FORCE_SCALE", 0.0),
            action_noise_std=config.get("ACTION_NOISE_STD", 0.0),
            transition_noise_std=config.get("TRANSITION_NOISE", 0.0),
        )
    elif env_name in ["PointRobotDiscrete-misc", "PointRobotDiscrete"]:
        from envs.point_robot import PointRobotDiscrete, EnvParams as PointRobotParams
        env = PointRobotDiscrete(fully_observable=config.get("FULLY_OBSERVABLE", True))
        dense = config.get("DENSE_REWARD", False) and ("Sparse" not in env_name)
        p_scale = config.get("POTENTIAL_SCALE", 1.0) if dense else 0.0
        env_params = PointRobotParams(
            dense_reward=dense,
            gamma=config.get("GAMMA", 0.99),
            potential_scale=p_scale,
            fully_observable=config.get("FULLY_OBSERVABLE", True),
            slip_prob=config.get("SLIP_PROB", 0.0),
            slip_force_scale=config.get("SLIP_FORCE_SCALE", 0.0),
            action_noise_std=config.get("ACTION_NOISE_STD", 0.0),
            transition_noise_std=config.get("TRANSITION_NOISE", 0.0),
        )
    elif "SpaceInvaders" in env_name:
        from envs.space_invaders import MinSpaceInvaders, EnvParams as SpaceInvadersParams
        env = MinSpaceInvaders()
        penalty_on_hit = config.get("PENALTY_ON_HIT", False) or ("Fixed" in env_name)
        sticky_prob = config.get("STICKY_ACTION_PROB", 0.0)
        env_params = SpaceInvadersParams(
            penalty_on_hit=penalty_on_hit,
            hit_penalty=config.get("HIT_PENALTY", 1.0),
            sticky_action_prob=sticky_prob,
            max_steps_in_episode=config.get("MAX_STEPS_IN_EPISODE", 1000),
            instant_bullets=config.get("INSTANT_BULLETS", False),
        )
    elif env_name in ["FourRooms-misc", "FourRooms"]:
        use_visual = config.get("USE_VISUAL_OBS", True)
        if use_visual:
            env, env_params = gymnax.make(env_name, use_visual_obs=True)
            config["NETWORK_TYPE"] = "cnn"
        else:
            env, env_params = gymnax.make(env_name, use_visual_obs=False)
    else:
        env, env_params = gymnax.make(env_name)

    if env_name == "MountainCar-v0":
        env = MountainCarNormalizeWrapper(env)
        env = MountainCarSparseRewardWrapper(env)
        config["NETWORK_TYPE"] = "mlp"

    env = TerminalInfoWrapper(env)
    env = LogWrapper(env, gamma=config.get("GAMMA", 0.99))

    if isinstance(env.action_space(env_params), spaces.Box):
        action_space = env.action_space(env_params)
        env = ClipAction(env, low=action_space.low, high=action_space.high)

    obs_shape = env.observation_space(env_params).shape
    if len(obs_shape) == 1:
        config["NETWORK_TYPE"] = "mlp"

    if config.get("NETWORK_TYPE", "mlp") == "mlp":
        if len(obs_shape) > 1:
            env = FlattenObservationWrapper(env)
    elif config.get("NETWORK_TYPE", "mlp") == "cnn":
        if len(obs_shape) == 2:
            env = AddChannelWrapper(env)
    if config.get("NORMALIZE_OBS", False):
        env = NormalizeObservationWrapper(env)

    return env, env_params
