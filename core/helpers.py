# helpers.py
# Technical helpers for RL training loops, GAE computation, PPO losses, and environment setup.
import gymnax
from gymnax.environments import spaces
from gymnax.wrappers.purerl import FlattenObservationWrapper
import jax
import jax.numpy as jnp

from envs.log_wrapper import LogWrapper
from envs.wrappers import (
    NormalizeObservationWrapper,
    AddChannelWrapper,
    ClipAction,
    TerminalInfoWrapper,
    MountainCarNormalizeWrapper,
    MountainCarSparseRewardWrapper,
)


def make_env(config):
    env_name = config["ENV_NAME"]
    env, env_params = gymnax.make(env_name)

    if "MAX_STEPS_IN_EPISODE" in config and hasattr(env_params, "max_steps_in_episode"):
        env_params = env_params.replace(
            max_steps_in_episode=int(config["MAX_STEPS_IN_EPISODE"])
        )

    if env_name == "MountainCar-v0":
        env = MountainCarNormalizeWrapper(env)
        env = MountainCarSparseRewardWrapper(env)
        config["NETWORK_TYPE"] = "mlp"

    env = TerminalInfoWrapper(env)
    env = LogWrapper(env, gamma=config.get("GAMMA", 0.99))

    if isinstance(env.action_space(env_params), spaces.Box):
        env = ClipAction(env)

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


def post_process_advantage(advantages, config, weights=None):
    """
    Standardizes and clips advantages for PPO policy optimization.
    """
    std_floor = config.get("ADV_STD_FLOOR", 0.1)
    adv_clip = config.get("ADV_CLIP", 3.0)

    if weights is not None:
        if weights.ndim == 1 and advantages.ndim == 2:
            weights = weights[:, None]
        w = weights / jnp.sum(weights)
        mean_adv = jnp.sum(w * advantages)
        var_adv = jnp.sum(w * (advantages - mean_adv) ** 2)
        std_adv = jnp.sqrt(var_adv + 1e-8)
    else:
        mean_adv = jnp.mean(advantages)
        var_adv = jnp.var(advantages)
        std_adv = jnp.sqrt(var_adv + 1e-8)

    if std_floor is not None and std_floor > 0.0:
        denom = jnp.maximum(std_adv, std_floor)
    else:
        denom = std_adv + 1e-8

    adv_norm = (advantages - mean_adv) / denom

    if adv_clip is not None and adv_clip > 0.0:
        adv_norm = jnp.clip(adv_norm, -adv_clip, adv_clip)

    return jax.lax.stop_gradient(adv_norm)


def pi_loss_fn(params, network, traj_batch, gae, config):
    pi = network.apply(params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = post_process_advantage(gae, config)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    return loss_actor, entropy


def ppo_clipped_v_loss(traj_batch, value_pred, targets, config):
    e = config["VF_CLIP"]
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value
    ).clip(-e, e)
    value_losses = jnp.square(value_pred - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()


def v_loss_fn(params, network, traj_batch, gae, targets, config):
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    return ppo_clipped_v_loss(traj_batch, value_pred, targets, config)


def _loss_fn(params, network, traj_batch, gae, targets, config):
    value_loss = v_loss_fn(params, network, traj_batch, gae, targets, config)
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * loss_actor
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)


def e_critic_loss(v_i, targets_i, v_j, targets_j, done, gamma):
    """
    Computes sampled E-loss (magnitude anchor + Dirichlet/Laplacian smoothness).
    - For ongoing transitions and timeouts: smooths e_i against e_j.
    - For true terminal transitions (done=True): absorbing state error is 0,
      so (e_i - e_j)^2 = (e_i - 0)^2 = (r_T - v_T)^2, smoothing v_T directly to reward.
    """
    e_i = targets_i - v_i
    e_j = jnp.where(done, 0.0, targets_j - v_j)

    magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)
    laplacian_loss = 0.5 * gamma * jnp.mean((e_i - e_j) ** 2)

    value_loss = magnitude_loss + laplacian_loss
    return value_loss, magnitude_loss, laplacian_loss


def e_loss_fn(
    params,
    network,
    obs,
    action,
    log_prob_old,
    next_obs,
    done,
    next_target,
    advantages,
    targets,
    config,
):
    """
    Combined loss for PPO with Sampled E critic.
    """
    # 1. Actor Loss (PPO clipped surrogate)
    pi = network.apply(params, obs, method=network.policy)
    log_prob = pi.log_prob(action)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - log_prob_old)

    adv_norm = post_process_advantage(advantages, config)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    # 2. Critic Loss (Sampled E-loss)
    gamma = config.get("GAMMA", 0.99)
    v_i = network.apply(params, obs, method=network.value)
    v_j = network.apply(params, next_obs, method=network.value)
    # Terminal absorbing state has value 0
    v_j = jnp.where(done, 0.0, v_j)

    value_loss, magnitude_loss, laplacian_loss = e_critic_loss(
        v_i, targets, v_j, next_target, done, gamma
    )

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * actor_loss
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )
    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "laplacian_loss": laplacian_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
    }
    return total_loss, losses


def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x

    minibatches = jax.tree.map(
        lambda x: preprocess_transition(x, rng), transitions
    )
    return minibatches


def calculate_gae(traj_batch, γ, λ):
    def _get_advantages(gae, transition):
        done = transition.done
        is_timeout = transition.info["is_timeout"]

        # MASK 1: Value Bootstrapping
        true_terminal = done & ~is_timeout
        bootstrap_mask = 1.0 - true_terminal

        # MASK 2: GAE Accumulation (Trajectory Boundary)
        # Sever the GAE chain if the environment reset for ANY reason (terminal or timeout).
        boundary_mask = 1.0 - done

        # 1. Compute TD Error (Safely bootstraps through timeouts)
        delta = (
            transition.reward
            + γ * transition.next_value * bootstrap_mask
            - transition.value
        )

        # 2. Accumulate GAE (Safely breaks at episode resets)
        gae = delta + (γ * λ * boundary_mask * gae)

        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    _, advantages = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )

    return (advantages, advantages + traj_batch.value)
