# jumanji_rl/core/jumanji_utils.py
# Utilities and transition containers for Jumanji RL

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import NamedTuple, Any, Optional


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    is_start: jnp.ndarray
    is_timeout: jnp.ndarray
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None
    aux_targets: Optional[jnp.ndarray] = None
    v_aux: Optional[jnp.ndarray] = None


class QTransition(NamedTuple):
    """Transition container for discrete Q-PPO algorithms."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray                 # Analytical marginalized V(s) = sum_a pi(a|s) Q(s,a)
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    is_start: jnp.ndarray
    is_timeout: jnp.ndarray
    q_taken: Optional[jnp.ndarray] = None
    next_action: Optional[jnp.ndarray] = None
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None
    next_reward: Optional[jnp.ndarray] = None
    next_next_obs: Optional[jnp.ndarray] = None
    next_true_terminal: Optional[jnp.ndarray] = None
    is_horizon: Optional[jnp.ndarray] = None
    dest_obs: Optional[jnp.ndarray] = None
    dest_action: Optional[jnp.ndarray] = None
    dest_targets: Optional[jnp.ndarray] = None
    dest_reward: Optional[jnp.ndarray] = None
    dest_next_obs: Optional[jnp.ndarray] = None
    dest_true_terminal: Optional[jnp.ndarray] = None
    hit_true_terminal: Optional[jnp.ndarray] = None
    valid_jump: Optional[jnp.ndarray] = None
    aux_targets: Optional[jnp.ndarray] = None


def prepare_q_jump_transitions(traj_batch: QTransition, targets: jnp.ndarray, last_obs: jnp.ndarray, last_action: jnp.ndarray, gamma: float, lmbda: float, rng: jax.Array) -> QTransition:
    """
    Samples geometric jump indices m ~ Geometric(1 - gamma * lambda) along trajectories
    for discrete E(lambda)-Q, handling spatial observation tensors (e.g. 5D [T, B, H, W, C])
    and boundary conditions (timeouts vs absorbing terminals).
    """
    num_steps, num_envs = traj_batch.obs.shape[0], traj_batch.obs.shape[1]
    p_success = 1.0 - (gamma * lmbda)
    p_success = jnp.clip(p_success, 1e-6, 1.0)

    # Sample geometric jump distance m in {1, 2, ...}
    uniform = jax.random.uniform(rng, (num_steps, num_envs), minval=1e-7, maxval=1.0)
    m = jnp.floor(jnp.log(uniform) / jnp.log(1.0 - p_success)).astype(jnp.int32) + 1

    t_grid = jnp.broadcast_to(jnp.arange(num_steps)[:, None], (num_steps, num_envs))
    dest_t = t_grid + m

    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    is_absorbing = true_terminal

    # Extended trajectory buffers with continuation state at horizon
    next_target_T = jnp.where(true_terminal[-1], 0.0, traj_batch.next_value[-1])
    obs_ext = jnp.concatenate([traj_batch.obs, last_obs[None]], axis=0)
    act_ext = jnp.concatenate([traj_batch.action, last_action[None]], axis=0)
    targets_ext = jnp.concatenate([targets, next_target_T[None]], axis=0)

    t_clamped = jnp.clip(dest_t, 0, num_steps)
    t_prev = jnp.minimum(t_grid + m - 1, num_steps - 1)
    done_cumsum = jnp.cumsum(traj_batch.done.astype(jnp.int32), axis=0)
    has_reset_between = (
        jnp.take_along_axis(done_cumsum, t_prev, axis=0) - done_cumsum
    ) > 0

    # Expand gather indices for multi-dimensional spatial observations (T, B, H, W, C)
    gather_idx_obs = t_clamped.reshape(t_clamped.shape + (1,) * (traj_batch.obs.ndim - 2))
    dest_obs_gathered = jnp.take_along_axis(obs_ext, gather_idx_obs, axis=0)
    dest_act_gathered = jnp.take_along_axis(act_ext, t_clamped, axis=0)
    dest_targets_gathered = jnp.take_along_axis(targets_ext, t_clamped, axis=0)

    # Gather successor reward / next_obs / true_terminal for v_aux jump proxy
    rew_ext = jnp.concatenate([traj_batch.reward, jnp.zeros((1, num_envs), dtype=traj_batch.reward.dtype)], axis=0)
    next_obs_ext = jnp.concatenate([traj_batch.next_obs, last_obs[None]], axis=0)
    term_ext = jnp.concatenate([true_terminal, jnp.zeros((1, num_envs), dtype=bool)], axis=0)
    dest_rew_gathered = jnp.take_along_axis(rew_ext, t_clamped, axis=0)
    dest_next_obs_gathered = jnp.take_along_axis(next_obs_ext, gather_idx_obs, axis=0)
    dest_term_gathered = jnp.take_along_axis(term_ext, t_clamped, axis=0)

    # Broadcast timeout mask for spatial observations
    timeout_mask_obs = traj_batch.is_timeout.reshape(traj_batch.is_timeout.shape + (1,) * (traj_batch.obs.ndim - 2))
    dest_obs = jnp.where(timeout_mask_obs, traj_batch.next_obs, dest_obs_gathered)
    dest_act = jnp.where(traj_batch.is_timeout, traj_batch.next_action, dest_act_gathered)
    dest_targets = jnp.where(traj_batch.is_timeout, traj_batch.next_value, dest_targets_gathered)

    valid_jump = jnp.where(
        true_terminal,
        False,
        jnp.where(traj_batch.is_timeout, m == 1, (dest_t < num_steps) & ~has_reset_between),
    )

    return traj_batch._replace(
        dest_obs=dest_obs,
        dest_action=dest_act,
        dest_targets=dest_targets,
        dest_reward=dest_rew_gathered,
        dest_next_obs=dest_next_obs_gathered,
        dest_true_terminal=dest_term_gathered,
        hit_true_terminal=is_absorbing,
        valid_jump=valid_jump,
    )


def shuffle_and_batch(batch, batch_size: int, n_minibatches: int, rng: jax.Array):
    """Flattens rollout and splits into randomized minibatches."""
    flat_batch = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
    permutation = jax.random.permutation(rng, batch_size)
    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), flat_batch)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [n_minibatches, -1] + list(x.shape[1:])),
        shuffled_batch,
    )
    return minibatches


def scale_rms(x: jnp.ndarray) -> jnp.ndarray:
    """Scales input by its Root Mean Square (RMS) without centering."""
    rms = jnp.sqrt(jnp.mean(jnp.square(x)))
    return x / (rms + 1e-8)


def post_process_advantage(gae: jnp.ndarray, config: dict) -> jnp.ndarray:
    """Post-processes advantage estimates (normalization or RMS scaling)."""
    if config.get("NORMALIZE_ADVANTAGE", True):
        return (gae - gae.mean()) / (gae.std() + 1e-8)
    return gae


def compute_runtime_metrics(
    train_state,
    network,
    traj_batch,
    loss_info,
    start_obs,
    config,
) -> dict[str, jax.Array]:
    """Computes runtime metrics for Jumanji Q-PPO training."""
    info = traj_batch.info
    if "returned_episode" in info and "returned_episode_returns" in info:
        done_mask = info["returned_episode"].astype(jnp.float32)
        n_done = jnp.sum(done_mask)
        has_done = n_done > 0

        returned_episode_returns = jnp.where(
            has_done,
            jnp.sum(info["returned_episode_returns"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_episode_returns"].mean(),
        )
        returned_discounted_episode_returns = jnp.where(
            has_done,
            jnp.sum(info["returned_discounted_episode_returns"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_discounted_episode_returns"].mean(),
        )
        returned_episode_lengths = jnp.where(
            has_done,
            jnp.sum(info["returned_episode_lengths"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_episode_lengths"].mean(),
        )
    elif "returned_episode_returns" in info:
        returned_episode_returns = info["returned_episode_returns"].mean()
        returned_discounted_episode_returns = info.get(
            "returned_discounted_episode_returns", returned_episode_returns
        ).mean()
        returned_episode_lengths = info.get("returned_episode_lengths", jnp.array(0.0)).mean()
    else:
        returned_episode_returns = traj_batch.reward.sum(axis=0).mean()
        returned_discounted_episode_returns = returned_episode_returns
        returned_episode_lengths = jnp.array(traj_batch.obs.shape[0], dtype=jnp.float32)

    if start_obs is not None:
        v_start_pred = network.apply(train_state.params, start_obs, method=network.value).mean()
    else:
        v_start_pred = traj_batch.value[0].mean()

    if isinstance(loss_info, dict):
        total_loss = loss_info.get("total_loss", jnp.array(0.0)).mean()
        value_loss = loss_info.get("value_loss", loss_info.get("val_loss", jnp.array(0.0))).mean()
        actor_loss = loss_info.get("actor_loss", jnp.array(0.0)).mean()
        entropy = loss_info.get("entropy", jnp.array(0.0)).mean()
    else:
        total_loss = loss_info[0].mean()
        value_loss = loss_info[1].mean()
        actor_loss = loss_info[2].mean()
        entropy = loss_info[3].mean()

    pi = network.apply(train_state.params, traj_batch.obs, method=network.policy)
    new_log_prob = pi.log_prob(traj_batch.action)
    log_ratio = new_log_prob - traj_batch.log_prob
    ratio = jnp.exp(log_ratio)

    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_eps = config.get("CLIP_EPS", 0.1)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    n_terminations = jnp.sum(true_terminal.astype(jnp.float32))
    n_truncations = jnp.sum(traj_batch.is_timeout.astype(jnp.float32))

    metrics = {
        "returned_episode_returns": returned_episode_returns,
        "returned_discounted_episode_returns": returned_discounted_episode_returns,
        "returned_episode_lengths": returned_episode_lengths,
        "mean_rew": traj_batch.reward.mean(),
        "V_start": v_start_pred,
        "v_pred_start": v_start_pred,
        "v_pred": traj_batch.value.mean(),
        "total_loss": total_loss,
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "n_terminations": n_terminations,
        "n_truncations": n_truncations,
    }

    if hasattr(network, "both"):
        v_main, aux_head = network.apply(train_state.params, traj_batch.obs, method=network.both)
        target = traj_batch.aux_targets if traj_batch.aux_targets is not None else traj_batch.targets
        if target is not None:
            v_aux = aux_head
            metrics["mse_v_main"] = jnp.mean((v_main - target) ** 2)
            metrics["mse_v_aux"] = jnp.mean((v_aux - target) ** 2)
            norm_main = jnp.linalg.norm(v_main)
            norm_aux = jnp.linalg.norm(v_aux)
            cos_sim = jnp.dot(v_main.flatten(), v_aux.flatten()) / (norm_main * norm_aux + 1e-8)
            metrics["v_similarity"] = cos_sim

    return metrics
