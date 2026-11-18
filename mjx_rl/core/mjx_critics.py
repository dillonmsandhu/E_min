"""
Critic and actor loss functions for MuJoCo Playground (MJX) PPO.

Reuses the validated E(0), E(lambda), TD(0), TD(lambda), and BSMC critic objectives
from `brax_rl.core.brax_critics`, while extending:
  - `prepare_geometric_jump_transitions`: PyTree-safe observation gathering so both
    flat array observations and dict observations (`{'state': ..., 'privileged_state': ...}`)
    work seamlessly for E(lambda).
  - `ppo_actor_loss_fn`: supports `raw_action` log-prob evaluation (for `tanh_normal`
    Brax parity) and configurable advantage normalization (`'mean_std'` or `'rms'`).
"""

import jax
import jax.numpy as jnp

from brax_rl.core.brax_critics import (
    CRITIC_LOSS_FNS,
    SPLIT_CRITIC_LOSS_FNS,
    bsmc_critic_loss_fn,
    calculate_gae,
    calculate_targets,
    compute_aux_head_loss,
    e_0_basic_critic_loss_fn,
    e_0_critic_loss_fn,
    e_aux_backward_surrogate,
    e_lambda_geometric_critic_loss,
    e_lambda_geometric_critic_loss_fn,
    get_critic_loss_fn,
    get_critic_target_lambda,
    ppo_critic_loss_fn,
    ppo_loss_fn,
    ppo_value_loss,
    td_0_critic_loss_fn,
    td_0_loss_fn,
    td_0_value_loss,
)
from brax_rl.core.brax_utils import scale_rms


def prepare_geometric_jump_transitions(
    traj_batch,
    targets: jax.Array,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    last_val: jax.Array = None,
):
    """
    Samples geometric lookahead jumps K ~ Geometric(1 - gamma * lambda) on the rollout buffer
    and extracts jump observation targets (supporting both flat array and PyTree/dict obs),
    continuation states, and validity masks for transition-level minibatching.
    """
    T, B = targets.shape[:2]
    gl = gamma * lmbda

    u = jax.random.uniform(rng, shape=(T, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    jumps = jnp.maximum(jumps, 0)

    t_arr = jnp.arange(T)[:, None]
    target_idx = t_arr + jumps + 1
    t_clamped = jnp.minimum(target_idx, T)

    t_prev = jnp.minimum(t_arr + jumps, T - 1)
    done_cumsum = jnp.cumsum(traj_batch.done.astype(jnp.int32), axis=0)
    has_reset_between = (
        jnp.take_along_axis(done_cumsum, t_prev, axis=0) - done_cumsum
    ) > 0

    if isinstance(traj_batch.info, dict):
        truncation = traj_batch.info.get("truncation", traj_batch.info.get("is_timeout", None))
        if truncation is not None:
            truncation = truncation > 0.5
        else:
            truncation = jnp.zeros_like(traj_batch.done, dtype=bool)
    else:
        truncation = jnp.zeros_like(traj_batch.done, dtype=bool)

    true_terminal = (traj_batch.done > 0.5) & ~truncation
    is_absorbing = true_terminal

    val_T = last_val if last_val is not None else traj_batch.value[-1]
    next_target_T = jnp.where(true_terminal[-1], 0.0, val_T)
    targets_ext = jnp.concatenate([targets, next_target_T[None]], axis=0)
    targets_jump = jnp.take_along_axis(targets_ext, t_clamped, axis=0)

    def _gather_obs_leaf(obs_leaf, next_obs_leaf):
        obs_ext = jnp.concatenate([obs_leaf, next_obs_leaf[-1][None]], axis=0)
        idx_expanded = t_clamped.reshape(t_clamped.shape + (1,) * (obs_leaf.ndim - 2))
        idx_broadcast = jnp.broadcast_to(idx_expanded, (T, B) + obs_leaf.shape[2:])
        obs_jump_gathered = jnp.take_along_axis(obs_ext, idx_broadcast, axis=0)
        trunc_mask = truncation.reshape(truncation.shape + (1,) * (obs_leaf.ndim - 2))
        return jnp.where(trunc_mask, next_obs_leaf, obs_jump_gathered)

    obs_jump = jax.tree_util.tree_map(_gather_obs_leaf, traj_batch.obs, traj_batch.next_obs)

    valid_jump = jnp.where(
        true_terminal,
        False,
        jnp.where(truncation, jumps == 0, ~has_reset_between),
    )

    return obs_jump, targets_jump, valid_jump, is_absorbing


def prepare_critic_transitions(
    critic_type: str,
    traj_batch,
    targets: jax.Array,
    last_val: jax.Array,
    gamma: float,
    e_lambda: float,
    rng: jax.Array,
):
    key = critic_type.lower()
    if key in ["e_lambda", "elambda", "e_geometric"]:
        obs_jump, targets_jump, valid_jump, is_absorbing = prepare_geometric_jump_transitions(
            traj_batch=traj_batch,
            targets=targets,
            gamma=gamma,
            lmbda=e_lambda,
            rng=rng,
            last_val=last_val,
        )
        new_info = {
            **traj_batch.info,
            "obs_jump": obs_jump,
            "targets_jump": targets_jump,
            "valid_jump": valid_jump,
            "is_absorbing": is_absorbing,
        }
        return traj_batch._replace(info=new_info)
    return traj_batch


def ppo_actor_loss_fn(actor_params, critic_params, network, traj_batch, gae, config, rng: jax.Array | None = None):
    """
    PPO clipped surrogate actor loss supporting:
      - `raw_action` (stored in `traj_batch.info['raw_action']`) for `tanh_normal`
        distributions so log-probs are computed without `arctanh` saturation.
      - Advantage normalization via `config['NETWORK_CONFIG']['ADVANTAGE_NORM']`:
        `'mean_std'` (Brax PPO standard: `(adv - mean) / (std + 1e-8)`) or `'rms'`.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    pi = network.apply(full_params, traj_batch.obs, method=network.policy)
    action_for_lp = (
        traj_batch.info["raw_action"]
        if isinstance(traj_batch.info, dict) and "raw_action" in traj_batch.info
        else traj_batch.action
    )
    log_prob = pi.log_prob(action_for_lp)
    entropy = pi.entropy(seed=rng).mean()

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    adv_norm_mode = config["NETWORK_CONFIG"].get("ADVANTAGE_NORM", "mean_std").lower()
    if adv_norm_mode == "mean_std":
        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
    elif adv_norm_mode == "rms":
        gae_norm = scale_rms(gae)
    elif adv_norm_mode == "none":
        gae_norm = gae
    else:
        raise ValueError(f"Unsupported ADVANTAGE_NORM '{adv_norm_mode}'.")

    clip_eps = config["CLIP_EPS"]
    surr1 = ratio * gae_norm
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    ent_coef = config["ENT_COEF"]
    total_actor_loss = actor_loss - ent_coef * entropy
    return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}


__all__ = [
    "CRITIC_LOSS_FNS",
    "SPLIT_CRITIC_LOSS_FNS",
    "bsmc_critic_loss_fn",
    "calculate_gae",
    "calculate_targets",
    "compute_aux_head_loss",
    "e_0_basic_critic_loss_fn",
    "e_0_critic_loss_fn",
    "e_aux_backward_surrogate",
    "e_lambda_geometric_critic_loss",
    "e_lambda_geometric_critic_loss_fn",
    "get_critic_loss_fn",
    "get_critic_target_lambda",
    "ppo_actor_loss_fn",
    "ppo_critic_loss_fn",
    "ppo_loss_fn",
    "ppo_value_loss",
    "prepare_critic_transitions",
    "prepare_geometric_jump_transitions",
    "td_0_critic_loss_fn",
    "td_0_loss_fn",
    "td_0_value_loss",
]
