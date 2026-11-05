# gymnax_q_critics.py
# Q-function critic losses for Q-PPO:
# 1. SARSA(lambda) regression
# 2. E(0) Q-minimization (no V_aux needed in deterministic dynamics)
# 3. E(lambda) Geometric jump Q-minimization
from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any


def extract_q_taken(q_all: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """Extracts Q(s, a) from Q(s, ·) tensor of shape (..., |A|)."""
    return jnp.take_along_axis(q_all, action[..., None], axis=-1).squeeze(-1)


# -------------------------------------------------------------
# 1. SARSA(lambda) Critic Loss
# -------------------------------------------------------------
def sarsa_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    Fitted on-policy SARSA(lambda) regression loss for Q(s_t, a_t):
        L = 0.5 * (targets - Q(s_t, a_t))^2
    targets are lambda-returns G_t.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_all = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_taken = extract_q_taken(q_all, traj_batch.action)

    targets_i = targets if targets is not None else traj_batch.targets
    loss = 0.5 * jnp.mean((targets_i - q_taken) ** 2)
    delta_mag = jnp.abs(targets_i - q_taken).mean()
    scaled_loss = config.get("VF_COEF", 0.5) * loss
    return scaled_loss, {"val_loss": loss, "delta_mag": delta_mag}


# -------------------------------------------------------------
# 2. E(0) State-Action Dirichlet Critic Loss (No V_aux needed!)
# -------------------------------------------------------------
def e_0_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    Sampled E(0) Dirichlet error loss on state-action pairs:
        e_i = G_t - Q(s_t, a_t)
        e_j = G_{t+1} - Q(s_{t+1}, a_{t+1})  (0.0 on terminals/timeouts)
        L = (1 - gamma) * E[e_i^2] + 0.5 * gamma * E[(e_i - e_j)^2]
    Full gradients flow directly through Q(s_t, a_t) and Q(s_{t+1}, a_{t+1}).
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_all_i = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_i = extract_q_taken(q_all_i, traj_batch.action)

    q_all_j = network.apply(full_params, traj_batch.next_obs, method=network.q_values)
    q_j = extract_q_taken(q_all_j, traj_batch.next_action)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout

    gamma = config["GAMMA"]

    e_i = targets_i - q_i
    e_j = jnp.where(true_terminal | is_timeout, 0.0, targets_j - q_j)

    weight_mc = 1.0 - gamma
    weight_dir = 0.5 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    laplacian_loss = weight_dir * jnp.mean((e_i - e_j) ** 2)

    value_loss = magnitude_loss + laplacian_loss
    scaled_loss = config.get("VF_COEF", 0.5) * value_loss

    delta_mag = jnp.abs(targets_i - q_i).mean()
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "delta_mag": delta_mag,
    }
    return scaled_loss, aux_info


# -------------------------------------------------------------
# 3. E(lambda) Geometric Jump State-Action Critic Loss
# -------------------------------------------------------------
def e_lambda_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    Sampled E(lambda) Geometric Jump critic loss on state-action pairs:
        e_i = G_t - Q(s_t, a_t)
        e_jump = G_{t+m} - Q(s_{t+m}, a_{t+m})
    Jump horizon m ~ Geometric(1 - gamma * lambda).
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_all_i = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_i = extract_q_taken(q_all_i, traj_batch.action)

    q_all_jump = network.apply(full_params, traj_batch.dest_obs, method=network.q_values)
    q_jump = extract_q_taken(q_all_jump, traj_batch.dest_action)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_jump = traj_batch.dest_targets

    gamma = config["GAMMA"]
    lmbda = config.get("E_LAMBDA", config.get("VALUE_LAMBDA", 0.8))
    gl = gamma * lmbda

    e_i = targets_i - q_i
    e_jump = targets_jump - q_jump

    is_absorbing = traj_batch.hit_true_terminal
    valid_jump = traj_batch.valid_jump

    diff_sq = jnp.where(
        is_absorbing,
        e_i ** 2,
        jnp.where(valid_jump, (e_i - e_jump) ** 2, e_i ** 2),
    )

    weight_mag = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    weight_dir = (0.5 * gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)

    magnitude_loss = weight_mag * jnp.mean(e_i ** 2)
    dirichlet_loss = weight_dir * jnp.mean(diff_sq)

    value_loss = magnitude_loss + dirichlet_loss
    scaled_loss = config.get("VF_COEF", 0.5) * value_loss

    delta_mag = jnp.abs(targets_i - q_i).mean()
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "delta_mag": delta_mag,
    }
    return scaled_loss, aux_info


Q_CRITIC_LOSS_FNS = {
    "sarsa": sarsa_critic_loss_fn,
    "td": sarsa_critic_loss_fn,
    "td_0": sarsa_critic_loss_fn,
    "td_lambda": sarsa_critic_loss_fn,
    "fitted": sarsa_critic_loss_fn,
    "mc": sarsa_critic_loss_fn,
    "e_0": e_0_q_critic_loss_fn,
    "e_0_q": e_0_q_critic_loss_fn,
    "e0": e_0_q_critic_loss_fn,
    "e": e_0_q_critic_loss_fn,
    "e_lambda": e_lambda_q_critic_loss_fn,
    "e_lambda_q": e_lambda_q_critic_loss_fn,
    "elambda": e_lambda_q_critic_loss_fn,
}


def get_q_critic_loss_fn(critic_type: str):
    key = critic_type.lower()
    if key not in Q_CRITIC_LOSS_FNS:
        raise ValueError(f"Unknown Q-critic type '{critic_type}'. Available: {list(Q_CRITIC_LOSS_FNS.keys())}")
    return Q_CRITIC_LOSS_FNS[key]
